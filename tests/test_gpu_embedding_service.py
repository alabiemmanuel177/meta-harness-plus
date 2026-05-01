"""Smoke tests for harness.embedding.GPUEmbeddingService.

The service exists to let a ThreadPoolExecutor over instances share a
single GPU-resident embedder safely. The tests verify:

  - Lock semantics: two concurrent threads serialize on the lock.
  - Correctness: per-thread results match what the underlying embedder
    would produce in a serial call.
  - Real-GPU smoke (skipped if no GPU / no sentence-transformers): 4
    threads concurrently calling embed_documents, no hang, results
    match the serial baseline.

Why threads-not-processes: V10_DESIGN.md §9 documents the multi-process
GPU-Hang verdict on AMD ROCm. ThreadPoolExecutor avoids that trap; the
GIL releases inside model.encode's C++/CUDA layer so I/O parallelism
isn't lost.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from harness.embedding import (
    EmbedderProtocol,
    GPUEmbeddingService,
    LocalEmbedder,
)


# ---------------------------------------------------------------------------
# Fake embedder used to assert lock semantics without touching the GPU.
# ---------------------------------------------------------------------------


class _SlowFakeEmbedder:
    """Mimics LocalEmbedder's API. Each embed_documents call sleeps a
    fixed amount; the test uses the wall-clock pattern to verify
    that two concurrent calls serialized through the service take
    ~2x the per-call time, not ~1x (which would mean the lock is
    broken)."""

    model_name = "fake-bge-large"
    DEFAULT_DOC_TRUNC_CHARS = 2048

    def __init__(self, per_call_seconds: float = 0.2):
        self.per_call_seconds = per_call_seconds
        self.in_flight = 0
        self.max_in_flight = 0
        self._observe_lock = threading.Lock()

    def embed_documents(self, texts, *, batch_size=None, truncate_chars=None):
        with self._observe_lock:
            self.in_flight += 1
            if self.in_flight > self.max_in_flight:
                self.max_in_flight = self.in_flight
        try:
            time.sleep(self.per_call_seconds)
            # Return deterministic embeddings keyed on text length so we
            # can assert correctness across threads.
            return [[float(len(t)), 0.0, 0.0] for t in texts]
        finally:
            with self._observe_lock:
                self.in_flight -= 1

    def embed_query(self, text):
        with self._observe_lock:
            self.in_flight += 1
            if self.in_flight > self.max_in_flight:
                self.max_in_flight = self.in_flight
        try:
            time.sleep(self.per_call_seconds)
            return [float(len(text)), 1.0, 0.0]
        finally:
            with self._observe_lock:
                self.in_flight -= 1


def test_service_serializes_concurrent_embed_documents():
    """Lock semantics: two threads calling embed_documents at the same
    time must not both be inside the underlying embedder
    simultaneously."""
    fake = _SlowFakeEmbedder(per_call_seconds=0.15)
    service = GPUEmbeddingService(fake)  # type: ignore[arg-type]

    def worker(payload: list[str]) -> list[list[float]]:
        return service.embed_documents(payload)

    # Four threads, each with a distinct text payload.
    payloads = [
        ["abc", "defg"],
        ["hijklm"],
        ["nop", "qrstuv", "wxyz"],
        ["AAA"],
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, p) for p in payloads]
        results = [f.result(timeout=10) for f in futures]

    # The fake observed at most ONE concurrent caller because of the
    # service's lock. If max_in_flight > 1, the lock is broken.
    assert fake.max_in_flight == 1, (
        f"GPUEmbeddingService allowed {fake.max_in_flight} concurrent "
        "GPU calls — the lock is not serializing access."
    )

    # And each thread got the right result for its payload.
    for payload, result in zip(payloads, results):
        assert len(result) == len(payload)
        for text, emb in zip(payload, result):
            assert emb == [float(len(text)), 0.0, 0.0], (
                f"Thread result for {text!r} mismatched: {emb!r}"
            )


def test_service_serializes_mixed_documents_and_query():
    """Threads alternating embed_documents / embed_query must also
    serialize cleanly — both call paths take the same lock."""
    fake = _SlowFakeEmbedder(per_call_seconds=0.1)
    service = GPUEmbeddingService(fake)  # type: ignore[arg-type]

    def call_docs(payload):
        return service.embed_documents(payload)

    def call_query(text):
        return service.embed_query(text)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(call_docs, ["foo", "bar"]),
            pool.submit(call_query, "alpha"),
            pool.submit(call_docs, ["baz"]),
            pool.submit(call_query, "beta-gamma"),
        ]
        for f in futures:
            f.result(timeout=10)

    assert fake.max_in_flight == 1, (
        "embed_query and embed_documents must share the same lock; "
        f"observed max_in_flight={fake.max_in_flight}"
    )


def test_service_passes_through_kwargs():
    """The service forwards batch_size and truncate_chars kwargs."""

    class _RecordingFake:
        model_name = "recording-fake"
        DEFAULT_DOC_TRUNC_CHARS = 2048

        def __init__(self):
            self.calls: list[dict] = []

        def embed_documents(self, texts, *, batch_size=None, truncate_chars=None):
            self.calls.append({
                "texts": list(texts),
                "batch_size": batch_size,
                "truncate_chars": truncate_chars,
            })
            return [[0.0] for _ in texts]

        def embed_query(self, text):
            return [0.0]

    fake = _RecordingFake()
    service = GPUEmbeddingService(fake)  # type: ignore[arg-type]
    service.embed_documents(["hello"], batch_size=128, truncate_chars=512)

    assert fake.calls == [{
        "texts": ["hello"],
        "batch_size": 128,
        "truncate_chars": 512,
    }]


# ---------------------------------------------------------------------------
# Real-GPU smoke test — skipped if sentence-transformers / GPU absent.
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _gpu_available(), reason="GPU + sentence-transformers required")
def test_real_gpu_4_threads_no_hang():
    """4 threads concurrently embedding short payloads on the real GPU.
    Asserts: no hang (10 s timeout per future), and per-thread results
    match a serial baseline."""
    serial = LocalEmbedder()
    # Warmup the model (model load itself is slow; want the per-call
    # part of the test to be the contended path).
    baseline = serial.embed_documents(["warmup"])
    assert baseline and len(baseline[0]) == 1024  # bge-large-en-v1.5 dim

    service = GPUEmbeddingService(serial)

    payloads = [
        ["tiny one"],
        ["another one", "with two"],
        ["solo three"],
        ["four", "items", "in", "this batch"],
    ]
    serial_expected = [serial.embed_documents(p) for p in payloads]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(service.embed_documents, p) for p in payloads]
        results = [f.result(timeout=30) for f in futures]

    # bge-large-en-v1.5 + normalize_embeddings=True is deterministic for
    # the same input shape; threaded output must match serial output
    # bit-for-bit on the first few coordinates.
    for serial_emb, threaded_emb in zip(serial_expected, results):
        assert len(serial_emb) == len(threaded_emb)
        for s_vec, t_vec in zip(serial_emb, threaded_emb):
            assert len(s_vec) == len(t_vec) == 1024
            # Same sequence in same order should produce identical
            # vectors. A small fp tolerance is fine on AMD.
            for s, t in zip(s_vec[:8], t_vec[:8]):
                assert abs(s - t) < 1e-5, (
                    f"threaded != serial: serial={s} threaded={t}"
                )
