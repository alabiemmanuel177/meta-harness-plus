"""Local embedding retrieval for Phase 1 stage 1b.

Per V10_DESIGN.md §11.2 and the user-acked Stage 1b constraint: the
default embedding model is local (BAAI/bge-large-en-v1.5 via
sentence-transformers). text-embedding-3-large is an opt-in config knob
in ``harness/config/models.yaml`` and NOT the default — anyone cloning
the repo and running ``make eval`` should not need an OpenAI key.

Reasoning the user gave:
  - Reproducibility: the local model is determinate and self-hosted.
  - Leaderboard eligibility: reviewers re-run the harness; an OpenAI
    dependency on the embedding step is friction.
  - Cost: ~$0 per inference on the local CPU/GPU, vs. $0.13/M tokens
    for text-embedding-3-large.

Embedding-3-large is available via the same interface for ablation
runs that need to test "would a stronger embedder change the answer."
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from typing import Optional


DEFAULT_LOCAL_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_LOCAL_DIM = 1024  # bge-large-en-v1.5 outputs 1024-dim embeddings


@dataclass
class EmbeddingResult:
    """Per-file embedding similarity to a query."""
    file_path: str
    cosine_similarity: float
    rank: int  # 1-indexed


class EmbedderProtocol:
    """Minimal interface every embedder must satisfy."""

    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Local embedder (default)
# ---------------------------------------------------------------------------


class LocalEmbedder(EmbedderProtocol):
    """Wraps sentence-transformers for local embedding inference.

    Lazy-loads the model on first use so importing the module is cheap.
    Caches embeddings keyed on (model_name, sha1(text)) in
    ``repo_cache/v10_embeddings/``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        *,
        device: str | None = None,
        cache_dir: Optional[str] = None,
        default_batch_size: int = 64,
    ):
        self.model_name = model_name
        # Auto-pick GPU if available; fall back to CPU. Explicit device=
        # override still wins (Phase 0+ runs may want device='cpu' for
        # deterministic comparison).
        if device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.cache_dir = cache_dir
        self.default_batch_size = default_batch_size
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for the local "
                    "embedder default. Install with `pip install "
                    "sentence-transformers`. To use a remote embedder "
                    "instead, see harness/config/models.yaml."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    DEFAULT_DOC_TRUNC_CHARS = 2048

    def embed_documents(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        truncate_chars: int | None = DEFAULT_DOC_TRUNC_CHARS,
    ) -> list[list[float]]:
        """Embed a list of documents.

        ``batch_size`` (default: self.default_batch_size = 64) controls
        the model.encode batch dimension. On the 32 GB AMD card we run
        eval at 256; CPU runs stay at 64.

        ``truncate_chars`` (default 2048) caps each document's length
        before tokenization. bge-large-en-v1.5's max sequence is 512
        tokens (~2K chars), so truncation is lossless for embedding
        quality.
        """
        model = self._ensure_model()
        if batch_size is None:
            batch_size = self.default_batch_size
        if truncate_chars is not None:
            texts = [t[:truncate_chars] if len(t) > truncate_chars else t for t in texts]
        embs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine sim becomes dot product
        )
        return [e.tolist() for e in embs]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        # bge-large-en-v1.5 recommends a query prefix for retrieval.
        # https://huggingface.co/BAAI/bge-large-en-v1.5
        prefix = "Represent this sentence for searching relevant passages: "
        emb = model.encode(
            [prefix + text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return emb.tolist()


# ---------------------------------------------------------------------------
# GPU embedding service — thread-safe wrapper for parallel orchestrators
# ---------------------------------------------------------------------------


class GPUEmbeddingService(EmbedderProtocol):
    """Thread-safe wrapper around a single ``LocalEmbedder``.

    Why this exists: V10's parallel eval (commit 11c) runs a
    ``ThreadPoolExecutor`` over instances. The instance worker is mostly
    I/O- and CPU-bound (sandbox setup, file dump, BM25, traceback
    parse, reranker call), but the embedding step touches the GPU. The
    AMD ROCm post-mortem (V10_DESIGN.md §9, 2026-04-30) showed that
    multi-PROCESS GPU contention deadlocks the card. A single-process
    multi-threaded design avoids the multi-process trap, but multiple
    Python threads racing into ``model.encode`` would still produce
    interleaved kernel launches.

    The fix is the simplest one that works: serialize the actual GPU
    call behind a ``threading.Lock``. The model is loaded once into
    VRAM; every thread that wants to embed waits its turn. The non-GPU
    work (file I/O, BM25, reranker LLM call) runs concurrently because
    those threads aren't holding the lock. CPython's GIL makes this
    safe — and ``model.encode`` releases the GIL inside the C++/CUDA
    kernel call, so blocking on the embedding lock doesn't starve the
    other threads from progressing on their I/O.

    Future optimization: an internal queue + multi-instance batching
    (3 instances at batch=768 instead of 3 separate batch=256 calls).
    Out of scope for commit 11a — the lock alone gives us the
    correctness + no-hang guarantee. Promote if microbenchmarks show
    >2x speedup.
    """

    def __init__(self, embedder: LocalEmbedder):
        self._embedder = embedder
        self._lock = threading.Lock()
        self.model_name = embedder.model_name

    def embed_documents(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        truncate_chars: int | None = LocalEmbedder.DEFAULT_DOC_TRUNC_CHARS,
    ) -> list[list[float]]:
        with self._lock:
            return self._embedder.embed_documents(
                texts,
                batch_size=batch_size,
                truncate_chars=truncate_chars,
            )

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            return self._embedder.embed_query(text)


# ---------------------------------------------------------------------------
# Remote embedder (opt-in via config)
# ---------------------------------------------------------------------------


class RemoteOpenAIEmbedder(EmbedderProtocol):
    """text-embedding-3-large via the OpenAI API. Opt-in only — needs
    OPENAI_API_KEY. Not the default; not used by ``make eval``.

    Defined here primarily so the API contract matches LocalEmbedder
    for ablation runs that compare local vs. remote embeddings.
    """

    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.model_name = model_name
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "openai SDK required for the remote embedder. Install "
                    "with `pip install openai`. The local embedder "
                    "(LocalEmbedder) is the default — switch back to it "
                    "via harness/config/models.yaml."
                ) from exc
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "RemoteOpenAIEmbedder needs OPENAI_API_KEY in env. "
                    "If you don't have an OpenAI key, switch to the local "
                    "embedder default in harness/config/models.yaml."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        client = self._ensure_client()
        # Batch in chunks of 64 to respect the API.
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            res = client.embeddings.create(model=self.model_name, input=chunk)
            out.extend([d.embedding for d in res.data])
        return out

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover
        client = self._ensure_client()
        res = client.embeddings.create(model=self.model_name, input=[text])
        return res.data[0].embedding


# ---------------------------------------------------------------------------
# Retrieval over file embeddings
# ---------------------------------------------------------------------------


def cosine_similarities_to(
    query_emb: list[float],
    doc_embs: list[list[float]],
) -> list[float]:
    """Pure-python cosine similarity. Embeddings from LocalEmbedder are
    pre-normalized (normalize_embeddings=True) so this is a dot product.
    For RemoteOpenAIEmbedder we'd want to L2-normalize first; for now
    we only ship local."""
    sims: list[float] = []
    for de in doc_embs:
        if len(de) != len(query_emb):
            raise ValueError(
                f"embedding dim mismatch: query {len(query_emb)} vs doc {len(de)}"
            )
        sims.append(sum(q * d for q, d in zip(query_emb, de)))
    return sims


def retrieve_by_embedding(
    embedder: EmbedderProtocol,
    *,
    file_paths: list[str],
    file_contents: list[str],
    query: str,
    top_k: int = 30,
) -> list[EmbeddingResult]:
    """End-to-end embedding retrieval. Caller-provided file content is
    embedded once; query is embedded per call. Returns top-K
    EmbeddingResult sorted by similarity descending.

    For very large repos, file_contents may need chunking. Phase 1
    starts with whole-file embedding; chunked embedding is a Stage 1b
    follow-up if the dev-50 retrieval eval shows the whole-file shape
    underperforms.
    """
    if len(file_paths) != len(file_contents):
        raise ValueError("file_paths and file_contents length mismatch")
    if not file_paths:
        return []
    doc_embs = embedder.embed_documents(file_contents)
    q_emb = embedder.embed_query(query)
    sims = cosine_similarities_to(q_emb, doc_embs)
    ranked = sorted(
        range(len(file_paths)),
        key=lambda i: (-sims[i], file_paths[i]),
    )
    out: list[EmbeddingResult] = []
    for rank, i in enumerate(ranked[:top_k], start=1):
        out.append(EmbeddingResult(
            file_path=file_paths[i],
            cosine_similarity=float(sims[i]),
            rank=rank,
        ))
    return out


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_LOCAL_DIM",
    "EmbeddingResult",
    "EmbedderProtocol",
    "LocalEmbedder",
    "GPUEmbeddingService",
    "RemoteOpenAIEmbedder",
    "cosine_similarities_to",
    "retrieve_by_embedding",
]
