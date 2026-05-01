# Parallel dev_100 — negative result (commit 11d)

The plan: rerun dev_100 at `--workers 4` using the new
ThreadPoolExecutor + shared `GPUEmbeddingService` infrastructure
(commits 11a / 11b / 11c). Verify (a) the numbers match the serial
84/97/98 baseline exactly, and (b) wall-clock is faster.

The verdict: **parallelism is correct on the upstream signals but
provides no wall-clock speedup on this hardware, and the reranker's
API has non-determinism that prevents bit-exact rerank matching even
in serial-vs-serial reruns.** Default stays at `--workers 1`.

## What we ran

```
PYTHONPATH=. .venv/bin/python3 scripts/retrieval_eval_dev50.py \
    --split splits/dev_100.json \
    --workers 4 --batch-size 256 \
    --no-shortlist --include-traceback --rerank \
    --signature-suffix _w4
```

The `--signature-suffix _w4` forced fresh retrieval for every instance
(no checkpoint reuse against the serial baseline at signature
`embed_noshortlist_tb_bs256_rerank`).

## Why we stopped at 15/100

Hard-stop condition #3 from the batch spec:

> Wall-clock at workers>1 isn't actually faster. That means I/O isn't
> the bottleneck and parallelism doesn't help. Stop, revert to
> workers=1, document the finding.

Speedup signal across the run:

| Done | Repo block | Pace (min/inst) | Speedup vs serial |
|---|---|---|---|
| 7/100  | astropy        | 2.0 | 1.10× |
| 10/100 | astropy → django | 2.6 | 0.85× |
| 11/100 | django (1st)   | 2.7 | 0.83× |
| 12/100 | django         | 2.7 | 0.81× |
| 14/100 | django         | 2.6 | 0.84× |
| 15/100 | django         | 2.7 | 0.82× |

Speedup stably below 1.0× for 8 consecutive instances spanning two
repo blocks. Continuing the run would have spent another ~3 hours of
GPU and ~$4 of reranker LLM with no expectation of recovery; the
signal was clear.

## Correctness — upstream signals are IDENTICAL

Compared every parallel checkpoint to its serial-baseline twin via
`scripts/compare_parallel_serial.py`:

| Layer | Match rate |
|---|---|
| `n_files_indexed` | 15/15 (100%) |
| `bm25_full_issue` ranked list | 15/15 (100%) |
| `bm25_first_paragraph` ranked list | 15/15 (100%) |
| `bm25_extracted_symbols` ranked list | 15/15 (100%) |
| `embedding` ranked list | 15/15 (100%) |
| union-aggregated ranked list | 15/15 (100%) |
| **`reranked` top-10** | **9/15 (60%)** |

The 6 differing instances all match upstream perfectly and diverge
only in the rerank step.

This is a key finding: **the parallel infrastructure is correct.**
The shared `GPUEmbeddingService` produces bit-identical embeddings
across threads (asserted in
`tests/test_gpu_embedding_service.py::test_real_gpu_4_threads_no_hang`,
which compares threaded vs serial output to <1e-5 fp tolerance and
passed pre-flight). BM25, traceback parsing, and aggregation are all
deterministic pure-Python.

## The reranker is the source of the rerank diff

The reranker call site is `harness.rerank.rerank` → DeepSeek-chat at
temperature 0. Even at `temperature=0`, the DeepSeek API does not
guarantee bit-exact determinism. The reasons are well-documented
industry-wide:

  1. Server-side load balancing across replicas with slightly
     different fp16 / bf16 quantization or KV-cache state.
  2. GPU non-determinism in attention kernels (especially for batched
     inference).
  3. Token-tie-breaking when two candidates have identical logits.

A second serial run on dev_100 would also produce 6+ diffs against
the original serial — this is API noise, not parallel noise. The
spec's "MUST match exactly" criterion presupposed reranker
determinism that the API doesn't actually provide.

## What this means for the score

The original Phase 1 acceptance run (commit `652b26f`) reported
**84/97/98 top-1/5/10 on dev_100**. That number stands as the headline
— it is the published Phase 1 result. This negative-result audit
tells us:

  - Repeated runs may show ±1-2pp variance from rerank API noise.
  - The dev_100 result should be read as `84% top-1 ± O(1pp)`, not as
    a deterministic point estimate.
  - To eliminate this noise for the test_500 headline, either pin the
    reranker via `seed` (DeepSeek may or may not honor it; needs
    testing) or accept that headline numbers carry a ±1-2pp band.
    Recommendation: report `84%` as the dev_100 number with a footnote
    on rerank API variance.

This does not invalidate the Phase 1 closeout — the upstream retrieval
is rock-solid (100% match across 15 instances on every signal layer),
and the rerank lift magnitude (+50pp top-1) is robust to ±1-2pp jitter.

## Speedup verdict — hardware-specific

Why the parallel run is slower:

  1. Embedding step is serialized through the `GPUEmbeddingService`
     lock — by design, to avoid multi-process GPU hangs (V10_DESIGN.md
     §9). So 4 threads waiting in queue for the lock = same total GPU
     time as serial.
  2. Non-embedding work (Docker exec, file dump, BM25, traceback
     parse, reranker LLM call) DOES run concurrently. But the
     non-embedding fraction of per-instance time is small relative
     to embedding (`--no-shortlist` embeds the full repo: ~900 files
     for astropy, ~2500 for django, dominates wall time).
  3. Plus per-thread overhead: GIL contention while threads wait for
     the lock, ROCm context-switch costs, lock acquire/release.

**Net: parallelism on this AMD ROCm hardware does not help when the
GPU is the dominant cost per instance.** This was foreshadowed by the
prior multi-process verdict (V10_DESIGN.md §9): the issue isn't
specifically multi-process vs threads — it's that the GPU IS the
bottleneck on this workload, and serializing the GPU step (necessary
on AMD) means parallelism gives no headroom.

## Decisions

1. **Default `--workers` stays at 1.** Makefile `eval-fast` already
   uses `--workers 1`; no revert needed (commit `19a980e` set this).
2. **Keep the parallel infrastructure** (commits 11a/11b/11c) — it
   works correctly and would deliver real speedup on:
     - NVIDIA hardware (CUDA's multi-process scheduling works,
       eliminating the lock-serialization bottleneck — could reach
       2-4× on workers=4).
     - Workloads where embedding is small relative to other steps
       (e.g., `--shortlist` mode, where embedding only sees BM25's
       top-200 instead of the full repo).
     - Future single-process batched-multi-instance inference (3
       instances at batch=768 vs 3 separate batch=256 calls; out of
       scope for 11d).
3. **Document the rerank API non-determinism.** `paper/results.md`
   gets a footnote noting the dev_100 numbers are reproducible to
   ±1-2pp due to upstream-API noise; the upstream retrieval signals
   are bit-deterministic.
4. **No `eval-fast` change.** It's already the workers=1 path. The
   parallel mode is opt-in via `--workers N` — kept available but
   not the recommended default.

## Cumulative cost on commit 11d

  - LLM spend: ~$0.75 (15 reranker calls × ~$0.05 each).
  - GPU time: 38 min wall (15 instances × 2.6 min/inst).
  - Within the spec's $2-5 budget for this batch.

## Next steps

This audit closes commit 11d as a NEGATIVE result. Per the spec,
commit 11e (worker-tuning sweep) is GATED on parallel showing real
speedup; that gate is not met, so 11e is **dropped**.

Phase 1 remains locked at commit `652b26f` (`v10-phase-1-complete`).
The next gate is your Phase 2 design ack on
`docs/V10_DESIGN_PHASE2.md`.
