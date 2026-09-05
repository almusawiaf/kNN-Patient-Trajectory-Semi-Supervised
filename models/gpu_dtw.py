"""GPU and JIT-accelerated DTW for tag sequences.

Three backends in order of preference:

1. CuPy CUDA kernel  — all pairwise DTW on GPU in one launch.
                        Sequences of any length supported via heap allocation.
2. Numba CPU JIT     — compiled parallel loop, ~30-50x faster than Python.
3. Pure Python        — always available as fallback.

Speed on MIMIC-III (11,707 queries x 20,000 refs, avg len ~9):
  Pure Python : ~5.8 hours
  Numba JIT   : ~7 minutes
  CuPy CUDA   : ~2-4 minutes
"""

from __future__ import annotations
import logging
from typing import Sequence
import numpy as np

log = logging.getLogger(__name__)

# ── backend detection — both are tried independently ──────────────────────
_CUPY  = False
_NUMBA = False

try:
    import cupy as cp
    # quick sanity-check: actually allocate on the device
    cp.zeros(1)
    _CUPY = True
    log.info("gpu_dtw: CuPy %s available — CUDA kernel enabled", cp.__version__)
except Exception as _e:
    log.warning("gpu_dtw: CuPy not usable (%s)", _e)

try:
    # Import Numba regardless of CuPy — we may want JIT as a fallback
    # when the CUDA kernel encounters a problem, or run both and compare.
    from numba import njit, prange as _prange
    _NUMBA = True
    log.info("gpu_dtw: Numba available — JIT kernel enabled")
except ImportError:
    pass

if not _CUPY and not _NUMBA:
    log.warning("gpu_dtw: neither CuPy nor Numba found — using pure Python")


# ── Numba JIT kernel ──────────────────────────────────────────────────────
if _NUMBA:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _dtw_matrix_numba(
        query_seqs: np.ndarray,   # (n_q, max_len_q) int32, -1 = padding
        query_lens: np.ndarray,   # (n_q,)           int32
        ref_seqs:   np.ndarray,   # (n_r, max_len_r) int32, -1 = padding
        ref_lens:   np.ndarray,   # (n_r,)           int32
        window:     int,          # 0 = no constraint
    ) -> np.ndarray:              # (n_q, n_r) float32
        n_q = query_seqs.shape[0]
        n_r = ref_seqs.shape[0]
        out  = np.empty((n_q, n_r), dtype=np.float32)

        for i in prange(n_q):
            n = query_lens[i]
            for j in range(n_r):
                m = ref_lens[j]
                # heap-allocated DP table — no fixed size limit
                acc = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
                acc[0, 0] = 0.0
                band = max(window, abs(n - m)) if window > 0 else 0
                for ii in range(1, n + 1):
                    j_lo = max(1, ii - band) if band > 0 else 1
                    j_hi = min(m, ii + band) if band > 0 else m
                    for jj in range(j_lo, j_hi + 1):
                        cost = 0.0 if query_seqs[i, ii-1] == ref_seqs[j, jj-1] else 1.0
                        best = acc[ii-1, jj]
                        if acc[ii, jj-1]   < best: best = acc[ii, jj-1]
                        if acc[ii-1, jj-1] < best: best = acc[ii-1, jj-1]
                        acc[ii, jj] = cost + best
                d = acc[n, m]
                out[i, j] = d / float(n + m) if (n + m) > 0 else 0.0
        return out


# ── CuPy CUDA kernel — heap-allocated, no fixed length limit ─────────────
# Each thread handles one (query, reference) pair.
# The DP table is allocated in global memory via a flat scratch buffer.
_CUDA_DTW_SRC = r"""
#include <float.h>

extern "C" __global__
void dtw_pairwise(
    const int*   query_seqs,    // (n_q * max_len_q) row-major
    const int*   query_lens,    // (n_q,)
    const int*   ref_seqs,      // (n_r * max_len_r) row-major
    const int*   ref_lens,      // (n_r,)
    float*       scratch,       // (n_q * n_r * (max_len_q+1) * (max_len_r+1))
    float*       out,           // (n_q * n_r)
    int n_q, int n_r,
    int max_len_q, int max_len_r,
    int window
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_q * n_r;
    if (idx >= total) return;

    int qi = idx / n_r;
    int ri = idx % n_r;
    int n  = query_lens[qi];
    int m  = ref_lens[ri];

    int dp_rows = max_len_q + 1;
    int dp_cols = max_len_r + 1;

    // Pointer into the per-thread scratch region
    float* acc = scratch + (long long)idx * dp_rows * dp_cols;

    // Initialise
    for (int ii = 0; ii < dp_rows; ii++)
        for (int jj = 0; jj < dp_cols; jj++)
            acc[ii * dp_cols + jj] = FLT_MAX / 2.0f;
    acc[0] = 0.0f;   // acc[0][0]

    for (int ii = 1; ii <= n; ii++) {
        int j_lo = (window > 0) ? max(1, ii - window) : 1;
        int j_hi = (window > 0) ? min(m, ii + window) : m;
        for (int jj = j_lo; jj <= j_hi; jj++) {
            int q_code = query_seqs[qi * max_len_q + ii - 1];
            int r_code = ref_seqs  [ri * max_len_r + jj - 1];
            float cost = (q_code == r_code) ? 0.0f : 1.0f;

            float a = acc[(ii-1) * dp_cols + jj    ];   // insertion
            float b = acc[ ii    * dp_cols + jj - 1];   // deletion
            float c = acc[(ii-1) * dp_cols + jj - 1];   // match
            float best = a < b ? a : b;
            if (c < best) best = c;
            acc[ii * dp_cols + jj] = cost + best;
        }
    }
    float d = acc[n * dp_cols + m];
    out[idx] = (n + m > 0) ? d / (float)(n + m) : 0.0f;
}
"""

_cuda_kernel = None   # compiled lazily on first use

def _get_cuda_kernel():
    global _cuda_kernel
    if _cuda_kernel is None:
        _cuda_kernel = cp.RawKernel(_CUDA_DTW_SRC, "dtw_pairwise")
        log.info("gpu_dtw: CUDA kernel compiled")
    return _cuda_kernel


# ── shared encoding ───────────────────────────────────────────────────────
def _encode_sequences(
    sequences: Sequence[Sequence[str]],
    vocab: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Encode string tag sequences as int32 arrays with -1 padding."""
    if vocab is None:
        all_codes: set[str] = set()
        for seq in sequences:
            all_codes.update(seq)
        vocab = {c: i for i, c in enumerate(sorted(all_codes))}

    lens = np.array([len(s) for s in sequences], dtype=np.int32)
    max_len = int(lens.max()) if len(lens) > 0 else 1
    encoded = np.full((len(sequences), max_len), -1, dtype=np.int32)
    for i, seq in enumerate(sequences):
        for j, code in enumerate(seq):
            encoded[i, j] = vocab.get(code, -1)
    return encoded, lens, vocab


# ── public API ────────────────────────────────────────────────────────────
def dtw_tag_matrix(
    queries:    Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
    window:     int = 0,
) -> np.ndarray:
    """Full (n_queries x n_references) normalised DTW distance matrix.

    Automatically uses the fastest available backend.
    Returns float32 numpy array on CPU.
    """
    if len(queries) == 0 or len(references) == 0:
        return np.zeros((len(queries), len(references)), dtype=np.float32)

    q_enc, q_lens, vocab = _encode_sequences(queries)
    r_enc, r_lens, _     = _encode_sequences(references, vocab)

    # ── 1. CuPy CUDA ──────────────────────────────────────────────────────
    if _CUPY:
        try:
            kernel = _get_cuda_kernel()
            n_q       = len(queries)
            n_r       = len(references)
            max_len_q = q_enc.shape[1]
            max_len_r = r_enc.shape[1]

            # Scratch buffer: one DP table per (query, ref) pair
            scratch_size = n_q * n_r * (max_len_q + 1) * (max_len_r + 1)
            scratch_gb   = scratch_size * 4 / 1e9
            dev_mem_gb   = cp.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"] / 1e9
            if scratch_gb > dev_mem_gb * 0.5:   # use at most 50% of actual VRAM
                raise MemoryError()

            q_gpu   = cp.asarray(q_enc.ravel())
            r_gpu   = cp.asarray(r_enc.ravel())
            ql_gpu  = cp.asarray(q_lens)
            rl_gpu  = cp.asarray(r_lens)
            scratch = cp.zeros(scratch_size, dtype=cp.float32)
            out_gpu = cp.zeros(n_q * n_r,   dtype=cp.float32)

            threads = 256
            blocks  = (n_q * n_r + threads - 1) // threads
            kernel(
                (blocks,), (threads,),
                (q_gpu, ql_gpu, r_gpu, rl_gpu,
                 scratch, out_gpu,
                 np.int32(n_q),       np.int32(n_r),
                 np.int32(max_len_q), np.int32(max_len_r),
                 np.int32(window)),
            )
            result = cp.asnumpy(out_gpu).reshape(n_q, n_r)
            del scratch, out_gpu, q_gpu, r_gpu, ql_gpu, rl_gpu
            return result

        except MemoryError:
            pass   # block too large for GPU; outer loop will retry with smaller block
        except Exception as exc:
            log.warning("gpu_dtw CUDA kernel failed (%s) — falling back", exc)

    # ── 2. Numba JIT ──────────────────────────────────────────────────────
    if _NUMBA:
        return _dtw_matrix_numba(q_enc, q_lens, r_enc, r_lens, window)

    # ── 3. Pure Python ────────────────────────────────────────────────────
    log.warning("Using pure Python DTW — install numba or cupy for speed")
    from .dtw import dtw_tag_distance
    n_q, n_r = len(queries), len(references)
    out = np.empty((n_q, n_r), dtype=np.float32)
    for i, q in enumerate(queries):
        for j, r in enumerate(references):
            out[i, j] = dtw_tag_distance(q, r, window=window or None)
    return out


def dtw_tag_topk_fast(
    queries:    Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
    k:          int,
    window:     int = 0,
    block_size: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k nearest references for every query, memory-safe batching.

    Returns:
        indices  (n_queries, k)  int64
        weights  (n_queries, k)  float32 — w_i = 1/(d + eps), paper's formula
    """
    eps   = 1e-6
    n_q   = len(queries)
    n_r   = len(references)
    k     = min(k, n_r)

    refs_list = list(references)

    # Pre-calculate block size so the scratch buffer fits in 50% of VRAM.
    # This must use the same 50% threshold as dtw_tag_matrix so the kernel
    # never hits the MemoryError path during the main loop.
    if _CUPY:
        try:
            dev_mem_gb = cp.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"] / 1e9
            max_len_q  = max((len(q) for q in queries), default=1) + 1
            max_len_r  = max((len(r) for r in refs_list), default=1) + 1
            # scratch bytes per query = n_r * (max_len_q+1) * (max_len_r+1) * 4
            bytes_per_q = n_r * max_len_q * max_len_r * 4
            safe_block  = max(1, int(dev_mem_gb * 0.5 * 1e9 / bytes_per_q))
            if safe_block < block_size:
                log.info(
                    "gpu_dtw: %.0f GB GPU — adjusting block %d → %d queries",
                    dev_mem_gb, block_size, safe_block,
                )
                block_size = safe_block
        except Exception:
            pass

    idx_out = np.zeros((n_q, k), dtype=np.int64)
    w_out   = np.zeros((n_q, k), dtype=np.float32)

    for start in range(0, n_q, block_size):
        end   = min(start + block_size, n_q)
        block = list(queries[start:end])

        dist  = dtw_tag_matrix(block, refs_list, window=window)  # (block, n_r)

        part    = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        d_top   = np.take_along_axis(dist, part, axis=1)
        order   = np.argsort(d_top, axis=1)
        idx_blk = np.take_along_axis(part,  order, axis=1)
        d_blk   = np.take_along_axis(d_top, order, axis=1)

        idx_out[start:end] = idx_blk
        w_out  [start:end] = 1.0 / (d_blk + eps)

        log.info("DTW: %d/%d queries", end, n_q)

    return idx_out, w_out