"""GPU availability detection and memory helpers.

Call `gpu_info()` at the start of any GPU-accelerated script to confirm
the device and log memory. Call `to_gpu(array)` / `to_cpu(array)` to move
data without writing backend-specific code everywhere.
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)

def gpu_info() -> dict:
    """Return device info dict; logs a warning and returns empty if no GPU."""
    try:
        import cupy as cp
        dev = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        info = {
            "backend": "cupy",
            "name": props["name"].decode(),
            "total_memory_gb": props["totalGlobalMem"] / 1e9,
            "device_id": dev.id,
        }
        log.info("GPU: %(name)s  %(total_memory_gb).1f GB", info)
        return info
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            info = {
                "backend": "torch",
                "name": torch.cuda.get_device_name(0),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "device_id": 0,
            }
            log.info("GPU: %(name)s  %(total_memory_gb).1f GB", info)
            return info
    except Exception:
        pass
    log.warning("No GPU detected — running on CPU")
    return {}


def to_gpu(array):
    """Move a numpy / scipy array to GPU. Returns unchanged if no GPU."""
    try:
        import cupy as cp
        from scipy import sparse
        if sparse.issparse(array):
            import cupyx.scipy.sparse as csp
            return csp.csr_matrix(array)
        return cp.asarray(array)
    except Exception:
        return array


def to_cpu(array):
    """Move a GPU array back to numpy. Returns unchanged if already CPU."""
    try:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        try:
            import cupyx.scipy.sparse as csp
            if isinstance(array, csp.spmatrix):
                return array.get()
        except Exception:
            pass
    except Exception:
        pass
    return array
