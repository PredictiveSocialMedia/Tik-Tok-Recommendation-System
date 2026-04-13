"""Shared PyTorch device selection for training and inference.

Preference order: CUDA (NVIDIA) → MPS (Apple Metal) → CPU.

LightGBM GPU training remains CUDA-only; FAISS GPU remains CUDA-only (faiss-gpu).
"""

from __future__ import annotations

from typing import Literal

TorchDeviceName = Literal["cuda", "mps", "cpu"]


def preferred_torch_device() -> TorchDeviceName:
    """Return the best available device string for generic PyTorch workloads."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def torch_empty_cache() -> None:
    """Release cached blocks on CUDA or MPS after large tensor work."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            empty = getattr(torch.mps, "empty_cache", None)
            if callable(empty):
                empty()
    except Exception:
        pass
