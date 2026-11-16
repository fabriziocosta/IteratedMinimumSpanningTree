"""Backward-compatible MPS optimizer entry point."""

from ._optimizer_torch import optimize_embedding_torch


def optimize_embedding_mps(*args, **kwargs):
    """Optimize coordinates on MPS using the shared PyTorch GPU backend."""
    return optimize_embedding_torch(*args, device_type="mps", **kwargs)
