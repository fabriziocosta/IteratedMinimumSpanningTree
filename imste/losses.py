"""Default differentiable losses for IMSTE."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _torch_module():
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "Differentiable IMSTE losses require PyTorch. Install it with "
            "`python -m pip install 'imste[gpu]'`."
        ) from error
    return torch


def log_attraction_loss(
    positive_squared_distances: torch.Tensor,
    edge_weights: torch.Tensor,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Mean rank-weighted log penalty on squared positive distances."""
    torch = _torch_module()
    return torch.mean(
        edge_weights * torch.log1p(positive_squared_distances + epsilon)
    )


def logistic_repulsion_loss(
    negative_squared_distances: torch.Tensor,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Negative-pair logistic loss, high below the margin and low above it."""
    torch = _torch_module()
    return torch.mean(
        torch.nn.functional.softplus(
            (margin - negative_squared_distances) / temperature
        )
    )
