"""Input-dependent band recalibration, with no separator implementation."""

import torch
from torch import nn

from .allocation import _integer


class BandRecalibration(nn.Module):
    """Shared scalar gating of features shaped [batch, bands, features, time].

    Temporal mean -> Linear -> ReLU -> Linear -> Sigmoid -> multiplication.
    The same MLP is applied independently to each band's temporal descriptor.
    No band is removed, and there is no Top-K selection or residual addition.

    The attribute name ``scorer`` and its layer indices are retained so an
    existing standalone BandImportanceGate state_dict can be loaded directly.
    """

    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.feature_dim = _integer(feature_dim, "feature_dim")
        hidden_dim = _integer(hidden_dim, "hidden_dim")
        self.scorer = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, return_gate=False):
        """Return weighted features, optionally with [batch, bands] weights."""
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("x must be a floating-point PyTorch tensor")
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, bands, features, time]")
        if any(size == 0 for size in x.shape):
            raise ValueError("all input dimensions must be nonempty")
        if x.shape[2] != self.feature_dim:
            raise ValueError(f"expected feature dimension {self.feature_dim}, got {x.shape[2]}")
        band_representation = x.mean(dim=-1)
        gate = self.scorer(band_representation)
        weighted_x = x * gate.unsqueeze(-1)
        if return_gate:
            return weighted_x, gate.squeeze(-1)
        return weighted_x


# Original research name, provided only for the standalone gate interface.
BandImportanceGate = BandRecalibration
