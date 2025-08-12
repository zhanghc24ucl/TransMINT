from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class SharpeLoss(nn.Module):
    """
    Negative Sharpe-ratio loss.
    """

    def __init__(
            self,
            output_steps: Optional[int] = None,
            epsilon: float = 1e-9,
    ) -> None:
        super().__init__()
        self.output_steps = output_steps
        self.epsilon      = epsilon

    # ---------------------------- forward ------------------------------- #
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        y_pred, y_true : (B, T, D)
        """
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have identical shapes")

        B, T, D = y_pred.shape
        k = self.output_steps or T

        pnl = y_pred[:, -k:, :] * y_true[:, -k:, :]

        r = pnl.reshape(-1)
        mean = r.mean()
        var  = r.var(unbiased=False)
        return -(mean / (var.add(self.epsilon).sqrt()))


class UtilityLoss(nn.Module):

    def __init__(
            self,
            output_steps: Optional[int] = None,
            risk_factor: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.output_steps      = output_steps
        self.risk_factor       = risk_factor or 0.0

        # ---------------------------- forward ------------------------------- #
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        y_pred, y_true : (B, T, D)
        """
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have identical shapes")

        B, T, D = y_pred.shape
        k = self.output_steps or T

        device, dtype = y_pred.device, y_pred.dtype
        rf = torch.as_tensor(self.risk_factor, device=device, dtype=dtype)

        pnl = y_pred[:, -k:, :] * y_true[:, -k:, :]

        r = pnl.reshape(-1)
        u = r.mean()
        if self.risk_factor:
            u -= rf * r.var(unbiased=False)
        return -u


class DecayedUtilityLoss(nn.Module):
    def __init__(
            self,
            output_steps: Optional[int] = None,
            risk_factor: Optional[float] = None,     # None => mean-only utility
            expdecay_factor: Optional[float] = None, # λ in (0,1]; None => uniform weights
            eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.output_steps    = output_steps
        self.risk_factor     = risk_factor
        self.expdecay_factor = expdecay_factor
        self.eps             = eps

    def _build_weights(self, k: int, device, dtype):
        """
        Build time-axis weights with recent_is_heavier=True and normalize_weights=True.
        Definition: w_t ∝ λ^(k-1 - t), for t=0..k-1; the most recent step t=k-1 has weight 1.
        """
        if self.expdecay_factor is None:
            w = torch.ones(k, device=device, dtype=dtype)
        else:
            lam = torch.as_tensor(float(self.expdecay_factor), device=device, dtype=dtype)
            t = torch.arange(k, device=device, dtype=dtype)  # 0..k-1
            exponents = (k - 1) - t
            w = lam ** exponents
        # Normalize to probability weights (sum to 1)
        w = w / (w.sum() + self.eps)
        return w.view(1, k, 1)  # shape (1,k,1) for broadcasting with (B,k,D)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        y_pred, y_true : (B, T, D)
        """
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have identical shapes")

        B, T, D = y_pred.shape
        k = self.output_steps or T

        device, dtype = y_pred.device, y_pred.dtype

        pnl = y_pred[:, -k:, :] * y_true[:, -k:, :]   # (B,k,D)
        w = self._build_weights(k, device, dtype)     # (1,k,1), sum(w)=1

        # Weighted mean over the time axis -> (B,D)
        mu = (pnl * w).sum(dim=1)                     # (B,D)

        if self.risk_factor is None:
            u = mu                                    # mean-only utility
        else:
            rf = torch.as_tensor(float(self.risk_factor), device=device, dtype=dtype)
            diff = pnl - mu.unsqueeze(1)              # (B,k,D)
            # Weighted variance using the same weights
            var = (w * diff * diff).sum(dim=1)        # (B,D)
            u = mu - rf * var

        # Aggregate to a scalar (mean over batch and target dimensions)
        return -u.mean()
