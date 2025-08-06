from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class SharpeLoss(nn.Module):
    """
    Negative Sharpe-ratio loss with optional linear transaction cost and
    quadratic slippage penalties.

    Modes
    -----
    * Full-sequence   → output_steps=None
    * Horizon-k       → output_steps=k (k ≥ 1)
        * PNL      : weights   w_{-k ... -1}  ×  returns r_{-k ... -1}
        * Initial  : weights   w_{-k-1}       ×  returns r_{-k-1}
        * Turnover : k trades  Δw_{-k ... -1}

    Aggregation
    -----------
    global_sharpe=True (default)  – one Sharpe on the pooled batch
    global_sharpe=False           – Sharpe per sample → `reduction`
    """

    def __init__(
        self,
        output_steps: Optional[int] = None,
        cost_factor: Optional[float] = None,
        slippage_factor: Optional[float] = None,
        eps: float = 1e-9,
        reduction: str = "mean",
        global_sharpe: bool = True,
    ) -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        if output_steps is None:
            assert cost_factor is None and slippage_factor is None

        self.output_steps      = output_steps
        self.cost_factor       = cost_factor or 0.0
        self.slippage_factor   = slippage_factor or 0.0
        self.eps               = eps
        self.reduction         = reduction
        self.global_sharpe     = global_sharpe

    # --------------------------- helpers -------------------------------- #
    def _neg_sharpe(self, r: Tensor, dim: Optional[int]) -> Tensor:
        """Return *negative* Sharpe."""
        if dim is None:
            r = r.reshape(-1)
            mean = r.mean()
            var  = r.var(unbiased=False)
        else:
            mean = r.mean(dim=dim)
            var  = r.var(dim=dim, unbiased=False)

        return -(mean / (var.add(self.eps).sqrt()))

    def _reduce(self, x: Tensor) -> Tensor:
        if self.reduction == "mean":
            return x.mean()
        if self.reduction == "sum":
            return x.sum()
        return x          # "none"

    # ---------------------------- forward ------------------------------- #
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        y_pred, y_true : (B, T, D)
        """
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have identical shapes")

        B, T, D = y_pred.shape
        k = self.output_steps

        device, dtype = y_pred.device, y_pred.dtype
        c1 = torch.as_tensor(self.cost_factor,      device=device, dtype=dtype)
        c2 = torch.as_tensor(self.slippage_factor,  device=device, dtype=dtype)

        # ------------------- PNL & trading costs ------------------- #
        if k is None:                                   # full-sequence
            pnl = y_pred * y_true                       # (B,T,D)
        else:                                           # horizon-k
            if T < k + 1:
                raise ValueError(f"T={T} must be ≥ k+1={k+1}")

            # PNL from the k + 1 weights/returns
            pnl = (y_pred[:, -k-1:, :] * y_true[:, -k-1:, :]).sum(dim=1)  # (B,D)

            if c1 or c2:
                # k trades inside the horizon
                vol = torch.diff(y_pred[:, -k-1:, :], dim=1)                  # (B,k,D)
                pnl = pnl - c1 * vol.abs().sum(dim=1)
                pnl = pnl - c2 * vol.square().sum(dim=1)

        # ---------------------- Sharpe aggregation ---------------- #
        if self.global_sharpe:
            return self._neg_sharpe(pnl, dim=None)       # scalar

        # per-sample Sharpe → reduction
        loss_vec = self._neg_sharpe(pnl.reshape(B, -1), dim=1)
        return self._reduce(loss_vec)


class UtilityLoss(nn.Module):

    def __init__(
        self,
        output_steps: Optional[int] = None,
        risk_factor: Optional[float] = None,
        cost_factor: Optional[float] = None,
        slippage_factor: Optional[float] = None,
    ) -> None:
        super().__init__()

        if output_steps is None:
            assert cost_factor is None and slippage_factor is None

        self.output_steps      = output_steps
        self.risk_factor       = risk_factor or 0.0
        self.cost_factor       = cost_factor or 0.0
        self.slippage_factor   = slippage_factor or 0.0

        # ---------------------------- forward ------------------------------- #
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        y_pred, y_true : (B, T, D)
        """
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have identical shapes")

        B, T, D = y_pred.shape
        k = self.output_steps

        device, dtype = y_pred.device, y_pred.dtype
        rf = torch.as_tensor(self.risk_factor,      device=device, dtype=dtype)
        c1 = torch.as_tensor(self.cost_factor,      device=device, dtype=dtype)
        c2 = torch.as_tensor(self.slippage_factor,  device=device, dtype=dtype)

        # ------------------- PNL & trading costs ------------------- #
        c = None
        if k is None:  # full-sequence
            r = y_pred * y_true  # (B,T,D)
        else:  # horizon-k
            if T < k + 1:
                raise ValueError(f"T={T} must be ≥ k+1={k + 1}")

            # PNL from the k + 1 weights/returns
            r = (y_pred[:, -k - 1:, :] * y_true[:, -k - 1:, :])  # (B,k+1,D)

            if c1 or c2:
                # k trades inside the horizon
                vol = torch.diff(y_pred[:, -k - 1:, :], dim=1)  # (B,k,D)
                c = c1 * vol.abs() + c2 * vol.square()  # (B,k,D)

        # r: (B, k+1, D) or (B, T, D)
        # c: (B, k, D) or None

        u = r.mean()
        if rf:
            u = u - rf * r.var(unbiased=False)
        if c is not None:
            u = u - c.mean()
        return -u
