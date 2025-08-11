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
        rf = torch.as_tensor(self.risk_factor,      device=device, dtype=dtype)

        pnl = y_pred[:, -k:, :] * y_true[:, -k:, :]

        r = pnl.reshape(-1)
        u = r.mean()
        if rf:
            u -= rf * r.var(unbiased=False)
        return -u
