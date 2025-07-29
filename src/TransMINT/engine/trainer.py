from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
from tqdm import tqdm

from ..data_utils.datamodule import NamedInputDataLoader
from ..data_utils.spec import InputSpec, NamedInput
from ..utils import set_seed

__all__ = ["TrainerConfig", "Trainer"]


@dataclass
class TrainerConfig:
    """Collection of hyper‑parameters and factory handles for a single run."""

    # ----- Core components --------------------------------------------------
    model_class: Callable[..., torch.nn.Module]
    loss_class: Callable[..., torch.nn.Module] = torch.nn.MSELoss
    valid_loss_class: Callable[..., torch.nn.Module] = torch.nn.MSELoss
    optimizer_class: Callable[..., torch.optim.Optimizer] = torch.optim.Adam

    # ----- Component‑specific kwargs ---------------------------------------
    model_params: Dict[str, Any] = field(default_factory=dict)
    loss_params: Dict[str, Any] = field(default_factory=dict)
    valid_loss_params: Dict[str, Any] = field(default_factory=dict)
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3})

    # ----- Runtime options --------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 10
    grad_clip_norm: Optional[float] = None  # e.g. 1.0 for stable training
    best_min_delta: float = 1e-4
    early_stop_patience: int = 25
    seed: int = 42

    # ----- Logging & misc ---------------------------------------------------
    log_interval: int = 50  # batches


class Trainer:
    def __init__(
            self,
            cfg: TrainerConfig,
            input_spec: InputSpec,
            train_loader: NamedInputDataLoader,
            valid_loader: Optional[NamedInputDataLoader] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        set_seed(cfg.seed)

        self.train_loader = train_loader

        self.model = cfg.model_class(input_spec, **cfg.model_params).to(self.device)
        self.criterion = cfg.loss_class(**cfg.loss_params).to(self.device)
        self.optimizer = cfg.optimizer_class(
            self.model.parameters(), **cfg.optimizer_params
        )

        self.valid_loader = valid_loader
        self.valid_loss = cfg.valid_loss_class(**cfg.valid_loss_params).to(self.device)

        self.best_val_metric: float = float("inf")  # lower is better (loss)
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def _train_step(self, batch) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        x, y = batch  # unpack user‑provided tuple
        x, y = x.to(self.device), y.to(self.device)

        preds = self.model(x)
        loss = self.criterion(preds, y.targets)
        loss.backward()

        if self.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

        self.optimizer.step()
        return loss.item()

    def fit(self) -> torch.nn.Module:
        """Run the training loop and return the model loaded with *best* weights."""
        set_seed(self.cfg.seed)

        patience = self.cfg.early_stop_patience
        wait_epochs = 0
        best_min_delta = self.cfg.best_min_delta
        n_epochs = self.cfg.epochs

        from numpy import inf
        best_train_loss = inf
        epoch_pbar = tqdm(range(1, n_epochs + 1), desc="Epochs", position=0, leave=True)
        for epoch in epoch_pbar:
            running = 0.0
            cnt = 0

            batch_pbar = tqdm(self.train_loader, desc="Batches", position=1, leave=False)
            for batch in batch_pbar:
                running += self._train_step(batch)
                cnt += 1

                if cnt % 100 == 0:
                    batch_pbar.set_postfix(train_loss=f'{running / cnt:.03f}', refresh=False)
            batch_pbar.close()

            train_loss = running / cnt
            if train_loss < best_train_loss:
                best_train_loss = train_loss

            progress_messages = {'train_loss': f'{train_loss:.04f}/{best_train_loss:.04f}'}

            if self.valid_loader is not None:
                val_loss = self.evaluate(self.valid_loader)

                if val_loss < self.best_val_metric - best_min_delta:
                    self.best_val_metric = val_loss
                    self.best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    wait_epochs = 0
                    progress_messages['status'] = 'update_best'
                else:
                    # check if early stop is needed
                    wait_epochs += 1
                    progress_messages['status'] = f'waiting({wait_epochs}/{patience})'
                progress_messages['valid_loss'] = f'{val_loss:.04f}/{self.best_val_metric:.04f}'
            epoch_pbar.set_postfix(**progress_messages)

            if 0 < patience <= wait_epochs:
                print(f"Early-Stopping triggered at epoch {epoch} "
                      f"(best val_loss={self.best_val_metric:.4f})")
                break
        epoch_pbar.close()

        # ----- Load best weights before returning -------------------------
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        return self.model

    @torch.no_grad()
    def evaluate(self, dataloader: NamedInputDataLoader) -> float:
        """Evaluate current model on a given dataloader, returning aggregated loss."""
        self.model.eval()
        preds, ys = [], []
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            preds.append(self.model(x))
            ys.append(y.targets)
        preds = torch.cat(preds)
        ys = torch.cat(ys)
        return self.valid_loss(preds, ys).item()

    @torch.no_grad()
    def predict(self, x: NamedInput) -> torch.Tensor:
        self.model.eval()
        return self.model(x.to(self.device)).cpu()
