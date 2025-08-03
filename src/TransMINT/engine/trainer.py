import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List

import torch
from tqdm import tqdm

from ..data_utils.datamodule import NamedInputDataLoader
from ..data_utils.spec import InputSpec, NamedInput
from ..utils import set_seed, set_random_state, get_random_state, set_deterministic_flags

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


@dataclass
class Snapshot:
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, Any]

    # * seed
    # * random_state
    # * torch_state
    random_state: Dict[str, Any]

    # * epochs: List
    # * completed: bool
    trainer_state: Dict[str, Any]

    best_model: Optional[Dict[str, torch.Tensor]] = None


class Trainer:
    def __init__(
            self,
            cfg: TrainerConfig,
            input_spec: InputSpec,
            train_loader: NamedInputDataLoader,
            valid_loader: Optional[NamedInputDataLoader] = None,
            callbacks: List[Callable[['Trainer'], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.input_spec = input_spec
        self.device = torch.device(cfg.device)

        set_deterministic_flags()

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.callbacks = callbacks or []

        self.model = None
        self.optimizer = None

        self.criterion = cfg.loss_class(**cfg.loss_params).to(self.device)
        self.valid_loss = cfg.valid_loss_class(**cfg.valid_loss_params).to(self.device)

        self.snapshot: Optional[Snapshot] = None

    def initialize(self, snapshot: Optional[Snapshot] = None):
        if snapshot is None:
            self._init_snapshot()
        else:
            self._set_snapshot(snapshot)

    def _set_snapshot(self, snapshot):
        cfg = self.cfg
        self.model = cfg.model_class(self.input_spec, **cfg.model_params).to(self.device)
        self.optimizer = cfg.optimizer_class(
            self.model.parameters(), **cfg.optimizer_params
        )

        self.snapshot = snapshot
        set_random_state(snapshot.random_state)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)

    def _init_snapshot(self):
        assert self.snapshot is None
        cfg = self.cfg

        set_seed(self.cfg.seed)
        self.model = cfg.model_class(self.input_spec, **cfg.model_params).to(self.device)
        self.optimizer = cfg.optimizer_class(
            self.model.parameters(), **cfg.optimizer_params
        )

        model_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
        }
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())

        from numpy import inf
        trainer_state = {
            'epochs': [],
            'completed': False,
            'wait_epochs': 0,
            'best_train_loss': inf,
            'best_val_metric': inf,
        }
        self.snapshot = Snapshot(
            model_state=model_state,
            optimizer_state=optimizer_state,
            trainer_state=trainer_state,
            random_state=get_random_state(),
        )

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
        assert self.snapshot is not None, 'Please initialize first.'

        if self.snapshot.trainer_state['completed']:
            if self.snapshot.best_model is not None:
                self.model.load_state_dict(self.snapshot.best_model)
            return self.model

        patience = self.cfg.early_stop_patience
        best_min_delta = self.cfg.best_min_delta
        n_epochs = self.cfg.epochs

        # load start state
        snapshot = self.snapshot
        trainer_state = snapshot.trainer_state
        finished = len(trainer_state['epochs'])

        if finished > 0:
            set_random_state(snapshot.random_state)
            self.model.load_state_dict(snapshot.model_state)
            self.optimizer.load_state_dict(snapshot.optimizer_state)

        epoch_pbar = tqdm(range(finished, n_epochs),
                          initial=finished, total=n_epochs,
                          desc="Epochs", position=0, leave=True)
        for epoch in epoch_pbar:
            epoch += 1  # change range from [0, n-1] to [1, n] for human readability

            running = 0.0
            cnt = 0
            epoch_state = {'id': epoch}

            batch_pbar = tqdm(self.train_loader, desc="Batches", position=1, leave=False)
            for batch in batch_pbar:
                running += self._train_step(batch)
                cnt += 1

                if cnt % 100 == 0:
                    batch_pbar.set_postfix(train_loss=f'{running / cnt:.03f}', refresh=False)
            batch_pbar.close()

            epoch_state['train_loss'] = train_loss = running / cnt
            if train_loss < trainer_state['best_train_loss']:
                trainer_state['best_train_loss'] = train_loss

            progress_messages = {'train_loss': f'{train_loss:.04f}/{trainer_state['best_train_loss']:.04f}'}

            snapshot.model_state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }
            snapshot.optimizer_state = copy.deepcopy(self.optimizer.state_dict())

            if self.valid_loader is not None:
                epoch_state['val_loss'] = val_loss = self.evaluate(self.valid_loader)

                if val_loss < trainer_state['best_val_metric'] - best_min_delta:
                    trainer_state['best_val_metric'] = val_loss
                    snapshot.best_model = {k: v.clone() for k, v in snapshot.model_state.items()}
                    trainer_state['wait_epochs'] = 0
                    progress_messages['status'] = 'update_best'
                else:
                    # check if early stop is needed
                    trainer_state['wait_epochs'] += 1
                    progress_messages['status'] = f'waiting({trainer_state['wait_epochs']}/{patience})'
                progress_messages['valid_loss'] = f'{val_loss:.04f}/{trainer_state['best_val_metric']:.04f}'
            epoch_pbar.set_postfix(**progress_messages)

            trainer_state['epochs'].append(epoch_state)
            snapshot.random_state = get_random_state()

            self._callback()

            if 0 < patience <= trainer_state['wait_epochs']:
                print(f"Early-Stopping triggered at epoch {epoch} "
                      f"(best val_loss={trainer_state['best_val_metric']:.4f})")
                break
        epoch_pbar.close()

        trainer_state['completed'] = True
        self._callback()

        if snapshot.best_model is not None:
            self.model.load_state_dict(snapshot.best_model)
        return self.model

    def _callback(self):
        for c in self.callbacks:
            try:
                c(self)
            except Exception as xe:
                print(f"Exception in callback {c.__class__.__name__}: {repr(xe)}")

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
