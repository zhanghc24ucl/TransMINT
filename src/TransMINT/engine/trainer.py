import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List

import torch
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
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

    # ----- Component‑specific kwargs ----------------------------------------
    model_params: Dict[str, Any] = field(default_factory=dict)
    loss_params: Dict[str, Any] = field(default_factory=dict)
    valid_loss_params: Dict[str, Any] = field(default_factory=dict)
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3})

    # ----- Scheduler --------------------------------------------------------
    scheduler_name: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    scheduler_step_on: str = "batch"  # can be either "batch" or "epoch"

    # ----- Runtime options --------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 30
    min_epochs: int = 0
    grad_clip_norm: Optional[float] = None  # e.g. 1.0 for stable training
    best_min_delta: float = 1e-4
    early_stop_patience: int = 5
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
    scheduler_state: Optional[Dict[str, Any]] = None


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
        self.scheduler = None

        self.criterion = cfg.loss_class(**cfg.loss_params).to(self.device)
        self.valid_loss = cfg.valid_loss_class(**cfg.valid_loss_params).to(self.device)

        self.snapshot: Optional[Snapshot] = None

    def _steps_per_epoch(self) -> int:
        try:
            return len(self.train_loader)
        except TypeError:
            raise RuntimeError("train_loader must implement __len__ for scheduler usage.")

    def _total_steps(self) -> int:
        return self._steps_per_epoch() * self.cfg.epochs

    def _current_lrs(self) -> List[float]:
        if self.optimizer is None:
            return []
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def _build_onecycle_scheduler(self):
        steps_per_epoch = self._steps_per_epoch()

        return OneCycleLR(
            self.optimizer,
            epochs=self.cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            **self.cfg.scheduler_params,
        )

    def _build_wm_cosine_scheduler(self):
        """ Params:
        * warmup_pct: float
        * min_lr_ratio: float
        """
        from math import cos, pi

        total_steps = self._total_steps()
        params = self.cfg.scheduler_params
        warmup_pct = params.get('warmup_pct', 0.1)
        min_lr_ratio = params.get('min_lr_ratio', 1e-6)
        warmup_steps = max(0, int(total_steps * warmup_pct))

        def lr_lambda(step: int):
            # linear warmup from 1/n_warmup_steps -> 1
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1.) / warmup_steps
            # cosine decay 1 -> min_lr_ratio
            t = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + cos(pi * t))  # 1 -> 0
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _build_scheduler(self):
        """Instantiate scheduler according to cfg. Must be called after optimizer creation."""
        name = self.cfg.scheduler_name
        if name is None:
            self.scheduler = None
            return

        step_on = self.cfg.scheduler_step_on
        if step_on not in ("batch", "epoch"):
            raise ValueError("scheduler_step_on must be 'batch' or 'epoch'")

        name = name.lower()
        if name == 'one_cycle':
            self.scheduler = self._build_onecycle_scheduler()
        elif name == 'warmup_cosine':
            self.scheduler = self._build_wm_cosine_scheduler()
        else:
            raise ValueError(f"Unknown scheduler name: {name}")

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
        self._build_scheduler()

        self.snapshot = snapshot
        set_random_state(snapshot.random_state)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)

        if self.scheduler:
            if snapshot.scheduler_state is None:
                raise RuntimeError("No scheduler state found in snapshot")
            self.scheduler.load_state_dict(snapshot.scheduler_state)

    def _init_snapshot(self):
        assert self.snapshot is None
        cfg = self.cfg

        set_seed(self.cfg.seed)
        self.model = cfg.model_class(self.input_spec, **cfg.model_params).to(self.device)
        self.optimizer = cfg.optimizer_class(
            self.model.parameters(), **cfg.optimizer_params
        )
        self._build_scheduler()

        model_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
        }
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        scheduler_state = copy.deepcopy(self.scheduler.state_dict()) \
            if self.scheduler else None

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
            scheduler_state=scheduler_state,
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

        if self.scheduler and self.cfg.scheduler_step_on == 'batch':
            self.scheduler.step()

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
        min_epochs = self.cfg.min_epochs
        n_epochs = self.cfg.epochs

        # load start state
        snapshot = self.snapshot
        trainer_state = snapshot.trainer_state
        finished = len(trainer_state['epochs'])

        if finished > 0:
            set_random_state(snapshot.random_state)
            self.model.load_state_dict(snapshot.model_state)
            self.optimizer.load_state_dict(snapshot.optimizer_state)

            if self.scheduler:
                if snapshot.scheduler_state is None:
                    raise RuntimeError("No scheduler state found in snapshot")
                self.scheduler.load_state_dict(snapshot.scheduler_state)

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
                    batch_pbar.set_postfix(train_loss=f'{running / cnt:.03e}', refresh=False)
            batch_pbar.close()

            if self.scheduler and self.cfg.scheduler_step_on == "epoch":
                self.scheduler.step()

            epoch_state['train_loss'] = train_loss = running / cnt
            if train_loss < trainer_state['best_train_loss']:
                trainer_state['best_train_loss'] = train_loss

            epoch_state['global_step'] = self._steps_per_epoch() * epoch
            epoch_state['lrs'] = lrs = self._current_lrs()

            progress_messages = {'train_loss': f'{train_loss:.03e}/{trainer_state["best_train_loss"]:.03e}'}
            if len(lrs) > 0:
                progress_messages['lr0'] = f'{lrs[0]:.03e}'

            snapshot.model_state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }
            snapshot.optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            if self.scheduler is not None:
                snapshot.scheduler_state = copy.deepcopy(self.scheduler.state_dict())

            if self.valid_loader is not None:
                eval_metrics = self.evaluate(self.valid_loader)
                epoch_state['val_loss'] = val_loss = eval_metrics['loss']
                epoch_state['tanh_derivative'] = eval_metrics['tanh_derivative']
                epoch_state['tanh_margin'] = eval_metrics['tanh_margin']

                if val_loss < trainer_state['best_val_metric'] - best_min_delta:
                    trainer_state['best_val_metric'] = val_loss
                    snapshot.best_model = {k: v.clone() for k, v in snapshot.model_state.items()}
                    trainer_state['wait_epochs'] = 0
                    progress_messages['status'] = 'update_best'
                else:
                    # check if early stop is needed
                    trainer_state['wait_epochs'] += 1
                    progress_messages['status'] = f'waiting({trainer_state["wait_epochs"]}/{patience})'
                progress_messages['valid_loss'] = f'{val_loss:.04f}/{trainer_state["best_val_metric"]:.04f}'
            epoch_pbar.set_postfix(**progress_messages)

            trainer_state['epochs'].append(epoch_state)
            snapshot.random_state = get_random_state()

            self._callback()

            if 0 < patience <= trainer_state['wait_epochs'] and epoch >= min_epochs:
                print(f"Early-Stopping triggered at epoch {epoch} "
                      f"(best val_loss={trainer_state['best_val_metric']:.4f})")
                break
        epoch_pbar.close()

        trainer_state['completed'] = True
        self._callback()

        if snapshot.best_model is not None:
            self.model.load_state_dict(snapshot.best_model)
        return self.model

    def n_parameters(self):
        model = self.model
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return trainable_params, non_trainable_params

    def lr_range_test(
            self,
            lr_start=1e-6, lr_end=1e-2,
            num_steps=3000, min_steps=0.1, smooth_beta=0.95, rel_worse=0.5,
            dataloader=None,
    ):
        from numpy import isfinite
        model, optimizer = self.model, self.optimizer
        device = self.device

        min_steps = int(min_steps * num_steps)
        init_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        init_opt = optimizer.state_dict()

        lr = lr_start
        mult = (lr_end / lr_start) ** (1 / max(1, num_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        lrs, losses = [], []
        avg_loss, best = 0.0, float("inf")
        step = 0

        dataloader = dataloader or self.train_loader
        for batch in dataloader:
            if step >= num_steps:
                break

            model.train()
            optimizer.zero_grad(set_to_none=True)

            x, y = batch
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = self.criterion(pred, y.targets)

            check_y = (1 - pred.detach().reshape(-1).square()).mean().item()
            print(check_y)

            step += 1
            avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss.item()
            smoothed = avg_loss / (1 - smooth_beta ** step)

            if not isfinite(smoothed):
                break

            lrs.append(lr)
            losses.append(smoothed)
            if smoothed < best:
                best = smoothed
            rw = (smoothed - best) / (abs(best) + 1e-8)
            if rw > rel_worse and step > min_steps:
                print('break', rw, rel_worse, smoothed, best)
                break

            loss.backward()
            optimizer.step()

            lr *= mult
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        model.load_state_dict(init_state)
        optimizer.load_state_dict(init_opt)
        return lrs, losses

    def _callback(self):
        for c in self.callbacks:
            try:
                c(self)
            except Exception as xe:
                print(f"Exception in callback {c.__class__.__name__}: {repr(xe)}")

    @torch.no_grad()
    def evaluate(self, dataloader: NamedInputDataLoader) -> Dict[str, float]:
        """Evaluate current model on a given dataloader, returning aggregated loss."""
        self.model.eval()
        preds, ys = [], []
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            preds.append(self.model(x))
            ys.append(y.targets)
        preds = torch.cat(preds)
        ys = torch.cat(ys)
        loss = self.valid_loss(preds, ys).item()
        deriv = (1 - preds.square()).mean().item()
        margin = (1 - preds.abs()).mean().item()
        return {
            'loss': loss,
            'tanh_derivative': deriv,
            'tanh_margin': margin,
        }

    @torch.no_grad()
    def predict(self, x: NamedInput) -> torch.Tensor:
        self.model.eval()
        return self.model(x.to(self.device)).cpu()
