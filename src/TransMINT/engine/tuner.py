from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from .backtest import Backtest, BacktestConfig, TrainerConfig
from ..data_utils.datamodule import DataLoaderConfig
from ..utils import mkpath

FLOAT_RANGE_TYPE = Union[float, Tuple[float, float]]

@dataclass
class TunerConfig:
    expr_id: str

    windows: List[List[Tuple[Any, Any, Any, Any]]]

    data_config: DataLoaderConfig
    trainer_config: TrainerConfig

    batch_sizes: List[int] = field(default_factory=lambda: [64])
    n_epochs: List[int] = field(default_factory=lambda: [100])
    d_models: List[int] = field(default_factory=lambda: [32])

    lr_range: FLOAT_RANGE_TYPE = 1.0
    dropout_range: FLOAT_RANGE_TYPE = 0.0

    n_trials: int = 30

    store_db: Optional[str] = None


def _suggest_float_range(trial, name, range_: FLOAT_RANGE_TYPE, **kwargs):
    if isinstance(range_, float):
        return range_
    return trial.suggest_float(name, *range_, **kwargs)


def _finished(state: TrialState):
    # we want to rerun failed trial, but ignore pruned one
    return state == TrialState.COMPLETE or state == TrialState.PRUNED


class Tuner:
    def __init__(self, tuner_config: TunerConfig, data_provider):
        self.config = tuner_config
        self.data_provider = data_provider

        sampler = TPESampler(seed=tuner_config.trainer_config.seed)

        if tuner_config.store_db is not None:
            mkpath(tuner_config.store_db)
            storage = f'sqlite:///{tuner_config.store_db}'
        else:
            storage = None

        self.study = optuna.create_study(
            study_name=tuner_config.expr_id,
            direction="maximize",
            sampler=sampler,
            storage=storage,
            load_if_exists=True
        )

    def best_config(self) -> Tuple[DataLoaderConfig, TrainerConfig]:
        params = self.study.best_params
        return self._build_config(params)

    def _build_config(self, params) -> Tuple[DataLoaderConfig, TrainerConfig]:
        data_cfg = deepcopy(self.config.data_config)
        trainer_cfg = deepcopy(self.config.trainer_config)

        data_cfg.batch_size = params.get('batch_size', self.config.batch_sizes[0])
        trainer_cfg.epochs = params.get('epochs', self.config.n_epochs[0])

        trainer_cfg.optimizer_params['lr'] = params.get('lr', self.config.lr_range)

        trainer_cfg.model_params['dropout'] = params.get('dropout', self.config.dropout_range)
        trainer_cfg.model_params['d_model'] = params.get('d_model', self.config.d_models[0])

        return data_cfg, trainer_cfg

    def _objective(self, trial):
        params = dict(
            batch_size=trial.suggest_categorical('batch_size', self.config.batch_sizes),
            epochs=trial.suggest_categorical('n_epochs', self.config.n_epochs),
            lr=_suggest_float_range(trial, 'lr', self.config.lr_range),
            dropout=_suggest_float_range(trial, 'dropout', self.config.dropout_range),
            d_model=trial.suggest_categorical('d_model', self.config.d_models),
        )

        dcfg, tcfg = self._build_config(params)

        metrics = []
        for ws in self.config.windows:
            backtest = Backtest(BacktestConfig(ws, dcfg, tcfg), self.data_provider)
            backtest.run()
            perf = backtest.performance()
            metrics.append(perf.sharpe_ratio)

        from numpy import mean
        score = mean(metrics)
        return score

    def tune(self):
        # ---- Launch optimisation --------------------------------------------------------
        n_finished = len([r for r in self.study.trials if _finished(r.state)])
        n_needed = self.config.n_trials - n_finished
        if n_needed > 0:
            print(f'{n_finished} trials finished, launching {n_needed} trials...')
            self.study.optimize(self._objective, n_trials=n_needed, timeout=None)
        else:
            print('All trials are finished.')

        print()
        print("Best trial:")
        best = self.study.best_trial
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print(f"Validation metric: {best.value:.4f}")
        return self._build_config(best.params)
