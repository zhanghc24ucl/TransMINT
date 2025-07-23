from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Union

import optuna
from optuna.samplers import TPESampler

from .backtest import Backtest, BacktestConfig, TrainerConfig
from ..data_utils.datamodule import DataLoaderConfig

FLOAT_RANGE_TYPE = Union[float, Tuple[float, float]]

@dataclass
class TunerConfig:
    windows: List[List[Tuple[Any, Any, Any, Any]]]

    data_config: DataLoaderConfig
    trainer_config: TrainerConfig

    batch_sizes: List[int] = field(default_factory=lambda: [64])
    n_epochs: List[int] = field(default_factory=lambda: [100])
    d_models: List[int] = field(default_factory=lambda: [32])

    lr_range: FLOAT_RANGE_TYPE = 1.0
    dropout_range: FLOAT_RANGE_TYPE = 0.0

    n_trials: int = 30


def _suggest_float_range(trial, name, range_: FLOAT_RANGE_TYPE, **kwargs):
    if isinstance(range_, float):
        return range_
    return trial.suggest_float(name, *range_, **kwargs)


class Tuner:
    def __init__(self, tuner_config: TunerConfig, data_provider):
        self.config = tuner_config
        self.data_provider = data_provider

    def _suggest_config(self, trial) -> Tuple[DataLoaderConfig, TrainerConfig]:
        data_cfg = deepcopy(self.config.data_config)
        trainer_cfg = deepcopy(self.config.trainer_config)

        data_cfg.batch_size = trial.suggest_categorical('batch_size', self.config.batch_sizes)
        trainer_cfg.epochs = trial.suggest_categorical('n_epochs', self.config.n_epochs)

        trainer_cfg.optimizer_params['lr'] = _suggest_float_range(trial, 'lr', self.config.lr_range)

        trainer_cfg.model_params['dropout'] = _suggest_float_range(trial, 'dropout', self.config.dropout_range)
        trainer_cfg.model_params['d_model'] = trial.suggest_categorical('d_model', self.config.d_models)

        return data_cfg, trainer_cfg

    def _objective(self, trial):
        dcfg, tcfg = self._suggest_config(trial)

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
        cfg = self.config
        sampler = TPESampler(seed=cfg.trainer_config.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # ---- Launch optimisation --------------------------------------------------------
        study.optimize(self._objective, n_trials=self.config.n_trials, timeout=None)  # set timeout if desired

        print()
        print("Best trial:")
        best = study.best_trial
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print(f"Validation metric: {best.value:.4f}")
