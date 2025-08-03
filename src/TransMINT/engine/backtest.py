from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .trainer import Trainer, TrainerConfig
from ..data_utils.datamodule import DataLoaderConfig
from ..utils import mkpath


class BacktestTickerResult:
    def __init__(self, ticker, times, dates, positions, target_returns, captured_returns):
        self.ticker = ticker
        self.times = times
        self.dates = dates
        self.positions = positions
        self.target_returns = target_returns
        self.captured_returns = captured_returns

    def daily_performance(self) -> 'DailyPerformance':
        from pandas import DataFrame
        from numpy import abs, diff, insert
        df = DataFrame({
            'dates': self.dates,
            'captured_returns': self.captured_returns,
            'volumes': abs(diff(insert(self.positions, 0, 0.0))),
        })

        agg_df = df.groupby('dates', sort=True)[['captured_returns', 'volumes']].sum().reset_index()

        agg_dates = agg_df['dates'].to_numpy()
        agg_returns = agg_df['captured_returns'].to_numpy()
        agg_vols = agg_df['volumes'].to_numpy()
        return DailyPerformance(agg_dates, agg_returns, agg_vols)

    @classmethod
    def concatenate(cls, results):
        if len(results) == 0:
            return None
        elif len(results) == 1:
            return results[0]
        ticker = results[0].ticker

        from numpy import concatenate
        times = []
        dates = []
        positions = []
        tgt_returns = []
        cap_returns = []

        for r in results:
            times.append(r.times)
            dates.append(r.dates)
            positions.append(r.positions)
            tgt_returns.append(r.target_returns)
            cap_returns.append(r.captured_returns)

        return cls(
            ticker,
            concatenate(times),
            concatenate(dates),
            concatenate(positions),
            concatenate(tgt_returns),
            concatenate(cap_returns),
        )


class DailyPerformance:
    def __init__(self, dates, returns, volumes):
        self.dates = dates
        self.returns = returns
        self.volumes = volumes
        assert len(self.dates) == len(self.returns) == len(self.volumes)

    def __len__(self):
        return len(self.dates)

    def __str__(self):
        d0, d1 = self.dates[0], self.dates[-1]
        return f'Performance({d0}->{d1}): Sharpe={self.sharpe_ratio:.2f}'

    @property
    def sharpe_ratio(self):
        ret = self.returns
        from math import sqrt
        return ret.mean() / ret.std() * sqrt(252)

    @property
    def turnover(self):
        from numpy import mean
        return mean(self.volumes)

    @classmethod
    def aggregate(cls, perfs: List['DailyPerformance'], method='mean'):
        if len(perfs) == 1:
            return perfs[0]
        dates = perfs[0].dates
        if method == 'mean':
            from numpy import mean
            agg = mean
        elif method == 'sum':
            from numpy import sum
            agg = sum
        else:
            raise ValueError(f'unknown aggregation method: {method}')
        ret = agg([p.returns for p in perfs], axis=0)
        vol = agg([p.volumes for p in perfs], axis=0)
        return DailyPerformance(dates, ret, vol)

    @classmethod
    def concatenate(cls, perfs):
        if len(perfs) == 1:
            return perfs[0]

        from numpy import concatenate as cat
        dates = []
        rets = []
        vols = []
        for p in perfs:
            dates.append(p.dates)
            rets.append(p.returns)
            vols.append(p.volumes)
        return DailyPerformance(cat(dates), cat(rets), cat(vols))


@dataclass
class BacktestRunConfig:
    train_start: Any
    test_start:  Any
    test_end:    Any

    data_cfg:    DataLoaderConfig
    trainer_cfg: TrainerConfig

    valid_start: Optional[Any] = None


class BacktestRun:
    def __init__(
            self,
            run_config: BacktestRunConfig,
            data_provider,
            store_path: Optional[str] = None,
    ):
        if run_config.valid_start is None:
            self.train_dataloader = data_provider.get_dataloader(
                run_config.data_cfg,
                run_config.train_start,
                run_config.test_start,
            )
            self.valid_dataloader = None
        else:
            self.train_dataloader = data_provider.get_dataloader(
                run_config.data_cfg,
                run_config.train_start,
                run_config.valid_start,
            )
            self.valid_dataloader = data_provider.get_dataloader(
                run_config.data_cfg,
                run_config.valid_start,
                run_config.test_start,
            )

        self.test_dataloader = data_provider.get_dataloader(
            run_config.data_cfg,
            run_config.test_start,
            run_config.test_end,
        )

        self.config = run_config

        self.trainer = None
        self.results = None
        self.performance = None

        if store_path is None:
            self.snapshot_path = None
            self.result_path = None
        else:
            mkpath(f'{store_path}/')
            self.snapshot_path = f'{store_path}/trainer.pt'
            self.result_path = f'{store_path}/results.bin'
            self._load_results()

    def _save_snapshot(self, trainer):
        if self.snapshot_path is not None:
            from torch import save
            save(trainer.snapshot, self.snapshot_path)

    def _load_snapshot(self):
        if self.snapshot_path is not None:
            from torch import load
            try:
                # Here we use the unsafe `weights_only`=False, since we use custom
                # data class Snapshot.
                return load(self.snapshot_path, weights_only=False, map_location='cpu')
            except FileNotFoundError:
                return None
        else:
            return None

    def _save_results(self):
        if self.result_path is not None:
            from pickle import dump
            with open(self.result_path, 'wb') as fh:
                dump((self.performance, self.results), fh)

    def _load_results(self):
        if self.result_path is not None:
            from pickle import load
            try:
                with open(self.result_path, 'rb') as fh:
                    self.performance, self.results = load(fh)
            except FileNotFoundError:
                self.results = None
                self.performance = None
        else:
            self.results = None
            self.performance = None

    def run(self):
        if self.results is not None:
            print('Backtest has completed already.')
            return self.results

        # train
        run_config = self.config
        m = Trainer(
            cfg=run_config.trainer_cfg,
            input_spec=run_config.data_cfg.input_spec,
            train_loader=self.train_dataloader,
            valid_loader=self.valid_dataloader,
            callbacks=[self._save_snapshot],
        )
        snapshot = self._load_snapshot()
        if snapshot is not None:
            print(f'resume from snapshot with {len(snapshot.trainer_state["epochs"])} epochs')
        m.initialize(snapshot)
        m.fit()

        results = defaultdict(list)

        for x, y in self.test_dataloader:
            pred_position = m.predict(x)

            pred_position = pred_position[:, -1, 0].numpy()
            target_return = y.targets[:, -1, 0].numpy()
            capture_return = pred_position * target_return

            res = BacktestTickerResult(
                y.ticker,
                y.details['time'],
                y.details['date'],
                pred_position,
                target_return,
                capture_return,
            )
            results[y.ticker].append(res)

        self.results = results = {
            k: BacktestTickerResult.concatenate(v)
            for k, v in results.items()
        }
        perfs = [r.daily_performance() for r in results.values()]
        self.performance = DailyPerformance.aggregate(perfs, method='mean')
        self._save_results()
        return results

    def ticker_performance(self):
        return {k: r.daily_performance() for k, r in self.results.items()}


@dataclass
class BacktestConfig:
    windows:     List[Tuple[Any, Any, Any, Any]]

    data_cfg:    DataLoaderConfig
    trainer_cfg: TrainerConfig

    def run_configs(self):
        for train_start, valid_start, test_start, test_end in self.windows:
            yield BacktestRunConfig(
                train_start=train_start,
                valid_start=valid_start,
                test_start=test_start,
                test_end=test_end,
                data_cfg=self.data_cfg,
                trainer_cfg=self.trainer_cfg,
            )


class Backtest:
    def __init__(self, config: BacktestConfig, data_provider, store_path=None):
        self.config = config
        runs = []
        for c in config.run_configs():
            if store_path is not None:
                run_store_path = f'{store_path}/{c.test_start}_{c.test_end}'
            else:
                run_store_path = None
            r = BacktestRun(c, data_provider, store_path=run_store_path)
            runs.append(r)
        self.runs = runs
        self.results = None
        self.performances = None

    def run(self):
        if self.results is not None:
            assert self.performances is not None
            return self.results

        results = defaultdict(list)
        perfs = []
        for r in self.runs:
            # print(f'Backtest: {r.config.test_start}->{r.config.test_end}')
            result = r.run()
            for k, v in result.items():
                results[k].append(v)
            perfs.append(r.performance)
        self.results = results = {
            k: BacktestTickerResult.concatenate(v)
            for k, v in results.items()
        }
        self.performances = perfs
        return results

    def performance(self):
        return DailyPerformance.concatenate(self.performances)

    def ticker_performance(self):
        return {k: r.daily_performance() for k, r in self.results.items()}
