from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .trainer import Trainer, TrainerConfig
from ..data_utils.datamodule import DataLoaderConfig


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
        df = DataFrame({
            'dates': self.dates,
            'captured_returns': self.captured_returns,
        })

        agg_df = df.groupby('dates', sort=True)['captured_returns'].sum().reset_index()

        agg_dates = agg_df['dates'].to_numpy()
        agg_returns = agg_df['captured_returns'].to_numpy()
        return DailyPerformance(agg_dates, agg_returns)

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
    def __init__(self, dates, returns):
        self.dates = dates
        self.returns = returns
        assert len(self.dates) == len(self.returns)

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

    def rescaled_returns(self, expected_vol):
        from math import sqrt
        ret = self.returns
        ann_vol = ret.std() * sqrt(252)
        norm_ret = expected_vol * ret / ann_vol
        return norm_ret

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
        return DailyPerformance(dates, ret)

    @classmethod
    def concatenate(cls, perfs, expected_vol=None):
        if len(perfs) == 1 and expected_vol is None:
            return perfs[0]

        from numpy import concatenate as cat
        dates = []
        rets = []
        for p in perfs:
            dates.append(p.dates)
            r = p.returns if expected_vol is None else \
                p.rescaled_returns(expected_vol)
            rets.append(r)
        return DailyPerformance(cat(dates), cat(rets))


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

        self.trainer = Trainer(
            cfg=run_config.trainer_cfg,
            input_spec=run_config.data_cfg.input_spec,
            train_loader=self.train_dataloader,
            valid_loader=self.valid_dataloader,
        )
        self.results = None
        self.performance = None

    def run(self):
        if self.results is not None:
            return self.results

        # train
        m = self.trainer
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
        return results


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
    def __init__(self, config: BacktestConfig, data_provider):
        self.config = config
        self.runs = [BacktestRun(c, data_provider) for c in config.run_configs()]
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

    def performance(self, expected_vol=None):
        return DailyPerformance.concatenate(self.performances, expected_vol)
