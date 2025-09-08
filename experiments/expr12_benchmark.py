import sys
from collections import defaultdict

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig, BacktestTickerResult, DailyPerformance
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.linear import MinLinear
from TransMINT.model.loss import DecayedUtilityLoss, SharpeLoss
from TransMINT.model.lstm import FusionLSTM, MinLSTM
from TransMINT.model.transformer import FusionTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data
from TransMINT.tasks.cn_futs.settings import InSampleWindows, OutOfSampleWindows
from TransMINT.utils import decay_factor
import numpy as np



version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)
input_spec = build_input_spec(version)

data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 256,
    time_step = 180,  # 15 hours
)

n_random = []
rngs = []

class RandomTrader:
    def __init__(self, seed):
        self.RNG = np.random.default_rng(seed)

    def predict(self, x):
        batch_size = x.batch_size
        positions = self.RNG.choice([-0.3, 0.3], size=batch_size)
        return positions


class HoldLongTrader:
    def __init__(self):
        pass

    def predict(self, x):
        batch_size = x.batch_size
        return np.full(batch_size, 0.3, dtype=np.float32)


class ReverseTrader:
    def __init__(self):
        pass

    def predict(self, x):
        batch_size = x.batch_size
        pos = np.full(batch_size, -0.3, dtype=np.float32)
        ret_5m = x['norm_ret_5m'][:, -1, 0]
        pos[ret_5m < 0] = 0.3
        return pos


def backtest(traders):
    keys = [e[0] for e in traders]
    perfs = {k: [] for k in keys}

    for _, _, start, stop in OutOfSampleWindows:
        results = {k: defaultdict(list) for k in keys}

        for x, y in data_provider.get_dataloader(data_cfg, start, stop):
            for key, trader in traders:
                pred_position = trader.predict(x)

                target_return = y.details['target_return']
                capture_return = pred_position * target_return

                res = BacktestTickerResult(
                    y.ticker,
                    y.details['time'],
                    y.details['date'],
                    pred_position,
                    target_return,
                    capture_return,
                )
                results[key][y.ticker].append(res)

        for k in keys:
            z = {
                ticker: BacktestTickerResult.concatenate(v)
                for ticker, v in results[k].items()
            }
            ticker_perfs = [r.daily_performance() for r in z.values()]
            performance = DailyPerformance.aggregate(ticker_perfs, method='mean')

            perfs[k].append(performance)

    all_perf = {k: DailyPerformance.concatenate(v) for k, v in perfs.items()}
    return keys, perfs, all_perf



rng = np.random.default_rng(42)
rand_traders = []
for s in rng.integers(0, 1000000, size=100, dtype=int):
    rand_traders.append((f'seed={s}', RandomTrader(s)))

hold_trader = ('hold_long', HoldLongTrader())
reverse_trader = ('reverse', ReverseTrader())

all_traders = [hold_trader, reverse_trader] + rand_traders

expr_keys, _, expr_perfs = backtest(all_traders)

hold_long_perf = expr_perfs[hold_trader[0]]
reverse_perf = expr_perfs[reverse_trader[0]]

rand_perfs = [expr_perfs[k] for k, _ in rand_traders]
sort_ix = np.argsort([rp.sharpe_ratio for rp in rand_perfs])
rand_best_perf = rand_perfs[sort_ix[0]]
rand_worst_perf = rand_perfs[sort_ix[-1]]
rand_median_perf = rand_perfs[sort_ix[len(sort_ix) // 2]]


from pickle import dump
with open('vault/benchmark.bin', 'wb') as fh:
    dump((
        hold_long_perf,
        reverse_perf,
        rand_best_perf,
        rand_median_perf,
        rand_worst_perf,
    ), fh)
