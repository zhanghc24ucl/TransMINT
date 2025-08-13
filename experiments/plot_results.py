from TransMINT.engine.backtest import BacktestConfig, Backtest
from TransMINT.tasks.cn_futs.settings import InSampleWindows

paths = [
    # 'to_be_deleted/20250812_batch/b128',
    # 'to_be_deleted/20250812_rf/k0.02',
    # 'to_be_deleted/20250812_rf/k0.1',

    # "20250813_lr_utility/l0.002",
    # "20250813_lr_utility/l0.001",
    # "20250813_lr_utility/l0.0007",
    # "20250813_lr_utility/l0.0003",
    # "20250813_lr_utility/l0.0001",
    # "20250813_lr_utility/l7e-05",

    # "20250811_loss_utility/s42_l0.0001_oNone",
    # "20250811_loss_utility/s63_l0.0001_oNone",
    # "20250811_loss_utility/s191_l0.0001_oNone",
    # "20250813_lr_sharpe/l1e-05",
    # "20250813_lr_sharpe/l8e-06",
    # "20250813_lr_sharpe/l5e-06",
    # "20250813_lr_sharpe/l3e-06",

    "20250813_loss_sharpe/s42",
    "20250813_loss_sharpe/s63",
    "20250813_loss_sharpe/s191",

    "20250815_loss_utility/s42",
    "20250815_loss_utility/s63",
    "20250815_loss_utility/s191",
]
bt_cfg = BacktestConfig(
    # windows=InSampleWindows,
    windows=[
        ('2016-03-01', '2018-07-01', '2019-01-01', '2019-07-01'),
        ('2016-07-01', '2019-01-01', '2019-07-01', '2020-01-01'),
        # ('2017-01-01', '2019-07-01', '2020-01-01', '2020-07-01'),
        # ('2017-07-01', '2020-01-01', '2020-07-01', '2021-01-01'),
    ],

    data_cfg=None,
    trainer_cfg=None,
)

from matplotlib import pyplot as plt
from TransMINT.viz.backtest import plot_performance, plot_ticker_performance

bts = []
rv = {}
for path in paths:
    bt = Backtest(bt_cfg, None, store_path=f'vault/{path}')
    bts.append(bt)
    bt.run()

    perfs = bt.ticker_performance()
    perf = bt.performance()
    perf = perf.adjust_for_costs(0.5e-4)
    plot_performance(perf, title=f'{path} risk=100: Sharpe = {perf.sharpe_ratio:.03f}', figsize=(14, 7))
    rv[path.split('_')[-1]] = perf
plot_ticker_performance(rv, figsize=(14, 7))
plt.show()
