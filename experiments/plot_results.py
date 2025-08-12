from TransMINT.engine.backtest import BacktestConfig, Backtest
from TransMINT.tasks.cn_futs.settings import InSampleWindows

paths = [
    '20250811_loss_sharpe/s63_l0.001_oNone',
    '20250811_loss_utility/s42_l0.001_oNone',
]
bt_cfg = BacktestConfig(
    # windows=InSampleWindows,
    windows=[
        ('2016-03-01', '2018-07-01', '2019-01-01', '2019-07-01'),
        ('2016-07-01', '2019-01-01', '2019-07-01', '2020-01-01'),
        ('2017-01-01', '2019-07-01', '2020-01-01', '2020-07-01'),
        ('2017-07-01', '2020-01-01', '2020-07-01', '2021-01-01'),
    ],

    data_cfg=None,
    trainer_cfg=None,
)

from matplotlib import pyplot as plt
from TransMINT.viz.backtest import plot_performance, plot_ticker_performance

bts = []
for path in paths:
    bt = Backtest(bt_cfg, None, store_path=f'vault/{path}')
    bts.append(bt)
    bt.run()

    perfs = bt.ticker_performance()
    perf = bt.performance()
    plot_performance(perf, title=f'{path} risk=100: Sharpe = {perf.sharpe_ratio:.03f}', figsize=(14, 7))
    plot_ticker_performance(perfs, title=path, figsize=(14, 7))
plt.show()
