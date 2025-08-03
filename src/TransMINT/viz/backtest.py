from typing import Dict

from matplotlib import pyplot as plt

from ..engine.backtest import DailyPerformance


def plot_performance(perf: DailyPerformance, title=None, **fig_kw):
    fig, ax0 = plt.subplots(**fig_kw)

    cumulative_return = (1 + perf.returns).cumprod()
    ax0.plot(perf.dates, cumulative_return, label='Cumulative Return')
    ax0.set_xlabel('Date')
    ax0.set_ylabel('Cumulative Return', color='blue')
    if title is None:
        title = f'Sharpe = {perf.sharpe_ratio:.03f}'
    ax0.set_title(title)
    ax0.grid(True)

    ax1 = ax0.twinx()
    ax1.plot(perf.dates, perf.volumes, color='red', alpha=0.5, label='Volume')
    ax1.set_ylabel('Volume (Turnover)', color='red')

    ax0.legend(loc='upper left')
    ax1.legend(loc='upper right')
    return fig


def plot_ticker_performance(perfs: Dict[str, DailyPerformance], title=None, cmap_name='plasma', **fig_kw):
    fig, ax0 = plt.subplots(**fig_kw)
    n = len(perfs)
    cmap = plt.cm.get_cmap(cmap_name, n)

    lines = []
    data_list = []

    for idx, (k, v) in enumerate(perfs.items()):
        cr = (1 + v.returns).cumprod()
        color = cmap(idx)

        line, = ax0.plot(v.dates, cr, color=color)
        lines.append(line)

        final_date = v.dates[-1]
        final_return = cr[-1]
        data_list.append((final_date, final_return, k, color))

    data_list.sort(key=lambda x: x[1])
    # add annotation
    from numpy import timedelta64
    date_tick = timedelta64(604800, 's')  # 7 days
    for i, (date, y, label, color) in enumerate(data_list):
        ax0.annotate(label, xy=(date, y), xytext=(date + date_tick, y), fontsize=9, color=color)

    ax0.set_xlabel('Date')
    ax0.set_ylabel('Cumulative Return', color='blue')
    ax0.grid(True)
    if title:
        ax0.set_title(title)
    return fig
