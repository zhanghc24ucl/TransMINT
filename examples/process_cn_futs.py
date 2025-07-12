import numpy as np

from TransMINT.settings import cn_futs
from TransMINT.utils import DAY, HOUR, MINUTE, mkpath


def convert_raw1m_to_targets(ticker, horizon=5, phase=0):
    raw_data = np.load(f'../data/raw/cn_futs_1m/{ticker}_1m.npy')
    date_data = np.load(f'../data/raw/cn_futs_meta/date.npy')

    start_time = raw_data['time'] - MINUTE
    price = raw_data['open']

    indices = ((start_time // MINUTE) % horizon) == phase
    N = sum(indices)

    output = np.empty(N, [
        ('time', int),
        ('price', float),
        ('date', 'datetime64[D]'),

        # sequential
        ('DoM', int),  # day of month
        ('MoY', int),  # month of year
        ('WoY', int),  # week of year

        # cyclical
        ('DoW', int),  # day of week
        ('HoD', int),  # hour of day
        ('MoH', int),  # minute of hour

        ('NToD', float),  # normalized time of day, within [0, 1]
    ])
    output['time'] = time = start_time[indices]
    output['price'] = price[indices]

    output['MoH'] = (time % HOUR) // MINUTE
    output['HoD'] = (time % DAY) // HOUR

    date_t0 = date_data['t0']
    assert date_t0[0] < time[0]

    date_delta = np.empty(len(date_data), dtype=int)
    date_delta[:-1] = np.diff(date_data['date'])
    date_delta[-1] = 3  # The last day is 2022-12-30, and there are 3 holidays ahead.
    date_delta[date_delta == 2] = 0  # tricky one to exclude all weekends.

    date_indices = date_t0.searchsorted(time, side='right') - 1
    for k in ['date', 'DoM', 'MoY', 'DoW', 'WoY']:
        output[k] = date_data[k][date_indices]

    ntod = output['NToD']
    date = output['date']

    prev_date = date[0]
    prev_start = 0
    for i in range(N):
        if date[i] != prev_date:
            T = i - prev_start  # T data points in previous day
            ntod[prev_start:i] = np.linspace(0, 1, T)
            prev_start = i
            prev_date = date[i]

        if i == N-1:
            ntod[prev_start:] = np.linspace(0, 1, N - prev_start)

    output_path = f'../data/cn_futs/targets_{horizon}_{phase}/{ticker}.npy'
    mkpath(output_path)
    np.save(output_path, output)

def convert_1m_run():
    for horizon in [5]:
        for phase in range(horizon):
            for ticker in cn_futs.CN_FUTS_TICKERS_FULL:
                convert_raw1m_to_targets(ticker, horizon, phase)
                print('converted', ticker, horizon, phase)
