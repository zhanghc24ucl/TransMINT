import numpy as np

from . import settings as cn_futs
from ...utils import DAY, HOUR, MINUTE, mkpath


def convert_raw1m_to_targets(ticker, datadir, *, horizon=5, phase=0):
    raw_data = np.load(f'{datadir}/raw/cn_futs_1m/{ticker}_1m.npy')
    date_data = np.load(f'{datadir}/raw/cn_futs_meta/date.npy')

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

    output_path = f'{datadir}/cn_futs/targets_{horizon}_{phase}/{ticker}.npy'
    mkpath(output_path)
    np.save(output_path, output)

def build_1m_price(datadir):
    for horizon in [5]:
        for phase in range(horizon):
            for ticker in cn_futs.CN_FUTS_TICKERS_FULL:
                convert_raw1m_to_targets(ticker, datadir, horizon=horizon, phase=phase)
                print('converted', ticker, horizon, phase)


def build_1m_tabular_v1(ticker, datadir, *, horizon=5, phase=0, offset=0):
    # load targets' prices
    tgts = np.load(f'{datadir}/cn_futs/targets_{horizon}_{phase}/{ticker}.npy')

    # load full 1m features
    feature1m = np.load(f'{datadir}/cn_futs/features_1m/{ticker}.npy')

    # validate all features are ready
    start_date = np.datetime64('2016-02-01')
    tgt_start_ix = tgts['date'].searchsorted(start_date)
    f1m_start_ix = feature1m['time'].searchsorted(tgts['time'][tgt_start_ix])
    for k in feature1m.dtype.names:
        assert not np.isnan(feature1m[k][f1m_start_ix]), (k, f1m_start_ix)

    tgts = tgts[tgt_start_ix:]
    n = len(tgts) - 1

    dtype = [
        ('time', int),
        ('date', 'datetime64[D]'),
        ('target_return', float),

        # features
        ('norm_time_of_day', float),
        ('norm_ret_1m', float),
        ('norm_ret_5m', float),
        ('ew_vol_1m', float),
    ]
    tabular = np.empty(n, dtype=dtype)
    tabular['time'] = tgt_time = tgts['time'][:-1]
    tabular['date'] = tgts['date'][:-1]

    tgt_logp = np.log(tgts['price'])
    tabular['target_return'] = tgt_logp[1:] - tgt_logp[:-1]

    tabular['norm_time_of_day'] = tgts['NToD'][:-1]
    eff_time = tgt_time - offset * MINUTE
    feature1m_ix = feature1m['time'].searchsorted(eff_time, side='right') - 1
    assert len(feature1m_ix) == n
    tabular['norm_ret_1m'] = feature1m['normret_1m'][feature1m_ix]
    tabular['norm_ret_5m'] = feature1m['normret_5m'][feature1m_ix]
    tabular['ew_vol_1m'] = feature1m['ew_vol_1m'][feature1m_ix]

    fn = f'{datadir}/cn_futs/tabular/v1/{ticker}.npy'
    mkpath(fn)
    np.save(fn, tabular)
    return tabular


def build_all_1m_tabular(datadir, *, horizon=5, phase=0, offset=0, version='v1'):
    build_func = {
        'v1': build_1m_tabular_v1,
    }[version]
    for ticker in cn_futs.CN_FUTS_TICKERS_FULL:
        build_func(ticker, datadir, horizon=horizon, phase=phase, offset=offset)
        print('built tabular data: ', ticker, horizon, phase, offset)
