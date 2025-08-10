import numpy as np

from . import settings as cn_futs
from .utils import expw_standardize
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


def build_1m_tabular(ticker, datadir, *, horizon=5, phase=0, offset=0, clip=5., epsilon=1e-12):
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
        # time
        ('norm_time_of_day', float),

        # return
        ('norm_ret_1m[0]', float),
        ('norm_ret_1m[1]', float),
        ('norm_ret_1m[2]', float),
        ('norm_ret_1m[3]', float),
        ('norm_ret_1m[4]', float),
        ('norm_ret_5m',    float),
        ('norm_ret_30m',   float),
        ('norm_ret_240m',  float),

        # volume/value
        ('norm_log_value_5m', float),
        ('norm_mfv_30m',      float),

        # statistics
        ('norm_er_5m',        float),
        ('norm_clv_30m',      float),
        ('norm_log_ewvol_1m', float),
        ('norm_log_gkvol_1m', float),
        ('macd_8_24_16_1m', float),
        ('macd_16_48_16_1m', float),
        ('macd_32_96_16_1m', float),
    ]
    tabular = np.empty(n, dtype=dtype)
    tabular['time'] = tgt_time = tgts['time'][:-1]
    tabular['date'] = tgts['date'][:-1]

    tgt_logp = np.log(tgts['price'])
    tabular['target_return'] = tgt_logp[1:] - tgt_logp[:-1]

    eff_time = tgt_time - offset * MINUTE
    feature1m_ix = feature1m['time'].searchsorted(eff_time, side='right') - 1
    assert len(feature1m_ix) == n
    assert feature1m_ix[0] >= 300  # sufficient warm-up period

    # Time features
    # uniform distributed ~ [0, 1] with mu = 0.5 and sigma = sqrt(1/12)
    tabular['norm_time_of_day'] = (tgts['NToD'][:-1] - 0.5) / np.sqrt(1/12)

    # Return features
    ret_sigma_1m = feature1m['ew_vol_1m']
    norm_ret_1m = feature1m['logret_1m'] / ret_sigma_1m
    if clip:
        norm_ret_1m = np.clip(norm_ret_1m, -clip, clip)
    for o in range(5):
        tabular[f'norm_ret_1m[{o}]'] = norm_ret_1m[feature1m_ix - o]

    for span in [5, 30, 240]:
        norm_ret = feature1m[f'logret_{span}m'] / (ret_sigma_1m * np.sqrt(span))
        if clip:
            norm_ret = np.clip(norm_ret, -clip, clip)
        tabular[f'norm_ret_{span}m'] = norm_ret[feature1m_ix]

    # Value/Volume features
    print('value_5m', np.isnan(feature1m['value_5m']).sum(), len(feature1m))
    norm_logv = expw_standardize(np.log(np.nan_to_num(feature1m['value_5m']) + epsilon), halflife=1000, clip=clip)
    tabular['norm_log_value_5m'] = norm_logv[feature1m_ix]

    print('er_5m',   np.isnan(feature1m['er_5m']).sum(), len(feature1m))
    print('clv_30m', np.isnan(feature1m['clv_30m']).sum(), len(feature1m))
    print('value_30m', np.isnan(feature1m['value_30m']).sum(), len(feature1m))
    mfv = np.nan_to_num(feature1m['clv_30m'] * feature1m['value_30m'])
    # for 30m feature, use longer halflife
    norm_mfv = expw_standardize(mfv, halflife=2000, clip=clip)
    tabular['norm_mfv_30m'] = norm_mfv[feature1m_ix]

    # Statistic features
    # er ~ [0, 1], normalized with mu=0.5
    norm_er = expw_standardize(np.nan_to_num(feature1m['er_5m']), mu=0.5, halflife=1000, clip=clip)
    tabular['norm_er_5m'] = norm_er[feature1m_ix]
    # clv ~ [-1, 1], normalized with mu=0
    norm_clv = expw_standardize(np.nan_to_num(feature1m['clv_30m']), mu=0, halflife=1000, clip=clip)
    tabular['norm_clv_30m'] = norm_clv[feature1m_ix]
    # normalize volatility
    print('volatility_1m', np.isnan(feature1m['ew_vol_1m']).sum(), len(feature1m))
    norm_ewvol = expw_standardize(np.nan_to_num(feature1m['ew_vol_1m']), halflife=1000, clip=clip)
    tabular['norm_log_ewvol_1m'] = norm_ewvol[feature1m_ix]
    norm_gkvol = expw_standardize(np.nan_to_num(feature1m['gk_vol_1m']), halflife=1000, clip=clip)
    tabular['norm_log_gkvol_1m'] = norm_gkvol[feature1m_ix]

    # statistics: MACD
    # to normalize macd, use fixed mu=0.
    for k in ['macd_8_24_16_1m', 'macd_16_48_16_1m', 'macd_32_96_16_1m']:
        tabular[k] = expw_standardize(feature1m[k][feature1m_ix], mu=0, halflife=1000, clip=clip)

    fn = f'{datadir}/cn_futs/tabular/v2/{ticker}.npy'
    mkpath(fn)
    np.save(fn, tabular)
    return tabular


def build_all_1m_tabular(datadir, **args):
    for ticker in cn_futs.CN_FUTS_TICKERS_FULL:
        build_1m_tabular(ticker, datadir, **args)
        print('built tabular data: ', ticker, args)
