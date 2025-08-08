import numpy as np

from . import settings as cn_futs
from .utils import exp_weight_cov_series
from ...utils import mkpath


def _fill_first_nan(n, value):
    rv = np.full(n, np.nan, dtype=value.dtype)
    rv[-len(value):] = value
    return rv


def build_1m_features(ticker, datadir):
    data_1m = np.load(f'{datadir}/raw/cn_futs_1m/{ticker}_1m.npy')

    n = len(data_1m)

    logprice = np.log(data_1m['close'])
    logret_1m = logprice[1:] - logprice[:-1]
    logret_5m = logprice[5:] - logprice[:-5]
    covs_1m = exp_weight_cov_series([logret_1m], halflife=1000, start=300, assume_zero_mean=True)
    vol_1m = np.empty(len(covs_1m), dtype=float)
    for j in range(len(covs_1m)):
        vol_1m[j] = np.sqrt(covs_1m[j].item())

    rv = np.empty(n, dtype=[
        ('time', int),

        ('ew_vol_1m', float),

        ('logret_1m', float),
        ('logret_5m', float),

        ('normret_1m', float),
        ('normret_5m', float),
    ])
    rv['time'] = data_1m['time']
    rv['ew_vol_1m'] = _fill_first_nan(n, vol_1m)
    rv['logret_1m'] = _fill_first_nan(n, logret_1m)
    rv['normret_1m'] = _fill_first_nan(n, logret_1m[-len(vol_1m):] / vol_1m)

    vol_5m = vol_1m * np.sqrt(5)
    rv['logret_5m'] = _fill_first_nan(n, logret_5m)
    rv['normret_5m'] = _fill_first_nan(n, logret_5m[-len(vol_5m):] / vol_5m)

    fn = f'{datadir}/cn_futs/features_1m/{ticker}.npy'
    mkpath(fn)
    np.save(fn, rv)
    return rv


def build_all_1m_features(datadir):
    for ticker in cn_futs.CN_FUTS_TICKERS_FULL:
        build_1m_features(ticker, datadir)
        print(f'built {ticker} 1m features')
