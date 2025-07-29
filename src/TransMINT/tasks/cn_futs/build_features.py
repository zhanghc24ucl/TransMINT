import numpy as np

from . import settings as cn_futs
from ...utils import dateint_to_datetime, mkpath

LN2 = np.log(2.0)

def exp_weight_cov_series(
        rets, *,
        decay_factor: float | None = None,
        halflife: float | None = None,
        start: int = 30,
        assume_zero_mean: bool = True,
    ):
    """
    Calculate daily EWMA covariance series starting from `start`

    Parameters
    ----------
    rets : (K, N) or (N, K) ndarray
        daily returns
    decay_factor : float, optional
        decay factor
    halflife : float, optional
        half life: decay_factor = 0.5 ** (1/halflife)
    start : int, default 30
        warm up window size
    assume_zero_mean : bool, default True
        if returns' mean is assumed to be 0.

    Returns
    -------
    covs : list[np.ndarray]
        covariance series with size = N - start + 1
    """
    if decay_factor is None and halflife is None:
        raise ValueError("Decay factor and halflife can not be specified together.")
    if decay_factor is None:
        decay_factor = 0.5 ** (1.0 / halflife)

    rets = np.asarray(rets, dtype=np.float64)
    if rets.shape[0] < rets.shape[1]:
        rets = rets.T                   # Shape -> (n_days, n_tickers)
    N, K = rets.shape

    S  = np.zeros((K, K))
    mu = np.zeros(K)
    covs = []

    for t in range(start):
        r = rets[t]
        if not assume_zero_mean:
            mu = decay_factor * mu + (1 - decay_factor) * r
            r  = r - mu
        S = decay_factor * S + (1 - decay_factor) * np.outer(r, r)

    for t in range(start, N):
        r = rets[t]
        if not assume_zero_mean:
            mu = decay_factor * mu + (1 - decay_factor) * r
            r  = r - mu
        S = decay_factor * S + (1 - decay_factor) * np.outer(r, r)
        covs.append(S.copy())

    return covs


def _split_and_fill(dates, splits):
    rv = np.empty(len(dates), dtype=int)
    prev_ix = 0
    for j in range(len(splits) - 1):
        ix = dates.searchsorted(splits[j+1][0])
        rv[prev_ix:ix] = splits[j][1]
        prev_ix = ix
    rv[prev_ix:] = splits[-1][1]
    return rv


def build_covariance(datadir):
    rets = []
    mins = []
    date = None
    tickers = cn_futs.CN_FUTS_TICKERS_FULL
    for ticker in tickers:
        r1d = np.load(f'{datadir}/raw/cn_futs_1d/{ticker}_1d.npy')
        ret = np.log(r1d['close'][1:] / r1d['close'][:-1])
        if date is None:
            date = r1d['date']
        else:
            assert len(r1d) == len(date)

        min_per_day = _split_and_fill(date, cn_futs.CN_FUTS_MINUTES_PER_DAY[ticker])
        mins.append(min_per_day)
        rets.append(ret)

    covs = exp_weight_cov_series(rets, halflife=10, start=25, assume_zero_mean=False)

    n = len(covs)
    k = len(tickers)
    mins = np.array(mins, dtype=int).T[-n:]

    dtypes = [
        ('date', 'datetime64[D]'),
        ('cov', float, (k, k)),
        ('vol', float, (k, )),
        ('minutes_per_day', int, (k, )),
    ]
    rv = np.empty(n, dtype=dtypes)
    rv['date'] = dateint_to_datetime(date[-n:])
    rv['cov'] = covs
    rv['vol'] = np.sqrt(np.diagonal(rv['cov'], axis1=1, axis2=2))
    rv['minutes_per_day'] = mins

    from pickle import dump
    with open(f'{datadir}/cn_futs/covariance.bin', 'wb') as fh:
        dump((rv, tickers), fh)
    return rv, tickers


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
