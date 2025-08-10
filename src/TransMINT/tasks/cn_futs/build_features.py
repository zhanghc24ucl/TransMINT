import numpy as np

from . import settings as cn_futs
from .utils import expw_cov_series, expw_garman_klass_volatility, macd, rolling_ohlc, rolling_sum
from ...utils import mkpath


def _fill_first_nan(n, value):
    rv = np.full(n, np.nan, dtype=value.dtype)
    rv[-len(value):] = value
    return rv


def build_1m_features(ticker, datadir, epsilon=1e-12):
    data_1m = np.load(f'{datadir}/raw/cn_futs_1m/{ticker}_1m.npy')

    n = len(data_1m)

    logprice = np.log(data_1m['close'])
    logret_1m = logprice[1:] - logprice[:-1]
    logret_5m = logprice[5:] - logprice[:-5]
    logret_30m = logprice[30:] - logprice[:-30]
    logret_240m = logprice[240:] - logprice[:-240]
    var_1m = expw_cov_series(logret_1m[:, None], halflife=1000, assume_zero_mean=True)[300:]
    vol_1m = np.sqrt(var_1m[:, 0, 0])

    gk_vol_1m = expw_garman_klass_volatility(
        data_1m['open'], data_1m['high'], data_1m['low'], data_1m['close'],
        halflife=1000, epsilon=epsilon,
    )[300:]

    rv = np.empty(n, dtype=[
        ('time', int),

        ('ew_vol_1m', float),
        ('gk_vol_1m', float),

        ('logret_1m', float),
        ('logret_5m', float),
        ('logret_30m', float),
        ('logret_240m', float),

        ('value_5m', float),
        ('value_30m', float),

        ('macd_8_24_16_1m', float),
        ('macd_16_48_16_1m', float),
        ('macd_32_96_16_1m', float),

        ('er_5m', float),
        ('clv_30m', float),
    ])
    rv['time'] = data_1m['time']

    rv['ew_vol_1m'] = _fill_first_nan(n, vol_1m)
    rv['gk_vol_1m'] = _fill_first_nan(n, gk_vol_1m)

    rv['logret_1m'] = _fill_first_nan(n, logret_1m)
    rv['logret_5m'] = _fill_first_nan(n, logret_5m)
    rv['logret_30m'] = _fill_first_nan(n, logret_30m)
    rv['logret_240m'] = _fill_first_nan(n, logret_240m)

    value = data_1m['value']
    rv['value_5m'] = _fill_first_nan(n, rolling_sum(value, 5))
    rv['value_30m'] = _fill_first_nan(n, rolling_sum(value, 30))

    rv['macd_8_24_16_1m'] = _fill_first_nan(
        n, macd(logprice, fast=8, slow=24, signal=16))
    rv['macd_16_48_16_1m'] = _fill_first_nan(
        n, macd(logprice, fast=8, slow=24, signal=16))
    rv['macd_32_96_16_1m'] = _fill_first_nan(
        n, macd(logprice, fast=32, slow=96, signal=16))

    o5, h5, l5, c5 = rolling_ohlc(
        data_1m['open'], data_1m['high'], data_1m['low'], data_1m['close'], window=5)
    o30, h30, l30, c30 = rolling_ohlc(
        data_1m['open'], data_1m['high'], data_1m['low'], data_1m['close'], window=30)
    rv['er_5m'] = _fill_first_nan(n, abs(c5 - o5) / (h5 - l5 + epsilon))
    rv['clv_30m'] = _fill_first_nan(n, (2 * c30 - h30 - l30) / (h30 - l30 + epsilon))

    fn = f'{datadir}/cn_futs/features_1m/{ticker}.npy'
    mkpath(fn)
    np.save(fn, rv)
    return rv


def build_all_1m_features(datadir):
    for ticker in cn_futs.CN_FUTS_TICKERS_FULL:
        build_1m_features(ticker, datadir)
        print(f'built {ticker} 1m features')
