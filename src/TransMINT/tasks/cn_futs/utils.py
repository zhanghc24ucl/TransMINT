import numpy as np
from scipy.signal import lfilter

ArrayLike = np.ndarray

LN2 = np.log(2.0)


def _alpha(
        alpha: float | None = None,
        decay: float | None = None,
        halflife: float | None = None,
        span: float | None = None,
):
    if alpha is not None:
        assert 0 < alpha < 1
        return alpha
    if decay is not None:
        assert 0 < decay < 1
        return 1.0 - decay
    if halflife is not None:
        return 1.0 - 0.5 ** (1.0 / float(halflife))
    if span is not None:
        assert span > 0
        return 2.0 / (span + 1.0)
    raise ValueError('No decay parameters specified')


def expw_mean(
        x: ArrayLike, *,
        alpha: float | None = None,
        decay: float | None = None,
        halflife: float | None = None,
        span: float | None = None,
):
    """
    Exponentially weighted moving average
    Parameters
    ----------
    x : numpy.ndarray
        Raw numerical sequence in chronological order.
    alpha : float, optional
        alpha for EWMA
    decay : float, optional
        decay factor
    halflife : float, optional
        half life: decay_factor = 0.5 ** (1/halflife)
    span : float, optional
        span: decay_factor = 1 / (span + 1)
    Returns
    -------
    y : numpy.ndarray
        EWMA series with the same length as the input.
    """
    alpha = _alpha(alpha=alpha, decay=decay, halflife=halflife, span=span)
    return lfilter([alpha], [1, -(1 - alpha)], x)


def expw_cov_series(
        rets: ArrayLike, *,
        alpha: float | None = None,
        decay: float | None = None,
        halflife: float | None = None,
        span: float | None = None,
        assume_zero_mean: bool = True
):
    """
    Calculate daily EWMA covariance series starting from `start`

    Parameters
    ----------
    rets : (N, K) ndarray
        daily returns
    alpha : float, optional
        alpha for EWMA
    decay : float, optional
        decay factor
    halflife : float, optional
        half life: decay_factor = 0.5 ** (1/halflife)
    span : float, optional
        span: decay_factor = 1 / (span + 1)
    assume_zero_mean : bool, default True
        if returns' mean is assumed to be 0.

    Returns
    -------
    covs : np.ndarray
        covariance series with size = N - start
    """
    alpha = _alpha(alpha=alpha, decay=decay, halflife=halflife, span=span)

    N, K = rets.shape

    S = np.zeros((K, K), dtype=float)
    mu = np.zeros(K, dtype=float)
    covs = np.empty((N, K, K), dtype=float)

    for t in range(N):
        r = rets[t]
        if not assume_zero_mean:
            mu = (1 - alpha) * mu + alpha * r
            r = r - mu
        S = (1 - alpha) * S + alpha * np.outer(r, r)
        covs[t] = S

    return covs


def expw_standardize(
        x: ArrayLike, *,
        alpha: float | None = None,
        decay: float | None = None,
        halflife: float | None = None,
        span: float | None = None,
        mu: float | None = None,
        clip: float | None = 5.,
        epsilon: float = 1e-12,
) -> ArrayLike:
    r"""
    Apply *online* standardization to an entire time-ordered sequence **x** using
    exponentially-weighted moving statistics.
    Internally, the function updates the mean $\mu_t$ and variance $\sigma_t^2$
    recursively, but returns the standardized series $z_t$ in one shot.

    Parameters
    ----------
    x : numpy.ndarray or Sequence[float]
        Raw numerical sequence in chronological order.
    alpha : float, optional
        alpha for EWMA
    decay : float, optional
        decay factor
    halflife : float, optional
        half life: decay_factor = 0.5 ** (1/halflife)
    span : float, optional
        span: decay_factor = 1 / (span + 1)
    mu : float, optional
        Fixed mean. If provided, standardizes (x - mu) by EWMA std of (x - mu).
    clip : float, optional
        Whether to clip values within [-clip, clip].
    epsilon : float, default 1e-12
        Small constant to prevent division by zero when the variance is tiny.

    Returns
    -------
    z : numpy.ndarray
        Standardized sequence
        $$
            z_t = \frac{x_t - \mu_t}{\sqrt{\sigma_t^2 + \varepsilon}},
        $$
        with the same length as the input.
    """
    alpha = _alpha(alpha=alpha, decay=decay, halflife=halflife, span=span)
    if mu is None:
        # --- EWMA mean:   μ_t = α·x_t + (1−α)·μ_{t−1}
        ewm_mu = lfilter([alpha], [1, -(1 - alpha)], x)
        # --- EWMA second moment: m2_t = α·x_t² + (1−α)·m2_{t−1}
        m2 = lfilter([alpha], [1, -(1 - alpha)], x * x)
        # --- Variance estimate (biased, adequate for standardization)
        var = m2 - ewm_mu * ewm_mu
        num = x - ewm_mu
    else:
        # Fixed-mean standardization: use EWMA of squared residuals
        mu = np.asarray(mu, dtype=float)
        # broadcast if needed
        res = x - mu
        # v_t = α·res_t² + (1−α)·v_{t−1} ≈ EWMA of (x − μ_fixed)²
        var = lfilter([alpha], [1, -(1 - alpha)], res * res)
        num = res

    var = np.maximum(var, 0.0) + epsilon
    z = num / np.sqrt(var)
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z


def expw_garman_klass_volatility(
        open_, high, low, close, *,
        alpha: float | None = None,
        decay: float | None = None,
        halflife: float | None = None,
        span: float | None = None,
        epsilon: float = 1e-12,
):
    """
    Compute Exponentially Weighted Garman–Klass volatility

    Parameters
    ----------
    open_, high, low, close : ndarray
        OHLC price arrays of the same length
    alpha : float, optional
        alpha for EWMA
    decay : float, optional
        decay factor
    halflife : float, optional
        half life: decay_factor = 0.5 ** (1/halflife)
    span : float, optional
        span: decay_factor = 1 / (span + 1)
    epsilon : float, default 1e-12
        Small constant to prevent division by zero when the variance is tiny.

    Returns
    -------
    vol : ndarray
        EWMA Garman–Klass volatility series starting from `start` index,
        length = L - start
    """
    # Parameter validation
    alpha = _alpha(alpha=alpha, decay=decay, halflife=halflife, span=span)

    # Ensure positive prices to avoid log issues
    O = np.clip(open_.astype(float), epsilon, None)
    H = np.clip(high.astype(float), epsilon, None)
    L = np.clip(low.astype(float), epsilon, None)
    C = np.clip(close.astype(float), epsilon, None)

    # Garman–Klass single-period variance estimate
    logHL = np.log(H / L)
    logCO = np.log(C / O)
    gk_var = 0.5 * (logHL ** 2) - (2.0 * np.log(2.0) - 1.0) * (logCO ** 2)
    gk_var = np.clip(gk_var, 0.0, None)  # avoid negatives due to bad ticks

    # EWMA variance recursion (adjust=False equivalent)
    ew_var = np.empty_like(gk_var)
    ew_var[0] = gk_var[0]
    for t in range(1, len(gk_var)):
        ew_var[t] = (1 - alpha) * ew_var[t - 1] + alpha * gk_var[t]

    # Return volatility (sqrt of variance) from start index onwards
    return np.sqrt(ew_var)


def macd(
        x: ArrayLike, *,
        fast=12, slow=26, signal=9,
):
    dif = expw_mean(x, span=fast) - expw_mean(x, span=slow)
    dea = expw_mean(dif, span=signal)
    return dif - dea


def rolling_sum(x: ArrayLike, window: int):
    c = np.cumsum(x)
    return c[window:] - c[:-window]


def _rolling_extreme(x: np.ndarray, window: int, *, pop_if) -> np.ndarray:
    from collections import deque
    N = x.shape[0]
    M = N - window + 1
    out = np.empty(M, dtype=float)
    dq = deque()
    for i in range(N):
        while dq and dq[0] <= i - window:
            dq.popleft()
        xi = x[i]
        while dq and pop_if(x[dq[-1]], xi):
            dq.pop()
        dq.append(i)
        if i >= window - 1:
            out[i - window + 1] = x[dq[0]]
    return out


def rolling_ohlc(open_, high, low, close, window: int):
    N = len(close)
    M = N - window + 1
    if M <= 0:
        # 没有任何满窗
        return (np.empty(0),) * 4

    # open/close
    o = open_[:M]  # 对应窗口 [k, k+window-1] 的开盘
    c = close[window - 1:]  # 对应窗口尾部的收盘

    # high/low
    h = _rolling_extreme(high, window, pop_if=lambda x, y: x <= y)
    l = _rolling_extreme(low,  window, pop_if=lambda x, y: x >= y)

    return o, h, l, c
