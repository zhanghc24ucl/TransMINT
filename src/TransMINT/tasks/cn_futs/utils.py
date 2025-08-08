import numpy as np
from scipy.signal import lfilter

ArrayLike = np.ndarray

LN2 = np.log(2.0)


def exp_weight_cov_series(rets, *, decay_factor: float | None = None, halflife: float | None = None, start: int = 30,
        assume_zero_mean: bool = True, ):
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
        rets = rets.T  # Shape -> (n_days, n_tickers)
    N, K = rets.shape

    S = np.zeros((K, K))
    mu = np.zeros(K)
    covs = []

    for t in range(start):
        r = rets[t]
        if not assume_zero_mean:
            mu = decay_factor * mu + (1 - decay_factor) * r
            r = r - mu
        S = decay_factor * S + (1 - decay_factor) * np.outer(r, r)

    for t in range(start, N):
        r = rets[t]
        if not assume_zero_mean:
            mu = decay_factor * mu + (1 - decay_factor) * r
            r = r - mu
        S = decay_factor * S + (1 - decay_factor) * np.outer(r, r)
        covs.append(S.copy())

    return covs


def ewma_standardize(x: ArrayLike, *,  # enforce keyword-only for the two decay parameters
        half_life: float = None, decay_factor: float = None,  # λ = 1 − α
        epsilon: float = 1e-12, ) -> ArrayLike:
    r"""
    Apply *online* standardization to an entire time-ordered sequence **x** using
    exponentially-weighted moving statistics.
    Internally, the function updates the mean $\mu_t$ and variance $\sigma_t^2$
    recursively, but returns the standardized series $z_t$ in one shot.

    Parameters
    ----------
    x : numpy.ndarray or Sequence[float]
        Raw numerical sequence in chronological order.
    half_life : float, optional
        Half-life $h$ (number of samples).  Exactly one of *half_life* or
        *decay_factor* must be provided.
    decay_factor : float, optional
        Exponential decay factor $\lambda \in (0,1)$, where a smaller
        $\lambda$ implies faster adaptation.  Related to the EWMA smoothing
        coefficient by $\alpha = 1 - \lambda$.
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
    # Exactly ONE of {half_life, decay_factor} must be specified
    if (half_life is None) == (decay_factor is None):
        raise ValueError("Exactly one of `half_life` or `decay_factor` must be specified.")

    # Convert half-life to decay factor if necessary
    if decay_factor is None:
        decay_factor = 0.5 ** (1.0 / half_life)  # λ = 0.5^{1/h}
    if not (0.0 < decay_factor < 1.0):
        raise ValueError("`decay_factor` must lie in the interval (0, 1).")

    alpha = 1.0 - decay_factor  # α = 1 − λ

    # --- EWMA mean:   μ_t = α · x_t + (1 − α) · μ_{t−1}
    mu = lfilter([alpha], [1, -(1 - alpha)], x)

    # --- EWMA second moment:  m2_t = α · x_t² + (1 − α) · m2_{t−1}
    m2 = lfilter([alpha], [1, -(1 - alpha)], np.asarray(x) * np.asarray(x))

    # --- Variance estimate (biased, but adequate for standardization)
    var = m2 - mu * mu  # σ_t² ≈ E[x²] − μ_t²

    # --- Standardize
    return (x - mu) / np.sqrt(var + epsilon)
