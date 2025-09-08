
import math
from collections import Counter
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from tabulate import tabulate

from ..engine.backtest import DailyPerformance


def compare_results(results: Dict[str, Dict[str, DailyPerformance]], keys=None):
    if keys is None:
        keys = sorted(results.keys())

    all_fields = set()
    for k, v in results.items():
        all_fields.update(v.keys())
    all_fields = sorted(all_fields)

    n = len(keys)
    cmp = {}
    for i in range(n):
        for j in range(i+1, n):
            v1 = results[keys[i]]
            v2 = results[keys[j]]
            deltas = []
            for f in all_fields:
                if f not in v1 or f not in v2:
                    continue
                deltas.append(v1[f].sharpe_ratio - v2[f].sharpe_ratio)
            cmp[keys[i], keys[j]] = summarize_deltas(deltas)
    return cmp


def _rankdata_average(x: Sequence[float]) -> np.ndarray:
    """Average ranks with tie-handling. Ranks start at 1."""
    x = np.asarray(x, float)
    order = np.argsort(x)
    ranks = np.empty_like(x, dtype=float)
    i = 0
    r = 1.0
    while i < len(x):
        j = i
        # find tie block [i, j)
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        # average rank for ties
        avg_rank = (r + r + (j - i)) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        r += (j - i + 1)
        i = j + 1
    return ranks

def exact_wilcoxon_signed_rank(deltas: Sequence[float]) -> Dict[str, float]:
    """
    Exact two-sided Wilcoxon signed-rank test via full sign enumeration (n <= 12 OK).
    Zeros are dropped (Pratt convention: ignore exact zeros).
    Returns: {'n_eff', 'W', 'Tplus', 'p_two_sided'}
    """
    x = np.asarray(deltas, float)
    # drop zeros
    mask = (x != 0.0)
    x = x[mask]
    n = len(x)
    if n == 0:
        return {'n_eff': 0, 'W': float('nan'), 'Tplus': float('nan'), 'p_two_sided': 1.0}
    absx = np.abs(x)
    ranks = _rankdata_average(absx)  # average ranks handle ties
    Tplus_obs = float(ranks[x > 0].sum())
    Tsum = float(ranks.sum())        # equals n*(n+1)/2 even with ties
    # enumerate all sign assignments (+/-) for exact null distribution
    # we compute the distribution of Tplus = sum of ranks with + sign
    Tplus_counts = Counter()
    # pre-list ranks for speed
    r = ranks.tolist()
    for bits in range(1 << n):
        Tplus = 0.0
        # bit i = 1 => + sign on obs i
        for i in range(n):
            if (bits >> i) & 1:
                Tplus += r[i]
        Tplus_counts[Tplus] += 1
    total = float(1 << n)
    # two-sided p: doubled min tail
    # (exact discrete two-sided; capped at 1)
    cdf_le = sum(c for t, c in Tplus_counts.items() if t <= Tplus_obs) / total
    cdf_ge = sum(c for t, c in Tplus_counts.items() if t >= Tplus_obs) / total
    p_two = 2.0 * min(cdf_le, cdf_ge)
    p_two = min(1.0, p_two)
    W = min(Tplus_obs, Tsum - Tplus_obs)  # usual Wilcoxon statistic
    return {'n_eff': n, 'W': W, 'Tplus': Tplus_obs, 'p_two_sided': p_two}

def exact_sign_test(deltas: Sequence[float]) -> Dict[str, float]:
    """
    Exact two-sided binomial sign test (zeros ignored).
    Returns: {'n_eff','k_pos','win_rate','p_two_sided'}
    """
    x = np.asarray(deltas, float)
    k_pos = int((x > 0).sum())
    k_neg = int((x < 0).sum())
    n = k_pos + k_neg
    if n == 0:
        return {'n_eff': 0, 'k_pos': 0, 'win_rate': float('nan'), 'p_two_sided': 1.0}
    win_rate = k_pos / n
    # exact two-sided p: double the smaller tail
    def upper_tail(k, n):
        return sum(math.comb(n, j) for j in range(k, n + 1)) / (2.0 ** n)
    p_upper = upper_tail(k_pos, n)
    p_lower = upper_tail(n - k_pos, n)  # symmetry
    p_two = 2.0 * min(p_upper, p_lower)
    p_two = min(1.0, p_two)
    return {'n_eff': n, 'k_pos': k_pos, 'win_rate': win_rate, 'p_two_sided': p_two}

def summarize_deltas(deltas: Sequence[float]) -> Dict[str, Any]:
    """
    Compute descriptive stats + exact Wilcoxon and Sign test.
    Returns a dict; print it or consume programmatically.
    """
    x = np.asarray(deltas, float)
    x_nz = x[x != 0.0]  # for robust quantiles, zeros are allowed; keep them
    n = len(x)
    if n == 0:
        raise ValueError("Empty deltas.")
    median = float(np.median(x_nz)) if len(x_nz) else 0.0
    mean = float(np.mean(x))  # mean uses all, including zeros
    q1 = float(np.quantile(x_nz, 0.25, method='linear')) if len(x_nz) else 0.0
    q3 = float(np.quantile(x_nz, 0.75, method='linear')) if len(x_nz) else 0.0
    iqr = q3 - q1
    win_rate = float((x > 0).mean())  # include zeros in denominator (common)
    wilc = exact_wilcoxon_signed_rank(x)
    sign = exact_sign_test(x)
    return {
        'n': n,
        'mean': mean,
        'median': median,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'win_rate': win_rate,
        'wilcoxon_exact': wilc,
        'sign_test_exact': sign,
    }


def rank_results(results: Dict[str, Dict[str, DailyPerformance]], keys=None, **rank_args):
    if keys is None:
        keys = sorted(results.keys())
    cmp_results = compare_results(results, keys)
    cmp_values = {k: v['median'] for k, v in cmp_results.items()}
    rank = rank_winloss(cmp_values, keys, **rank_args)
    return rank


def rank_winloss(pairs, items, method="copeland"):
    """
    Rank items based on pairwise win/loss outcomes.

    Parameters:
    -----------
    pairs : dict
        Keys are tuples (A, B), values are {+1: A beats B, -1: A loses to B, 0: tie or unknown}.
        If (A, B) is not in the dictionary, it is treated as 0 (no match played).
    items : iterable
        List of item identifiers, e.g., ['A', 'B', 'C', ...]
    method : str
        Ranking method: "copeland" or "rank_centrality"

    Returns:
    --------
    ranking : list
        List of items sorted by rank (best first).
    scores : dict
        Dictionary mapping each item to its score.
    """
    items = list(items)
    idx = {k: i for i, k in enumerate(items)}  # Map item to index
    n = len(items)

    # Normalize: construct matrix W[i,j] ∈ {-1, 0, +1}
    # W[i,j] = +1 if i beats j, -1 if i loses to j, 0 otherwise
    W = np.zeros((n, n), dtype=int)
    for (a, b), v in pairs.items():
        if a not in idx or b not in idx or a == b:
            continue
        i, j = idx[a], idx[b]
        v = 1 if v > 0 else (-1 if v < 0 else 0)  # Normalize to -1, 0, +1
        W[i, j] = np.sign(v)
        W[j, i] = -np.sign(v)  # Ensure antisymmetry

    if method == "copeland":
        # Copeland score: wins minus losses
        wins = (W == 1).sum(axis=1)   # Number of wins for each item
        losses = (W == -1).sum(axis=1)  # Number of losses for each item
        s = wins - losses
        order = np.argsort(-s)  # Sort in descending order of score
        scores = {items[i]: int(s[i]) for i in range(n)}
        ranking = [items[i] for i in order]
        return ranking, scores

    elif method == "rank_centrality":
        # Rank Centrality (a Bradley-Terry-Luce variant using random walks)
        # Transition matrix P: probability i -> j proportional to times i lost to j

        L = np.maximum(W, 0)  # L[i,j] = 1 if i beat j (win matrix)
        F = np.maximum(-W, 0)  # F[i,j] = 1 if i lost to j (loss matrix)
        P = F.astype(float)   # Transition probabilities based on losses

        # Row normalization: P[i,j] = prob to go from i to j
        for i in range(n):
            s = P[i].sum()
            if s > 0:
                P[i] /= s
            else:
                # If no losses (unbeaten), distribute uniformly (teleport)
                P[i] = 1.0 / n

        # Add damping factor (PageRank-style) to handle disconnected components
        alpha = 0.85  # Damping factor
        J = np.full((n, n), 1.0 / n)  # Uniform transition matrix
        P = alpha * P + (1 - alpha) * J  # Final transition matrix

        # Power iteration to compute stationary distribution
        pi = np.full(n, 1.0 / n)  # Initial uniform distribution
        for _ in range(200):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi, 1) < 1e-10:
                break
            pi = pi_new

        s = pi  # Rank Centrality scores (stationary probabilities)
        order = np.argsort(-s)  # Descending order
        scores = {items[i]: float(s[i]) for i in range(n)}
        ranking = [items[i] for i in order]
        return ranking, scores

    else:
        raise ValueError("Unknown method")


def print_comparison(cmp_results: Dict[Tuple[str, str], Dict[str, Any]], pair_keys=None, **tab_args):
    if pair_keys is None:
        pair_keys = sorted(cmp_results.keys())

    tbl = []
    for pair_a, pair_b in pair_keys:
        r = cmp_results[pair_a, pair_b]
        tbl.append((
            f'{pair_a} vs. {pair_b}',
            r['n'],
            f'{r["median"]:.3f}[{r["q1"]:.3f}~{r["q3"]:.3f}]',
            r["win_rate"],
            r["wilcoxon_exact"]["p_two_sided"],
        ))
    print(tabulate(
        tbl,
        headers=['Pair', 'N', 'Median[IQR]', 'Win Rate', 'Wilcoxon P (two-sided, exact)'],
        floatfmt=[".0f", ".0f", ".0f", ".02f", ".04f"],
        **tab_args
    ))
