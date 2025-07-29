
def mkpath(filename):
    from os import makedirs
    from os.path import dirname
    makedirs(dirname(filename), exist_ok=True)


SECOND = 1_000_000_000
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE
DAY = 24 * HOUR


def set_seed(seed: int):
    from random import seed as sys_seed
    sys_seed(seed)
    from numpy import random as np_rand
    np_rand.seed(seed)
    # for torch
    from torch import manual_seed, cuda, backends
    manual_seed(seed)             # pytorch CPU random seed
    cuda.manual_seed(seed)        # pytorch GPU random seed
    cuda.manual_seed_all(seed)    # multiple GPU
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False


def dateint_to_datetime(dateint):
    from numpy import empty, datetime64
    from datetime import datetime
    n = len(dateint)
    rv = empty(n, dtype='datetime64[D]')
    for i in range(n):
        d = dateint[i]
        yy = d // 10000
        mm = (d % 10000) // 100
        dd = d % 100
        rv[i] = datetime64(datetime(yy, mm, dd), 'D')
    return rv


def cov_to_corr(cov):
    from numpy import diag, sqrt, outer
    stds = sqrt(diag(cov))
    std_matrix = outer(stds, stds)
    return cov / std_matrix
