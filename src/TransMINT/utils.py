from typing import Any, Dict


def mkpath(filename):
    from os import makedirs
    from os.path import dirname
    makedirs(dirname(filename), exist_ok=True)


SECOND = 1_000_000_000
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE
DAY = 24 * HOUR


def set_deterministic_flags():
    from torch import backends
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False


def set_seed(seed: int):
    from random import seed as sys_seed
    sys_seed(seed)
    from numpy import random as np_rand
    np_rand.seed(seed)
    # for torch
    from torch import manual_seed, cuda
    manual_seed(seed)             # pytorch CPU random seed
    cuda.manual_seed(seed)        # pytorch GPU random seed
    cuda.manual_seed_all(seed)    # multiple GPU


def get_random_state() -> Dict[str, Any]:
    from random import getstate as py_state
    from numpy.random import get_state as np_state
    from torch import get_rng_state as torch_state
    from torch.cuda import get_rng_state_all as cuda_state
    return {
        'python_random': py_state(),
        'numpy_random': np_state(),
        'torch_random': torch_state(),
        'torch_cuda_random': cuda_state()
    }


def set_random_state(random_state: Dict[str, Any]):
    from random import setstate as py_state
    from numpy.random import set_state as np_state
    from torch import set_rng_state as torch_state
    from torch.cuda import set_rng_state_all as cuda_state
    py_state(random_state['python_random'])
    np_state(random_state['numpy_random'])
    torch_state(random_state['torch_random'])
    cuda_state(random_state['torch_cuda_random'])


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


def merge_features(feature_data: Dict[str, Any], select=None, end_time=None):
    if select is None:
        select = feature_data.keys()
    data = []
    for v in select:
        d = feature_data[v]
        if end_time:
            d = d[:d['time'].searchsorted(end_time)]
        data.append(d)
    if len(data) == 1:
        return data[0]
    from numpy import concatenate
    return concatenate(data)
