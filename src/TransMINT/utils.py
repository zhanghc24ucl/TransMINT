
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
