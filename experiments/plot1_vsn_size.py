import sys

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.loss import DecayedUtilityLoss, SharpeLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data, random_spec
from TransMINT.tasks.cn_futs.settings import InSampleWindows

version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)

def get_n_params(is_lite, d_model, n_features):
    input_spec = build_input_spec(version)
    args = dict(
        d_model=d_model,
        num_heads=4,
        dropout=0.2,
        trainable_skip_add=False,
        is_lite=is_lite,
    )
    model = MINTransformer(random_spec(2, n_features), **args)
    return model.n_parameters()



from matplotlib import pyplot as plt

xs = [10, 20, 50, 100]
ns = [16, 32, 64, 128]
# for n in ns:
tbl = []
for x in xs:
    v1 = []
    # for x in xs:
    for n in ns:
        p1 = get_n_params(True, n, x)[0]
        p2 = get_n_params(False, n, x)[0]
        v1.append(p2 / p1)

    plt.plot(ns, v1, '-o', label=f'{x} features')
    # plt.plot(xs, v1, '-o', label=f'd_model={n}')
    plt.legend(loc='upper right')
plt.show()
