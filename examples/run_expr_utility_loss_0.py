import copy

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.base import MINLinear
from TransMINT.model.loss import SharpeLoss, UtilityLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.model.lstm import MINLSTM
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data
from TransMINT.viz.backtest import plot_performance, plot_ticker_performance

raw_data = load_data('../data')

data_provider = CNFutDataProvider(raw_data)

base_args = dict(
    optimizer_class=torch.optim.Adam,
    optimizer_params=dict(
        lr=0.001,
    ),
    loss_class=UtilityLoss,
    loss_params=dict(
        risk_factor=0.0,  # no risk
    ),
    valid_loss_class=SharpeLoss,
    valid_loss_params=dict(
        output_steps=1,
    ),
    grad_clip_norm=1,

    device='cuda',
    epochs=10,
    seed=63,

    early_stop_patience=3,
)

input_spec = build_input_spec()
data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 64,
    time_step = 180,  # 15 hours
)

trainer_cfg_lstm = TrainerConfig(
    model_class=MINLSTM,
    model_params=dict(
        d_model=16,
        dropout=0.2,
        num_layers=2,
    ),
    **base_args,
)

trainer_cfg_linear = TrainerConfig(
    model_class=MINLinear,
    model_params=dict(
        time_step=data_cfg.time_step,
    ),
    **base_args,
)

trainer_cfg_trans = TrainerConfig(
    model_class=MINTransformer,
    model_params=dict(
        d_model=16,
        num_heads=4,
        dropout=0.2,
        trainable_skip_add=False,
    ),
    **base_args,
)


base_bt_cfg = BacktestConfig(
        windows=[
            ('2017-01-01', '2019-07-01', '2020-01-01', '2020-07-01'),
            ('2017-07-01', '2020-01-01', '2020-07-01', '2021-01-01'),
            ('2018-01-01', '2020-07-01', '2021-01-01', '2021-07-01'),
        ],
        data_cfg=data_cfg,
        trainer_cfg=trainer_cfg_lstm,
)
labels = ['LSTM', 'Linear', 'MINTransformer']
models = [trainer_cfg_lstm, trainer_cfg_linear, trainer_cfg_trans]

from matplotlib import pyplot as plt
for label, model in zip(labels, models):
    bt_cfg = copy.deepcopy(base_bt_cfg)
    bt_cfg.trainer_cfg = model

    bt = Backtest(bt_cfg, data_provider, store_path=f'experiments/20250803_norisk/{label}')
    bt.run()
    perfs = bt.ticker_performance()
    perf = bt.performance()
    plot_performance(perf, title=f'{label}, no risk : Sharpe = {perf.sharpe_ratio:.03f}', figsize=(14, 7))
    plot_ticker_performance(perfs, title=label, figsize=(14, 7))

plt.show()
