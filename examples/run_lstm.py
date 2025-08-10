import copy

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.loss import SharpeLoss, UtilityLoss
from TransMINT.model.lstm import MinFusionLSTM, MinLSTM
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data
from TransMINT.viz.backtest import plot_performance, plot_ticker_performance

version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)

base_args = dict(
    optimizer_class=torch.optim.Adam,
    optimizer_params=dict(
        lr=0.001,
    ),
    loss_class=UtilityLoss,
    loss_params=dict(
        risk_factor=0.1,
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

input_spec = build_input_spec(version)
data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 64,
    time_step = 180,  # 15 hours
)

trainer_cfg_lstm = TrainerConfig(
    model_class=MinLSTM,
    model_params=dict(
        d_model=16,
        dropout=0.2,
        num_layers=2,
    ),
    **base_args,
)

trainer_cfg_lstm_fusion = TrainerConfig(
    model_class=MinFusionLSTM,
    model_params=dict(
        d_model=16,
        dropout=0.2,
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
labels = ['LSTM_raw', 'LSTM_fusion']
models = [trainer_cfg_lstm, trainer_cfg_lstm_fusion]

bts = []

for label, model in zip(labels, models):
    bt_cfg = copy.deepcopy(base_bt_cfg)
    bt_cfg.trainer_cfg = model

    bt = Backtest(bt_cfg, data_provider, store_path=f'experiments/20250810_lstm_norm/{label}')
    bt.run()
    bts.append(bt)

from matplotlib import pyplot as plt
for label, bt in zip(labels, bts):
    perfs = bt.ticker_performance()
    perf = bt.performance()
    plot_performance(perf, title=f'{label} risk=100: Sharpe = {perf.sharpe_ratio:.03f}', figsize=(14, 7))
    plot_ticker_performance(perfs, title=label, figsize=(14, 7))
plt.show()
