import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.loss import SharpeLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data

raw_data = load_data('../data')

data_provider = CNFutDataProvider(raw_data)

input_spec = build_input_spec()
data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 64,
    time_step = 180,  # 15 hours
)

trainer_cfg = TrainerConfig(
    model_class=MINTransformer,
    model_params=dict(
        d_model=32,
        num_heads=4,
        output_size=1,
        dropout=0.2,
        trainable_skip_add=False,
    ),
    optimizer_class=torch.optim.Adam,
    optimizer_params=dict(
        lr=0.001,
    ),
    loss_class=SharpeLoss,
    loss_params=dict(
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


bt_cfg = BacktestConfig(
        windows=[
            ('2020-01-01', '2020-12-01', '2021-01-01', '2021-02-01'),
            ('2020-02-01', '2021-01-01', '2021-02-01', '2021-03-01'),
            ('2020-03-01', '2021-02-01', '2021-03-01', '2021-04-01'),
        ],
        data_cfg=data_cfg,
        trainer_cfg=trainer_cfg,
)
bt = Backtest(bt_cfg, data_provider, store_path='/tmp/demo_cn_futs')
bt.run()
perf = bt.performance()

from matplotlib import pyplot as plt

cumulative_return = (1 + perf.returns).cumprod()
plt.figure(figsize=(12, 6))
plt.plot(perf.dates, cumulative_return)
plt.title(f'Cumulative Portfolio Return: Sharpe={perf.sharpe_ratio:.03f}')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()
