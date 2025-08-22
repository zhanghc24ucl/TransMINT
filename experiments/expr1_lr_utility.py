import copy
import sys

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.loss import DecayedUtilityLoss, SharpeLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data

seed = int(sys.argv[1])
is_lite = len(sys.argv) > 2
print(seed, ', is_lite:', is_lite)

version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)

trainer_cfg = TrainerConfig(
    model_class=MINTransformer,
    model_params=dict(
        d_model=16,
        num_heads=4,
        dropout=0.2,
        trainable_skip_add=False,
        is_lite=is_lite,
    ),
    optimizer_class=torch.optim.AdamW,
    optimizer_params=dict(lr=3e-4),
    loss_class=DecayedUtilityLoss,
    loss_params=dict(risk_factor=0.1, expdecay_factor=None),
    valid_loss_class=SharpeLoss,
    valid_loss_params=dict(output_steps=1),
    grad_clip_norm=1,
    device='cuda',
    epochs=20,
    early_stop_patience=0,
    seed=seed,
)

input_spec = build_input_spec(version)
data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 128,
    time_step = 180,  # 15 hours
)

base_bt_cfg = BacktestConfig(
    windows=[
        ('2016-07-01', '2019-01-01', '2019-07-01', '2020-01-01'),
        ('2017-01-01', '2019-07-01', '2020-01-01', '2020-07-01'),
    ],
    data_cfg=data_cfg,
    trainer_cfg=trainer_cfg,
)

suffix = '_lite' if is_lite else ''
bts = []
for lr in [0.0010, 0.0007, 0.0003, 0.0001, 0.00007]:
    bt_cfg = copy.deepcopy(base_bt_cfg)
    bt_cfg.trainer_cfg.optimizer_params['lr'] = lr

    bt = Backtest(bt_cfg, data_provider, store_path=f'vault/20250819_lr/utility_{lr:.0e}_{seed}{suffix}')
    bts.append(bt)

for bt in bts:
    bt.run()
