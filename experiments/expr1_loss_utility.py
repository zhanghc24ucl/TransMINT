import copy
import sys

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.loss import SharpeLoss, UtilityLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data
from TransMINT.tasks.cn_futs.settings import InSampleWindows

seed = int(sys.argv[1])

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
    ),
    optimizer_class=torch.optim.AdamW,
    optimizer_params=dict(lr=0.001),
    loss_class=UtilityLoss,
    loss_params=dict(risk_factor=0.1),
    valid_loss_class=SharpeLoss,
    valid_loss_params=dict(output_steps=1),
    grad_clip_norm=1,
    device='cuda',
    epochs=20,
    early_stop_patience=20,
    seed=63,
)

input_spec = build_input_spec(version)
data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 64,
    time_step = 180,  # 15 hours
)

base_bt_cfg = BacktestConfig(
    windows=InSampleWindows,
    data_cfg=data_cfg,
    trainer_cfg=trainer_cfg,
)

bts = []
for lr in [0.001, 0.0001]:
    for o in [None]:
        bt_cfg = copy.deepcopy(base_bt_cfg)
        bt_cfg.trainer_cfg.seed = seed
        bt_cfg.trainer_cfg.optimizer_params['lr'] = lr
        bt_cfg.trainer_cfg.loss_params['output_steps'] = o

        bt = Backtest(bt_cfg, data_provider, store_path=f'vault/20250811_loss_utility/s{seed}_l{lr}_o{o or "None"}')
        bts.append(bt)

for bt in bts:
    bt.run()
