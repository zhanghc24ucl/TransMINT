import sys

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.loss import DecayedUtilityLoss, SharpeLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data
from TransMINT.tasks.cn_futs.settings import InSampleWindows

seed = int(sys.argv[1])
choice = sys.argv[2]
is_lite = len(sys.argv) > 3

choices = {
    'no':    (0.0001, None, None),
    'w1': (0.0001, 0.05, 0.05),
    'w3': (0.0003, 0.10, 0.05),
}
lr, pct, min_lr_ratio = choices[choice]

version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)

if pct is None:
    args = dict()
else:
    args = dict(
        scheduler_name='warmup_cosine',
        scheduler_params={
            "warmup_pct": pct,
            "min_lr_ratio": min_lr_ratio,
        },
    )

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
    optimizer_params=dict(lr=lr),
    loss_class=DecayedUtilityLoss,
    loss_params=dict(risk_factor=0.1),
    valid_loss_class=SharpeLoss,
    valid_loss_params=dict(output_steps=1),
    grad_clip_norm=1,
    device='cuda',
    epochs=30,
    early_stop_patience=0,
    seed=seed,
    **args,
)

input_spec = build_input_spec(version)
data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 128,
    time_step = 180,  # 15 hours
)

bt_cfg = BacktestConfig(
    windows=InSampleWindows,
    data_cfg=data_cfg,
    trainer_cfg=trainer_cfg,
)

suffix = '_lite' if is_lite else ''
print(bt_cfg.trainer_cfg.scheduler_name, bt_cfg.trainer_cfg.scheduler_params)
bt = Backtest(bt_cfg, data_provider, store_path=f'vault/20250819_scheduler/s{seed}_{choice}{suffix}')
bt.run()
