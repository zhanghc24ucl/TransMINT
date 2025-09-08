import sys

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.linear import MinLinear
from TransMINT.model.loss import DecayedUtilityLoss, SharpeLoss
from TransMINT.model.lstm import FusionLSTM, MinLSTM
from TransMINT.model.transformer import FusionTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data
from TransMINT.tasks.cn_futs.settings import InSampleWindows, OutOfSampleWindows
from TransMINT.utils import decay_factor

seed = int(sys.argv[1])
arch = sys.argv[2]
model = sys.argv[3]

is_lite = arch == 'lite'
print(f'seed: {seed}, arch: {arch}/{is_lite}, model: {model}')

version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)

base_args = dict(
    optimizer_class=torch.optim.AdamW,
    optimizer_params=dict(lr=3e-4),
    loss_class=DecayedUtilityLoss,
    loss_params=dict(risk_factor=0.2, expdecay_factor=decay_factor(180)),
    valid_loss_class=SharpeLoss,
    valid_loss_params=dict(output_steps=1),
    scheduler_name='warmup_cosine',
    scheduler_params={
        "warmup_pct": 0.10,
        "min_lr_ratio": 0.05,
    },
    grad_clip_norm=1,
    device='cuda',
    epochs=30,
    min_epochs=25,
    early_stop_patience=5,
    seed=seed,
)

model_cfg = {
    'fLSTM': dict(
        model_class=FusionLSTM,
        model_params=dict(
            d_model=32,
            dropout=0.2,
            is_lite=is_lite,
        ),
    ),
    'fTrans': dict(
        model_class=FusionTransformer,
        model_params=dict(
            d_model=32,
            num_heads=4,
            dropout=0.2,
            trainable_skip_add=False,
            is_lite=is_lite,
        ),
    ),
}[model]


trainer_cfg = TrainerConfig(
    **model_cfg,
    **base_args,
)

input_spec = build_input_spec(version)

data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 256,
    time_step = 180,  # 15 hours
)


windows = OutOfSampleWindows
bt_cfg = BacktestConfig(
    windows=windows,
    data_cfg=data_cfg,
    trainer_cfg=trainer_cfg,
)

print(trainer_cfg)
bt = Backtest(bt_cfg, data_provider, store_path=f'vault/20250829_abla/s{seed}_{model}_{arch}')
bt.run()
