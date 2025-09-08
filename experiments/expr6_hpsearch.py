import sys

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.loss import DecayedUtilityLoss, SharpeLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data
from TransMINT.tasks.cn_futs.settings import InSampleWindows
from TransMINT.utils import decay_factor

# demo: python3 expr6_hpsearch.py 16 128 63 3 lite
# d_model=16, batch_size=128, seed=63, lr=3e-4, is lite: True
d_model = int(sys.argv[1])
b_size = int(sys.argv[2])
seed = int(sys.argv[3])
lr = round(float(sys.argv[4]) * 1e-4, 6)
is_lite = len(sys.argv) > 5
print(f'd_model={d_model}, b_size={b_size}, seed={seed}, lr={lr}, is lite: {is_lite}')

version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)

base_args = dict(
    optimizer_class=torch.optim.AdamW,
    optimizer_params=dict(lr=lr),
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


trainer_cfg = TrainerConfig(
    model_class=MINTransformer,
    model_params=dict(
        d_model=d_model,
        num_heads=4,
        dropout=0.2,
        is_lite=is_lite,
    ),
    **base_args,
)

input_spec = build_input_spec(version)

data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = b_size,
    time_step = 180,  # 15 hours
)


bt_cfg = BacktestConfig(
    windows=InSampleWindows,
    data_cfg=data_cfg,
    trainer_cfg=trainer_cfg,
)

print(trainer_cfg)
suffix = '_lite' if is_lite else ''
bt = Backtest(bt_cfg, data_provider, store_path=f'vault/20250827_hpsearch/s{seed}_d{d_model}_b{b_size}_l{lr}{suffix}')
bt.run()
