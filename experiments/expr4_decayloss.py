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

mode = sys.argv[1]
print(mode)

mode_candidates = {
    'k05_none': (0.05, None),
    'k05_180':  (0.05, 180),
    'k05_20':   (0.05, 20),
    'k10_none': (0.1, None),
    'k10_180':  (0.1, 180),
    'k10_20':   (0.1, 20),
    'k20_none': (0.2, None),
    'k20_180':  (0.2, 180),
    'k20_20':   (0.2, 20),
    'k30_none': (0.3, None),
    'k30_180':  (0.3, 180),
    'k30_20':   (0.3, 20),
    'k50_none': (0.5, None),
    'k50_180':  (0.5, 180),
    'k50_20':   (0.5, 20),
}

version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)

loss_k, loss_hl = mode_candidates[mode]
loss_decay = None if loss_hl is None else decay_factor(loss_hl)

trainer_cfg = TrainerConfig(
    model_class=MINTransformer,
    model_params=dict(
        d_model=16,
        num_heads=4,
        dropout=0.2,
        trainable_skip_add=False,
    ),
    optimizer_class=torch.optim.AdamW,
    optimizer_params=dict(lr=2e-4),
    loss_class=DecayedUtilityLoss,
    loss_params=dict(risk_factor=loss_k, expdecay_factor=loss_decay),
    scheduler_name='warmup_cosine',
    scheduler_params={
        "warmup_pct": 0.10,
        "min_lr_ratio": 0.05,
    },
    valid_loss_class=SharpeLoss,
    valid_loss_params=dict(output_steps=1),
    grad_clip_norm=1,
    device='cuda',
    epochs=30,
    min_epochs=25,
    early_stop_patience=5,
    seed=63,
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

print(mode, loss_hl, bt_cfg.trainer_cfg.loss_params)
bt = Backtest(bt_cfg, data_provider, store_path=f'vault/20250815_decayloss/{mode}')
bt.run()
