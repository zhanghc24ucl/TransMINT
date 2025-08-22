import sys

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.loss import DecayedUtilityLoss, SharpeLoss
from TransMINT.model.lstm import MinLSTM
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data
from TransMINT.tasks.cn_futs.settings import OutOfSampleWindows
from TransMINT.utils import decay_factor

seed = int(sys.argv[1])

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


trainer_cfg = TrainerConfig(
    model_class=MinLSTM,
    model_params=dict(
        d_model=16,
        num_layers=2,
        dropout=0.2,
    ),
    **base_args,
)

input_spec = build_input_spec(version)

data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 128,
    time_step = 180,  # 15 hours
)


bt_cfg = BacktestConfig(
    windows=OutOfSampleWindows,
    data_cfg=data_cfg,
    trainer_cfg=trainer_cfg,
)

print(trainer_cfg)
bt = Backtest(bt_cfg, data_provider, store_path=f'vault/20250821_oos/s{seed}_lstm')
bt.run()
