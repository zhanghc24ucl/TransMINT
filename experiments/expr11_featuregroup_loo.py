import sys

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.linear import MinLinear
from TransMINT.model.loss import DecayedUtilityLoss, SharpeLoss
from TransMINT.model.lstm import FusionLSTM, MinLSTM
from TransMINT.model.transformer import FusionTransformer, MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data
from TransMINT.tasks.cn_futs.settings import InSampleWindows, OutOfSampleWindows
from TransMINT.utils import decay_factor


seed = int(sys.argv[1])
arch = sys.argv[2]
feature_opts = sys.argv[3]

is_lite = arch == 'lite'

excludes = {
    'no_static': [
        'ticker', 'sector'
    ],
    'no_time': [
        'norm_time_of_day'
    ],
    'no_return': [
        'norm_ret_1m[0]', 'norm_ret_1m[1]', 'norm_ret_1m[2]', 'norm_ret_1m[3]', 'norm_ret_1m[4]',
        'norm_ret_5m', 'norm_ret_30m', 'norm_ret_240m'
    ],
    'no_macd': [
        'macd_8_24_16_1m', 'macd_16_48_16_1m', 'macd_32_96_16_1m'
    ],
    'no_vol': [
        'norm_log_ewvol_1m', 'norm_log_gkvol_1m',
    ],
    'no_other': [
        'norm_log_value_5m', 'norm_mfv_30m', 'norm_er_5m', 'norm_clv_30m',
    ]
}

version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)
input_spec = build_input_spec(version, exclude=excludes[feature_opts])

print(f'seed: {seed}, arch: {arch}/{is_lite}, features: {feature_opts}/{input_spec.count(feature_class="static")}/{input_spec.count(feature_class="observed")}')
print([f.name for f in input_spec.features])

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
    model_class=MINTransformer,
    model_params=dict(
        d_model=32,
        num_heads=4,
        dropout=0.2,
        is_lite=is_lite,
    ),
    **base_args,
)

data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 256,
    time_step = 180,  # 15 hours
)

bt_cfg = BacktestConfig(
    windows=OutOfSampleWindows,
    data_cfg=data_cfg,
    trainer_cfg=trainer_cfg,
)

print(trainer_cfg)
bt = Backtest(bt_cfg, data_provider, store_path=f'vault/20250829_loo/s{seed}_{arch}_{feature_opts}')
bt.run()
