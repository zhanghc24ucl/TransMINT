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

feature_option = sys.argv[1]  # full, no_static, no_time, no_return, no_other

version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)

base_args = dict(
    optimizer_class=torch.optim.AdamW,
    optimizer_params=dict(lr=2e-4),
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
    seed=63,
)


trainer_cfg = TrainerConfig(
    model_class=MINTransformer,
    model_params=dict(
        d_model=16,
        num_heads=4,
        dropout=0.2,
        is_lite=True,
    ),
    **base_args,
)

args = {
    'full': dict(exclude=None),
    'no_static': dict(exclude=['ticker', 'sector']),
    'no_time': dict(exclude=['norm_time_of_day']),
    'no_return': dict(exclude=['norm_ret_1m[0]', 'norm_ret_1m[1]', 'norm_ret_1m[2]', 'norm_ret_1m[3]', 'norm_ret_1m[4]',
                               'norm_ret_5m', 'norm_ret_30m', 'norm_ret_240m']),
    'no_other': dict(exclude=['norm_log_value_5m', 'norm_mfv_30m',
                              'norm_er_5m', 'norm_clv_30m', 'norm_log_ewvol_1m', 'norm_log_gkvol_1m',
                              'macd_8_24_16_1m', 'macd_16_48_16_1m', 'macd_32_96_16_1m']),
}[feature_option]

input_spec = build_input_spec(version, **args)

print('==== feature_option')
for f in input_spec.features:
    print(f.name)

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

print(trainer_cfg)
bt = Backtest(bt_cfg, data_provider, store_path=f'vault/20250816_features/{feature_option}')
bt.run()
