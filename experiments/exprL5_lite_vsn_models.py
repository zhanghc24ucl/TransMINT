import sys

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.loss import DecayedUtilityLoss, SharpeLoss
from TransMINT.model.lstm import FusionLSTM, MinLSTM
from TransMINT.model.transformer import FusionTransformer, MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data

option = sys.argv[1]  # FusionLSTM_16_16_16, FusionTrans_16_16_4, MINTrans_16_16_1
m_name, d_model, d_static, d_observed = option.split('_')
d_model = int(d_model)
d_static = int(d_static)
d_observed = int(d_observed)

version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)

base_args = dict(
    optimizer_class=torch.optim.AdamW,
    optimizer_params=dict(lr=2e-4),
    loss_class=DecayedUtilityLoss,
    loss_params=dict(risk_factor=0.1),
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


trainer_cfg = {
    'FusionLSTM': TrainerConfig(
        model_class=FusionLSTM,
        model_params=dict(
            d_model=d_model,
            dropout=0.2,
            d_static=d_static,
            d_observed=d_observed,
            is_lite=True,
        ),
        **base_args,
    ),
    'FusionTrans': TrainerConfig(
        model_class=FusionTransformer,
        model_params=dict(
            d_model=d_model,
            num_heads=4,
            dropout=0.2,
            d_static=d_static,
            d_observed=d_observed,
            is_lite=True,
        ),
        **base_args,
    ),
    'MINTrans': TrainerConfig(
        model_class=MINTransformer,
        model_params=dict(
            d_model=d_model,
            num_heads=4,
            dropout=0.2,
            d_static=d_static,
            d_observed=d_observed,
            is_lite=True,
        ),
        **base_args,
    ),
}[m_name]

input_spec = build_input_spec(version)

data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 128,
    time_step = 180,  # 15 hours
)


ws = [
        ('2017-01-01', '2019-07-01', '2020-01-01', '2020-07-01'),

        ('2016-03-01', '2018-07-01', '2019-01-01', '2019-07-01'),
        ('2016-07-01', '2019-01-01', '2019-07-01', '2020-01-01'),
        # ('2017-01-01', '2019-07-01', '2020-01-01', '2020-07-01'),
        ('2017-07-01', '2020-01-01', '2020-07-01', '2021-01-01'),
]

bt_cfg = BacktestConfig(
    windows=ws,
    data_cfg=data_cfg,
    trainer_cfg=trainer_cfg,
)

print(trainer_cfg)
bt = Backtest(bt_cfg, data_provider, store_path=f'vault/20250815_lite_vsn_models/{option}')
bt.run()
