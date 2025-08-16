import sys

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.loss import DecayedUtilityLoss, SharpeLoss
from TransMINT.model.lstm import FusionLSTM
from TransMINT.model.transformer import FusionTransformer, MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data
from TransMINT.tasks.cn_futs.settings import InSampleWindows

option = sys.argv[1]
model, static_args = option.split('_')
print(model, static_args)

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


trainer_configs = {
    'MINTrans': TrainerConfig(
        model_class=MINTransformer,
        model_params=dict(
            d_model=16,
            num_heads=4,
            dropout=0.2,
            trainable_skip_add=False,
        ),
        **base_args,
    ),
    'FusionTrans': TrainerConfig(
        model_class=FusionTransformer,
        model_params=dict(
            d_model=16,
            num_heads=4,
            dropout=0.2,
            trainable_skip_add=False,
        ),
        **base_args,
    ),
    'FusionLSTM': TrainerConfig(
        model_class=FusionLSTM,
        model_params=dict(
            d_model=16,
            dropout=0.2,
        ),
        **base_args,
    ),
}

args = {
    'full': dict(),
    'nosector': dict(exclude=['sector']),
    'nostatic': dict(exclude=['ticker', 'sector']),
}[static_args]

trainer_cfg = trainer_configs[model]
input_spec = build_input_spec(version, **args)
print(input_spec.count(feature_class='static'), input_spec.count(feature_class='observed'))
print(input_spec.get(feature_class='static'))

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
bt = Backtest(bt_cfg, data_provider, store_path=f'vault/20250815_no_static/{option}')
bt.run()
