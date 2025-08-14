import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.model.loss import DecayedUtilityLoss, SharpeLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data

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
    optimizer_params=dict(lr=0.0001),
    loss_class=DecayedUtilityLoss,
    loss_params=dict(risk_factor=0.1),
    valid_loss_class=SharpeLoss,
    valid_loss_params=dict(output_steps=1),
    # scheduler_name='one_cycle',
    # scheduler_params={
    #     "max_lr": 0.0001,
    #     "pct_start": 0.3,
    #     "div_factor": 10.0,          # initial lr = 1e-5
    #     "final_div_factor": 1000.,   #   final lr = 1e-7
    #     "anneal_strategy": "cos",
    # },
    scheduler_name='warmup_cosine',
    scheduler_params={
        "warmup_pct": 0.1,
        "min_lr_ratio": 1e-3,
    },
    grad_clip_norm=1,
    device='cuda',
    epochs=10,
    early_stop_patience=0,
    seed=63,
)

input_spec = build_input_spec(version)
data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 128,
    time_step = 180,  # 15 hours
)

bt_cfg = BacktestConfig(
    windows=[
        ('2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01'),
    ],
    data_cfg=data_cfg,
    trainer_cfg=trainer_cfg,
)

# bt = Backtest(bt_cfg, data_provider, store_path=f'experiments/demo_scheduler/one_cycle')
# bt = Backtest(bt_cfg, data_provider, store_path=f'experiments/demo_scheduler/one_cycle_ck')
# bt = Backtest(bt_cfg, data_provider, store_path=f'experiments/demo_scheduler/warmup')
bt = Backtest(bt_cfg, data_provider, store_path=f'experiments/demo_scheduler/warmup_ck')
bt.run()
