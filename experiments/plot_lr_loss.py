import copy
import sys

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig, Trainer
from TransMINT.model.loss import DecayedUtilityLoss, SharpeLoss, UtilityLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data
from TransMINT.tasks.cn_futs.settings import InSampleWindows


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
    optimizer_params=dict(lr=0.001),
    # loss_class=UtilityLoss,
    # loss_params=dict(risk_factor=0.1),
    loss_class=DecayedUtilityLoss,
    loss_params=dict(risk_factor=0.1, expdecay_factor=None),
    # loss_class=SharpeLoss,
    # loss_params=dict(),
    valid_loss_class=SharpeLoss,
    valid_loss_params=dict(output_steps=1),
    grad_clip_norm=1.0,
    device='cuda',
    epochs=20,
    early_stop_patience=20,
    seed=63,
)

input_spec = build_input_spec(version)
data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 64,
    time_step = 180,  # 15 hours
)

trainer = Trainer(
    trainer_cfg, input_spec,
    data_provider.get_dataloader(data_cfg, "2016-03-01", "2020-01-01"))
trainer.initialize()
lrs, losses = trainer.lr_range_test(num_steps=500, min_steps=500, rel_worse=0.8)

from matplotlib import pyplot as plt
plt.plot(lrs, losses)
plt.xscale('log')
plt.show()
