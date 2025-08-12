import copy

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig, Trainer
from TransMINT.model.base import MinLinear
from TransMINT.model.loss import SharpeLoss, UtilityLoss
from TransMINT.model.lstm import MinFusionLSTM, MinLSTM
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data

version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)

base_args = dict(
    optimizer_class=torch.optim.Adam,
    optimizer_params=dict(
        lr=0.001,
    ),
    loss_class=UtilityLoss,
    loss_params=dict(
        risk_factor=0.1,
    ),
    valid_loss_class=SharpeLoss,
    valid_loss_params=dict(
        output_steps=1,
    ),
    grad_clip_norm=1,

    device='cuda',
    epochs=10,
    seed=63,

    early_stop_patience=3,
)

input_spec = build_input_spec(version)
data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 64,
    time_step = 180,  # 15 hours
)

trainer_cfg_lstm = TrainerConfig(
    model_class=MinLSTM,
    model_params=dict(
        d_model=16,
        dropout=0.2,
        num_layers=2,
    ),
    **base_args,
)
trainer_cfg_lstm_fusion = TrainerConfig(
    model_class=MinFusionLSTM,
    model_params=dict(
        d_model=16,
        dropout=0.2,
    ),
    **base_args,
)
trainer_cfg_linear = TrainerConfig(
    model_class=MinLinear,
    model_params=dict(),
    **base_args,
)
trainer_cfg_trans = TrainerConfig(
    model_class=MINTransformer,
    model_params=dict(
        d_model=16,
        num_heads=4,
        dropout=0.2,
        trainable_skip_add=False,
    ),
    **base_args,
)

labels = ['lstm', 'fusion_lstm', 'fusion_trans']
for cfg, label in zip([trainer_cfg_lstm, trainer_cfg_lstm_fusion, trainer_cfg_trans], labels):
    for d_model in [16, 32, 64, 128, 256]:
        cfg = copy.deepcopy(cfg)
        cfg.model_params['d_model'] = d_model
        trainer = Trainer(cfg, input_spec, None, None)
        trainer.initialize()
        print(label, cfg.model_params['d_model'], trainer.n_parameters())
