import copy

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
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
    loss_class=DecayedUtilityLoss,
    loss_params=dict(risk_factor=0.1, expdecay_factor=None),
    valid_loss_class=SharpeLoss,
    valid_loss_params=dict(output_steps=1),
    grad_clip_norm=1,
    device='cuda',
    epochs=15,
    early_stop_patience=15,
    seed=63,
)

input_spec = build_input_spec(version)
data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 64,
    time_step = 180,  # 15 hours
)

base_bt_cfg = BacktestConfig(
    windows=InSampleWindows,
    data_cfg=data_cfg,
    trainer_cfg=trainer_cfg,
)

bts = []
hl = [None, 180, 60, 20]
dfs = []
for k in hl:
    if k is None:
        df = None
    else:
        df = 0.5 ** (1 / k)
    print(k, df)
    dfs.append(df)

for df, hl in zip(dfs, hl):
    bt_cfg = copy.deepcopy(base_bt_cfg)
    bt_cfg.trainer_cfg.loss_params['expdecay_factor'] = df

    bt = Backtest(bt_cfg, data_provider, store_path=f'vault/20250812_decay_loss/hl{hl}')
    bts.append(bt)

for bt in bts:
    bt.run()
