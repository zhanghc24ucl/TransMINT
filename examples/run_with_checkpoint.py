from copy import deepcopy

import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.trainer import Trainer, TrainerConfig
from TransMINT.model.loss import SharpeLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.mon_trans.data import MomTransDataProvider, build_input_spec, load_data
from TransMINT.utils import set_seed

r = load_data('../data/mom_trans/quandl_cpd_126lbw.csv')
data_provider = MomTransDataProvider(r)

trainer_cfg_10 = TrainerConfig(
    model_class=MINTransformer,
    model_params=dict(
        d_model=32,
        num_heads=4,
        output_size=1,
        dropout=0.2,
        trainable_skip_add=False,
    ),
    optimizer_class=torch.optim.Adam,
    optimizer_params=dict(lr=0.001),
    loss_class=SharpeLoss,
    loss_params=dict(),
    valid_loss_class=SharpeLoss,
    valid_loss_params=dict(output_steps=1),
    grad_clip_norm=1,

    device='cuda',
    seed=63,
    epochs=10,

    early_stop_patience=3,
)

trainer_cfg_5 = deepcopy(trainer_cfg_10)
trainer_cfg_5.epochs = 5

input_spec = build_input_spec(r.ticker.cat.categories.size, 126, 21)
data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 64,
    time_step = 252,
)

print('start first 5 steps')
case1_5 = Trainer(
    cfg=trainer_cfg_5,
    input_spec=input_spec,
    train_loader=data_provider.get_dataloader(data_cfg, '2017-01-01', '2019-07-01'),
    valid_loader=data_provider.get_dataloader(data_cfg, '2019-07-01', '2020-01-01'),
)
case1_5.initialize()
case1_5.fit()
print('finish first 5 steps')

snapshot = case1_5.snapshot
snapshot.trainer_state['completed'] = False
torch.save(snapshot, '/tmp/case1_5.pt')
snapshot = torch.load('/tmp/case1_5.pt', weights_only=False, map_location='cpu')

set_seed(0)

print('start last 5 steps')
case1_10 = Trainer(
    cfg=trainer_cfg_10,
    input_spec=input_spec,
    train_loader=data_provider.get_dataloader(data_cfg, '2017-01-01', '2019-07-01'),
    valid_loader=data_provider.get_dataloader(data_cfg, '2019-07-01', '2020-01-01'),
)
case1_10.initialize(snapshot)
case1_10.fit()
print('finish last 5 steps')

set_seed(0)

print('start 10 steps')
case2_10 = Trainer(
    cfg=trainer_cfg_10,
    input_spec=input_spec,
    train_loader=data_provider.get_dataloader(data_cfg, '2017-01-01', '2019-07-01'),
    valid_loader=data_provider.get_dataloader(data_cfg, '2019-07-01', '2020-01-01'),
)
case2_10.initialize()
case2_10.fit()
print('finish 10 steps')
