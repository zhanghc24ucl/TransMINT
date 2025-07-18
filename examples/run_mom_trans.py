import torch

from TransMINT.engine.trainer import Trainer, TrainerConfig
from TransMINT.model.loss import SharpeLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.mon_trans.data import build_input_spec, create_data_loader, load_data

r = load_data('../data/mom_trans/quandl_cpd_126lbw.csv')

input_spec = build_input_spec(r.ticker.cat.categories.size)

train_loader = create_data_loader(
    r, input_spec,
    '2018-02-01', '2020-12-31',
    time_step=16, batch_size=32
)

for x, y in train_loader:
    print(x)

train_cfg = TrainerConfig(
    model_class=MINTransformer,
    model_params=dict(
        d_model=16,
        num_heads=4,
        output_size=1,
        dropout=0.1,
        trainable_skip_add=False,
    ),
    optimizer_class=torch.optim.Adam,
    optimizer_params=dict(
        lr=0.001,
    ),
    loss_class=SharpeLoss,
    device='cpu',
)

m = Trainer(train_cfg, input_spec, train_loader)
m.fit()
