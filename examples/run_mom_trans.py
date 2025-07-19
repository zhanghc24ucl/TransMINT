import torch

from TransMINT.engine.trainer import Trainer, TrainerConfig
from TransMINT.model.loss import SharpeLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.mon_trans.data import build_input_spec, create_data_loader, load_data

r = load_data('../data/mom_trans/quandl_cpd_126lbw.csv')

input_spec = build_input_spec(r.ticker.cat.categories.size, 126, 21)

train_loader = create_data_loader(
    r, input_spec,
    '2017-01-01', '2020-01-01',
    time_step=252, batch_size=64,
)
valid_loader = create_data_loader(
    r, input_spec,
    '2020-01-01', '2021-01-01',
    time_step=252, batch_size=64,
)
print(len(valid_loader))

train_cfg = TrainerConfig(
    model_class=MINTransformer,
    model_params=dict(
        d_model=32,
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
    device='cuda',
    log_interval=10,
    epochs=100,
)

m = Trainer(train_cfg, input_spec, train_loader)
m.fit()
metrics = m.evaluate(valid_loader)
