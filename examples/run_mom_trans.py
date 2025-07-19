import torch

from TransMINT.engine.trainer import Trainer, TrainerConfig
from TransMINT.model.loss import SharpeLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.mon_trans.data import build_input_spec, create_data_loader, load_data

r = load_data('../data/mom_trans/quandl_cpd_126lbw.csv')

input_spec = build_input_spec(r.ticker.cat.categories.size, 126, 21)

batch_size = 64
train_loader = create_data_loader(
    r, input_spec,
    '2017-01-01', '2020-01-01',
    time_step=252, batch_size=batch_size,
)
valid_loader = create_data_loader(
    r, input_spec,
    '2020-01-01', '2021-01-01',
    time_step=252, batch_size=batch_size,
)
test_loader = create_data_loader(
    r, input_spec,
    '2021-01-01', '2022-01-01',
    time_step=252, batch_size=batch_size,
)

print(len(train_loader), len(valid_loader), len(test_loader))

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

    early_stop_patience=30,
)

m = Trainer(train_cfg, input_spec, train_loader, valid_loader)
m.fit()
print(f'out-of-sample test results: {m.evaluate(test_loader): .4f}')
