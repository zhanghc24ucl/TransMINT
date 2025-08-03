import torch

from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.engine.backtest import Backtest, BacktestConfig
from TransMINT.engine.trainer import TrainerConfig
from TransMINT.engine.tuner import Tuner, TunerConfig
from TransMINT.model.loss import SharpeLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.mon_trans.data import MomTransDataProvider, build_input_spec, load_data

r = load_data('../data/mom_trans/quandl_cpd_126lbw.csv')
data_provider = MomTransDataProvider(r)

trainer_cfg = TrainerConfig(
    model_class=MINTransformer,
    model_params=dict(
        d_model=32,
        num_heads=4,
        output_size=1,
        dropout=0.2,
        trainable_skip_add=False,
    ),
    optimizer_class=torch.optim.Adam,
    optimizer_params=dict(
        lr=0.001,
    ),
    loss_class=SharpeLoss,
    loss_params=dict(
    ),
    valid_loss_class=SharpeLoss,
    valid_loss_params=dict(
        output_steps=1,
    ),
    grad_clip_norm=1,

    device='cuda',
    epochs=100,
    seed=63,

    early_stop_patience=30,
)

input_spec = build_input_spec(r.ticker.cat.categories.size, 126, 21)
data_cfg = DataLoaderConfig(
    input_spec=input_spec,
    batch_size = 64,
    time_step = 252,
)

windows = [
    # fold 1
    [
        ('2017-01-01', '2019-07-01', '2020-01-01', '2020-07-01'),
    ],
    # fold 2
    [
        ('2017-01-01', '2020-01-01', '2020-07-01', '2021-01-01'),
    ]
]

tuner_cfg = TunerConfig(
    expr_id='demo',

    windows=windows,
    data_config=data_cfg,
    trainer_config=trainer_cfg,

    batch_sizes=[32, 64, 128],
    d_models=[16, 32, 64],
    n_epochs=[100],

    lr_range=0.001,
    dropout_range=(0.0, 0.2),

    n_trials=10,
    store_db='demo/tune.db',
)
tuner = Tuner(tuner_cfg, data_provider)
tuner.tune()

best_dcfg, best_tcfg = tuner.best_config()
bcfg = BacktestConfig(
    windows=[
        ('2017-01-01', '2020-07-01', '2021-01-01', '2021-07-01'),
        ('2017-01-01', '2021-01-01', '2021-07-01', '2022-01-01'),
    ],
    data_cfg=best_dcfg,
    trainer_cfg=best_tcfg,
)
backtest = Backtest(bcfg, data_provider)
backtest.run()
perf = backtest.performance()

from matplotlib import pyplot as plt

cumulative_return = (1 + perf.returns).cumprod()
plt.figure(figsize=(12, 6))
plt.plot(perf.dates, cumulative_return)
plt.title('Cumulative Portfolio Return')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()

