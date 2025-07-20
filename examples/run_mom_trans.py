import pandas as pd
import torch

from TransMINT.engine.trainer import Trainer, TrainerConfig, ValidationSharpeLoss
from TransMINT.model.loss import SharpeLoss
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.mon_trans.data import build_input_spec, create_data_loader, load_data

def run_single_expr(raw_data, intervals, train_cfg, batch_size=64, time_step=252):
    train_start, valid_start, test_start, test_end = intervals
    train_loader, _ = create_data_loader(
        raw_data, input_spec,
        train_start, valid_start,
        time_step=time_step, batch_size=batch_size,
    )
    valid_loader, _ = create_data_loader(
        raw_data, input_spec,
        valid_start, test_start,
        time_step=time_step, batch_size=batch_size,
    )
    test_loader, test_details = create_data_loader(
        raw_data, input_spec,
        test_start, test_end,
        time_step=time_step, batch_size=batch_size,
    )
    # only calculate Sharpe loss of the last time step.
    valid_loss = ValidationSharpeLoss(output_steps=1)
    m = Trainer(train_cfg, input_spec, train_loader, valid_loader, valid_loss=valid_loss)
    m.fit()
    print(f'out-of-sample test loss: {m.evaluate(test_loader): .4f}')

    pred_position = m.predict_all(test_loader)[:, -1, 0].numpy()
    ret = pred_position * test_details['return']

    df = pd.DataFrame({
        'date': test_details['date'],
        'ticker': test_details['ticker'],
        'pred_position': pred_position,
        'target_return': test_details['return'],
        'daily_return': ret,
    })
    df.to_csv(f'{test_end}.csv')

    daily_portfolio_return = df.groupby('date')['daily_return'].mean()
    daily_portfolio_return = daily_portfolio_return.sort_index()
    from math import sqrt
    sharpe = daily_portfolio_return.mean() / daily_portfolio_return.std() * sqrt(252)
    print(f'out-of-sample Sharpe ratio: {sharpe:.02f}')
    return daily_portfolio_return


r = load_data('../data/mom_trans/quandl_cpd_126lbw.csv')
input_spec = build_input_spec(r.ticker.cat.categories.size, 126, 21)

expr_intervals = [
    # train_start, train_end/valid_start, valid_end/test_start, test_end
    ('2017-01-01', '2019-07-01', '2020-01-01', '2021-01-01'),
    ('2017-01-01', '2020-07-01', '2021-01-01', '2022-01-01'),
]

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
    loss_params=dict(
    ),
    device='cuda',
    log_interval=10,
    epochs=100,

    early_stop_patience=30,
)
batch_size = 64
time_step = 252

prets = []
for e in expr_intervals:
    pret = run_single_expr(r, e, train_cfg, batch_size=batch_size, time_step=time_step)

    from math import sqrt
    ann_vol = pret.std() * sqrt(252)
    norm_pret = 0.15 * pret / ann_vol
    prets.append(norm_pret)

from matplotlib import pyplot as plt

prets = pd.concat(prets)
cumulative_return = (1 + prets).cumprod()
plt.figure(figsize=(12, 6))
cumulative_return.plot(title='Cumulative Portfolio Return')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()
