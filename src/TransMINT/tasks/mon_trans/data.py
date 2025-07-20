import numpy as np
import pandas as pd

from TransMINT.data_utils.spec import FeatureSpec, InputSpec

def load_data(path):
    df = pd.read_csv(path)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.ticker = df.ticker.astype('category')
    return df

def build_input_spec(ticker_size, lbw1=None, lbw2=None, time_feature=False):
    specs = [
        FeatureSpec('ticker', 'static', 'categorical', category_size=ticker_size),
        FeatureSpec('norm_daily_return',     'observed', 'real'),
        FeatureSpec('norm_monthly_return',   'observed', 'real'),
        FeatureSpec('norm_quarterly_return', 'observed', 'real'),
        FeatureSpec('norm_biannual_return',  'observed', 'real'),
        FeatureSpec('norm_annual_return',    'observed', 'real'),
        FeatureSpec('macd_8_24',             'observed', 'real'),
        FeatureSpec('macd_16_48',            'observed', 'real'),
        FeatureSpec('macd_32_96',            'observed', 'real'),
    ]
    if time_feature:
        specs.extend([
            FeatureSpec('day_of_week',       'time_pos', 'cyclical'),
            FeatureSpec('day_of_month',      'time_pos', 'sequential'),
            FeatureSpec('week_of_year',      'time_pos', 'sequential'),
            FeatureSpec('month_of_year',     'time_pos', 'sequential'),
            FeatureSpec('year',              'time_pos', 'sequential'),
        ])
    if lbw1:
        specs.append(FeatureSpec(f'cp_score_{lbw1}', 'observed', 'real')),
        specs.append(FeatureSpec(f'cp_rl_{lbw1}',    'observed', 'real')),
    if lbw2:
        specs.append(FeatureSpec(f'cp_score_{lbw2}', 'observed', 'real'))
        specs.append(FeatureSpec(f'cp_rl_{lbw2}',    'observed', 'real'))
    return InputSpec(specs)


def _create_ticker_data_loader(raw_data, input_spec, **dataloader_args):
    from ...data_utils.datamodule import SlidingTableDataLoader

    N = len(raw_data)
    # target_returns
    tgt_rets = np.empty(N, dtype=[
        ('time', 'datetime64[ns]'),
        ('return', float),
    ])
    tgt_rets['time'] = raw_data.index.to_numpy()
    tgt_rets['return'] = raw_data.target_returns.to_numpy()

    static_features = {'ticker': raw_data.ticker.cat.codes.iloc[0]}
    table_features = {}

    for f in input_spec.features:
        if f.feature_class == 'static':
            # only static feature: ticker
            continue
        table_features[f.name] = raw_data[f.name].to_numpy()

    dataloader = SlidingTableDataLoader(
        input_spec,
        tgt_rets,
        static_features,
        table_features,
        **dataloader_args
    )
    return dataloader

def create_data_loader(raw_data, input_spec, start_time, stop_time, time_step, **dataloader_args):
    start_time = pd.to_datetime(start_time)
    stop_time = pd.to_datetime(stop_time)

    ticker_dataloaders = []
    details = []
    for ticker in raw_data.ticker.cat.categories:
        ticker_data = raw_data[raw_data.ticker == ticker]

        bix = ticker_data.index.searchsorted(start_time)
        eix = ticker_data.index.searchsorted(stop_time)
        bix = max(time_step - 1, bix)

        if bix >= eix:
            continue

        # slice ticker data with looking back window = `time_step`
        bix_ts = bix - time_step + 1
        ticker_data = ticker_data.iloc[bix_ts:eix]

        if not ticker_data.index.is_monotonic_increasing:
            ticker_data = ticker_data.sort_index()

        dloader = _create_ticker_data_loader(
            ticker_data,
            input_spec,
            time_step=time_step,
            **dataloader_args
        )
        detail = np.empty(
            eix - bix,
            dtype=[
                ('ticker', 'U20'),
                ('date',   'datetime64[D]'),
                ('return', float),
            ]
        )
        detail['ticker'] = ticker
        sub_data = ticker_data.iloc[time_step - 1:]
        detail['date'] = sub_data.index.to_numpy()
        detail['return'] = sub_data.target_returns.to_numpy()

        ticker_dataloaders.append(dloader)
        details.append(detail)

    details = np.concatenate(details)
    from ...data_utils.datamodule import CompositeNamedInputDataLoader
    return CompositeNamedInputDataLoader(ticker_dataloaders), details
