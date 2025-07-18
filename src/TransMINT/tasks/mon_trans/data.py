import numpy as np
import pandas as pd

from TransMINT.data_utils.spec import FeatureSpec, InputSpec


def load_data(path):
    df = pd.read_csv(path)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.ticker = df.ticker.astype('category')
    return df


def build_input_spec(ticker_size):
    return InputSpec([
        FeatureSpec('ticker',            'static',   'categorical', category_size=ticker_size),
        FeatureSpec('day_of_week',       'time_pos', 'cyclical'),
        FeatureSpec('day_of_month',      'time_pos', 'sequential'),
        FeatureSpec('norm_daily_return', 'observed', 'real'),
        FeatureSpec('macd_8_24',         'observed', 'real'),
    ])


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

def create_data_loader(raw_data, input_spec, start_time, stop_time, **dataloader_args):
    from ...data_utils.datamodule import CompositeNamedInputDataLoader
    start_time = pd.to_datetime(start_time)
    stop_time = pd.to_datetime(stop_time)
    raw_data = raw_data.iloc[(raw_data.index >= start_time) & (raw_data.index < stop_time)]

    ticker_dataloaders = []
    for ticker in raw_data.ticker.cat.categories:
        ticker_data = raw_data[raw_data.ticker == ticker]

        if not ticker_data.index.is_monotonic_increasing:
            ticker_data = ticker_data.sort_index()

        dloader = _create_ticker_data_loader(
            ticker_data,
            input_spec,
            **dataloader_args
        )
        ticker_dataloaders.append(dloader)

    return CompositeNamedInputDataLoader(ticker_dataloaders)
