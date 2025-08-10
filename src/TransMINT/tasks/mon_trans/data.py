import numpy as np
import pandas as pd

from TransMINT.data_utils.datamodule import DataLoaderConfig, DataProvider, NamedInputDataLoader
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


class MomTransDataProvider(DataProvider):
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def get_dataloader(
            self,
            config: DataLoaderConfig,
            start_time,
            stop_time,
    ) -> NamedInputDataLoader:
        start_time = pd.to_datetime(start_time)
        stop_time = pd.to_datetime(stop_time)

        raw_data = self.raw_data
        time_step = config.time_step

        ticker_dataloaders = []
        for j, ticker in enumerate(sorted(raw_data.ticker.cat.categories)):
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

            dloader = self._create_ticker_data_loader(config, j, ticker, ticker_data)
            ticker_dataloaders.append(dloader)

        from ...data_utils.datamodule import CompositeNamedInputDataLoader
        return CompositeNamedInputDataLoader(ticker_dataloaders)

    @staticmethod
    def _create_ticker_data_loader(config, ticker_ix, ticker, ticker_data):
        from ...data_utils.datamodule import SlidingTableDataLoader

        N = len(ticker_data)
        # target_returns
        tgt_rets = np.empty(N, dtype=[
            ('time', 'datetime64[ns]'),
            ('date', 'datetime64[D]'),
            ('target_return', float),
            ('norm_target_return', float),
        ])
        tgt_rets['time'] = ticker_data.index.to_numpy()
        tgt_rets['date'] = ticker_data.index.to_numpy()
        tgt_rets['target_return'] = ticker_data.target_returns.to_numpy()
        # FIXME: we do not have normalized target return in mom_trans data.
        tgt_rets['norm_target_return'] = ticker_data.target_returns.to_numpy()

        static_features = {'ticker': ticker_ix}
        table_features = {}

        for f in config.input_spec.features:
            if f.feature_class == 'static':
                # only static feature: ticker
                continue
            table_features[f.name] = ticker_data[f.name].to_numpy()

        dataloader = SlidingTableDataLoader(
            config,
            tgt_rets,
            static_features,
            table_features,
            ticker=ticker,
        )
        return dataloader
