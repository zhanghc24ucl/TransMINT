
import numpy as np

from . import settings as cn_futs
from ...data_utils.datamodule import DataLoaderConfig, DataProvider, NamedInputDataLoader
from ...data_utils.spec import FeatureSpec, InputSpec


def build_input_spec():
    n_tickers = len(cn_futs.CN_FUTS_TICKERS_FULL)
    n_sectors = len(set(cn_futs.CN_FUTS_SECTORS.values()))
    specs = [
        FeatureSpec('ticker', 'static', 'categorical', category_size=n_tickers),
        FeatureSpec('sector', 'static', 'categorical', category_size=n_sectors),

        FeatureSpec('norm_ret_1m',      'observed', 'real'),
        FeatureSpec('norm_ret_5m',      'observed', 'real'),
        FeatureSpec('ew_vol_1m',        'observed', 'real'),
        FeatureSpec('norm_time_of_day', 'observed', 'real'),
    ]
    return InputSpec(specs)


def load_data(data_dir, version='v1'):
    rv = {}
    for k in cn_futs.CN_FUTS_TICKERS_FULL:
        rv[k] = np.load(f'{data_dir}/cn_futs/tabular/{version}/{k}.npy')
    return rv


class CNFutDataProvider(DataProvider):
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.tickers = sorted(raw_data.keys())

        sectors = sorted(set(cn_futs.CN_FUTS_SECTORS.values()))
        sector_ix = {s: i for i, s in enumerate(sectors)}
        self.ticker_sectors = [
            sector_ix[cn_futs.CN_FUTS_SECTORS[ticker]]
            for ticker in self.tickers
        ]

    def get_dataloader(
            self,
            config: DataLoaderConfig,
            start_time,
            stop_time,
    ) -> NamedInputDataLoader:
        tickers = self.tickers
        raw_data = self.raw_data
        time_step = config.time_step

        start_time = np.datetime64(start_time)
        stop_time = np.datetime64(stop_time)

        ticker_dataloaders = []
        for j, ticker in enumerate(tickers):
            ticker_data = raw_data[ticker]

            date = ticker_data['date']

            bix = date.searchsorted(start_time)
            eix = date.searchsorted(stop_time)
            bix = max(time_step - 1, bix)

            if bix >= eix:
                continue

            # slice ticker data with looking back window = `time_step`
            bix_ts = bix - time_step + 1
            ticker_data = ticker_data[bix_ts:eix]

            sector_ix = self.ticker_sectors[j]

            dloader = self._create_ticker_data_loader(
                config, j, sector_ix, ticker, ticker_data)
            ticker_dataloaders.append(dloader)

        from ...data_utils.datamodule import CompositeNamedInputDataLoader
        return CompositeNamedInputDataLoader(ticker_dataloaders)

    @staticmethod
    def _create_ticker_data_loader(config, ticker_ix, sector_ix, ticker, ticker_data):
        from ...data_utils.datamodule import SlidingTableDataLoader

        N = len(ticker_data)
        # target_returns
        tgt_rets = np.empty(N, dtype=[
            ('time', 'datetime64[ns]'),
            ('date', 'datetime64[D]'),
            ('return', float),
        ])
        tgt_rets['time'] = ticker_data['time'].astype('datetime64[ns]')
        tgt_rets['date'] = ticker_data['date']
        tgt_rets['return'] = ticker_data['target_return']

        static_features = {'ticker': ticker_ix, 'sector': sector_ix}
        table_features = {}

        for f in config.input_spec.features:
            if f.feature_class == 'static':
                # only static feature: ticker
                continue
            table_features[f.name] = ticker_data[f.name]

        dataloader = SlidingTableDataLoader(
            config,
            tgt_rets,
            static_features,
            table_features,
            ticker=ticker,
        )
        return dataloader
