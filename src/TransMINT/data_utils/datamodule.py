from dataclasses import dataclass
from typing import Dict, Iterator, Tuple, List, Any

from numpy import full
from torch import Tensor, from_numpy

from .spec import NamedInput, TargetReturn, InputSpec
from ..utils import MINUTE


@dataclass
class DataLoaderConfig:
    input_spec: InputSpec
    batch_size: int = 32
    time_step: int = 16
    offset: int = 0 * MINUTE


class NamedInputDataLoader:
    def __len__(self):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Tuple[NamedInput, TargetReturn]]:
        raise NotImplementedError()


class DataProvider:
    def get_dataloader(
            self,
            config: DataLoaderConfig,
            start_time,
            stop_time,
    ) -> NamedInputDataLoader:
        raise NotImplementedError()


class CompositeNamedInputDataLoader(NamedInputDataLoader):
    def __init__(self, data_loaders):
        self.data_loaders = data_loaders

    def __len__(self):
        return sum((len(dl) for dl in self.data_loaders), 0)

    def __iter__(self) -> Iterator[Tuple[NamedInput, TargetReturn]]:
        for dl in self.data_loaders:
            yield from dl


def safe_sliding_view(arr, start: int, batch_size: int, time_step: int):
    """
    Safe sliding window using copy.
    """
    from numpy.lib.stride_tricks import as_strided

    selected = arr[start:start + batch_size + time_step - 1]
    if selected.ndim == 1:
        selected = selected[:, None]

    stride0, stride1 = selected.strides
    out = as_strided(
        selected,
        shape=(batch_size, time_step, selected.shape[1]),
        strides=(stride0, stride0, stride1)
    ).copy()
    return out


class SlidingTableDataLoader(NamedInputDataLoader):
    def __init__(
            self,
            config: DataLoaderConfig,
            target_returns,
            static_features: Dict[str, int],
            table_features,
            ticker: str = '',
    ):
        self.config = config
        self.input_spec = config.input_spec
        self.ticker = ticker

        # The target_returns should at least have 3 fields:
        # * time
        # * date
        # * return
        self.target_returns = target_returns

        self.static_features = static_features

        # The table_features can be:
        # * Dict[str, ndarray]
        # * rec_array
        # * DataFrame
        # All of them should be the same length as target_returns
        self.table_features = table_features

        self.time_step = config.time_step
        self.batch_size = config.batch_size

        assert len(self.target_returns) >= self.time_step, 'no enough data'
        self.n_sample = len(self.target_returns) - self.time_step + 1

    def __len__(self):
        B = self.batch_size
        return (self.n_sample + B - 1) // B

    def __iter__(self) -> Iterator[Tuple[NamedInput, TargetReturn]]:
        B = self.batch_size
        T = self.time_step
        N = self.n_sample

        raw_tgt_returns = self.target_returns

        for batch_start in range(0, N, B):
            batch_stop = min(batch_start + B, N)
            actual_batch_size = batch_stop - batch_start

            tgt_value = safe_sliding_view(
                raw_tgt_returns['norm_target_return'],
                batch_start, actual_batch_size, T
            )

            # extract details of the last time_step
            last_start = batch_start + T - 1
            last_stop = last_start + actual_batch_size

            details = raw_tgt_returns[last_start:last_stop][['time', 'date', 'target_return']]
            tgt_value = TargetReturn(self.ticker, tgt_value, details)

            raw_data = {}
            for f in self.input_spec.features:
                if f.feature_class == 'static':
                    v = full(
                        actual_batch_size,
                        self.static_features[f.name], dtype=int
                    )
                else:
                    v = safe_sliding_view(
                        self.table_features[f.name],
                        batch_start, actual_batch_size, T
                    )
                raw_data[f.name] = v

            fvalue = NamedInput(
                raw_data=raw_data,
                input_spec=self.input_spec,
                batch_size=actual_batch_size,
                time_step=T,
            )
            yield fvalue, tgt_value
