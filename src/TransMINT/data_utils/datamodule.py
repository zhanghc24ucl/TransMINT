from typing import Dict, Iterator, Tuple

from numpy import full
from torch import Tensor, from_numpy

from .spec import NamedInput
from ..utils import MINUTE


class NamedInputDataLoader:
    def __len__(self):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Tuple[NamedInput, Tensor]]:
        raise NotImplementedError()


class CompositeNamedInputDataLoader(NamedInputDataLoader):
    def __init__(self, data_loaders):
        self.data_loaders = data_loaders

    def __len__(self):
        return sum((len(dl) for dl in self.data_loaders), 0)

    def __iter__(self) -> Iterator[Tuple[NamedInput, Tensor]]:
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
            input_spec,
            target_returns,
            static_features: Dict[str, int],
            table_features,
            time_step=16,
            batch_size=32,
    ):
        self.input_spec = input_spec
        self.target_returns = target_returns

        self.static_features = static_features
        self.table_features = table_features

        self.time_step = time_step
        self.batch_size = batch_size

    def __len__(self):
        N = len(self.target_returns) - self.time_step + 1
        B = self.batch_size
        return (N + B - 1) // B

    def __iter__(self) -> Iterator[Tuple[NamedInput, Tensor]]:
        B = self.batch_size
        T = self.time_step
        N = len(self.target_returns) - self.time_step + 1  # num of samples

        for batch_start in range(0, N, B):
            batch_stop = min(batch_start+B, N)
            actual_batch_size = min(T, batch_stop-batch_start)

            tgt_value = from_numpy(safe_sliding_view(
                self.target_returns['return'],
                batch_start, actual_batch_size, T
            ))

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

class SlidingTimeAlignedDataLoader(NamedInputDataLoader):
    def __init__(
            self,
            input_spec,
            target_returns,
            static_features: Dict[str, int],
            time_features,
            observed_features,
            time_step=16,
            batch_size=32,
            offset=MINUTE,
    ):
        self.input_spec = input_spec
        self.target_returns = target_returns

        self.static_features = static_features
        self.time_features = time_features
        assert len(time_features) == len(target_returns)

        self.observed_features = observed_features
        self._obs_time = observed_features['time']

        self.time_step = time_step
        self.batch_size = batch_size
        self.offset = offset

    def _get_feature(self, feature_spec, ix):
        from numpy import full, arange
        if feature_spec.feature_class == 'static':
            return full(self.time_step, self.static_features[feature_spec.name])
        elif feature_spec.feature_class == 'time_pos':
            return self.time_features[feature_spec.name]

        if feature_spec.lag_size is None:
            return self._obs_time[feature_spec.field][ix]
        raw = self._obs_time[feature_spec.field]
        # convert from 1-dim ix [15,25,35,45] with lag_size=3 to
        # [[15, 14, 13],
        #  [25, 24, 23],
        #  ...
        ix_with_lag = ix[:, None] - arange(feature_spec.lag_size)[::-1]
        return raw[ix_with_lag]

    def __len__(self):
        return len(self.target_returns) - self.time_step + 1

    def __iter__(self) -> Iterator[Tuple[NamedInput, Tensor]]:
        for i in range(len(self)):
            tgt = self.target_returns[i:i+self.time_step]
            tgt_value = tgt['return']

            tgt_time = tgt['time']
            decision_time = tgt_time - self.offset
            feature_ix = self._obs_time.searchsorted(decision_time)

            feature_value = NamedInput(
                raw_data={
                    f.name: self._get_feature(f, feature_ix)
                    for f in self.input_spec.features
                },
                input_spec=self.input_spec,
                batch_size=self.batch_size,
                time_step=self.time_step
            )
            yield feature_value, tgt_value
