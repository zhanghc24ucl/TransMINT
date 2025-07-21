from dataclasses import dataclass
from typing import Dict, List, Optional

from numpy import ndarray


@dataclass
class FeatureSpec:
    field: str
    feature_class: str         # 'static' | 'time_pos' | 'observed'
    feature_type: str          # 'real' | 'categorical' | 'cyclical' | 'sequential'
    feature_tag: Optional[str] = None       # None | 'time' | 'return' | 'volume' | ...
    category_size: Optional[int] = None     # only for feature_type == 'categorical'
    lag_size: Optional[int] = None

    @property
    def name(self):
        return self.field if self.lag_size is None else f'{self.field}[{self.lag_size}]'

    def validate(self):
        assert self.feature_class in ["static", "time_pos", "observed"]
        assert self.feature_type in ["real", "categorical", "cyclical", "sequential"]
        if self.feature_type == 'categorical':
            assert self.category_size is not None

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

@dataclass
class InputSpec:
    features: List[FeatureSpec]

    def get(self, **kw) -> List[FeatureSpec]:
        return [f for f in self.features if all(getattr(f, k) == v for k, v in kw.items())]

    def count(self, **kw) -> int:
        return len(self.get(**kw))

    def validate(self):
        names = set()
        for f in self.features:
            f.validate()
            f_name = f.name
            if f_name in names:
                raise ValueError(f"Duplicate feature: {f_name}")
            names.add(f_name)

    def __str__(self):
        from tabulate import tabulate
        ctn = []
        for f in self.features:
            ctn.append([f.name, f.feature_class, f.feature_type, f.feature_tag, f.category_size])
        return tabulate(ctn, headers=['name', 'class', 'type', 'tag', 'category_size'], tablefmt="github")


class NamedInput:
    def __init__(
            self,
            raw_data: Dict[str, ndarray],
            input_spec: InputSpec,
            batch_size: int,
            time_step: int,
            device=None
    ):
        self.input_spec = input_spec

        self.batch_size = batch_size
        self.time_step = time_step

        if device is None:
            from torch import device as _d
            device = _d('cpu')
        self.device = device

        self._values = {}
        for f in self.input_spec.features:
            self._init_value(f, raw_data[str(f)])

    def _init_value(self, fspec, value):
        from torch import from_numpy
        if value.ndim == 1:  # static: [B,]
            assert value.shape[0] == self.batch_size
        elif value.ndim == 2:  # temporal: [B, T]
            assert value.shape[0] == self.batch_size
            assert value.shape[1] == self.time_step
        elif value.ndim == 3:  # lagged temporal: [B, T, L]
            assert value.shape[0] == self.batch_size
            assert value.shape[1] == self.time_step
        else:
            raise ValueError(f"value.ndim={value.ndim}")
        self._values[fspec.name] = from_numpy(value).to(self.device)

    def to(self, device):
        for k, v in self._values.items():
            self._values[k] = v.to(device)
        self.device = device
        return self

    def __getitem__(self, item):
        return self._values[item]


class TargetReturn:
    def __init__(
            self,
            ticker: str,
            targets: ndarray,
            details,
            device=None
    ):
        self.ticker = ticker

        from torch import from_numpy
        self.targets = tgt = from_numpy(targets)  # (B, T, 1)
        self.details = details                    # (B,     )

        assert len(targets) == len(self.details)

        if device is None:
            from torch import device as _d
            device = _d('cpu')
        self.device = device

        self.batch_size = tgt.size(0)
        self.time_step = tgt.size(1)

    def to(self, device):
        self.targets = self.targets.to(device)
        self.device = device
        return self
