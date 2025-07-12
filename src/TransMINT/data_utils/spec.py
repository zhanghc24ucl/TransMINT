from dataclasses import dataclass
from typing import Dict, List, Optional

from numpy import ndarray


@dataclass
class FeatureSpec:
    name: str
    feature_class: str         # 'static' | 'time_pos' | 'observed' | 'known'
    feature_type: str          # 'real' | 'categorical' | 'cyclical' | 'sequential'
    feature_tag: Optional[str] = None       # None | 'time' | 'return' | 'volume' | ...
    category_size: Optional[int] = None     # only for feature_type == 'categorical'

    def validate(self):
        assert self.feature_class in ["static", "time_pos", "observed", "known"]
        assert self.feature_type in ["real", "categorical", "cyclical", "sequential"]
        if self.feature_type == 'categorical':
            assert self.category_size is not None

    def __hash__(self):
        return hash(self.name)

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
            if f.name in names:
                raise ValueError(f"Duplicate feature name: {f.name}")
            names.add(f.name)

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
            self._init_value(f, raw_data[f.name])

    def _init_value(self, fspec, value):
        from torch import from_numpy
        if value.ndim == 1:  # static: [B,]
            assert value.shape[0] == self.batch_size
        elif value.ndim == 2:  # temporal: [B, T]
            assert value.shape[0] == self.batch_size
            assert value.shape[1] == self.time_step
        else:
            raise ValueError(f"value.ndim={value.ndim}")
        self._values[fspec.name] = from_numpy(value).to(self.device)

    def to(self, device):
        for k, v in self._values.items():
            self._values[k] = v.to(device)
        self.device = device
    def __getitem__(self, item):
        return self._values[item]
