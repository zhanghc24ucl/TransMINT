from numpy import zeros
from torch import device

from TransMINT.data_utils.spec import FeatureSpec, InputSpec, NamedInput
from TransMINT.model.transformer import MINTransformer

features = [
    FeatureSpec('ticker', 'static', 'categorical', None, 35),
    FeatureSpec('sector', 'static', 'categorical', None, 8),
    FeatureSpec('DoM', 'observed', 'real', 'time'),
    FeatureSpec('MoY', 'observed', 'real', 'time'),
    FeatureSpec('DoW', 'observed', 'real', 'time'),
    FeatureSpec('WoY', 'observed', 'real', 'time'),
]
input_spec = InputSpec(features)
input_spec.validate()

# (batch, T)
batch, timestep = 3, 5
d_model = 8

demo_static = zeros((batch,), dtype=[
    ('ticker', int),
    ('sector', int),
])
demo_temporal = zeros((batch, timestep), dtype=[
    ('DoM', int),
    ('MoY', int),
    ('DoW', int),
    ('WoY', int),
])
inputs = {k: demo_static[k] for k in demo_static.dtype.names}
inputs.update({k: demo_temporal[k] for k in demo_temporal.dtype.names})

# test GPU
dev = device("cuda")
demo_inputs = NamedInput(inputs, input_spec, batch_size=batch, time_step=timestep, device=dev)

mint = MINTransformer(
    input_spec,
    d_model=d_model,
    num_heads=4,
    output_size=1,
    dropout=0.1,
).to(dev)
mint.forward(demo_inputs)
