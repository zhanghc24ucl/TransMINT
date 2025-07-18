import torch
from numpy import zeros

from TransMINT.data_utils.spec import FeatureSpec, InputSpec, NamedInput
from TransMINT.model.transformer import MINTransformer
from TransMINT.model.loss import SharpeLoss

features = [
    FeatureSpec('ticker', 'static', 'categorical', None, 35),
    FeatureSpec('sector', 'static', 'categorical', None, 8),
    FeatureSpec('DoM', 'time_pos', 'real', 'time'),
    FeatureSpec('MoY', 'time_pos', 'real', 'time'),
    FeatureSpec('DoW', 'time_pos', 'real', 'time'),
    FeatureSpec('WoY', 'time_pos', 'real', 'time'),
    FeatureSpec('ret1m', 'observed', 'real', 'return', lag_size=10)
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
    ('ret1m[10]', float, 10),
])
inputs = {k: demo_static[k] for k in demo_static.dtype.names}
inputs.update({k: demo_temporal[k] for k in demo_temporal.dtype.names})

# test GPU
dev = torch.device("cuda")
demo_inputs = NamedInput(inputs, input_spec, batch_size=batch, time_step=timestep, device=dev)

mint = MINTransformer(
    input_spec,
    d_model=d_model,
    num_heads=4,
    output_size=1,
    dropout=0.1,
).to(dev)
result = mint.forward(demo_inputs)
print(result.shape)

loss1 = SharpeLoss()
loss2 = SharpeLoss(output_steps=3, global_sharpe=False)
loss3 = SharpeLoss(output_steps=1, cost_factor=0.1, slippage_factor=0.1, global_sharpe=True)
loss4 = SharpeLoss(output_steps=4, cost_factor=0.1, slippage_factor=0.1, reduction="mean")
y_true = torch.randn(batch, timestep, 1).to(dev)
print(loss1(result, y_true))
print(loss2(result, y_true))
print(loss3(result, y_true))
print(loss4(result, y_true))
