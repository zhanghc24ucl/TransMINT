from TransMINT.engine.trainer import Snapshot
from torch import load


# s = load('vault/20250811_loss_utility/s42_l0.001_oNone/2019-01-01_2019-07-01/trainer.pt', map_location='cpu', weights_only=False)
s = load('vault/20250811_loss_sharpe/s63_l0.001_oNone/2019-01-01_2019-07-01/trainer.pt', map_location='cpu', weights_only=False)
assert isinstance(s, Snapshot)

from matplotlib import pyplot as plt

trains = []
valids = []
for e in s.trainer_state['epochs']:
    trains.append(e['train_loss'])
    valids.append(e['val_loss'])

plt.plot(trains)
ax = plt.twinx()
ax.plot(valids, color='red')
plt.show()
