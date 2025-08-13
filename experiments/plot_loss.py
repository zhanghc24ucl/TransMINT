from TransMINT.engine.trainer import Snapshot
from torch import load
from TransMINT.utils import find_all


from matplotlib import pyplot as plt


base_dirs = [
    # 'vault/20250811_loss_sharpe/s42_l0.001_oNone',
    # 'vault/20250811_loss_sharpe/s63_l0.001_oNone',
    # 'vault/20250811_loss_sharpe/s191_l0.001_oNone',
    # 'vault/20250811_loss_utility/s63_l0.0001_oNone',
    # 'vault/20250811_loss_utility/s42_l0.0001_oNone',
    # 'vault/20250811_loss_utility/s191_l0.0001_oNone',
    # 'vault/20250811_loss_utility/s63_l0.001_oNone',
    # 'vault/20250811_loss_utility/s42_l0.001_oNone',
    # 'vault/20250811_loss_utility/s191_l0.001_oNone',
    # 'vault/20250813_lr_utility/l0.003',
    # 'vault/20250813_lr_sharpe/l0.0001',
    # 'vault/20250813_lr_sharpe/l3e-05',
    # 'vault/20250813_lr_sharpe/l1e-05',
    'vault/20250813_lr_sharpe/l8e-06',
    # 'vault/20250813_lr_sharpe/l5e-06',
    # 'vault/20250813_lr_sharpe/l3e-06',
]
all_snapshot_paths = find_all(base_dirs, 'trainer.pt')
all_snapshots = [
    load(path, map_location='cpu', weights_only=False)
    for path in all_snapshot_paths
]

def aggregate_losses(snapshots):
    if len(snapshots) == 0:
        return []

    from collections import defaultdict
    trains = defaultdict(list)
    valids = defaultdict(list)
    for s in snapshots:
        for j, e in enumerate(s.trainer_state['epochs']):
            trains[j].append(e['train_loss'])
            valids[j].append(e['val_loss'])
    max_j = max(max(trains.keys()), max(valids.keys()))

    rv = []
    for j in range(max_j):
        rv.append((j, trains[j], valids[j]))
    return rv


def plot_aggregated_losses(losses):
    fig, axs = plt.subplots(1, 1)
    trains = []
    valids = []
    labels = []

    from numpy import mean, max, min, nan
    def _stats(x):
        if len(x) == 0:
            return nan, nan, nan
        else:
            return mean(x), max(x), min(x)

    for j, train, valid in losses:
        trains.append(_stats(train))
        valids.append(_stats(valid))
        labels.append(j)
    train_mean, train_max, train_min = zip(*trains)
    axs.plot(labels, train_mean, label=f'Train', color='blue')
    axs.fill_between(labels, train_min, train_max, alpha=0.2, color='blue')

    valid_axs = axs.twinx()
    valid_mean, valid_max, valid_min = zip(*valids)
    valid_axs.plot(labels, valid_mean, label=f'Valid', color='red')
    valid_axs.fill_between(labels, valid_min, valid_max, alpha=0.2, color='red')

    axs.legend(loc='upper right')
    valid_axs.legend(loc='upper left')
    return fig


# ls = aggregate_losses(all_snapshots)
# plot_aggregated_losses(ls)
# plt.show()

for s in all_snapshots:
    trains = []
    valids = []
    for e in s.trainer_state['epochs']:
        trains.append(e['train_loss'])
        valids.append(e['val_loss'])
    plt.plot(trains, color='blue')
    ax = plt.twinx()
    ax.plot(valids, color='red')
plt.show()
