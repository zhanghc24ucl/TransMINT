
def load_all_trainer_snapshots(root_dirs):
    from ..utils import find_all
    from torch import load
    files = find_all(root_dirs, 'trainer.pt')
    return [load(path, map_location='cpu', weights_only=False) for path in files]


def extract_loss(snapshot):
    from numpy import array
    trains = []
    valids = []
    for e in snapshot.trainer_state['epochs']:
        trains.append(e['train_loss'])
        valids.append(e['val_loss'])
    return array(trains, dtype=float), array(valids, dtype=float)


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
    for j in range(max_j + 1):
        rv.append((j, trains[j], valids[j]))
    return rv


def load_results(store_path, windows):
    from ..engine.backtest import Backtest, BacktestConfig
    bt_cfg = BacktestConfig(
        windows=windows,
        data_cfg=None,
        trainer_cfg=None,
    )

    # load results
    bt = Backtest(bt_cfg, None, store_path=store_path)
    bt.run()

    return bt.performance(), bt.results, bt.ticker_performance()
