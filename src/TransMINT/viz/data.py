import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


def plot_heatmap(matrix, labels, *, title="Heatmap", figsize=(10, 8), cmap="RdYlBu_r", annot=True):
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(matrix,
                          xticklabels=labels,
                          yticklabels=labels,
                          cmap=cmap,
                          annot=annot,
                          fmt='.2f',
                          square=True,
                          vmin=-1.0,
                          vmax=1.0,
                          cbar_kws={"shrink": .8})

    plt.title(title)
    plt.tight_layout()

    return plt.gcf()


def plot_feature_distribution(
        feature_data, feature_keys, *,
        n_column=5, title="Feature Distribution", figsize=(10, 8)):
    N = len(feature_keys)
    n_row = (N + n_column - 1) // n_column
    fig, axs = plt.subplots(n_row, n_column, figsize=figsize)
    for i, (key, ax) in enumerate(zip(feature_keys, axs.flatten())):
        ax.hist(feature_data[key], bins=100, histtype='step', density=True)
        ax.set_title(key, pad=10)
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    return fig


def _feature_stats(x, y):
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import mutual_info_regression

    # 1. Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(x, y)

    # 2. Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(x, y)

    # 3. Mutual Information
    # mutual_info_regression
    mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0]

    return (pearson_corr, pearson_p), (spearman_corr, spearman_p), mi


def show_feature_info(feature_data, feature_keys, target_key):
    tgt = feature_data[target_key]
    tbl = []
    for k in feature_keys:
        print(k)
        (c1, p1), (c2, p2), mi = _feature_stats(feature_data[k], tgt)
        tbl.append([
            k,
            f'{c1:.4f} / {p1:.3f}',
            f'{c2:.4f} / {p2:.3f}',
            f'{mi:.4f}',
        ])
    keys = [
        'Feature',
        'Pearson Corr', 'Spearman Corr',
        'Mutual Information']
    print(tabulate(tbl, headers=keys, tablefmt='github'))
