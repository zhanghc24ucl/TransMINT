import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(matrix, labels, title="Heatmap", figsize=(10, 8), cmap="RdYlBu_r", annot=True):
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(matrix,
                          xticklabels=labels,
                          yticklabels=labels,
                          cmap=cmap,
                          annot=annot,
                          fmt='.2f',
                          square=True,
                          cbar_kws={"shrink": .8})

    plt.title(title)
    plt.tight_layout()

    return plt.gcf()
