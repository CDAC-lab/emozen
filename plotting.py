import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def tsne_plot_similar_words(
    title, labels, embedding_clusters, word_clusters, a, filename=None
):
    plt.figure(figsize=(32, 18))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(
        labels, embedding_clusters, word_clusters, colors
    ):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(
                word,
                alpha=0.5,
                xy=(x[i], y[i]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
                size=8,
            )
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(False)
    if filename:
        plt.savefig(filename, format="png", dpi=150, bbox_inches="tight")
