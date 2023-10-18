import argparse
import os

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from data_preprocessing import load_or_preprocess_data
from model import embed_model
from plotting import tsne_plot_similar_words
from training import train_model
from word_embeddings import calculate_word_embeddings


def main():
    parser = argparse.ArgumentParser(description="description")

    # Add the arguments
    parser.add_argument("-s", "--suffix", type=str, help="suffix")
    parser.add_argument("--max-features", default=50000, type=int, help="max features")
    parser.add_argument("--max-length", default=60, type=int, help="max length")
    parser.add_argument("--embed-size", default=300, type=int, help="embedding size")
    parser.add_argument("--batch-size", default=2048, type=int, help="batch size")
    parser.add_argument("--epochs", default=300, type=int, help="epochs")
    parser.add_argument("--patience", default=15, type=int, help="patience")
    parser.add_argument("--gpu-device", default="0", type=str, help="GPU device")
    parser.add_argument(
        "--set-gpu-device", default=False, action="store_true", help="set GPU device?"
    )
    parser.add_argument("--dataset", type=str, help="dataset path", required=True)
    parser.add_argument(
        "--load-model", default=False, action="store_true", help="load model"
    )
    parser.add_argument("--load-model-path", type=str, help="load model path")
    parser.add_argument("--plot", default=True, type=bool, help="plot?")

    args = parser.parse_args()

    (
        tokenizer1,
        X_train,
        X_test,
        y_train,
        y_test,
        x_train1,
        x_test1,
        mapping,
        y_train1,
        y_test1,
    ) = load_or_preprocess_data(args)
    model = embed_model(args.max_features, args.embed_size, args.max_length)

    if args.load_model:
        model.load_weights(f"models/weights_{args.load_model_path}.hdf5")
        print(f"Model loaded from models/weights_{args.load_model_path}.hdf5")

    model.summary()

    _ = train_model(
        model, x_train1, y_train1, x_test1, y_test1, args.batch_size, args.epochs, args
    )

    if args.plot:
        cs, _ = calculate_word_embeddings(
            model, tokenizer1, args.max_features, args.suffix
        )
        emm = pd.read_csv(f"embeddings/embeddings_{args.suffix}.csv", sep="\t")
        keys = [
            "certainly",
            "angry",
            "anticipation",
            "fear",
            "joy",
            "trust",
            "surprise",
            "sad",
            "disgust",
        ]

        embedding_clusters = []
        word_clusters = []
        for word in keys:
            embeddings = []
            words = []
            for similar_word in emm.index.values[
                cs[tokenizer1.word_index[word] - 1].argsort()[-20:][::-1]
            ]:
                words.append(similar_word)
                embeddings.append(embed_model.loc[similar_word].values)
            embedding_clusters.append(embeddings)
            word_clusters.append(words)

        embedding_clusters = np.array(embedding_clusters)
        n, m, k = embedding_clusters.shape
        tsne_model_en_2d = TSNE(
            perplexity=15, n_components=2, init="pca", n_iter=3500, random_state=32
        )
        embeddings_en_2d = np.array(
            tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
        ).reshape(n, m, 2)

        tsne_plot_similar_words(
            "Similar words",
            keys,
            embeddings_en_2d,
            word_clusters,
            0.7,
            f"plot/{args.suffix}.png",
        )


if __name__ == "__main__":
    main()
