import os
import pickle

import pandas as pd
from keras.preprocessing import sequence, text
from sklearn.model_selection import train_test_split


def load_or_preprocess_data(args):
    fname = args.dataset.split("//")[-1][:-4]

    if not os.path.isfile(f"preprocessed/{fname}/y_test1.pickle"):
        # Load and preprocess the dataset
        dataset = pd.read_csv(args.dataset, encoding="utf-8")
        max_features = args.max_features
        maxlen = args.max_length

        tokenizer1 = text.Tokenizer(
            num_words=max_features, filters="", lower=False, oov_token="UNK"
        )
        tokenizer1.fit_on_texts(dataset["tweet"].tolist())

        # Save tokenizer
        with open(f"preprocessed/{fname}/tokenizer.pickle", "wb") as f:
            pickle.dump(tokenizer1, f)

        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            dataset["tweet"],
            dataset["target"],
            test_size=0.10,
            random_state=42,
            stratify=dataset["target"],
        )

        # Save preprocessed data
        with open(f"preprocessed/{fname}/X_train.pickle", "wb") as f:
            pickle.dump(X_train, f)
        with open(f"preprocessed/{fname}/X_test.pickle", "wb") as f:
            pickle.dump(X_test, f)
        with open(f"preprocessed/{fname}/y_train.pickle", "wb") as f:
            pickle.dump(y_train, f)
        with open(f"preprocessed/{fname}/y_test.pickle", "wb") as f:
            pickle.dump(y_test, f)

        # Text to sequences and padding
        X_train1 = tokenizer1.texts_to_sequences(X_train.tolist())
        X_test1 = tokenizer1.texts_to_sequences(X_test.tolist())

        x_train1 = sequence.pad_sequences(
            X_train1, maxlen=maxlen, padding="post", truncating="post"
        )
        x_test1 = sequence.pad_sequences(
            X_test1, maxlen=maxlen, padding="post", truncating="post"
        )

        # Save preprocessed sequences
        with open(f"preprocessed/{fname}/x_train1.pickle", "wb") as f:
            pickle.dump(x_train1, f)
        with open(f"preprocessed/{fname}/x_test1.pickle", "wb") as f:
            pickle.dump(x_test1, f)

        # One-hot encoding mapping
        mapping = pd.get_dummies(y_train).columns.values.tolist()
        with open(f"preprocessed/{fname}/mapping.pickle", "wb") as f:
            pickle.dump(mapping, f)

        y_train1 = pd.get_dummies(y_train)[mapping].values
        with open(f"preprocessed/{fname}/y_train1.pickle", "wb") as f:
            pickle.dump(y_train1, f)

        y_test1 = pd.get_dummies(y_test)[mapping].values
        with open(f"preprocessed/{fname}/y_test1.pickle", "wb") as f:
            pickle.dump(y_test1, f)

    else:
        # Load preprocessed data if it exists
        with open(f"preprocessed/{fname}/tokenizer.pickle", "rb") as f:
            tokenizer1 = pickle.load(f)

        with open(f"preprocessed/{fname}/X_train.pickle", "rb") as f:
            X_train = pickle.load(f)

        with open(f"preprocessed/{fname}/X_test.pickle", "rb") as f:
            X_test = pickle.load(f)

        with open(f"preprocessed/{fname}/y_train.pickle", "rb") as f:
            y_train = pickle.load(f)

        with open(f"preprocessed/{fname}/y_test.pickle", "rb") as f:
            y_test = pickle.load(f)

        with open(f"preprocessed/{fname}/x_train1.pickle", "rb") as f:
            x_train1 = pickle.load(f)

        with open(f"preprocessed/{fname}/x_test1.pickle", "rb") as f:
            x_test1 = pickle.load(f)

        with open(f"preprocessed/{fname}/mapping.pickle", "rb") as f:
            mapping = pickle.load(f)

        with open(f"preprocessed/{fname}/y_train1.pickle", "rb") as f:
            y_train1 = pickle.load(f)

        with open(f"preprocessed/{fname}/y_test1.pickle", "rb") as f:
            y_test1 = pickle.load(f)

    return (
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
    )
