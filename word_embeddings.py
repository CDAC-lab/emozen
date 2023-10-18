from itertools import islice

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def calculate_word_embeddings(model, tokenizer1, max_features, fname):
    embeddings = model.layers[1].get_weights()[0]
    words_embeddings = {
        w: embeddings[idx]
        for w, idx in islice(tokenizer1.word_index.items(), None, max_features - 1)
    }
    emm = pd.DataFrame.from_dict(words_embeddings, orient="index")
    emm.to_csv(f"embeddings/embeddings_{fname}.csv", sep="\t")

    # Calculate cosine similarity
    cs = cosine_similarity(emm)
    word_index = pd.read_csv(f"word_index/word_index_{fname}.csv", encoding="utf-8")
    word_index = {w: i for i, w in enumerate(word_index["0"].values.tolist())}
    return cs, word_index
