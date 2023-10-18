import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras.layers import GRU, Bidirectional, Dense, Embedding, Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from metrics import fmeasure, precision, recall


def embed_model(max_features, embed_size, maxlen):
    model_input = Input(shape=(maxlen,), dtype="int32")
    embed = Embedding(max_features, embed_size, mask_zero=True, name="embedding")
    x = embed(model_input)
    x = Bidirectional(GRU(maxlen, return_sequences=False), name="bi_lstm_0")(x)
    outputs = [Dense(8, activation="softmax", name="softmax")(x)]
    model = Model(inputs=[model_input], outputs=outputs, name="Emozen")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(clipnorm=1, lr=0.001),
        metrics=[metrics.categorical_accuracy, fmeasure, recall, precision],
    )
    return model
