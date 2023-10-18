from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def datagenerator(text, labels, batchsize):
    while True:
        start = 0
        end = batchsize

        while start < len(text):
            # load your text from numpy arrays or read from directory
            x = text[start:end]
            y = labels[start:end]
            yield x, y

            start += batchsize
            end += batchsize


def train_model(model, x_train1, y_train1, x_test1, y_test1, batch_size, epochs, args):
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", verbose=1, factor=0.5, patience=3, min_lr=0.0001
    )
    checkpointer = ModelCheckpoint(
        filepath=f"models/weights_{args.suffix}.hdf5",
        monitor="val_loss",
        verbose=1,
        mode="min",
        save_best_only=False,
    )
    earlyStopper = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=args.patience, verbose=1, mode="min"
    )

    steps = x_train1.shape[0] // batch_size
    val_steps = x_test1.shape[0] // batch_size

    # Train the model
    hist = model.fit_generator(
        datagenerator(x_train1, y_train1, batch_size),
        validation_data=datagenerator(x_test1, y_test1, batch_size),
        callbacks=[earlyStopper, checkpointer, reduce_lr],
        verbose=1,
        steps_per_epoch=steps,
        validation_steps=val_steps,
        epochs=epochs,
    )

    return hist
