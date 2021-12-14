from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [EarlyStopping(), ModelCheckpoint(model_path, save_best_only=True)]


fashion_model = make_fashion_mnist_model()
fashion_model.compile(optimizer="adam", loss="categorical_crossentropy")
fashion_model.fit(
    train_iterator,
    validation_data=valid_iterator,
    steps_per_epoch=10,
    epochs=2,
    callbacks=callbacks,
)
