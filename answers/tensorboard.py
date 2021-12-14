from tensorflow.keras.callbacks import TensorBoard

callbacks = [TensorBoard(run_dir, write_graph=True)]


fashion_model = make_fashion_mnist_model()
fashion_model.compile(optimizer="adam", loss="categorical_crossentropy")
fashion_model.fit(
    x_train, y_train, epochs=2, validation_data=(x_test, y_test), callbacks=callbacks
)
