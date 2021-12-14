from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback


class WeightStdCallback(Callback):
    def __init__(self):
        super(WeightStdCallback, self).__init__()

    def on_epoch_end(self, batch, logs={}):
        print(
            f"\nWeight std of last layer: {self.model.weights[-1].numpy().std():1.2E}"
        )
        return


fashion_model = make_fashion_mnist_model()
fashion_model.compile(optimizer="adam", loss="categorical_crossentropy")
fashion_model.fit(
    x_train[:1000, :, :, :],
    y_train[:1000],
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[WeightStdCallback()],
)
