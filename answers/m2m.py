from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Lambda


def make_m2m_model():
    """Function for making many-to-one RNN"""
    model = Sequential()
    model.add(
        SimpleRNN(
            200, activation="relu", input_shape=(LOOK_BACK, 1), return_sequences=True
        )
    )
    model.add(Dropout(0.2))
    model.add(SimpleRNN(30, activation="relu", return_sequences=True))
    model.add(Lambda(lambda x: x[:, -2:, :]))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1)))

    return model


np.random.seed(7)

model = make_m2m_model()
model.compile(loss="mse", optimizer="nadam")


callbacks = [
    TensorBoard(m2o_dir, write_graph=False),
    ModelCheckpoint(filepath=model_path, save_best_only=True),
]
history = model.fit(
    X_train_many,
    y_train_many,
    batch_size=32,
    epochs=100,
    shuffle=True,
    validation_data=(X_test_many, y_test_many),
    callbacks=callbacks,
)
