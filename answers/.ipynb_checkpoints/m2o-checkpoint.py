from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def make_m2o_model():
    """Function for making many-to-one RNN"""
    model = Sequential()
    model.add(
        SimpleRNN(
            200, activation="relu", input_shape=(LOOK_BACK, 1), return_sequences=True
        )
    )
    model.add(Dropout(0.2))
    model.add(SimpleRNN(30, activation="relu", return_sequences=False))
    model.add(Dropout(0.2))
    model.add((Dense(1)))

    return model


np.random.seed(7)

model = make_m2o_model()
model.compile(loss="mse", optimizer=Adam(learning_rate=1e-4, decay=1e-2))


callbacks = [
    TensorBoard(m2o_dir, write_graph=False),
    ModelCheckpoint(filepath=model_path, save_best_only=True),
]

# +
EPOCHS = 500

history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=EPOCHS,
    shuffle=True,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
)
