from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential


def make_lstm_model():
    model = Sequential()
    model.add(
        LSTM(64, input_shape=(timesteps, data_dim), dropout=0.1, recurrent_dropout=0.1)
    )
    model.add(Dense(n_classes, activation="softmax"))

    return model


np.random.seed(707)

callbacks = [ModelCheckpoint(filepath=model_path, save_best_only=True), EarlyStopping()]

model = make_lstm_model()
model.compile(optimizer="nadam", loss="categorical_crossentropy", metrics=["accuracy"])

EPOCHS = 5
EPOCHS = 1  # CI
history = model.fit(
    X_train,
    Y_train,
    batch_size=32,
    epochs=EPOCHS,
    validation_data=(X_test, Y_test),
    callbacks=callbacks,
)
