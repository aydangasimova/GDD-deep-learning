from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

# Make our model.
model = Sequential()
model.add(Embedding(NUM_WORDS, 32))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

# Callbacks.
checkpoint = callbacks.ModelCheckpoint(
    filepath="../output/imdb_lstm.h5", verbose=1, save_best_only=True
)
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0, patience=2, verbose=0, mode="auto"
)

# Compile and train.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping],
)

score, acc = model.evaluate(X_test, y_test, batch_size=64)

print("Test score:", score)
print("Test accuracy:", acc)
