from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

# Make our model.
mvm = Sequential()
mvm.add(Embedding(NUM_WORDS, 30))
mvm.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1))
mvm.add(Dense(1, activation="sigmoid"))

# Compile and train.
mvm.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
mvm.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=5,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
)

score, acc = mvm.evaluate(X_test, y_test, batch_size=64)

print("Test score:", score)
print("Test accuracy:", acc)
