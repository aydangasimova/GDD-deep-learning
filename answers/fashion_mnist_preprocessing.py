from tensorflow.keras.utils import to_categorical

if len(X_train.shape) != 4:
    X_train = np.expand_dims(X_train, axis=3)
if len(X_val.shape) != 4:
    X_val = np.expand_dims(X_val, axis=3)
if len(X_test.shape) != 4:
    X_test = np.expand_dims(X_test, axis=3)

y_train_onehot, y_val_onehot, y_test_onehot = to_categorical(y_train), to_categorical(y_val), to_categorical(y_test)