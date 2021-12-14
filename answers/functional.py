from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D
from tensorflow.keras.models import Model


def make_fashion_mnist_model():

    inputs = Input(shape=(28, 28, 1))

    x = Conv2D(64, kernel_size=2, padding="same", activation="relu")(inputs)
    x = MaxPool2D(pool_size=2)(x)

    x = Dropout(0.3)(x)
    x = Conv2D(filters=32, kernel_size=2, padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)

    x = Dropout(0.5)(x)
    predictions = Dense(10, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=predictions)

    return model


model = make_fashion_mnist_model()
model.summary()
