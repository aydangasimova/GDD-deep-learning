from tensorflow.keras import layers


def make_cnn_model():
    model = Sequential()
    # input layer transformation (BatchNormalization + Dropout)
    model.add(layers.BatchNormalization(input_shape=(32, 32, 3)))
    model.add(layers.Dropout(rate=0.3))

    # convolutional layer (Conv2D + MaxPooling2D + Flatten + Dropout)
    model.add(
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=0.3))

    # fully connected layer (Dense + BatchNormalization + Activation + Dropout)
    model.add(layers.Dense(150))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(rate=0.3))

    # output layer (Dense + BatchNormalization + Activation)
    model.add(layers.Dense(units=10))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("softmax"))

    
    return model


model = make_cnn_model()
model.summary()
