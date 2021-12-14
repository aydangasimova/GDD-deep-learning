from tensorflow.keras import layers
from tensorflow.keras.models import Model


def make_custom_model(base_model):
    # get base model output
    x = base_model.output
    # add GlobalAveragePooling2D layer
    x = layers.GlobalAveragePooling2D()(x)
    # add Dense layer of 512 units
    x = layers.Dense(units=512, activation="relu")(x)
    # add output Dense layer with 10 units and softmax activation function
    predictions = layers.Dense(units=10, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


model = make_custom_model(base_model)
model.summary()
