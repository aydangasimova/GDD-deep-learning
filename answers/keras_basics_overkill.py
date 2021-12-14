from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def make_overkill_model():
    model = Sequential()
    model.add(
        layers.Dense(name="HiddenLayer_1", units=30, activation="tanh", input_dim=2)
    )
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(name="HiddenLayer_2", units=10, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(name="HiddenLayer_3", units=3, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(name="OutputLayer", units=1, activation="sigmoid"))
    return model
