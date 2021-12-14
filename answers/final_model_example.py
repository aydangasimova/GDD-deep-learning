# +
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout, BatchNormalization

def create_model(): 
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(30,30,3)))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(NUM_CATEGORIES, activation='softmax'))
    return model

model = create_model()
train_and_evaluate(create_model)
