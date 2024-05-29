from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model(input_shape, layers, neurons, activation):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=input_shape))
    for _ in range(1, layers):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(1))
    return model
