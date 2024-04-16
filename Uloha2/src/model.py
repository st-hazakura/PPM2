from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model(layers, neurons, activation):
    model = Sequential()
    model.add(Dense(neurons, input_dim = 1, activation=activation))
    for _ in range(1, layers):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(1))
    return model


def train_model(x_train, y_train, x_val, y_val, layers, neurons, activation):
    model = create_model(layers, neurons, activation)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=4000, verbose = 0)
    return model

def evaluate_model(model, x_test, y_test):
    loss = model.evaluate(x_test, y_test, verbose=0)
    return loss