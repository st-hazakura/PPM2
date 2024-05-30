from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def create_model(layers, neurons, activation):
    model = Sequential()
    model.add(Dense(100, input_dim = 100, activation=activation)) # hodnot 
    for _ in range(1, layers):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(100)) # VYST HODNOT
    return model


def train_model(x_train, y_train, x_val, y_val, layers, neurons, activation):
    model = create_model(layers, neurons, activation)
    model.compile(optimizer='adam', loss='mse')
    checkpoint = ModelCheckpoint(filepath=f"src/model_{layers}_{neurons}_{activation}.keras",
                                 save_best_only=True)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=4000, verbose=0, callbacks=[checkpoint])
    return model


def evaluate_model(model, x_test, y_test):
    loss = model.evaluate(x_test, y_test, verbose=0)
    return loss