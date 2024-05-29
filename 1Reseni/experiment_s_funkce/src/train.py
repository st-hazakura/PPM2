from tensorflow.keras.callbacks import ModelCheckpoint
from .model import create_model


def train_model(x_train, y_train, x_val, y_val, config):
    model = create_model(input_shape=(x_train.shape[1],), **config)
    model.compile(optimizer='adam', loss='mse')
    checkpoint = ModelCheckpoint(filepath=f"models/model_{config['layers']}_{config['neurons']}_{config['activation']}.keras",
                                 save_best_only=True)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5000, callbacks=[checkpoint], verbose = 0)
    return model
