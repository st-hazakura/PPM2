# zkouseni jiz nalezene architektury na ruznych f-cich s ruznymi parametry
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from gen_dat import gen_ruz_dat

from src.train import train_model
from src.evaluate import evaluate_model


def load_data(data_path='experiment_s_archit/data_exp/'):
    x_train = np.load(f'{data_path}train_data.npy')
    y_train = np.load(f'{data_path}train_labels.npy')
    x_test = np.load(f'{data_path}test_data.npy')
    y_test = np.load(f'{data_path}test_labels.npy')
    
    x = np.load('experiment_s_archit/data_exp/x.npy')
    y = np.load('experiment_s_archit/data_exp/y.npy')    
    
    return x_train, y_train, x_test, y_test,x, y

def find_best_model(results_path='results/experiment_results.csv', models_path='models/'):
    results = pd.read_csv(results_path)
    best_model_record = results.loc[results['loss'].idxmin()]
    return best_model_record

def predict_and_scale(model, x_data, scaler_y):
    predictions = model.predict(x_data)
    return scaler_y.inverse_transform(predictions)


def scale_data(scaler_x, scaler_y, x_data, y_data):
    x_data_ns = scaler_x.inverse_transform(x_data)
    y_data_ns = scaler_y.inverse_transform(y_data)

    return x_data_ns, y_data_ns


def plot_results(x_train_ns, y_train_ns, y_predict_train, x_test_ns, y_predict_test, filename):
    plt.figure(figsize=(15,15))
    plt.scatter(x_train_ns, y_train_ns, label='Training Data')
    plt.scatter(x_test_ns, y_test_ns, label='Test Data')
    plt.scatter(x_train_ns, y_predict_train, label='Predicted Training Data', s=20)
    plt.scatter(x_test_ns, y_predict_test, label='Predicted Test Data', s=20)
    plt.title('Comparison of Actual and Predicted Values')
    plt.xlabel('Input Variable (x)')
    plt.ylabel('Output Variable (y)')
    plt.legend()
    plt.savefig(f'experiment_s_archit/img_result/{filename}.png')
    plt.close()


#####
funkce  = [ np.sqrt,np.tan ,np.exp]
uroven_sumu = [0.1*i for i in range(2,5)]
rozsah_hodnot = [i*np.pi for i in range(1,4)] # nemusime menit pocet hodnot, pak vsak skalujeme od 0 do 1
pocet_hodnot = [100 for i in range(1,4)]

parameters = list(zip(rozsah_hodnot, pocet_hodnot))
results = []

for rozsah, pocet_hodnot in parameters:
    for uroven in uroven_sumu:
        for f in funkce:
            scaler_x, scaler_y = gen_ruz_dat(rozsah = rozsah, pocet_hodnot = pocet_hodnot, uroven_sumu = uroven, funkce = f) # vratit pro spravne zpetne preskalovani 

            x_train, y_train, x_test, y_test, x, y = load_data()

            best_model = find_best_model()
            layers = best_model['layers']
            neurons = best_model['neurons']
            activation = best_model['activation']

            configs = [{"layers": layers, "neurons": neurons, "activation": activation}]

            function_name = f.__name__
            
            for config in configs:
                model = train_model(x_train, y_train, x_test, y_test, config)
                loss = evaluate_model(model, x_test, y_test)
                

            scaler_x.fit(x)
            scaler_y.fit(y)

            y_predict_train = predict_and_scale(model, x_train, scaler_y)
            y_predict_test = predict_and_scale(model, x_test, scaler_y)

            x_train_ns, y_train_ns = scale_data(scaler_x, scaler_y, x_train, y_train)
            x_test_ns, y_test_ns = scale_data(scaler_x, scaler_y, x_test, y_test)

            print(f'MSE: {mean_squared_error(y_train_ns, y_predict_train)}')
            results.append({'rozsah': rozsah, 'uroven sumu': uroven,'funkce': function_name, 'loss': loss, 'MSE': mean_squared_error(y_train_ns, y_predict_train)})
            plot_results(x_train_ns, y_train_ns, y_predict_train, x_test_ns, y_predict_test, filename = f'vysledek_{rozsah}_{uroven}_{function_name}')

results_df = pd.DataFrame(results)
results_df.to_csv('experiment_s_archit/experiment_results.csv', index=False)