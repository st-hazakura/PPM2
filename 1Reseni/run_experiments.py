# ! nezpoustet - 9 hodin hledani architektury
import numpy as np
import pandas as pd
from src.train import train_model
from src.evaluate import evaluate_model
from sklearn.metrics import mean_squared_error

exec(open('./src/clear_dirik.py').read())


x_train = np.load('data/train_data.npy')
y_train = np.load('data/train_labels.npy')
x_test = np.load('data/test_data.npy')
y_test = np.load('data/test_labels.npy')

konfigurace = [
    {'layers': i, 'neurons': n, 'activation': a} 
    for i in range(1, 5)  
    for n in [2**i for i in range(4, 8)]
    for a in ['relu', 'sigmoid', 'tanh'] 
]

vysledky = []

for konfig in konfigurace:
    model = train_model(x_train, y_train, x_test, y_test, konfig)
    loss = evaluate_model(model, x_test, y_test)
    vysledky.append({**konfig, 'loss': loss})

vysledky_df = pd.DataFrame(vysledky)
vysledky_df.to_csv('results/experiment_results.csv', index=False)
