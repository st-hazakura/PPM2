# pro nalezenou nejlepsi konfigurace na sin -1 1
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


##############
results = pd.read_csv('results/experiment_results.csv')
nejlep_zaznam_modelu = results.loc[results['loss'].idxmin()]

naz_souboru_nejl_modelu = f"model_{nejlep_zaznam_modelu['layers']}_{nejlep_zaznam_modelu['neurons']}_{nejlep_zaznam_modelu['activation']}.keras"
nejl_model_cesta = f"models/{naz_souboru_nejl_modelu}"
print(nejl_model_cesta)

loaded_model = tf.keras.models.load_model(nejl_model_cesta)
##########

x_train = np.load('data/train_data.npy')
y_train = np.load('data/train_labels.npy')
x_test = np.load('data/test_data.npy')
y_test = np.load('data/test_labels.npy')

x = np.load('data/x.npy')
y = np.load('data/y.npy')

scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

scaler_x.fit(x)
scaler_y.fit(y)


y_pred_train_non_sc = loaded_model.predict(x_train)
y_pred_test_non_sc = loaded_model.predict(x_test)

y_predict_train = scaler_y.inverse_transform(y_pred_train_non_sc)
y_predict_test = scaler_y.inverse_transform(y_pred_test_non_sc)


x_train_non_scale = scaler_x.inverse_transform(x_train)
y_train_non_scale = scaler_y.inverse_transform(y_train)

x_test_non_scale = scaler_x.inverse_transform(x_test)
y_test_non_scale = scaler_y.inverse_transform(y_test)

print(f'MSE: {mean_squared_error(y_train_non_scale, y_predict_train)}')

#################

fig = plt.figure (figsize=(15,15))
plt.scatter(x_train_non_scale,y_train_non_scale, label='Originalni hodnoty pouzite pro trenovani')
plt.scatter(x_test_non_scale,y_test_non_scale, label='Originalni hodnoty pouzite pro testovani')
# plot x vs yhat
plt.scatter(x_train_non_scale,y_predict_train, label='Predikovane hodnoty na trenovaci mnozine x_train', s=20)
plt.scatter(x_test_non_scale,y_predict_test, label='Predikovane hodnoty na testovaci mnnozine x_test', s=20)
plt.title('Vysledek nejlepseho nelezeneho modelu')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig("Nehlepsi_vysledek.png")
plt.show()

