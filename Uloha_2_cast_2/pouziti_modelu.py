#pouzivame model nauceny na 1000 funkce
import tensorflow as tf
import matplotlib.pyplot as plt
from gen_dat import gen_ruz
import numpy as np
import os
os.chdir('C:/Users/Home/Desktop/vyučovaní/Programovani/1 Letni semestr 2 rocnik/PPM2/Uloha_2_cast_2')

best_model_path = 'src/model_3_64_sigmoid.keras'
loaded_model = tf.keras.models.load_model(best_model_path)

scaler_x, x_s, y, y_bez_sumu = gen_ruz(1)
y_pred = loaded_model.predict(y)

x_unscaled = scaler_x.inverse_transform(x_s)

plt.figure(figsize= (15, 15))
plt.scatter(x_unscaled, y_pred, color= "red", label = "predict")
plt.scatter(x_unscaled, y, color = "blue", label = "zasumele")
plt.scatter(x_unscaled, y_bez_sumu, color = "green", label = "nezasumele puvodni hodnoty hodnoty")
plt.legend()
plt.show()