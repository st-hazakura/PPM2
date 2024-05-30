# uloha2 cast 2 ucim na model na 1000 nahodne zgenerovanych funkcich
from gen_dat import gen_ruz
import numpy as np
from src.model import train_model, evaluate_model
from sklearn.model_selection import train_test_split
import os
os.chdir('C:/Users/Home/Desktop/vyučovaní/Programovani/1 Letni semestr 2 rocnik/PPM2/Uloha_2_cast_2')


pocet_f = 1000
scaler_x, x_s, y_all, y_bez_sumu = gen_ruz(pocet_f)

config = {"layers":3, "neurons":64, "activation": "sigmoid"}

x_train, x_test, y_train, y_test = train_test_split( y_all ,y_bez_sumu, test_size=0.2, random_state=0) # y - y'
model = train_model(x_train, y_train, x_test, y_test, config['layers'], config['neurons'], config['activation'])
loss = evaluate_model(model, x_test, y_test)
print(f"Loss: {loss}")