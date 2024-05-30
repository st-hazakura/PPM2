from random import uniform
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def gen_ruz(num_datasets):
    klicove_body = 5
    klic_x = np.linspace(0, 2*np.pi, klicove_body)
    
    x = np.linspace(0, 2*np.pi, 100)
    x_all = []
    y_all = []
    y_clean_all = [] 
    
    for i in range(num_datasets):
        klic_y = np.random.uniform(0, 1, size=klicove_body)
        cs = CubicSpline(klic_x, klic_y)
        y_clean = cs(x)
        y_noisy = y_clean + np.random.uniform(-0.2, 0.2, size = y_clean.size)
        
        y_clean_all.append(y_clean)  
        y_all.append(y_noisy)
        x_all.append(x)
        
    y_all = np.array(y_all)
    x_all = np.array(x_all)
    y_clean_all = np.array(y_clean_all)
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    x_scaled = scaler_x.fit_transform(x_all)
    

    return  scaler_x, x_scaled, y_all, y_clean_all