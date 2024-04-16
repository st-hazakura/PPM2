import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.interpolate import CubicSpline

def gen_ruz(num_datasets):
    klicove_body = 5
    klic_x = np.linspace(0, 2*np.pi, klicove_body)
    
    x_all = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
    y_all = []
    y_clean_all = [] 
    
    for _ in range(num_datasets):
        klic_y = np.random.uniform(0, 1, size=klicove_body)
        cs = CubicSpline(klic_x, klic_y)
        y_clean = cs(x_all).reshape(-1, 1)
        y_noisy = y_clean + np.random.normal(-0.05, 0.05, x_all.shape)
        
        y_all.append(y_noisy.reshape(-1))  
        y_clean_all.append(y_clean.reshape(-1))  
    
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    x_scaled = scaler_x.fit_transform(x_all)

    return scaler_x, x_scaled, y_all, y_clean_all