import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def gen_ruz_dat(rozsah, pocet_hodnot, uroven_sumu, funkce):
    x = np.linspace(0, rozsah, pocet_hodnot)
    y = funkce(x) + np.random.uniform(-uroven_sumu, uroven_sumu, size=pocet_hodnot)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    x = scaler_x.fit_transform(x)
    y = scaler_y.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    np.save('experiment_s_archit/data_exp/train_data.npy', x_train)
    np.save('experiment_s_archit/data_exp/train_labels.npy', y_train)
    np.save('experiment_s_archit/data_exp/test_data.npy', x_test)
    np.save('experiment_s_archit/data_exp/test_labels.npy', y_test)
    np.save('experiment_s_archit/data_exp/x.npy', x)
    np.save('experiment_s_archit/data_exp/y.npy', y)
    return scaler_x, scaler_y
    
