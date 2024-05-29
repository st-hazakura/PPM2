import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pocet_hodnot = 100
x = np.linspace(0, 2 * np.pi, pocet_hodnot)
y = np.sin(x) + np.random.uniform(-0.2, 0.2, size=pocet_hodnot)

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

np.save('data/train_data.npy', x_train)
np.save('data/train_labels.npy', y_train)
np.save('data/test_data.npy', x_test)
np.save('data/test_labels.npy', y_test)
np.save('data/x.npy', x)
np.save('data/y.npy', y)

