import matplotlib.pyplot as plt
from gen_dat import gen_ruz
from src.model import train_model, evaluate_model
from sklearn.model_selection import train_test_split
import os
os.chdir('C:/Users/Home/Desktop/vyučovaní/Programovani/1 Letni semestr 2 rocnik/PPM2/Uloha2')

def plot_results(x, ys_pred_ns, filename, title):
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green']

    for i, y_pred in enumerate(ys_pred_ns):
        plt.scatter(x, y_pred, label=f'Funkce {i+1}', color=colors[i])

    plt.title(f'{title}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(f'{filename}.png')
    plt.show()


pocet_f = 1
scaler_x, x_scaled, y_all, y_puvodni = gen_ruz(pocet_f)

config = {"layers":3, "neurons":64, "activation": "sigmoid"}
ys_pred_all = []

for i in range(pocet_f):
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_all[i], test_size=0.2, random_state=0)
    model = train_model(x_train, y_train, x_test, y_test, config['layers'], config['neurons'], config['activation'])
    loss = evaluate_model(model, x_test, y_test)
    
    print(f"Model {i+1} Loss: {loss}")
    y_pred = model.predict(x_scaled)
    ys_pred_all.append(y_pred)
    

x_unscaled = scaler_x.inverse_transform(x_scaled)
plot_results(x_unscaled, ys_pred_all, filename='predicovane_vysledky', title = 'Predict hodnoty')
plot_results(x_unscaled, y_puvodni, filename='puvodni_vysledky', title = 'Puvodni Hodnoty')