import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

def multilayer_perceptron_regressor(X_train, y_train, X_test, y_test):
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=5000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    plot_predictions(y_test, y_pred)

    return mape

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Echte Waarden', color='blue', marker='o', linestyle='--', markersize=5)
    plt.plot(y_pred, label='Voorspelde Waarden', color='red', marker='x', linestyle='-', markersize=5)
    plt.xlabel('Data Punten')
    plt.ylabel('Waarden')
    plt.title('Echte Waarden vs. Voorspelde Waarden')
    plt.legend()
    plt.grid(True)
    plt.show()

