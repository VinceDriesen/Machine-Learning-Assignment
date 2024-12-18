import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error

def multilayer_perceptron_regressor(X_train, y_train, X_test, y_test, hidden_layer_sizes, activation, solver, learning_rate, max_iter=5000):
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    plot_predictions(y_test, y_pred)

    return mape

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Real values', color='blue', marker='o', linestyle='--', markersize=5)
    plt.plot(y_pred, label='Prediction values', color='red', marker='x', linestyle='-', markersize=5)
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

