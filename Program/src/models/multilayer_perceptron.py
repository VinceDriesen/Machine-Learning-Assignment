import csv
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def multilayer_perceptron_regressor(X_train, y_train, X_test, y_test, hidden_layer_sizes, activation, solver,
                                    learning_rate, max_iter=5000):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)  # R² toevoegen

    # plot_predictions(y_test, y_pred)  # Als je de plot wilt behouden

    return mape, r2  # Retourneer zowel MAPE als R²


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


def run_grid_search_mlp(X_train, y_train, X_test, y_test, output_csv="mlp.csv"):
    hidden_layer_sizes = [(10,), (20,), (50,), (100,), (50, 50)]
    activations = ['relu', 'tanh', 'logistic']
    solvers = ['adam', 'sgd', 'lbfgs']
    learning_rates = ['constant', 'invscaling', 'adaptive']
    max_iter = 5000

    best_mape = float('inf')
    best_r2 = -float('inf')  # Startwaarde voor R², want R² kan tussen -∞ en 1 liggen
    best_params = {}

    # Schrijf header naar CSV-bestand
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["hidden_layer_sizes", "activation", "solver", "learning_rate", "iteration", "MAPE", "R²"])

        for hidden_size in hidden_layer_sizes:
            for activation in activations:
                for solver in solvers:
                    for lr in learning_rates:
                        for iteration in range(1, 4):  # Herhaal elke combinatie 5 keer
                            # print(f"Training with hidden_layer_sizes={hidden_size}, activation={activation}, solver={solver}, learning_rate={lr}, iteration={iteration}")

                            # Bereken MAPE en R²
                            mape, r2 = multilayer_perceptron_regressor(
                                X_train, y_train, X_test, y_test, hidden_size, activation, solver, lr, max_iter
                            )

                            # Schrijf resultaten naar CSV-bestand
                            writer.writerow([hidden_size, activation, solver, lr, iteration, mape, r2])

                            # Update beste MAPE en beste R²
                            if mape < best_mape:
                                best_mape = mape
                                best_params['mape'] = {
                                    'hidden_layer_sizes': hidden_size,
                                    'activation': activation,
                                    'solver': solver,
                                    'learning_rate': lr
                                }
                            if r2 > best_r2:
                                best_r2 = r2
                                best_params['r2'] = {
                                    'hidden_layer_sizes': hidden_size,
                                    'activation': activation,
                                    'solver': solver,
                                    'learning_rate': lr
                                }

                            # print(f"Iteration {iteration} MAPE: {mape * 100:.2f}%, R²: {r2:.4f}\n")

    print(f"Best MAPE: {best_mape * 100:.2f}% with params: {best_params['mape']}")
    print(f"Best R²: {best_r2:.4f} with params: {best_params['r2']}")
    print(f"Results saved to {output_csv}")
