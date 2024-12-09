import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']


def support_vector_machine(X_train, y_train, X_test, y_test):
    mape_scores = []
    predictions = {}

    for kernel in KERNELS:
        # Train het model en maak voorspellingen
        model = SVR(kernel=kernel)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Bereken MAPE en sla gegevens op
        mape = mean_absolute_percentage_error(y_test, y_pred_test)
        mape_scores.append(mape)
        predictions[kernel] = {
            "train": (y_train, y_pred_train),
            "test": (y_test, y_pred_test)
        }

    # Vind de beste kernel
    best_kernel = KERNELS[np.argmin(mape_scores)]

    # Plot voor de beste kernel
    train_true, train_pred = predictions[best_kernel]["train"]
    test_true, test_pred = predictions[best_kernel]["test"]

    plt.figure(figsize=(12, 6))

    # Training data plotten
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(train_true)), train_true, color="blue", label="Echte waarden")
    plt.scatter(range(len(train_pred)), train_pred, color="green", label="Voorspellingen")
    plt.title(f"Training Data - Kernel: {best_kernel}")
    plt.xlabel("Index")
    plt.ylabel("Waarde")
    plt.legend()

    # Test data plotten
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(test_true)), test_true, color="blue", label="Echte waarden")
    plt.scatter(range(len(test_pred)), test_pred, color="orange", label="Voorspellingen")
    plt.title(f"Test Data - Kernel: {best_kernel}")
    plt.xlabel("Index")
    plt.ylabel("Waarde")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Beste kernel: {best_kernel}")
    return best_kernel, min(mape_scores)


def calculate_support_vector_machine(X_train, y_train, X_test, y_test, kernel_name):
    model = SVR(kernel=kernel_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Kernel: {kernel_name}")
    print(f"MAPE: {mape * 100:.2f}%")
    print(f"---------------------------------")

    return mape
