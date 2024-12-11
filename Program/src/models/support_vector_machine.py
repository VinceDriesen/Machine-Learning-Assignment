import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']


def support_vector_machine(X_train, y_train, X_test, y_test):
    mape_scores = []
    predictions = {}

    for kernel in KERNELS:
        model = SVR(kernel=kernel)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        mape = mean_absolute_percentage_error(y_test, test_pred)
        mape_scores.append(mape)

        predictions[kernel] = (y_train, train_pred, y_test, test_pred)

    best_kernel = KERNELS[np.argmin(mape_scores)]

    train_true, train_pred, test_true, test_pred = predictions[best_kernel]
    plot_predictions(train_true, train_pred, test_true, test_pred, best_kernel)

    return best_kernel, min(mape_scores)


def plot_predictions(train_true, train_pred, test_true, test_pred, best_kernel):
    plt.figure(figsize=(14, 6))

    # Training data plotten
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(train_true)), train_true, color="red", label="Echte waarden", s=5)
    plt.scatter(range(len(train_pred)), train_pred, color="blue", label="Voorspellingen", s=5)
    plt.title(f"Training Data - Kernel: {best_kernel}")
    plt.xlabel("Index")
    plt.ylabel("Waarde")
    plt.legend()

    # Test data plotten
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(test_true)), test_true, color="blue", label="Echte waarden", s=5)
    plt.scatter(range(len(test_pred)), test_pred, color="orange", label="Voorspellingen", s=5)
    plt.title(f"Test Data - Kernel: {best_kernel}")
    plt.xlabel("Index")
    plt.ylabel("Waarde")
    plt.legend()

    plt.tight_layout()
    plt.show()

