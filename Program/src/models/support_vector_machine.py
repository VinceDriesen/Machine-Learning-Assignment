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
    plot_predictions(test_true, test_pred)

    return best_kernel, min(mape_scores)


def plot_predictions(test_true, test_pred):
    plt.figure(figsize=(14, 6))

    plt.scatter(range(len(test_true)), test_true, color="blue", label="Real values", s=5)
    plt.scatter(range(len(test_pred)), test_pred, color="orange", label="Predictions", s=5)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()

    plt.tight_layout()
    plt.show()

