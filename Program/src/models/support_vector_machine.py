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
    for kernel in KERNELS:
        mape = calculate_support_vector_machine(X_train, y_train, X_test, y_test, kernel)
        mape_scores.append(mape)
    best_kernel = KERNELS[np.argmin(mape_scores)]
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