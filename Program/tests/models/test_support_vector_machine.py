import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error

from src.models.support_vector_machine import calculate_support_vector_machine, support_vector_machine

@pytest.fixture
def sample_data():
    # Voorbeelddata maken
    data = {
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100,
        'Low': np.random.rand(100) * 100,
        'Last Close': np.random.rand(100) * 100
    }
    df = pd.DataFrame(data)
    X = df[['Open', 'High', 'Low']]
    y = df['Last Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test

def test_calculate_support_vector_machine(sample_data):
    X_train, y_train, X_test, y_test = sample_data
    kernel = 'linear'
    mape = calculate_support_vector_machine(X_train, y_train, X_test, y_test, kernel)
    assert mape >= 0, "MAPE moet groter of gelijk aan 0 zijn"

def test_support_vector_machine(sample_data):
    X_train, y_train, X_test, y_test = sample_data
    best_kernel = support_vector_machine(X_train, y_train, X_test, y_test)
    assert best_kernel in ['linear', 'poly', 'rbf', 'sigmoid'], "Beste kernel moet een van de ondersteunde kernels zijn"
