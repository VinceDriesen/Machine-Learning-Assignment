from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

def naive_bayes_regressor(X_train, y_train, X_test, y_test):
    # Discretiseer de doelvariabele
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    y_train_discretized = discretizer.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = GaussianNB()
    model.fit(X_train, y_train_discretized)

    # Maak voorspellingen en converteer terug naar continue waarden
    y_pred_discretized = model.predict_proba(X_test)
    y_pred = np.dot(y_pred_discretized, discretizer.bin_edges_[0][:-1])

    # Bereken MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("Naive Bayes Regressor:")
    print(f"MAPE: {mape * 100:.2f}%")
    print("---------------------------------")
    return mape
