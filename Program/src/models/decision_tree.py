from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

def decision_tree_regressor(X_train, y_train, X_test, y_test):
    # Model training
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Voorspellingen maken
    y_pred = model.predict(X_test)

    # Berekening van MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"Decision Tree Regressor MAPE: {mape * 100:.2f}%")

    # Feature importance plotten
    feature_importances = model.feature_importances_
    plt.bar(np.arange(len(feature_importances)), feature_importances)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.show()

    return mape

# Hyperparameter-tuning met GridSearchCV
def tune_decision_tree(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42),
                               param_grid,
                               scoring='neg_mean_absolute_percentage_error',
                               cv=5)
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validated MAPE:", -grid_search.best_score_)
    return grid_search.best_estimator_