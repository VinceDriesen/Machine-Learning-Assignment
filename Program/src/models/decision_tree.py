from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

def decision_tree_regressor(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("Decision Tree Regressor:")
    print(f"MAPE: {mape * 100:.2f}%")
    print("---------------------------------")
    return mape
