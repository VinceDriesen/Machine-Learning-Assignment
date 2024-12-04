from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

def naive_bayes_classifier(X_train, y_train, X_test, y_test):
    mape = calculate_naive_bayes_classifier(X_train, y_train, X_test, y_test)
    return mape

def calculate_naive_bayes_classifier(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print("Naive Bayes Classifier:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"Naive bayes classifier MAPE: {mape * 100:.2f}%")
    print(f"---------------------------------")

    return mape
