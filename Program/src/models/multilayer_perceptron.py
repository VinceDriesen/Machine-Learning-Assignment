from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def multilayer_perceptron_classifier(X_train, y_train, X_test, y_test):
    accuracy = calculate_multilayer_perceptron_classifier(X_train, y_train, X_test, y_test)
    return accuracy

def calculate_multilayer_perceptron_classifier(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    print("Multilayer Perceptron Classifier:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"---------------------------------")

    return accuracy