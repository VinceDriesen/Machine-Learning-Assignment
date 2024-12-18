import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

Y_COLUMN = 'Last Close'
FEATURES = ['Date', 'Open', 'High', 'Low']
COLUMN_HEADERS = FEATURES + [Y_COLUMN]
TEST_SIZE = 0.2

def create_test_train_data(train_file, test_file):
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    try:
        get_column_names(train_data)
        get_column_names(test_data)
    except ValueError as e:
        print(e)

    convert_date_to_datetime(train_data)
    convert_date_to_datetime(test_data)
    plot_data(train_data)
    plot_data(test_data)

    train_data.drop('Date', axis=1, inplace=True)
    test_data.drop('Date', axis=1, inplace=True)

    X_train = train_data.drop(Y_COLUMN, axis=1)
    y_train = train_data[Y_COLUMN]
    X_test = test_data.drop(Y_COLUMN, axis=1)
    y_test = test_data[Y_COLUMN]

    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler

def convert_date_to_datetime(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day

def get_column_names(data):
    headers = data.columns.tolist()
    correct_headers = COLUMN_HEADERS
    missing_headers = [header for header in correct_headers if header not in headers]

    if missing_headers:
        raise ValueError(f"Missing column(s): {', '.join(missing_headers)}")
    print("All required columns are present.")
    return True

def plot_data(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Open'], label='Open Price', alpha=0.7)
    plt.plot(data['Date'], data['High'], label='High Price', alpha=0.7)
    plt.plot(data['Date'], data['Low'], label='Low Price', alpha=0.7)
    plt.plot(data['Date'], data['Last Close'], label='Last Close Price', alpha=0.7)
    plt.title("Stock Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
