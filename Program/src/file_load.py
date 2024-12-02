import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

Y_COLUMN = 'Last Close'
FEATURES = ['Date', 'Open', 'High', 'Low']
COLUMN_HEADERS = FEATURES + [Y_COLUMN]
TEST_SIZE = 0.2

def create_test_train_data(file_path):
    data = pd.read_csv(file_path)

    try:
        get_column_names(data)
    except ValueError as e:
        print(e)

    convert_date_to_datetime(data)
    plot_data(data)

    X = data.drop(Y_COLUMN, axis=1)
    y = data[Y_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    return X_train, y_train, X_test, y_test

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