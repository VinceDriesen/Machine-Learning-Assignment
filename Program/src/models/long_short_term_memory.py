import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model(num_epochs, model, train_loader, criterion, optimizer, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

def plot_predictions(y_test, y_pred, title="Model Predictions vs Actual Values"):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual Values", color='b', marker='o')
    plt.plot(y_pred, label="Predicted Values", color='r', marker='x')
    plt.fill_between(range(len(y_test)), y_test, y_pred, color='gray', alpha=0.2)
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_lstm_regressor(X_train, y_train, X_test, y_test, timesteps=5, epochs=50, batch_size=16, lr=0.001, hidden_dim=50, num_layers=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, timesteps)
    X_test_seq, y_test_seq = prepare_sequences(X_test, y_test, timesteps)

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    output_dim = 1

    model = LSTMRegressor(input_dim, hidden_dim, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_model(epochs, model, train_loader, criterion, optimizer, device)

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).squeeze().cpu().numpy()

    mape = mean_absolute_percentage_error(y_test_seq, y_pred_test)
    print(f"LSTM Regressor - hidden_dim={hidden_dim}, num_layers={num_layers}, lr={lr}:")
    print(f"MAPE: {mape * 100:.2f}%")
    print("---------------------------------")

    plot_predictions(y_test_seq, y_pred_test, title=f"LSTM Regressor - hidden_dim={hidden_dim}, num_layers={num_layers}, lr={lr} - MAPE: {mape * 100:.2f}%")

    return mape

def run_grid_search_lstm(X_train, y_train, X_test, y_test, timesteps=5, epochs=50, batch_size=16):
    hidden_dims = [10, 20, 50, 100]
    num_layers_list = [1, 2, 3, 4]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]

    best_mape = float('inf')
    best_params = {}

    for hidden_dim in hidden_dims:
        for num_layers in num_layers_list:
            for lr in learning_rates:
                print(f"Training with hidden_dim={hidden_dim}, num_layers={num_layers}, lr={lr}")
                mape = calculate_lstm_regressor(X_train, y_train, X_test, y_test, timesteps, epochs, batch_size, lr, hidden_dim, num_layers)
                if mape < best_mape:
                    best_mape = mape
                    best_params = {'hidden_dim': hidden_dim, 'num_layers': num_layers, 'lr': lr}
                print(f"MAPE: {mape * 100:.2f}%\n")

    print(f"Best MAPE: {best_mape * 100:.2f}% with params: {best_params}")

def prepare_sequences(X, y, timesteps):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i + timesteps])
        y_seq.append(y[i + timesteps])
    return np.array(X_seq), np.array(y_seq)
