import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn

# LSTM Model Class
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Output van de laatste tijdstap
        return out

def lstm_regressor(X_train, y_train, X_test, y_test):
    mape = calculate_lstm_regressor(X_train, y_train, X_test, y_test)
    return mape

def calculate_lstm_regressor(X_train, y_train, X_test, y_test, timesteps=5, epochs=50, batch_size=16, lr=0.001):
    # Data schalen
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Data voorbereiden voor LSTM
    X_train_seq, y_train_seq = prepare_sequences(X_train_scaled, y_train, timesteps)
    X_test_seq, y_test_seq = prepare_sequences(X_test_scaled, y_test, timesteps)

    # Conversie naar PyTorch tensors
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

    # Model initialiseren
    input_dim = X_train.shape[1]
    hidden_dim = 50
    num_layers = 1
    output_dim = 1
    model = LSTMRegressor(input_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluatie
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).squeeze().numpy()

    # MAPE berekenen
    mape = mean_absolute_percentage_error(y_test_seq, y_pred_test)
    print("LSTM Regressor:")
    print(f"MAPE: {mape * 100:.2f}%")
    print(f"---------------------------------")

    return mape

# Functie om sequenties te genereren
def prepare_sequences(X, y, timesteps):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i + timesteps])  # Maak de sequentie van de features
        y_seq.append(y[i + timesteps])   # Gebruik NumPy-indexering
    return np.array(X_seq), np.array(y_seq)


