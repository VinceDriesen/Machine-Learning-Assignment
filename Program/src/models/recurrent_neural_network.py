import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error
from src.models.long_short_term_memory import prepare_sequences


class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Alleen de laatste tijdstap nemen
        return out


def recurrent_neural_network_regressor(X_train, y_train, X_test, y_test):
    mape = calculate_recurrent_neural_network_regressor(X_train, y_train, X_test, y_test)
    return mape


def calculate_recurrent_neural_network_regressor(X_train, y_train, X_test, y_test, timesteps=5, epochs=50,
                                                 batch_size=16, lr=0.001):
    # Data voorbereiden
    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, timesteps)
    X_test_seq, y_test_seq = prepare_sequences(X_test, y_test, timesteps)

    # Omzetten naar PyTorch tensors
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

    # Model Initialiseren
    input_dim = X_train.shape[1]  # Aantal kenmerken
    hidden_dim = 50  # Aantal verborgen eenheden
    num_layers = 1  # Aantal lagen in RNN
    output_dim = 1  # Output dimensie (1 voor regressie)

    model = RNNRegressor(input_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.MSELoss()  # Je kunt ook L1Loss proberen
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

    mape = mean_absolute_percentage_error(y_test_seq, y_pred_test)
    print("Recurrent Neural Network Regressor:")
    print(f"MAPE: {mape * 100:.2f}%")
    print(f"---------------------------------")

    return mape
