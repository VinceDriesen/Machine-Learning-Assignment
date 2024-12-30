import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from models.long_short_term_memory import prepare_sequences

class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_size):
        super(RNNRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def train_model(num_epochs, model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
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
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.show()

def recurrent_neural_network_regressor(X_train, y_train, X_test, y_test, timesteps=5, epochs=50, batch_size=16, lr=0.001, hidden_size=20, num_layers=1):
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

    model = RNNRegressor(input_dim, hidden_size, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_model(epochs, model, train_loader, criterion, optimizer)

    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor
        y_pred_test = model(X_test_tensor).squeeze().cpu().numpy()

    mape = mean_absolute_percentage_error(y_test_seq, y_pred_test)

    plot_predictions(y_test_seq, y_pred_test, title=f"RNN Regressor - hidden_size={hidden_size}, num_layers={num_layers}, lr={lr} - MAPE: {mape * 100:.2f}%")
    return mape