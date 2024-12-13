import torch
import csv
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from src.models.long_short_term_memory import prepare_sequences

import multiprocessing
from itertools import product

def grid_search_worker(params):
    hidden_dim, num_layers, lr, ts, epoch, bs, X_train, y_train, X_test, y_test = params
    results = []
    for i in range(1, 4):  # Herhaal 3 keer
        mape, r2 = recurrent_neural_network_regressor(
            X_train, y_train, X_test, y_test, ts, epoch, bs, lr, hidden_dim, num_layers
        )
        results.append((hidden_dim, num_layers, lr, ts, epoch, bs, i, mape, r2))
        print(f"[Iteration {i}] MAPE: {mape * 100:.2f}%, R²: {r2:.4f} | hidden_dim={hidden_dim}, num_layers={num_layers}, "
              f"lr={lr}, timesteps={ts}, epochs={epoch}, batch_size={bs}")
    return results

def run_grid_search_rnn_parallel(X_train, y_train, X_test, y_test, output_csv="rnn.csv"):
    hidden_dims = [10, 50, 100]
    num_layers_list = [1, 2]
    learning_rates = [0.001, 0.01, 0.1]
    timesteps = [1, 5, 10]
    epochs = [50, 100, 150]
    batch_size = [4, 8, 16]

    param_combinations = list(product(hidden_dims, num_layers_list, learning_rates, timesteps, epochs, batch_size))
    param_combinations = [(hd, nl, lr, ts, ep, bs, X_train, y_train, X_test, y_test) for hd, nl, lr, ts, ep, bs in param_combinations]

    best_mape = float('inf')
    best_r2 = -float('inf')
    best_params = {}

    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["hidden_dim", "num_layers", "learning_rate", "timesteps", "epochs", "batch_size", "iteration", "MAPE", "R²"])

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(grid_search_worker, param_combinations)

        for (params, (mape, r2)) in zip(param_combinations, results):
            for iteration in range(1, 4):
                hidden_dim, num_layers, lr, ts, epoch, bs, _, _, _, _ = params
                writer.writerow([hidden_dim, num_layers, lr, ts, epoch, bs, 1, mape, r2])  # Adjust iteration as needed

                if mape < best_mape:
                    best_mape = mape
                    best_params = {
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'learning_rate': lr,
                        'timesteps': ts,
                        'epochs': epoch,
                        'batch_size': bs
                    }
                if r2 > best_r2:
                    best_r2 = r2

    print(f"Best MAPE: {best_mape * 100:.2f}% with params: {best_params}")
    print(f"Best R²: {best_r2:.4f}")
    print(f"Results saved to {output_csv}")


class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_size):
        super(RNNRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def train_model(num_epochs, model, train_loader, criterion, optimizer, device):
    model.train()

    # Print device information
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")

    # Use mixed precision if CUDA is available
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def recurrent_neural_network_regressor(X_train, y_train, X_test, y_test, timesteps=5, epochs=50, batch_size=16, lr=0.001, hidden_size=20, num_layers=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, timesteps)
    X_test_seq, y_test_seq = prepare_sequences(X_test, y_test, timesteps)

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    input_dim = X_train.shape[1]
    output_dim = 1

    model = RNNRegressor(input_dim, hidden_size, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_model(epochs, model, train_loader, criterion, optimizer, device)

    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        y_pred_test = model(X_test_tensor).squeeze().cpu().numpy()

    # MAPE
    mape = mean_absolute_percentage_error(y_test_seq, y_pred_test)
    # R2
    r2 = r2_score(y_test_seq, y_pred_test)

    print(f"Test MAPE: {mape * 100:.2f}%")
    print(f"Test R²: {r2:.4f}")
    print("---------------------------------")

    return mape, r2

def run_grid_search_rnn(X_train, y_train, X_test, y_test, output_csv="rnn.csv"):
    hidden_dims = [10, 50, 100]
    num_layers_list = [1, 2]
    learning_rates = [0.001, 0.01, 0.1]
    timesteps = [1, 5, 10]
    epochs = [50, 100, 150]
    batch_size = [4, 8, 16]

    best_mape = float('inf')
    best_r2 = -float('inf')
    best_params = {}

    # Write header to CSV file
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["hidden_dim", "num_layers", "learning_rate", "timesteps", "epochs", "batch_size", "iteration", "MAPE", "R²"])

        for hidden_dim in hidden_dims:
            for num_layers in num_layers_list:
                for lr in learning_rates:
                    for ts in timesteps:
                        for epoch in epochs:
                            for bs in batch_size:
                                for iteration in range(1, 4):  # Repeat each combination 3 times
                                    mape, r2 = recurrent_neural_network_regressor(
                                        X_train, y_train, X_test, y_test, ts, epoch, bs, lr, hidden_dim, num_layers
                                    )

                                    writer.writerow([hidden_dim, num_layers, lr, ts, epoch, bs, iteration, mape, r2])

                                    if mape < best_mape:
                                        best_mape = mape
                                        best_params = {
                                            'hidden_dim': hidden_dim,
                                            'num_layers': num_layers,
                                            'learning_rate': lr,
                                            'timesteps': ts,
                                            'epochs': epoch,
                                            'batch_size': bs
                                        }
                                    if r2 > best_r2:
                                        best_r2 = r2

    print(f"Best MAPE: {best_mape * 100:.2f}% with params: {best_params}")
    print(f"Best R²: {best_r2:.4f}")
    print(f"Results saved to {output_csv}")
