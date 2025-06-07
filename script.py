import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import uuid
from datetime import timedelta
from loguru import logger
import sys

# Configure logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("engine_knock_analysis.log", rotation="500 MB")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
logger.info("Random seeds set for reproducibility")

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load and preprocess data
logger.info("Loading data from CSV file")
df = pd.read_csv("./data/engine_knock_data_minute.csv")
logger.info(f"Loaded data shape: {df.shape}")
df = df.drop(columns=['Knock', 'IgnitionTiming'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
logger.info("Data preprocessing completed")

# Initialize the scaler
logger.info("Initializing MinMaxScaler")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.drop(columns=['Timestamp']))
df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=df.columns[1:])
df_scaled['Timestamp'] = df['Timestamp']
df_scaled = df_scaled[['Timestamp'] + [col for col in df_scaled.columns if col != 'Timestamp']]
logger.info("Data scaling completed")

# Define sequence length
sequence_length = 10
logger.info(f"Sequence length set to {sequence_length}")

# Function to create sequences
def create_sequences(data, seq_length):
    logger.debug(f"Creating sequences with length {seq_length}")
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    logger.debug(f"Created {len(X)} sequences")
    return np.array(X), np.array(y)

# LSTM Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=2, output_size=1, dropout=0.2, bidirectional=True):
        super(LSTMModel, self).__init__()
        logger.info(f"Initializing LSTM model with hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        logger.debug("LSTM model architecture created")

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Add early stopping class after the imports
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Add plotting functions after the EarlyStopping class
def plot_feature_predictions(df, df_scaled, scaler, feature, test_predictions, test_targets, results_df):
    plt.figure(figsize=(15, 8))
    feature_idx = df_scaled.columns.get_loc(feature) - 1
    actual_unscaled = scaler.inverse_transform(df_scaled.drop(columns=['Timestamp']).values)[:, feature_idx]
    test_start_idx = len(df_scaled) - len(test_predictions)
    
    plt.plot(df['Timestamp'], actual_unscaled, label='Actual', alpha=0.7)
    plt.plot(df['Timestamp'].iloc[test_start_idx:], results_df[f'{feature}_pred'].iloc[test_start_idx:], 
             label='Predicted', linestyle='--')
    plt.title(f'{feature} Actual vs Predicted')
    plt.xlabel('Timestamp')
    plt.ylabel(feature)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{feature}_predictions.png')
    plt.close()
    logger.info(f"Saved {feature} predictions plot to '{feature}_predictions.png'")

def plot_feature_forecast(feature, forecast_df, last_timestamp):
    plt.figure(figsize=(15, 8))
    plt.plot(forecast_df['Timestamp'], forecast_df[f'{feature}_forecast'], label='Forecast')
    plt.title(f'{feature} 7-Day Forecast')
    plt.xlabel('Timestamp')
    plt.ylabel(feature)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{feature}_forecast.png')
    plt.close()
    logger.info(f"Saved {feature} forecast plot to '{feature}_forecast.png'")

# Modify the train_model function to include early stopping
def train_model(X_train, y_train, X_test, y_test, feature_name, num_epochs=20, batch_size=16):
    logger.info(f"Starting training for feature: {feature_name}")
    model = LSTMModel(input_size=1, hidden_size=100, num_layers=2, output_size=1, 
                      dropout=0.2, bidirectional=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    logger.debug("Model, loss function, optimizer, and early stopping initialized")

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.debug("DataLoaders created")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_targets = []
        param_norms = []
        grad_norms = []

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            optimizer.zero_grad()
            loss.backward()

            # Log gradients and parameters
            current_param_norm = 0
            current_grad_norm = 0
            for p in model.parameters():
                if p.requires_grad:
                    current_param_norm += p.data.norm(2).item()**2
                    if p.grad is not None:
                        current_grad_norm += p.grad.data.norm(2).item()**2
            param_norms.append(current_param_norm**0.5)
            grad_norms.append(current_grad_norm**0.5)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_predictions.extend(outputs.squeeze().detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        train_r2 = r2_score(train_targets, train_predictions)

        # Testing phase
        model.eval()
        total_test_loss = 0
        test_predictions = []
        test_targets = []
        with torch.no_grad():
            for batch_X_test, batch_y_test in test_loader:
                batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                y_pred_batch = model(batch_X_test)
                test_loss_batch = criterion(y_pred_batch.squeeze(), batch_y_test)
                total_test_loss += test_loss_batch.item()
                test_predictions.extend(y_pred_batch.squeeze().detach().cpu().numpy())
                test_targets.extend(batch_y_test.detach().cpu().numpy())

        avg_test_loss = total_test_loss / len(test_loader)
        test_r2 = r2_score(test_targets, test_predictions)

        # Log training progress
        logger.info(f'Feature: {feature_name}, Epoch [{epoch+1}/{num_epochs}]')
        logger.info(f'  Training Loss: {avg_train_loss:.4f}, R²: {train_r2:.4f}')
        logger.info(f'  Test Loss: {avg_test_loss:.4f}, R²: {test_r2:.4f}')
        logger.info(f'  Avg Parameter Norm: {np.mean(param_norms):.4f}')
        logger.info(f'  Avg Gradient Norm: {np.mean(grad_norms):.4f}')

        # Early stopping check
        early_stopping(avg_test_loss)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered for {feature_name} at epoch {epoch+1}")
            break

    logger.info(f"Training completed for feature: {feature_name}")
    return model, test_predictions, test_targets

# Function to forecast next 7 days
def forecast_future(model, last_sequence, n_future=10080):
    logger.info(f"Starting forecast for next {n_future} time steps")
    model.eval()
    predictions = []
    current_sequence = last_sequence.copy()
    with torch.no_grad():
        for i in range(n_future):
            input_seq = torch.FloatTensor(current_sequence).reshape(1, sequence_length, 1).to(device)
            pred = model(input_seq).cpu().numpy()[0, 0]
            predictions.append(pred)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred
            if i % 1000 == 0:
                logger.debug(f"Forecast progress: {i}/{n_future}")
    logger.info("Forecast completed")
    return np.array(predictions)

# Initialize storage for results
all_test_predictions = {}
all_test_targets = {}
all_models = {}
all_forecasts = {}
features = ['RPM', 'CylinderPressure', 'BurnRate', 'Vibration', 'EGOVoltage', 'TempSensor']

# Initialize results_df with the same index as df_scaled
results_df = pd.DataFrame(index=df_scaled.index)
results_df['Timestamp'] = df['Timestamp']

# Modify the main training loop to include immediate plotting
for feature in features:
    print(f"\nProcessing feature: {feature}")
    feature_data = df_scaled[feature].values
    X, y = create_sequences(feature_data, sequence_length)
    
    # Split data
    train_size = int(len(X) * 0.9)
    X_train = torch.FloatTensor(X[:train_size]).reshape(-1, sequence_length, 1).to(device)
    y_train = torch.FloatTensor(y[:train_size]).to(device)
    X_test = torch.FloatTensor(X[train_size:]).reshape(-1, sequence_length, 1).to(device)
    y_test = torch.FloatTensor(y[train_size:]).to(device)

    # Train model
    model, test_predictions, test_targets = train_model(X_train, y_train, X_test, y_test, feature)
    all_models[feature] = model
    all_test_predictions[feature] = test_predictions
    all_test_targets[feature] = test_targets

    # Process and plot predictions
    test_pred_array = np.array(test_predictions).reshape(-1, 1)
    test_target_array = np.array(test_targets).reshape(-1, 1)
    feature_idx = df_scaled.columns.get_loc(feature) - 1
    
    # Inverse transform predictions
    test_pred_unscaled = scaler.inverse_transform(
        np.concatenate([np.zeros((len(test_pred_array), feature_idx)), test_pred_array, 
                        np.zeros((len(test_pred_array), len(features) - feature_idx - 1))], axis=1)
    )[:, feature_idx]
    
    # Update results_df for this feature
    results_df[f'{feature}_pred'] = np.nan
    test_start_idx = len(df_scaled) - len(test_pred_array)
    results_df.loc[test_start_idx:, f'{feature}_pred'] = test_pred_unscaled

    # Plot predictions for this feature
    plot_feature_predictions(df, df_scaled, scaler, feature, test_predictions, test_targets, results_df)

    # Forecast and plot for this feature
    last_sequence = feature_data[-sequence_length:]
    forecast = forecast_future(model, last_sequence)
    all_forecasts[feature] = forecast

    # Create and plot forecast for this feature
    last_timestamp = df['Timestamp'].iloc[-1]
    forecast_timestamps = [last_timestamp + timedelta(minutes=i+1) for i in range(10080)]
    forecast_df = pd.DataFrame({
        'Timestamp': forecast_timestamps,
        f'{feature}_forecast': scaler.inverse_transform(
            np.concatenate([np.zeros((len(forecast), feature_idx)), forecast.reshape(-1, 1), 
                            np.zeros((len(forecast), len(features) - feature_idx - 1))], axis=1)
        )[:, feature_idx]
    })
    
    # Plot forecast for this feature
    plot_feature_forecast(feature, forecast_df, last_timestamp)

    # Update all_forecast_df
    if feature == features[0]:
        all_forecast_df = forecast_df
    else:
        all_forecast_df = all_forecast_df.merge(forecast_df, on='Timestamp')

# Save results to CSV
results_df.to_csv('engine_knock_predictions.csv', index=False)
all_forecast_df.to_csv('engine_knock_forecasts.csv', index=False)
print("Saved predictions to 'engine_knock_predictions.csv' and forecasts to 'engine_knock_forecasts.csv'")