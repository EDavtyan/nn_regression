import pandas as pd
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna
import json
import time

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from utils import (
    load_and_preprocess_concrete,
    load_and_preprocess_bank32NH,
    load_and_preprocess_delta_elevators,
    load_and_preprocess_house_16,
    load_and_preprocess_housing,
    load_and_preprocess_insurance
)

#############################
# Logging Setup
#############################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

#############################
# SAINT Model Definition
#############################
class SAINT(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim=32,
        depth=4,
        num_heads=4,
        dropout=0.1,
        use_inter_sample_attention=False
    ):
        super(SAINT, self).__init__()
        self.embed_dim = embed_dim
        self.use_inter_sample_attention = use_inter_sample_attention

        # Learnable embedding matrix for each feature
        self.feature_embeddings = nn.Parameter(torch.randn(input_dim, embed_dim))
        nn.init.xavier_uniform_(self.feature_embeddings)

        # Transformer encoder for column-wise attention
        encoder_layer_cols = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder_cols = nn.TransformerEncoder(encoder_layer_cols, num_layers=depth)

        # Final regression head
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # Input: x -> (batch_size, input_dim)
        batch_size, input_dim = x.size()

        # Embed features: Multiply input values with learnable embeddings
        embedded_features = x.unsqueeze(-1) * self.feature_embeddings.unsqueeze(0)
        # Shape: (batch_size, input_dim, embed_dim)

        # Column-wise attention
        col_encoded = self.transformer_encoder_cols(embedded_features)  # (batch_size, input_dim, embed_dim)

        # Aggregate features using mean pooling
        pooled = col_encoded.mean(dim=1)  # Shape: (batch_size, embed_dim)

        # Predict regression target
        out = self.fc(pooled)  # Shape: (batch_size, 1)
        return out

#############################
# Training and Evaluation
#############################
def train_saint(X_train, y_train, X_test, y_test, params):
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    # Initialize model
    model = SAINT(
        input_dim=X_train.shape[1],
        embed_dim=params['embed_dim'],
        depth=params['depth'],
        num_heads=params['num_heads'],
        dropout=params['dropout']
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    # Measure training time for all epochs
    train_start = time.time()
    for epoch in range(params['epochs']):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
    train_end = time.time()

    # Print training time for these epochs (in seconds, no decimals)
    epoch_training_time = int(train_end - train_start)
    logger.info(f"Training time for current run: {epoch_training_time} seconds")

    # Evaluation on training and test sets
    model.eval()
    with torch.no_grad():
        pred_train = model(X_train_t).squeeze(-1).numpy()
        train_mse = mean_squared_error(y_train, pred_train)
        train_r2 = r2_score(y_train, pred_train)

        pred_test = model(X_test_t).squeeze(-1).numpy()
        test_mse = mean_squared_error(y_test, pred_test)
        test_r2 = r2_score(y_test, pred_test)

    return train_mse, train_r2, test_mse, test_r2

#############################
# Hyperparameter Optimization with Optuna
#############################
def objective(trial, X, y):
    # Split the data into train and validation sets
    kfold = KFold(n_splits=3, shuffle=True)
    scores = []

    # Define hyperparameters with constraints
    embed_dim = trial.suggest_categorical('embed_dim', [16, 32, 64])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])

    if embed_dim % num_heads != 0:
        raise optuna.TrialPruned(f"Invalid combination: embed_dim {embed_dim} is not divisible by num_heads {num_heads}")

    params = {
        'embed_dim': embed_dim,
        'depth': trial.suggest_int('depth', 2, 6),
        'num_heads': num_heads,
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'epochs': 50
    }

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        _, _, val_mse, _ = train_saint(X_train, y_train, X_val, y_val, params)
        scores.append(val_mse)

    return np.mean(scores)

#############################
# Main Execution
#############################
if __name__ == "__main__":
    overall_start = time.time()  # Start time of entire experiment

    datasets = [
        ("Concrete", load_and_preprocess_concrete),
        ("Bank32NH", load_and_preprocess_bank32NH),
        ("Delta Elevators", load_and_preprocess_delta_elevators),
        ("House 16", load_and_preprocess_house_16),
        ("Housing", load_and_preprocess_housing),
        ("Insurance", load_and_preprocess_insurance)
    ]

    results = {}

    for dataset_name, loader in datasets:
        logger.info(f"Processing dataset: {dataset_name}")
        try:
            X_train, X_test, y_train, y_test, _ = loader()

            # Time the hyperparameter optimization
            optimize_start = time.time()
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, np.vstack((X_train, X_test)), np.hstack((y_train, y_test))), n_trials=20)
            optimize_end = time.time()

            optimization_minutes = round((optimize_end - optimize_start) / 60, 1)
            logger.info(f"Time taken for hyperparameter optimization for {dataset_name}: {optimization_minutes} minutes")

            # Train with the best parameters
            best_params = study.best_params
            logger.info(f"Best parameters for {dataset_name}: {best_params}")

            train_mse, train_r2, test_mse, test_r2 = train_saint(X_train, y_train, X_test, y_test, best_params)
            logger.info(f"Final Train MSE: {train_mse:.4f}, R^2: {train_r2:.4f}")
            logger.info(f"Final Test MSE: {test_mse:.4f}, R^2: {test_r2:.4f}")

            # Store results in JSON
            results[dataset_name] = {
                "best_params": best_params,
                "train_mse": train_mse,
                "train_r2": train_r2,
                "test_mse": test_mse,
                "test_r2": test_r2
            }

            # Update results.json after processing this dataset
            with open("results/results_SAINT.json", "w") as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {str(e)}")

    overall_end = time.time()
    total_hours = round((overall_end - overall_start) / 3600, 1)
    logger.info(f"Total experiment time: {total_hours} hours")