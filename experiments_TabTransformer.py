import json
import logging
import os
import time

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from tab_transformer_pytorch import TabTransformer
from torch.utils.data import DataLoader, TensorDataset
from utils import (
    load_and_preprocess_bank32NH,
    load_and_preprocess_concrete,
    load_and_preprocess_delta_elevators,
    load_and_preprocess_house_16,
    load_and_preprocess_housing,
    load_and_preprocess_insurance,
    load_and_preprocess_movies
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
# Global Setting for Iterations
#############################
NUM_ITERATIONS = 15  # <--- How many times to run each dataset

#############################
# Helper Function for Encoding
#############################
def preprocess_data_for_tab_transformer(X_train, X_test, column_names, categorical_threshold=15):
    cat_indices = []
    num_indices = []

    # Identify categorical vs numerical columns
    for i, col in enumerate(column_names):
        try:
            temp = X_train[:, i].astype(float)
            unique_vals = np.unique(temp)
            if len(unique_vals) < categorical_threshold:
                cat_indices.append(i)
            else:
                num_indices.append(i)
        except ValueError:
            cat_indices.append(i)

    encoders = {}
    for i in cat_indices:
        all_vals = np.concatenate([X_train[:, i], X_test[:, i]])
        all_vals_str = all_vals.astype(str)
        le = LabelEncoder()
        le.fit(all_vals_str)
        X_train[:, i] = le.transform(X_train[:, i].astype(str))
        X_test[:, i] = le.transform(X_test[:, i].astype(str))
        encoders[i] = le

    # Ensure correct dtypes
    for i in num_indices:
        X_train[:, i] = X_train[:, i].astype(float)
        X_test[:, i] = X_test[:, i].astype(float)

    # Construct categorical and numerical arrays
    if cat_indices:
        categoricals_train = X_train[:, cat_indices].astype(int)
        categoricals_test = X_test[:, cat_indices].astype(int)
    else:
        categoricals_train = np.empty((X_train.shape[0], 0), dtype=int)
        categoricals_test = np.empty((X_test.shape[0], 0), dtype=int)

    if num_indices:
        numericals_train = X_train[:, num_indices].astype(float)
        numericals_test = X_test[:, num_indices].astype(float)
    else:
        numericals_train = np.empty((X_train.shape[0], 0), dtype=float)
        numericals_test = np.empty((X_test.shape[0], 0), dtype=float)

    cat_cardinalities = []
    for i in cat_indices:
        if i in encoders:
            cat_cardinalities.append(len(encoders[i].classes_))
        else:
            cat_cardinalities.append(int(np.max(X_train[:, i])) + 1)

    return categoricals_train, numericals_train, categoricals_test, numericals_test, cat_cardinalities

#############################
# Training and Evaluation
#############################
def train_tabtransformer(categoricals, numericals, y, params, cat_cardinalities):
    """
    Train the TabTransformer on the provided dataset.
    Returns the trained model.
    """
    categoricals_t = torch.tensor(categoricals, dtype=torch.long)
    numericals_t = torch.tensor(numericals, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(categoricals_t, numericals_t, y_t)
    dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    if params["embed_dim"] % params["num_heads"] != 0:
        raise ValueError(f"embed_dim {params['embed_dim']} must be divisible by num_heads {params['num_heads']}")

    model = TabTransformer(
        categories=cat_cardinalities,
        num_continuous=numericals.shape[1],
        dim=params["embed_dim"],
        dim_out=1,
        depth=params["depth"],
        heads=params["num_heads"],
        attn_dropout=params["dropout"],
        ff_dropout=params["dropout"]
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    train_start = time.time()
    model.train()
    for epoch in range(params["epochs"]):
        for cat_batch, num_batch, y_batch in dataloader:
            optimizer.zero_grad()
            preds = model(cat_batch, num_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
    train_end = time.time()

    logger.info(f"Training time for current run: {int(train_end - train_start)} seconds")

    return model

def evaluate_model(model, categoricals, numericals, y):
    """
    Evaluate the trained model on given data.
    Returns MSE and R^2.
    """
    model.eval()
    with torch.no_grad():
        categoricals_t = torch.tensor(categoricals, dtype=torch.long)
        numericals_t = torch.tensor(numericals, dtype=torch.float32)
        preds = model(categoricals_t, numericals_t).squeeze(-1).numpy()

        if np.isnan(preds).any():
            raise ValueError("Model predictions contain NaN values.")

        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)
    return mse, r2

#############################
# Hyperparameter Optimization with Optuna
#############################
def objective(trial, categoricals, numericals, y, cat_cardinalities):
    embed_dim = trial.suggest_categorical("embed_dim", [16, 32, 64])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    if embed_dim % num_heads != 0:
        # If not divisible, prune trial
        raise optuna.TrialPruned(
            f"Invalid combination: embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        )

    params = {
        "embed_dim": embed_dim,
        "depth": trial.suggest_int("depth", 2, 6),
        "num_heads": num_heads,
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": trial.suggest_int("epochs", 5, 35, step=10)
    }

    model = train_tabtransformer(categoricals, numericals, y, params, cat_cardinalities)
    mse, _ = evaluate_model(model, categoricals, numericals, y)
    return mse

#############################
# Main Execution
#############################
if __name__ == "__main__":
    overall_start = time.time()  # Start time of the entire experiment

    datasets = [
        ("Movies", load_and_preprocess_movies),
        ("Concrete", load_and_preprocess_concrete),
        ("Bank32NH", load_and_preprocess_bank32NH),
        ("House 16", load_and_preprocess_house_16),
        ("Delta Elevators", load_and_preprocess_delta_elevators),
        ("Housing", load_and_preprocess_housing),
        ("Insurance", load_and_preprocess_insurance),
    ]

    # Load existing results if they exist
    results_path = f"results/results_TabTransformer_low_epochs_{NUM_ITERATIONS}_itterations.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    for dataset_name, loader in datasets:
        logger.info(f"Processing dataset: {dataset_name}")

        try:
            X_train, X_test, y_train, y_test, column_names = loader()

            (categoricals_train, numericals_train,
             categoricals_test, numericals_test,
             cat_cardinalities) = preprocess_data_for_tab_transformer(
                 X_train, X_test, column_names
             )

            # Create a sub-dict in results if it doesn't exist yet
            if dataset_name not in results:
                results[dataset_name] = {}

            # Run multiple iterations
            for iteration_idx in range(1, NUM_ITERATIONS + 1):
                iteration_key = f"Iteration_{iteration_idx}"

                # Check if the current iteration already exists in JSON
                if iteration_key in results[dataset_name]:
                    logger.info(
                        f"{dataset_name}: {iteration_key} already exists in the JSON. Skipping..."
                    )
                    continue

                logger.info(f"  Iteration {iteration_idx} for {dataset_name}")

                # Time the hyperparameter optimization for this iteration
                optimize_start = time.time()
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda trial: objective(trial, categoricals_train, numericals_train, y_train, cat_cardinalities), 
                    n_trials=20
                )
                optimize_end = time.time()
                optimization_minutes = round((optimize_end - optimize_start) / 60, 1)

                logger.info(f"  Time taken for hyperparameter optimization: {optimization_minutes} minutes")

                best_params = study.best_params
                logger.info(f"  Best parameters: {best_params}")

                # Train final model with best params
                final_model = train_tabtransformer(
                    categoricals_train, numericals_train, y_train, best_params, cat_cardinalities
                )

                # Evaluate on train
                train_mse, train_r2 = evaluate_model(
                    final_model, categoricals_train, numericals_train, y_train
                )
                logger.info(f"  Final Train MSE: {train_mse:.4f}, R^2: {train_r2:.4f}")

                # Evaluate on test
                test_mse, test_r2 = evaluate_model(
                    final_model, categoricals_test, numericals_test, y_test
                )
                logger.info(f"  Test MSE: {test_mse:.4f}, R^2: {test_r2:.4f}")

                # Store iteration results
                results[dataset_name][iteration_key] = {
                    "best_params": best_params,
                    "train_mse": train_mse,
                    "train_r2": train_r2,
                    "test_mse": test_mse,
                    "test_r2": test_r2,
                }

                # Save results to JSON each time
                if not os.path.exists("results"):
                    os.makedirs("results")
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=4)

            # ----------------------------
            # AFTER all iterations for this dataset, 
            # average results across runs that were completed.
            # ----------------------------
            iteration_keys = [
                k for k in results[dataset_name].keys()
                if k.startswith("Iteration_")
            ]

            if iteration_keys:  # Only compute averages if we actually have data
                total_train_mse = 0.0
                total_train_r2 = 0.0
                total_test_mse = 0.0
                total_test_r2 = 0.0

                for k in iteration_keys:
                    run_data = results[dataset_name][k]
                    total_train_mse += run_data["train_mse"]
                    total_train_r2 += run_data["train_r2"]
                    total_test_mse += run_data["test_mse"]
                    total_test_r2 += run_data["test_r2"]

                num_runs = len(iteration_keys)
                avg_train_mse = total_train_mse / num_runs
                avg_train_r2 = total_train_r2 / num_runs
                avg_test_mse = total_test_mse / num_runs
                avg_test_r2 = total_test_r2 / num_runs

                # Store the averaged metrics
                results[dataset_name]["average_across_runs"] = {
                    "num_runs": num_runs,
                    "avg_train_mse": avg_train_mse,
                    "avg_train_r2": avg_train_r2,
                    "avg_test_mse": avg_test_mse,
                    "avg_test_r2": avg_test_r2
                }

                logger.info(f"Averaged results for {dataset_name}:")
                logger.info(f"  Average Train MSE: {avg_train_mse:.4f}, R^2: {avg_train_r2:.4f}")
                logger.info(f"  Average Test MSE: {avg_test_mse:.4f}, R^2: {avg_test_r2:.4f}")

                # Save again after we compute the averages
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=4)

        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {str(e)}")

    overall_end = time.time()
    total_hours = round((overall_end - overall_start) / 3600, 1)
    logger.info(f"Total experiment time: {total_hours} hours")