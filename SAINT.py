import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
# Data Loading and Preprocessing
#############################
def load_and_preprocess_concrete():
    """
    Load and preprocess the concrete dataset.
    Returns:
    - X_train (np.ndarray): Training feature matrix.
    - X_test (np.ndarray): Testing feature matrix.
    - y_train (np.ndarray): Training target vector.
    - y_test (np.ndarray): Testing target vector.
    - column_names (list): List of feature names.
    """
    logger.info("Loading the concrete dataset...")
    data = pd.read_csv("datasets/concrete_data.csv")
    target = "concrete_compressive_strength"
    
    if target not in data.columns:
        logger.error(f"Target column '{target}' not found in dataset!")
        raise ValueError(f"Target column '{target}' not found in dataset!")
    
    logger.info("Performing preprocessing...")
    data = data[data[target].notna()]

    X = data.drop(target, axis=1)
    y = data[target].values
    column_names = X.columns.tolist()

    # Standardize the features
    logger.info("Standardizing the features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    logger.info("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    logger.info("Data preprocessing completed successfully.")
    return X_train, X_test, y_train, y_test, column_names

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
        logger.info("Initializing the SAINT model...")
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
        logger.info("SAINT model initialized successfully.")

    def forward(self, x):
        # Input: x -> (batch_size, input_dim)
        batch_size, input_dim = x.size()

        # Embed features: Multiply input values with learnable embeddings
        embedded_features = x.unsqueeze(-1) * self.feature_embeddings.unsqueeze(0)
        # Shape: (batch_size, input_dim, embed_dim)

        # Column-wise attention (treat features as tokens)
        col_encoded = self.transformer_encoder_cols(embedded_features)  # (batch_size, input_dim, embed_dim)

        # Aggregate features using mean pooling
        pooled = col_encoded.mean(dim=1)  # Shape: (batch_size, embed_dim)

        # Predict regression target
        out = self.fc(pooled)  # Shape: (batch_size, 1)
        return out

#############################
# Training Procedure
#############################
def train_saint(X_train, y_train, X_test, y_test, input_dim):
    logger.info("Starting SAINT model training...")

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = SAINT(
        input_dim=input_dim,
        embed_dim=16,  # Reduced embedding dimension for 8 features
        depth=3,       # Simplified transformer depth
        num_heads=2,
        dropout=0.1
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 250
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Evaluation
    logger.info("Evaluating model performance on the test set...")
    model.eval()
    with torch.no_grad():
        pred_test = model(X_test_t).squeeze(-1).numpy()
        test_mse = mean_squared_error(y_test, pred_test)
        test_r2 = r2_score(y_test, pred_test)
        logger.info(f"Test MSE: {test_mse:.4f}, R^2: {test_r2:.4f}")

    logger.info("Training and evaluation completed successfully.")
    return model

#############################
# Main Execution
#############################
if __name__ == "__main__":
    logger.info("Program started...")
    try:
        X_train, X_test, y_train, y_test, column_names = load_and_preprocess_concrete()
        model = train_saint(X_train, y_train, X_test, y_test, input_dim=len(column_names))
        logger.info("SAINT model training pipeline completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")