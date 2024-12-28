# utils.py

import os
import pandas as pd  # For data manipulation and loading
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting datasets

#############################
# Preprocessing the Movies dataset
#############################
def preprocess_movies_data(df):
    # Drop high-cardinality or irrelevant columns
    drop_columns = [
        "movie_title", "color", "director_name", "actor_1_name",
        "actor_2_name", "actor_3_name", "language", "country",
        "content_rating", "aspect_ratio"
    ]
    df = df.drop(columns=drop_columns, errors="ignore")

    for column in df.columns:
        if df[column].dtype in [float, int]:
            # Instead of using inplace=True, assign the result back
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna("missing")

    # Check for remaining NaNs (should be none after this step)
    assert not df.isnull().any().any(), "NaN values still exist in the dataset after preprocessing!"

    # Separate categorical and numeric columns
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Apply Label Encoding to categorical columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders


#############################
# Dataset Loading & Preprocessing Functions
#############################

def load_and_preprocess_movies():
    """
    Load and preprocess the movies dataset using the provided preprocess_movies_data function.
    """
    if not os.path.exists("datasets/movies.csv"):
        raise FileNotFoundError("Movies dataset file is missing.")

    # Load raw data
    data = pd.read_csv("datasets/movies.csv")
    target = "gross"

    # Apply the provided preprocessing function
    movies_cleaned, movies_label_encoders = preprocess_movies_data(data)

    # Ensure target column is available and drop rows without target
    if target not in movies_cleaned.columns:
        raise ValueError(f"Target column '{target}' not found in Movies dataset.")

    movies_cleaned = movies_cleaned[movies_cleaned[target].notna()]

    # Extract feature matrix and target vector
    X = movies_cleaned.drop(target, axis=1).values
    y = movies_cleaned[target].values
    column_names = movies_cleaned.drop(target, axis=1).columns.tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Final NaN checks
    if (np.isnan(X_train).any() or np.isnan(X_test).any() or
        np.isnan(y_train).any() or np.isnan(y_test).any()):
        raise ValueError("NaN values found in Movies dataset after preprocessing and splitting.")

    return X_train, X_test, y_train, y_test, column_names


def load_and_preprocess_bank32NH():
    """
    Load and preprocess the bank32NH dataset.

    Returns:
    - X_train (np.ndarray): Training feature matrix.
    - X_test (np.ndarray): Testing feature matrix.
    - y_train (np.ndarray): Training target vector.
    - y_test (np.ndarray): Testing target vector.
    - column_names (list): List of feature names.
    """
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Check if files exist
    if not (os.path.exists("datasets/bank32NH/bank32nh.data") and 
            os.path.exists("datasets/bank32NH/bank32nh.test") and 
            os.path.exists("datasets/bank32NH/bank32nh.domain")):
        raise FileNotFoundError("Bank32NH dataset files are missing.")

    # Read the data files
    train_data = pd.read_csv(
        "datasets/bank32NH/bank32nh.data",
        delim_whitespace=True, 
        header=None
    )
    test_data = pd.read_csv(
        "datasets/bank32NH/bank32nh.test",
        delim_whitespace=True, 
        header=None
    )
    # Read the domain file
    domain_data = pd.read_csv(
        "datasets/bank32NH/bank32nh.domain",
        delim_whitespace=True, 
        header=None
    )

    # Extract column names from the domain file
    column_names = domain_data.iloc[:, 0].apply(lambda x: x.split()[0]).tolist()

    # Confirm that the domain file matches the training data
    if len(column_names) != train_data.shape[1]:
        raise ValueError(
            f"Column count mismatch. domain file has {len(column_names)} columns, "
            f"train_data has {train_data.shape[1]} columns."
        )

    # Assign column names to the DataFrames
    train_data.columns = column_names
    test_data.columns = column_names

    # Confirm the target column (last column "rej" according to the domain)
    target = "rej"
    if target not in train_data.columns:
        raise ValueError(f"Target column '{target}' not found in Bank32NH dataset columns.")

    # Convert all columns to numeric (coerce errors => NaN for non-numeric)
    train_data = train_data.apply(pd.to_numeric, errors='coerce')
    test_data = test_data.apply(pd.to_numeric, errors='coerce')

    # Drop any rows where the target is NaN
    train_data = train_data[train_data[target].notna()]
    test_data = test_data[test_data[target].notna()]

    # Debug prints after numeric conversion
    print("After numeric conversion:")
    print("train_data shape:", train_data.shape)
    print("test_data shape:", test_data.shape)

    # Since domain_data indicates all features are ": continuous",
    # we skip one-hot encoding or get_dummies().

    # If you truly have categorical columns, you could combine train/test
    # and apply pd.get_dummies() in a unified way. For now, we assume all continuous.

    # Separate features (X) and target (y)
    X_train = train_data.drop(columns=[target]).values
    y_train = train_data[target].values

    X_test = test_data.drop(columns=[target]).values
    y_test = test_data[target].values

    # Update the list of column names (excluding the target)
    column_names = [col for col in train_data.columns if col != target]

    # Final check for NaNs
    if (np.isnan(X_train).any() or np.isnan(X_test).any() or
        np.isnan(y_train).any() or np.isnan(y_test).any()):
        raise ValueError("NaN values found in Bank32NH processed data after all preprocessing steps.")

    # Print shapes again for clarity
    print("Final X_train shape:", X_train.shape)
    print("Final X_test shape:", X_test.shape)
    print("Final y_train shape:", y_train.shape)
    print("Final y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test, column_names


def load_and_preprocess_concrete():
    """
    Load and preprocess the concrete dataset.

    Returns:
    - X_train, X_test, y_train, y_test, column_names
    """
    if not os.path.exists("datasets/concrete_data.csv"):
        raise FileNotFoundError("Concrete dataset file is missing.")

    data = pd.read_csv("datasets/concrete_data.csv")
    target = "concrete_compressive_strength"

    data = data[data[target].notna()]

    X = data.drop(target, axis=1).values
    y = data[target].values
    column_names = data.drop(target, axis=1).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Check for NaNs
    if (np.isnan(X_train).any() or np.isnan(X_test).any() or
        np.isnan(y_train).any() or np.isnan(y_test).any()):
        raise ValueError("NaN values found in Concrete dataset after splitting.")

    return X_train, X_test, y_train, y_test, column_names


def load_and_preprocess_delta_elevators():
    """
    Load and preprocess the delta_elevators dataset.

    Returns:
    - X_train, X_test, y_train, y_test, column_names
    """
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    if not (os.path.exists("datasets/Elevators/delta_elevators.domain") and
            os.path.exists("datasets/Elevators/delta_elevators.data")):
        raise FileNotFoundError("Elevators dataset files are missing.")

    # Read the domain file
    domain_data = pd.read_csv(
        "datasets/Elevators/delta_elevators.domain",
        delim_whitespace=True, 
        header=None
    )
    # print("domain_data shape:", domain_data.shape)
    # print(domain_data)

    # Create the column names list
    column_names = domain_data.iloc[:, 0].apply(lambda x: x.split()[0]).tolist()

    # Read the data file
    data = pd.read_csv(
        "datasets/Elevators/delta_elevators.data",
        delim_whitespace=True, 
        header=None
    )
    # print("data shape (before columns assigned):", data.shape)
    # print(data.head())

    # If there's a mismatch, handle or raise an error
    if len(column_names) != data.shape[1]:
        raise ValueError(
            f"Column count mismatch. domain file has {len(column_names)} columns, "
            f"data file has {data.shape[1]} columns."
        )

    # Assign the columns
    data.columns = column_names

    # Set the target column
    target = "Se"
    if target not in data.columns:
        raise ValueError(
            f"Target column '{target}' not found in Elevators dataset columns: {data.columns.tolist()}"
        )

    # Convert all columns to numeric (NaN if conversion fails)
    data = data.apply(pd.to_numeric, errors='coerce')

    # Drop rows where the target is NaN
    data = data[data[target].notna()]

    # Separate features and target
    X = data.drop(target, axis=1).values
    y = data[target].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Final NaN check
    if (np.isnan(X_train).any() or np.isnan(X_test).any() or
        np.isnan(y_train).any() or np.isnan(y_test).any()):
        raise ValueError("NaN values found in Delta Elevators dataset after splitting.")

    # The final column list (excluding target)
    column_names = [col for col in data.columns if col != target]

    return X_train, X_test, y_train, y_test, column_names

def load_and_preprocess_house_16():
    """
    Load and preprocess the house_16 dataset.
    """
    if not os.path.exists("datasets/house_16H.csv"):
        raise FileNotFoundError("House_16H dataset file is missing.")

    data = pd.read_csv("datasets/house_16H.csv")
    target = "price"

    data = data.apply(pd.to_numeric, errors='coerce')
    data = pd.get_dummies(data, drop_first=True)

    data = data[data[target].notna()]

    X = data.drop(target, axis=1).values
    y = data[target].values
    column_names = data.drop(target, axis=1).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    if (np.isnan(X_train).any() or np.isnan(X_test).any() or
        np.isnan(y_train).any() or np.isnan(y_test).any()):
        raise ValueError("NaN values found in House_16 dataset after splitting.")

    return X_train, X_test, y_train, y_test, column_names


def load_and_preprocess_housing():
    """
    Load and preprocess the housing dataset.
    - Fills missing values in 'total_bedrooms' with the median.
    - One-hot encodes 'ocean_proximity'.
    - Ensures all columns are numeric.
    """
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    if not os.path.exists("datasets/housing.csv"):
        raise FileNotFoundError("Housing dataset file is missing.")

    # 1. Load data
    data = pd.read_csv("datasets/housing.csv")
    target = "median_house_value"

    # 2. Fill missing 'total_bedrooms'
    data["total_bedrooms"] = data["total_bedrooms"].fillna(data["total_bedrooms"].median())

    # 3. One-hot encode 'ocean_proximity'
    data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)

    # 4. Ensure all columns are numeric (convert to float64)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype(float)

    # 5. Drop rows where the target is NaN
    data = data[data[target].notna()]

    # Debugging: Check dtypes and sample data
    # print("Data types after conversion:\n", data.dtypes)
    # print("Sample data after processing:\n", data.head())

    # 6. Separate features (X) and target (y)
    X = data.drop(columns=[target]).values
    y = data[target].values

    # Debugging: Check X and y
    # print("Feature matrix (X) dtype:", X.dtype)
    # print("Target vector (y) dtype:", y.dtype)

    # 7. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # 8. Verify no NaN values remain
    if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(y_train).any() or np.isnan(y_test).any():
        raise ValueError("NaN values found in the Housing dataset after splitting.")

    # 9. Column names for features
    column_names = [col for col in data.columns if col != target]

    return X_train, X_test, y_train, y_test, column_names


def load_and_preprocess_insurance():
    """
    Load and preprocess the insurance dataset.

    Returns:
    - X_train (np.ndarray): Training feature matrix.
    - X_test (np.ndarray): Testing feature matrix.
    - y_train (np.ndarray): Training target vector.
    - y_test (np.ndarray): Testing target vector.
    - column_names (list): List of feature names.
    """
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    if not os.path.exists("datasets/insurance.csv"):
        raise FileNotFoundError("Insurance dataset file is missing.")

    # 1. Load data
    data = pd.read_csv("datasets/insurance.csv")
    target = "charges"

    # 2. One-hot encode categorical columns
    categorical_columns = ["sex", "smoker", "region"]
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # 3. Ensure all boolean columns are converted to int
    bool_columns = data.select_dtypes(include=["bool"]).columns
    for col in bool_columns:
        data[col] = data[col].astype(int)

    # Debugging: Check dtypes after conversion
    # print("Dtypes after one-hot encoding and conversion:\n", data.dtypes)

    # 4. Verify all columns are numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Debugging: Check for non-numeric columns
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        raise ValueError(f"Non-numeric columns found after numeric conversion: {list(non_numeric_cols)}")

    # 5. Drop rows where the target is NaN
    data = data[data[target].notna()]

    # 6. Separate features (X) and target (y)
    X = data.drop(columns=[target]).values
    y = data[target].values
    column_names = [col for col in data.columns if col != target]

    # Debugging: Check data types and samples of X and y
    # print("Feature matrix (X) dtype:", X.dtype)
    # print("Target vector (y) dtype:", y.dtype)

    # 7. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # 8. Final NaN check in train-test split
    if (
        np.isnan(X_train).any() or np.isnan(X_test).any() or
        np.isnan(y_train).any() or np.isnan(y_test).any()
    ):
        raise ValueError("NaN values found in Insurance dataset after splitting.")

    return X_train, X_test, y_train, y_test, column_names