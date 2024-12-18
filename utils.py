import pandas as pd  # For data manipulation and loading
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting datasets

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
    # Load training and testing data
    train_data = pd.read_csv(
        "datasets/bank32NH/bank32nh.data",
        delim_whitespace=True, header=None
    )
    test_data = pd.read_csv(
        "datasets/bank32NH/bank32nh.test",
        delim_whitespace=True, header=None
    )

    # Load domain file for column names
    domain_data = pd.read_csv(
        "datasets/bank32NH/bank32nh.domain",
        delim_whitespace=True, header=None
    )
    column_names = domain_data.iloc[:, 0].apply(lambda x: x.split()[0]).tolist()

    # Apply column names to the train and test dataframes
    train_data.columns = column_names
    test_data.columns = column_names

    # Set the target column (assuming 'rej' for this dataset)
    target = "rej"

    # Data Preprocessing
    train_data = train_data.apply(pd.to_numeric, errors='ignore')
    test_data = test_data.apply(pd.to_numeric, errors='ignore')

    # One-hot encoding for categorical variables
    train_data = pd.get_dummies(train_data, drop_first=True)
    test_data = pd.get_dummies(test_data, drop_first=True)

    # Ensure that both train and test have the same columns after encoding
    train_cols = set(train_data.columns)
    test_cols = set(test_data.columns)
    common_cols = list(train_cols & test_cols)

    # Align the columns
    train_data = train_data[common_cols]
    test_data = test_data[common_cols]

    # Separate features and target
    X_train, y_train = train_data.drop(target, axis=1).values, train_data[target].values
    X_test, y_test = test_data.drop(target, axis=1).values, test_data[target].values

    return X_train, X_test, y_train, y_test, common_cols

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
    # Load dataset
    data = pd.read_csv("datasets/concrete_data.csv")
    target = "concrete_compressive_strength"
    data = data[data[target].notna()]

    X = data.drop(target, axis=1).values
    y = data[target].values
    column_names = data.drop(target, axis=1).columns.tolist()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33
    )

    return X_train, X_test, y_train, y_test, column_names

def load_and_preprocess_delta_elevators():
    """
    Load and preprocess the delta_elevators dataset.

    Returns:
    - X_train (np.ndarray): Training feature matrix.
    - X_test (np.ndarray): Testing feature matrix.
    - y_train (np.ndarray): Training target vector.
    - y_test (np.ndarray): Testing target vector.
    - column_names (list): List of feature names.
    """
    # Load domain file for column names
    domain_data = pd.read_csv(
        "datasets/Elevators/delta_elevators.domain",
        delim_whitespace=True, header=None
    )
    column_names = domain_data.iloc[:, 0].apply(lambda x: x.split()[0]).tolist()

    # Load dataset
    data = pd.read_csv("datasets/Elevators/delta_elevators.data",
                       delim_whitespace=True, header=None)
    data.columns = column_names

    # Set the target column (assuming 'Se' for this dataset)
    target = "Se"

    # Data Preprocessing
    data = data.apply(pd.to_numeric, errors='ignore')

    # One-hot encoding for categorical variables if necessary
    # If the dataset contains categorical variables, uncomment the following lines
    # data = pd.get_dummies(data, drop_first=True)

    # Separate features and target
    X = data.drop(target, axis=1).values
    y = data[target].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33
    )

    return X_train, X_test, y_train, y_test, column_names

def load_and_preprocess_house_16():
    """
    Load and preprocess the house_16 dataset.

    Returns:
    - X_train (np.ndarray): Training feature matrix.
    - X_test (np.ndarray): Testing feature matrix.
    - y_train (np.ndarray): Training target vector.
    - y_test (np.ndarray): Testing target vector.
    - column_names (list): List of feature names.
    """
    # Load dataset
    data = pd.read_csv("datasets/house_16H.csv")
    target = "price"

    # Data Preprocessing
    data = data.apply(pd.to_numeric, errors='ignore')
    data = pd.get_dummies(data, drop_first=True)
    data = data[data[target].notna()]

    # Separate features and target
    X = data.drop(target, axis=1).values
    y = data[target].values
    column_names = data.drop(target, axis=1).columns.tolist()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33
    )

    return X_train, X_test, y_train, y_test, column_names

def load_and_preprocess_housing():
    """
    Load and preprocess the housing dataset.

    Returns:
    - X_train (np.ndarray): Training feature matrix.
    - X_test (np.ndarray): Testing feature matrix.
    - y_train (np.ndarray): Training target vector.
    - y_test (np.ndarray): Testing target vector.
    - column_names (list): List of feature names.
    """
    # Load dataset
    data = pd.read_csv("datasets/housing.csv")
    target = "median_house_value"

    # Data Preprocessing
    data['total_bedrooms'].fillna(data['total_bedrooms'].median(), inplace=True)
    data = pd.get_dummies(data, drop_first=True)
    data = data[data[target].notna()]

    # Separate features and target
    X = data.drop(target, axis=1).values
    y = data[target].values
    column_names = data.drop(target, axis=1).columns.tolist()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33
    )

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
    # Load dataset
    data = pd.read_csv("datasets/insurance.csv")
    target = "charges"

    # Data Preprocessing
    data = data.apply(pd.to_numeric, errors='ignore')
    data = pd.get_dummies(data, drop_first=True)
    data = data[data[target].notna()]

    # Separate features and target
    X = data.drop(target, axis=1).values
    y = data[target].values
    column_names = data.drop(target, axis=1).columns.tolist()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33
    )

    return X_train, X_test, y_train, y_test, column_names

def load_and_preprocess_movies():
    """
    Load and preprocess the movies dataset.

    Returns:
    - data (pd.DataFrame): Preprocessed dataset.
    - target (str): Target column name.
    """
    # Load dataset
    data = pd.read_csv("datasets/movies.csv")
    target = "gross"

    # Data Cleaning
    data.drop(['movie_title', 'color', 'director_name', 'actor_1_name',
               'actor_2_name', 'actor_3_name', 'language', 'country', 'content_rating', 'aspect_ratio'], axis=1,
              inplace=True)

    # Remove rows with missing target
    data = data[data[target].notna()]

    # Separate features and target
    X = data.drop(target, axis=1).values
    y = data[target].values
    column_names = data.drop(target, axis=1).columns.tolist()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33
    )

    return X_train, X_test, y_train, y_test, column_names