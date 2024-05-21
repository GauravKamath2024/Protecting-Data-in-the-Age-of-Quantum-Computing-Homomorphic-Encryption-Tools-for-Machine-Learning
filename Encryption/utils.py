import torch
import random

# Set initial random seeds for reproducibility
torch.random.manual_seed(2024)
random.seed(2024)

def clean_dataset(df, target):
    """
    Clean the dataset by filling missing values and balancing the classes.

    Parameters:
    df (pd.DataFrame): The input dataframe to be cleaned.
    target (str): The target column name in the dataframe.

    Returns:
    pd.DataFrame: The cleaned and balanced dataframe.
    """
    # Fill missing values in each column with the column mean
    df = df.apply(lambda col: col.fillna(col.mean()), axis=0)

    # Remove any remaining rows with missing values
    df = df.dropna()

    # Balance the dataset by sampling an equal number of instances for each class
    new_df = df.groupby(target)
    df = new_df.apply(lambda x: x.sample(new_df.size().min(), random_state=73).reset_index(drop=True))

    return df

def split_train_and_test(X, y, ratio=0.2):
    """
    Split the dataset into training and testing sets.

    Parameters:
    X (torch.Tensor): The input features.
    y (torch.Tensor): The target labels.
    ratio (float, optional): The ratio of the test set size to the total dataset size. Default is 0.2.

    Returns:
    tuple: A tuple containing:
        - X_train (torch.Tensor): Training features.
        - y_train (torch.Tensor): Training labels.
        - X_test (torch.Tensor): Testing features.
        - y_test (torch.Tensor): Testing labels.
    """
    # Create a list of indices and shuffle them
    idxs = [i for i in range(len(X))]
    random.shuffle(idxs)

    # Determine the delimiter index for splitting
    delim = int(len(X) * ratio)
    test_idxs, train_idxs = idxs[:delim], idxs[delim:]

    # Split the data into training and testing sets
    return X[train_idxs], y[train_idxs], X[test_idxs], y[test_idxs]

def standardize_data(df, target, ratio=0.2, seed=None):
    """
    Clean, standardize, and split the dataset into training and testing sets.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    target (str): The target column name in the dataframe.
    ratio (float, optional): The ratio of the test set size to the total dataset size. Default is 0.2.
    seed (int, optional): The random seed for reproducibility. Default is None.

    Returns:
    tuple: A tuple containing:
        - X_train (torch.Tensor): Standardized training features.
        - y_train (torch.Tensor): Standardized training labels.
        - X_test (torch.Tensor): Standardized testing features.
        - y_test (torch.Tensor): Standardized testing labels.
    """
    # Set the random seed if provided
    if seed is not None:
        torch.random.manual_seed(seed)
        random.seed(seed)

    # Clean the dataset
    df = clean_dataset(df, target)

    # Standardize the dataset (mean=0, std=1)
    df = (df - df.mean()) / df.std()

    # Convert dataframe to tensors
    X = torch.tensor(df.drop(target, axis=1).values).float()
    y = torch.tensor(df[target].values).float().unsqueeze(1)

    # Split the data into training and testing sets
    return split_train_and_test(X, y, ratio)

def randomize_data(m=1024, n=2):
    """
    Generate random, linearly separable data for binary classification.

    Parameters:
    m (int, optional): Number of training samples. Default is 1024.
    n (int, optional): Number of features. Default is 2.

    Returns:
    tuple: A tuple containing:
        - X_train (torch.Tensor): Random training features.
        - y_train (torch.Tensor): Training labels based on the separation line y=x.
        - X_test (torch.Tensor): Random testing features.
        - y_test (torch.Tensor): Testing labels based on the separation line y=x.
    """
    # Generate random training data
    X_train = torch.randn(m, n)
    X_test = torch.randn(m // 2, n)

    # Generate labels based on the line y = x
    y_train = (X_train[:, 0] >= X_train[:, 1]).float().unsqueeze(0).t()
    y_test = (X_test[:, 0] >= X_test[:, 1]).float().unsqueeze(0).t()

    return X_train, y_train, X_test, y_test
