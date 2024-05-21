import pandas as pd
import argparse

from utils import standardize_data
from EncryptData import encryptData
from train_model import train_encrypted_model
from EncryptedEvalutation import encrypted_evaluation

def encrypt_and_train_model(path, target, ratio, epochs=None, seed=None):
    """
    Function to encrypt data, train an encrypted model, and evaluate it.

    Parameters:
    path (str): Path to the dataset CSV file.
    target (str): Name of the target variable.
    ratio (float): Test split ratio.
    epochs (int, optional): Number of epochs to train the model.
    seed (int, optional): Seed value for reproducibility.

    Returns:
    float: Accuracy on the test set.
    float: Time taken for inference.
    """
    # Read the dataset
    df = pd.read_csv(path)

    # Standardize data and split into train and test sets
    X_train, y_train, X_test, y_test = standardize_data(df, target, ratio, seed)

    # Encrypt the data
    enc_X_train, enc_y_train, enc_time = encryptData(X_train, y_train)
    print(f"Encryption of training set took {round(enc_time, 3)} Seconds")
    enc_X_test, enc_y_test, enc_time = encryptData(X_test, y_test)
    print(f"Encryption of test set took {round(enc_time, 3)} Seconds")

    # Get the number of features
    n_features = X_train.shape[1]

    # Train the encrypted model
    eelr = train_encrypted_model(enc_X_train, enc_y_train, n_features, len(y_train), epochs, seed)

    # Evaluate the model
    accuracy, eval_time = encrypted_evaluation(eelr, enc_X_test, enc_y_test, len(y_test))
    print(f"Accuracy on test set is {accuracy} Inference was complete in {round(eval_time, 3)} Seconds")

    return accuracy, eval_time

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Taking the path and target variable')

    # Adding the arguments
    parser.add_argument('--path', type=str,required=True, help='The path to the dataset')
    parser.add_argument('--target', type=str,required=True, help='The target variable')
    parser.add_argument('--ratio', type=float, required=True, help='test split')
    parser.add_argument('--epochs', type=int,required=False, help='Number of epochs to train model')
    parser.add_argument('--seed', type=int,required=False, help='Set the seed value')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with provided arguments
    encrypt_and_train_model(args.path, args.target, args.ratio, args.epochs, args.seed)

