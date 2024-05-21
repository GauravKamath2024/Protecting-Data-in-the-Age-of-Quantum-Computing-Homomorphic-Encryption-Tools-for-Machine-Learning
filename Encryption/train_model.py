import tenseal as ts
from time import time
import torch


from EncryptedLogisticRegression import EncryptedLR
from LogisticRegression import LogisticRegression
from EncryptedEvalutation import encrypted_evaluation


# Set polynomial modulus degree and coefficient modulus bit sizes for CKKS scheme
poly_mod_degree = 8192
coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]

# Create TenSEAL context for training
ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_training.global_scale = 2 ** 21
ctx_training.generate_galois_keys()

times = []

def train_encrypted_model(enc_X_train, enc_y_train, n_features, length, epochs=100, seed=2024):
    """
    Train an encrypted logistic regression model using encrypted data.

    Parameters:
    enc_X_train (list of ts.CKKSTensor): List of encrypted feature tensors for training.
    enc_y_train (list of ts.CKKSTensor): List of encrypted labels for training.
    n_features (int): Number of features in the training data.
    length (int): Length of the training data.
    epochs (int, optional): Number of epochs to train the model. Default is 100.
    seed (int, optional): Seed for random number generation to ensure reproducibility. Default is 2024.

    Returns:
    EncryptedLR: Trained encrypted logistic regression model.
    """
    torch.manual_seed(seed)
    print(f"Model will be trained for {epochs} Epochs")
    eelr = EncryptedLR(LogisticRegression(n_features))
    for epoch in range(epochs):
        eelr.encrypt(ctx_training)
        t_start = time()
        et = time()
        for enc_X, enc_y in zip(enc_X_train, enc_y_train):
            enc_out = eelr.forward(enc_X)
            eelr.backward(enc_X, enc_out, enc_y)
        eelr.update_parameters()
        t_end = time()
        times.append(t_end - t_start)
        
        eelr.decrypt()
        train_accoracy, eval_time = encrypted_evaluation(eelr, enc_X_train, enc_y_train, length)
        print(f"Epoch [{epoch + 1}/{epochs}] Train Accuracy {train_accoracy} Completed in {round(time() - et, 3)} Seconds")
        et = time()


    return eelr