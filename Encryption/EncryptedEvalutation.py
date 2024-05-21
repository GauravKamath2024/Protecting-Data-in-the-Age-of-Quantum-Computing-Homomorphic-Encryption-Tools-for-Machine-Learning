from time import time
import torch

def encrypted_evaluation(model, enc_X, enc_y, length):
    """
    Evaluate the accuracy of an encrypted logistic regression model.

    Parameters:
    model (EncryptedLR): The encrypted logistic regression model to be evaluated.
    enc_X (list of ts.CKKSTensor): List of encrypted feature tensors for evaluation.
    enc_y (list of ts.CKKSTensor): List of encrypted labels for evaluation.
    length (int): Length of the evaluation data.

    Returns:
    tuple: A tuple containing:
        - accuracy (float): The accuracy of the model on the evaluation data.
        - evaluation_time (float): The time taken to complete the evaluation.
    """
    t_start = time()

    correct = 0
    for enc_X, y in zip(enc_X, enc_y):
        # encrypted evaluation
        enc_out = model(enc_X)
        # plain comparaison
        out = enc_out.decrypt()
        out = torch.tensor(out)
        out = torch.sigmoid(out)
        if torch.abs(out - torch.tensor(y.decrypt())) < 0.5:
            correct += 1

    t_end = time() - t_start
    return correct / length, t_end