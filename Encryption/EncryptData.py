import tenseal as ts
from time import time

poly_mod_degree = 8192
coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]

ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_training.global_scale = 2 ** 21
ctx_training.generate_galois_keys()

def encryptData(X, y):
    """
    Encrypt the training data using the CKKS scheme.

    Parameters:
    X (list of torch.Tensor): List of feature tensors for training.
    y (list of torch.Tensor): List of label tensors for training.

    Returns:
    tuple: A tuple containing:
        - enc_X (list of ts.CKKSTensor): List of encrypted feature tensors.
        - enc_y (list of ts.CKKSTensor): List of encrypted label tensors.
        - encryption_time (float): The time taken to encrypt the data.
    """
    t_start = time()
    enc_X = [ts.ckks_vector(ctx_training, x.tolist()) for x in X]
    enc_y = [ts.ckks_vector(ctx_training, y.tolist()) for y in y]
    encryption_time = time() - t_start


    return enc_X, enc_y, encryption_time