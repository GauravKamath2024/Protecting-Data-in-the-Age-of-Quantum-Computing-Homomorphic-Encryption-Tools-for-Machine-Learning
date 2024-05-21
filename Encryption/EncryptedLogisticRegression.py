import tenseal as ts
import torch

class EncryptedLR:
    """
    Encrypted Logistic Regression model.
    """

    def __init__(self, torch_lr):
        """
        Initialize the EncryptedLR model.

        Parameters:
        torch_lr (torch.nn.Module): PyTorch logistic regression model.
        """
        # Extract weights and biases from the PyTorch logistic regression model
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()

        # Initialize gradient accumulators and iteration count
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    def forward(self, enc_x):
        """
        Forward pass through the model.

        Parameters:
        enc_x (ts.CKKSTensor): Encrypted input features.

        Returns:
        ts.CKKSTensor: Encrypted output prediction.
        """
        # Compute weighted sum and add bias
        enc_out = enc_x.dot(self.weight) + self.bias
        # Apply sigmoid activation function
        enc_out = EncryptedLR.sigmoid(enc_out)
        return enc_out
    
    def backward(self, enc_x, enc_out, enc_y):
        """
        Backward pass to compute gradients.

        Parameters:
        enc_x (ts.CKKSTensor): Encrypted input features.
        enc_out (ts.CKKSTensor): Encrypted output prediction.
        enc_y (ts.CKKSTensor): Encrypted true labels.
        """
        # Compute difference between prediction and true labels
        out_minus_y = (enc_out - enc_y)
        # Accumulate gradients
        self._delta_w += enc_x * out_minus_y
        self._delta_b += out_minus_y
        # Increment count
        self._count += 1

    def update_parameters(self):
        """
        Update model parameters using accumulated gradients.
        """
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")
        # Update weights with gradient descent and a small regularization term
        self.weight -= self._delta_w * (1 / self._count) + self.weight * 0.05
        self.bias -= self._delta_b * (1 / self._count)
        # Reset gradient accumulators and iteration count
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    @staticmethod
    def sigmoid(enc_x):
        """
        Sigmoid activation function approximation using a polynomial of degree 3.

        Parameters:
        enc_x (ts.CKKSTensor): Encrypted input.

        Returns:
        ts.CKKSTensor: Encrypted output after applying sigmoid activation.
        """
        return enc_x.polyval([0.5, 0.197, 0, -0.004])

    def plain_accuracy(self, x_test, y_test):
        """
        Evaluate accuracy of the model on plain (non-encrypted) data.

        Parameters:
        x_test (torch.Tensor): Input features of the test data.
        y_test (torch.Tensor): True labels of the test data.

        Returns:
        float: Accuracy of the model on the test data.
        """
        # Convert weights and biases to torch tensors
        w = torch.tensor(self.weight)
        b = torch.tensor(self.bias)
        # Compute output using plain (non-encrypted) operations
        out = torch.sigmoid(x_test.matmul(w) + b).reshape(-1, 1)
        # Compute correct predictions
        correct = torch.abs(y_test - out) < 0.5
        # Compute accuracy
        return correct.float().mean()
    
    def encrypt(self, context):
        """
        Encrypt model parameters using TenSEAL context.

        Parameters:
        context (ts.Context): TenSEAL context for encryption.
        """
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)

    def decrypt(self):
        """
        Decrypt model parameters.
        """
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()

    def __call__(self, *args, **kwargs):
        """
        Callable method to forward pass through the model.

        Returns:
        ts.CKKSTensor: Encrypted output prediction.
        """
        return self.forward(*args, **kwargs)
