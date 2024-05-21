import torch

class LogisticRegression(torch.nn.Module):
    """
    Logistic Regression model class.
    """

    def __init__(self, n_features, *args, **kwargs):
        """
        Initialize the Logistic Regression model.

        Parameters:
        n_features (int): Number of input features.
        """
        super(LogisticRegression, self).__init__(*args, **kwargs)
        # Define a linear layer with input size n_features and output size 1
        self.lr = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying sigmoid activation.
        """
        # Compute output of the linear layer and apply sigmoid activation
        output = torch.sigmoid(self.lr(x))
        return output
