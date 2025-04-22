import torch
import torch.nn as nn

class GoldPriceLSTM(nn.Module):
    """
    LSTM-based deep learning model for gold price forecasting.
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        """
        Initializes the LSTM model.

        Args:
            input_size (int): Number of input features (from DataLoader).
            hidden_size (int): Number of hidden units in LSTM layers.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout rate to prevent overfitting.
        """
        super(GoldPriceLSTM, self).__init__()

        # LSTM Layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully Connected Output Layer
        self.fc = nn.Linear(hidden_size, 1)  # Predicting a single future price value

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input sequence tensor.

        Returns:
            torch.Tensor: Predicted gold price.
        """
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Taking only the last time step output
        return output

model = GoldPriceLSTM(input_size=4)  # Based on the 4 selected features
sample_input = torch.randn(32, 30, 4)  # (batch_size=32, sequence_length=30, num_features=4)
sample_output = model(sample_input)
print("✅ Model output shape:", sample_output.shape)  # Expected output: torch.Size([32, 1])
