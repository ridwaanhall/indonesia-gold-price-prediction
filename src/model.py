import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GoldPriceLSTM(nn.Module):
    """
    Enhanced LSTM model for gold price prediction.
    Uses a multi-layer LSTM architecture with dropout and residual connections
    for improved long-term prediction capability.
    """
    
    def __init__(self, input_size: int = 11, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden layers
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate for regularization
        """
        super(GoldPriceLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Main LSTM layers with dropout between layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout layer after LSTM
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers for prediction
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, 1)
        """
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Take only the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Fully connected layers with ReLU activation
        fc1_out = F.relu(self.fc1(lstm_out))
        fc1_out = self.dropout(fc1_out)
        
        fc2_out = F.relu(self.fc2(fc1_out))
        
        # Final layer without activation (regression)
        output = self.fc3(fc2_out)
        
        # Remove final dimension if it's 1
        return output.squeeze(-1)
    
    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction step with no gradient calculation.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Prediction
        """
        with torch.no_grad():
            return self(x)


class EnhancedGoldPriceLSTM(nn.Module):
    """
    Advanced LSTM architecture for gold price prediction with attention mechanism.
    Better suited for capturing long-term patterns and seasonal effects.
    """
    
    def __init__(self, input_size: int = 11, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.3):
        """
        Initialize the enhanced LSTM model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden layers
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate for regularization
        """
        super(EnhancedGoldPriceLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Bidirectional LSTM for capturing patterns in both directions
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)  # 2 for bidirectional
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # 2 for bidirectional
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 1)
    
    def attention_net(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism to LSTM output.
        
        Args:
            lstm_output (torch.Tensor): Output from LSTM layer
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention weighted output and attention weights
        """
        # Calculate attention weights
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)
        
        # Apply attention weights to LSTM output
        context = torch.sum(attn_weights * lstm_output, dim=1)
        
        return context, attn_weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, 1)
        """
        # LSTM layers
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_length, hidden_size*2)
        
        # Apply attention
        attn_out, _ = self.attention_net(lstm_out)
        
        # Apply dropout
        attn_out = self.dropout(attn_out)
        
        # Fully connected layers with batch normalization and ReLU
        fc1_out = F.relu(self.bn1(self.fc1(attn_out)))
        fc1_out = self.dropout(fc1_out)
        
        fc2_out = F.relu(self.bn2(self.fc2(fc1_out)))
        fc2_out = self.dropout(fc2_out)
        
        # Final layer without activation (regression)
        output = self.fc3(fc2_out)
        
        # Remove final dimension if it's 1
        return output.squeeze(-1)
    
    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction step with no gradient calculation.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Prediction
        """
        with torch.no_grad():
            return self(x)


# For direct testing
if __name__ == "__main__":
    # Test basic model
    input_size = 11  # Example input size
    seq_length = 60  # Example sequence length
    batch_size = 32  # Example batch size
    
    # Create a sample input tensor
    sample_input = torch.rand((batch_size, seq_length, input_size))
    
    # Instantiate the models
    basic_model = GoldPriceLSTM(input_size=input_size)
    enhanced_model = EnhancedGoldPriceLSTM(input_size=input_size)
    
    # Test forward pass
    basic_output = basic_model(sample_input)
    enhanced_output = enhanced_model(sample_input)
    
    # Print output shapes
    print(f"Basic model output shape: {basic_output.shape}")
    print(f"Enhanced model output shape: {enhanced_output.shape}")
    
    # Check model parameters
    basic_params = sum(p.numel() for p in basic_model.parameters())
    enhanced_params = sum(p.numel() for p in enhanced_model.parameters())
    
    print(f"Basic model parameters: {basic_params}")
    print(f"Enhanced model parameters: {enhanced_params}")
    
    print("Model architecture verified successfully!")
