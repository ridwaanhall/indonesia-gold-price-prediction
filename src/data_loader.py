import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class GoldPriceDataset(Dataset):
    """
    A PyTorch Dataset for Gold Price Prediction.

    Attributes:
        file_path (str): Path to the processed CSV file.
        sequence_length (int): Number of past days used for predictions.
        features (list): Selected feature columns for training.
    """

    def __init__(self, file_path: str, sequence_length: int = 30):
        """
        Initializes the GoldPriceDataset.

        Args:
            file_path (str): Path to the processed CSV file.
            sequence_length (int): Number of days to consider as input for forecasting.
        """
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.df = pd.read_csv(file_path, parse_dates=["date"])

        # Normalize numerical features for training stability
        self.features = ["sell", "sell_ma7", "sell_ma30", "sell_volatility_30"]
        self.normalize_data()

    def normalize_data(self):
        """
        Normalize dataset using Min-Max scaling.
        """
        for feature in self.features:
            self.df[feature] = (self.df[feature] - self.df[feature].min()) / (self.df[feature].max() - self.df[feature].min())

    def __len__(self):
        """
        Returns the total number of data points available.
        """
        return len(self.df) - self.sequence_length

    def __getitem__(self, idx):
        """
        Retrieves a data sample for model training.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: (input sequence tensor, target value tensor).
        """
        x = self.df[self.features].iloc[idx:idx + self.sequence_length].values
        y = self.df["sell"].iloc[idx + self.sequence_length]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Function to load data using PyTorch DataLoader
def get_data_loader(file_path: str, batch_size: int = 32, sequence_length: int = 30, shuffle: bool = True):
    """
    Creates a PyTorch DataLoader for batch processing.

    Args:
        file_path (str): Path to the processed CSV file.
        batch_size (int): Number of samples per batch.
        sequence_length (int): Length of time-series input.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    dataset = GoldPriceDataset(file_path, sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

file_path = "data/processed/gold_prices_cleaned.csv"
batch_size=32
sequence_length=30
shuffle=True

data_loader = get_data_loader(file_path, batch_size, sequence_length, shuffle)
for batch_idx, (inputs, targets) in enumerate(data_loader):
    print(f"Batch {batch_idx}: Inputs Shape {inputs.shape}, Targets Shape {targets.shape}")
    break  # Only print the first batch for validation
