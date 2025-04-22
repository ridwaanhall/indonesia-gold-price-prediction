import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import os

class GoldPriceDataset(Dataset):
    """
    Dataset class for gold price prediction with LSTM model.
    Handles data preprocessing, normalization, and sequence preparation.
    """
    
    def __init__(self, file_path: str, sequence_length: int = 60, train: bool = True):
        """
        Initialize dataset from CSV file.
        
        Args:
            file_path (str): Path to the CSV file with gold price data
            sequence_length (int): Number of days to include in each sequence
            train (bool): Whether this dataset is for training
        """
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.is_train = train
        
        # Read data
        self.df = self._load_data()
        
        # Extract features and normalize data
        self._prepare_features()
        
        # Create sequences for LSTM input
        self.X, self.y, self.dates = self._create_sequences()
        
        # Convert to tensors
        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y, dtype=torch.float32)
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load and verify the data from CSV file.
        
        Returns:
            pd.DataFrame: Loaded and verified DataFrame
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("Data must contain a 'date' column")
        
        # Ensure required price columns exist
        if 'sell' not in df.columns:
            raise ValueError("Data must contain a 'sell' column for gold sell price")
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _prepare_features(self) -> None:
        """
        Prepare features for model training/prediction.
        This includes feature selection and normalization.
        """
        # Define our feature set for the model
        self.features = [
            'sell',                  # Sell price - primary target for prediction
            'buy',                   # Buy price
            'sell_ma7',              # 7-day moving average
            'sell_ma30',             # 30-day moving average
            'sell_ma365',            # 365-day moving average
            'price_change_pct',      # Percentage price change (day-to-day)
            'sell_volatility_30',    # 30-day volatility
        ]

        # Add cyclical time features using sine/cosine encoding
        # Day of week (0-6)
        self.df['sin_day'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['cos_day'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)

        # Month (1-12)
        self.df['sin_month'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['cos_month'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        # Add these cyclical features to our feature list
        self.features.extend(['sin_day', 'cos_day', 'sin_month', 'cos_month'])
        
        # Create min/max dictionaries for normalization
        self.min_vals = {}
        self.max_vals = {}
        
        # Normalize all numerical features to [0,1] range
        for feature in self.features:
            self.min_vals[feature] = self.df[feature].min()
            self.max_vals[feature] = self.df[feature].max()
            
            # Avoid division by zero if min=max
            if self.max_vals[feature] > self.min_vals[feature]:
                self.df[f"{feature}_norm"] = (self.df[feature] - self.min_vals[feature]) / (self.max_vals[feature] - self.min_vals[feature])
            else:
                self.df[f"{feature}_norm"] = 0.0
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
        """
        Create sequences of data for LSTM input.
        Each sequence has length sequence_length, and the target is the next day's price.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, List[datetime]]: X sequences, y targets, and corresponding dates
        """
        X, y, dates = [], [], []
        
        # For each valid sequence position
        for i in range(len(self.df) - self.sequence_length):
            # Extract sequence and target
            sequence_df = self.df.iloc[i:i + self.sequence_length]
            target_df = self.df.iloc[i + self.sequence_length]
            
            # Get normalized features for sequence
            features = []
            for j in range(self.sequence_length):
                day_features = [sequence_df.iloc[j][f"{feature}_norm"] for feature in self.features]
                features.append(day_features)
            
            # Target is the normalized sell price for the next day
            target = target_df["sell_norm"]
            
            X.append(features)
            y.append(target)
            dates.append(target_df["date"])
        
        return np.array(X), np.array(y), dates
    
    def __len__(self) -> int:
        """
        Get number of sequences in the dataset.
        
        Returns:
            int: Number of sequences
        """
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target by index.
        
        Args:
            idx (int): Index of the sequence
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input sequence and target
        """
        return self.X_tensor[idx], self.y_tensor[idx]
    
    def get_last_sequence(self) -> Tuple[torch.Tensor, datetime]:
        """
        Get the most recent sequence in the dataset for prediction.
        
        Returns:
            Tuple[torch.Tensor, datetime]: Last sequence and its end date
        """
        last_sequence = self.X_tensor[-1]
        last_date = self.dates[-1] if self.dates else self.df.iloc[-1]['date']
        return last_sequence, last_date
    
    def denormalize_price(self, normalized_price: float) -> float:
        """
        Convert normalized price back to original scale.
        
        Args:
            normalized_price (float): Normalized price value
            
        Returns:
            float: Original scale price
        """
        min_val = self.min_vals['sell']
        max_val = self.max_vals['sell']
        return normalized_price * (max_val - min_val) + min_val


def get_data_loader(file_path: str, batch_size: int = 32, sequence_length: int = 60, shuffle: bool = True) -> Tuple[DataLoader, GoldPriceDataset]:
    """
    Create a DataLoader for the gold price dataset.
    
    Args:
        file_path (str): Path to CSV data file
        batch_size (int): Batch size for training
        sequence_length (int): Number of days in each sequence
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        Tuple[DataLoader, GoldPriceDataset]: DataLoader and the underlying Dataset
    """
    dataset = GoldPriceDataset(file_path, sequence_length=sequence_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader, dataset


# For testing
if __name__ == "__main__":
    # Test on the preprocessed data
    data_path = "data/processed/gold_prices_cleaned.csv"
    
    # Check if the file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        exit(1)
    
    # Load the dataset
    train_loader, dataset = get_data_loader(data_path, batch_size=32)
    
    # Print dataset stats
    print(f"\nDataset loaded successfully!")
    print(f"Number of sequences: {len(dataset)}")
    print(f"Sequence length: {dataset.sequence_length}")
    print(f"Feature count: {len(dataset.features)}")
    print(f"Features: {dataset.features}")
    
    # Basic data checks
    print("\nData ranges:")
    for feature in dataset.features:
        print(f"- {feature}: Min={dataset.min_vals[feature]:.2f}, Max={dataset.max_vals[feature]:.2f}")
    
    # Check a sample batch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"\nSample batch shape: Inputs {inputs.shape}, Targets {targets.shape}")
        print(f"Input value range: Min={inputs.min().item():.4f}, Max={inputs.max().item():.4f}")
        print(f"Target value range: Min={targets.min().item():.4f}, Max={targets.max().item():.4f}")
        break
    
    # Test the last sequence retrieval
    last_sequence, last_date = dataset.get_last_sequence()
    print(f"\nLast sequence shape: {last_sequence.shape}")
    print(f"Last sequence date: {last_date}")
    
    # Test denormalization
    test_norm_price = 0.8
    original_price = dataset.denormalize_price(test_norm_price)
    print(f"\nDenormalized test price (0.8) = {original_price:.2f} IDR per 0.01 gram")
    
    print("\nDataset preparation complete and working properly!")
