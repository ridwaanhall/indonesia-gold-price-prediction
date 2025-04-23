import pandas as pd
from typing import Dict, Any
import os

class GoldPricePreprocessor:
    """
    A class for preprocessing gold price data from CSV format.

    Attributes:
        input_file (str): Path to the raw CSV file.
        output_file (str): Path to save the processed CSV file.
        df (pd.DataFrame): DataFrame to store processed data.
    """

    def __init__(self, input_file: str, output_file: str):
        """
        Initializes the GoldPricePreprocessor with input and output file paths.

        Args:
            input_file (str): Path to the raw CSV file.
            output_file (str): Path to save the processed CSV file.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.df: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        # Check the file extension to determine how to load it
        if self.input_file.endswith('.csv'):
            return pd.read_csv(self.input_file, parse_dates=['date'])
        elif self.input_file.endswith('.json'):
            # For future compatibility if JSON files are used
            try:
                import json
                with open(self.input_file, 'r') as file:
                    data = json.load(file)
                # Extract price list if it exists in the expected format
                if isinstance(data, dict) and 'data' in data and 'priceList' in data['data']:
                    return pd.DataFrame(data['data']['priceList'])
                else:
                    # Direct JSON structure
                    return pd.DataFrame(data)
            except Exception as e:
                print(f"Error loading JSON: {e}")
                return pd.DataFrame()
        else:
            raise ValueError(f"Unsupported file format for {self.input_file}")

    def preprocess_data(self) -> None:
        """
        Processes the raw gold price data:
        - Converts columns to appropriate data types
        - Removes duplicates
        - Handles missing values using forward fill and smart replacements
        - Applies feature engineering (moving averages, volatility)
        - Saves processed data to a CSV file
        """
        # Load the data
        self.df = self.load_data()
        
        # Handle CSV that's already in the right format
        if 'date' in self.df.columns and 'sell' in self.df.columns and 'buy' in self.df.columns:
            print("Data already in expected format, applying additional processing...")
        # Handle JSON converted data with different column names
        elif 'lastUpdate' in self.df.columns and 'hargaJual' in self.df.columns:
            # Convert data types
            self.df['hargaJual'] = self.df['hargaJual'].astype(float)
            self.df['hargaBeli'] = self.df['hargaBeli'].astype(float)
            self.df['lastUpdate'] = pd.to_datetime(self.df['lastUpdate'])

            # Rename columns to match expected format
            self.df.rename(columns={
                'lastUpdate': 'date',
                'hargaJual': 'sell',
                'hargaBeli': 'buy'
            }, inplace=True)
        else:
            raise ValueError("Input data doesn't contain expected columns")

        # Ensure date column is datetime
        if self.df['date'].dtype != 'datetime64[ns]':
            self.df['date'] = pd.to_datetime(self.df['date'])

        # Remove duplicate timestamps
        self.df.drop_duplicates(subset=['date'], inplace=True)

        # Sort by date
        self.df = self.df.sort_values(by='date')
        
        # Handle zero values in sell and buy columns
        # First, create a mask for zero values
        sell_zeros_mask = self.df["sell"] == 0
        buy_zeros_mask = self.df["buy"] == 0

        # Replace zeros with NaN
        self.df.loc[sell_zeros_mask, "sell"] = float('nan')
        self.df.loc[buy_zeros_mask, "buy"] = float('nan')

        # Interpolate the missing values
        # Linear interpolation works well for short gaps
        self.df["sell"] = self.df["sell"].interpolate(method='linear')
        self.df["buy"] = self.df["buy"].interpolate(method='linear')

        # In case there are still NaNs at the beginning or end, use forward/backward fill
        self.df["sell"] = self.df["sell"].ffill().bfill()
        self.df["buy"] = self.df["buy"].ffill().bfill()

        # Keep only necessary columns
        self.df = self.df[['date', 'sell', 'buy']]

        # Feature Engineering: Moving Averages
        self.df["sell_ma7"] = self.df["sell"].rolling(window=7, min_periods=1).mean()
        self.df["sell_ma30"] = self.df["sell"].rolling(window=30, min_periods=1).mean()
        self.df["sell_ma365"] = self.df["sell"].rolling(window=365, min_periods=1).mean()

        # Price Change Percentage 
        self.df["price_change_pct"] = self.df["sell"].pct_change(fill_method=None) * 100
        # Replace any inf values with NaN and then fill NaN with 0
        self.df["price_change_pct"] = self.df["price_change_pct"].replace([float('inf'), float('-inf')], float('nan')).fillna(0)

        # Volatility (Rolling Standard Deviation over 30 days)
        self.df["sell_volatility_30"] = self.df["sell"].rolling(window=30, min_periods=1).std().fillna(0)

        # Time Features
        self.df["day_of_week"] = self.df["date"].dt.dayofweek
        self.df["quarter"] = self.df["date"].dt.quarter
        self.df["month"] = self.df["date"].dt.month.astype('int32')

    def save_data(self) -> None:
        """
        Saves the cleaned data to a CSV file.
        """
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        self.df.to_csv(self.output_file, index=False)
        print(f"✅ Preprocessing completed! Data saved to '{self.output_file}'")

    def run(self) -> None:
        """
        Executes the full preprocessing pipeline.
        """
        self.preprocess_data()
        self.save_data()

# Execute the preprocessing script
# If the data is already in CSV format, use it directly
raw_data_path = "data/processed/gold_prices_cleaned.csv" if os.path.exists("data/processed/gold_prices_cleaned.csv") else "data/raw/10_years_22042025.json"
processed_data_path = "data/processed/gold_prices_cleaned.csv"
preprocessor = GoldPricePreprocessor(raw_data_path, processed_data_path)
preprocessor.run()
