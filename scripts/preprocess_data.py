import json
import pandas as pd
from typing import Dict, Any

class GoldPricePreprocessor:
    """
    A class for preprocessing gold price data from JSON format.

    Attributes:
        input_file (str): Path to the raw JSON file.
        output_file (str): Path to save the processed CSV file.
        df (pd.DataFrame): DataFrame to store processed data.
    """

    def __init__(self, input_file: str, output_file: str):
        """
        Initializes the GoldPricePreprocessor with input and output file paths.

        Args:
            input_file (str): Path to the raw JSON file.
            output_file (str): Path to save the processed CSV file.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.df: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> Dict[str, Any]:
        """
        Loads data from the JSON file.

        Returns:
            dict: Parsed JSON data.
        """
        with open(self.input_file, 'r') as file:
            data = json.load(file)
        return data.get('data', {}).get('priceList', [])

    def preprocess_data(self) -> None:
        """
        Processes the raw gold price data:
        - Converts columns to appropriate data types
        - Removes duplicates
        - Handles missing values using forward fill and smart replacements
        - Applies feature engineering (moving averages, volatility)
        - Saves processed data to a CSV file
        """
        raw_data = self.load_data()
        self.df = pd.DataFrame(raw_data)

        # Convert data types
        self.df['hargaJual'] = self.df['hargaJual'].astype(float)
        self.df['hargaBeli'] = self.df['hargaBeli'].astype(float)
        self.df['lastUpdate'] = pd.to_datetime(self.df['lastUpdate'])

        # Remove duplicate timestamps
        self.df.drop_duplicates(subset=['lastUpdate'], inplace=True)

        # Sort by date
        self.df = self.df.sort_values(by='lastUpdate')
        
        # Handle zero values in hargaJual and hargaBeli columns
        # First, create a mask for zero values
        sell_zeros_mask = self.df["hargaJual"] == 0
        buy_zeros_mask = self.df["hargaBeli"] == 0

        # Replace zeros with NaN
        self.df.loc[sell_zeros_mask, "hargaJual"] = float('nan')
        self.df.loc[buy_zeros_mask, "hargaBeli"] = float('nan')

        # Interpolate the missing values
        # Linear interpolation works well for short gaps
        self.df["hargaJual"] = self.df["hargaJual"].interpolate(method='linear')
        self.df["hargaBeli"] = self.df["hargaBeli"].interpolate(method='linear')

        # In case there are still NaNs at the beginning or end, use forward/backward fill
        self.df["hargaJual"] = self.df["hargaJual"].ffill().bfill()
        self.df["hargaBeli"] = self.df["hargaBeli"].ffill().bfill()

        # Rename columns to match expected format
        self.df.rename(columns={
            'lastUpdate': 'date',
            'hargaJual': 'sell',
            'hargaBeli': 'buy'
        }, inplace=True)

        # Keep only necessary columns
        self.df = self.df[['date', 'sell', 'buy']]

        # Handle missing values using ffill (Forward Fill)
        self.df["sell"] = self.df["sell"].ffill()
        self.df["buy"] = self.df["buy"].ffill()

        # Feature Engineering: Moving Averages
        self.df["sell_ma7"] = self.df["sell"].rolling(window=7).mean().ffill()
        self.df["sell_ma30"] = self.df["sell"].rolling(window=30).mean().ffill()
        self.df["sell_ma365"] = self.df["sell"].rolling(window=365).mean().ffill()

        # Price Change Percentage - no need for separate handling of zeros now
        self.df["price_change_pct"] = self.df["sell"].pct_change(fill_method=None) * 100
        # Replace any inf values with NaN and then fill NaN with 0
        self.df["price_change_pct"] = self.df["price_change_pct"].replace([float('inf'), float('-inf')], float('nan')).fillna(0)

        # Volatility (Rolling Standard Deviation over 30 days)
        self.df["sell_volatility_30"] = self.df["sell"].rolling(window=30).std().fillna(0)

        # Time Features
        self.df["day_of_week"] = self.df["date"].dt.dayofweek
        self.df["quarter"] = self.df["date"].dt.quarter
        self.df["month"] = self.df["date"].dt.month.astype('int32')

    def save_data(self) -> None:
        """
        Saves the cleaned data to a CSV file.
        """
        self.df.to_csv(self.output_file, index=False)
        print(f"✅ Preprocessing completed! Data saved to '{self.output_file}'")

    def run(self) -> None:
        """
        Executes the full preprocessing pipeline.
        """
        self.preprocess_data()
        self.save_data()

# Execute the preprocessing script
raw_data_path = "data/raw/10_years_22042025.json"
processed_data_path = "data/processed/gold_prices_cleaned.csv"
preprocessor = GoldPricePreprocessor(raw_data_path, processed_data_path)
preprocessor.run()
