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
        - Renames columns to match expected output
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

        # Rename columns to match expected format
        self.df.rename(columns={
            'lastUpdate': 'date',
            'hargaJual': 'sell',
            'hargaBeli': 'buy'
        }, inplace=True)

        # Keep only necessary columns
        self.df = self.df[['date', 'sell', 'buy']]

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
