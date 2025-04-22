import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Union, Optional
import calendar
from model import GoldPriceLSTM
from data_loader import GoldPriceDataset

class GoldPricePredictor:
    """
    Class for predicting gold prices using a trained LSTM model.
    Supports next-day, multi-day, and long-term (up to 5 years) forecasting.
    """
    
    def __init__(self, model_path: str, data_path: str, sequence_length: int = 60):
        """
        Initialize the predictor with a trained model and historical data.
        
        Args:
            model_path (str): Path to the trained model (.pth file)
            data_path (str): Path to the processed data (.csv file)
            sequence_length (int): Length of input sequence for prediction
        """
        self.model_path = model_path
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.dataset = self._load_dataset()
        self.model = self._load_model()
        
        # Dicts to store historical seasonality data
        self._monthly_patterns = None
        self._weekly_patterns = None
        self._calculate_seasonal_patterns()

    def _load_dataset(self) -> GoldPriceDataset:
        """
        Load the dataset containing historical gold price data.
        
        Returns:
            GoldPriceDataset: Dataset object with normalized data
        """
        return GoldPriceDataset(self.data_path, sequence_length=self.sequence_length)

    def _load_model(self) -> GoldPriceLSTM:
        """
        Load the trained LSTM model.
        
        Returns:
            GoldPriceLSTM: Trained PyTorch model
        """
        # Determine input size from dataset features
        input_size = len(self.dataset.features)
        
        # Initialize model with the same architecture used for training
        model = GoldPriceLSTM(input_size=input_size)
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.eval()
        
        return model
    
    def _calculate_seasonal_patterns(self) -> None:
        """
        Calculate historical seasonal patterns (monthly and weekly) from historical data.
        These will be used to adjust long-term predictions.
        """
        # Original dataframe from dataset
        df = self.dataset.df
        
        if 'date' not in df.columns or 'sell' not in df.columns:
            print("Warning: Cannot calculate seasonal patterns, missing date or sell columns.")
            return
        
        # Calculate monthly seasonality - average deviation from yearly trend by month
        df['year_month'] = df['date'].dt.strftime('%Y-%m')
        
        monthly_patterns = {}
        monthly_df = df.groupby(df['date'].dt.month)['sell'].mean()
        
        # Store as percentage difference from overall mean
        overall_mean = monthly_df.mean()
        for month, value in monthly_df.items():
            monthly_patterns[month] = (value / overall_mean) - 1.0
        
        self._monthly_patterns = monthly_patterns
        
        # Calculate day-of-week patterns - average deviation from weekly mean by day
        weekly_patterns = {}
        weekly_df = df.groupby(df['date'].dt.dayofweek)['sell'].mean()
        
        # Store as percentage difference from overall mean
        weekly_mean = weekly_df.mean()
        for day, value in weekly_df.items():
            weekly_patterns[day] = (value / weekly_mean) - 1.0
            
        self._weekly_patterns = weekly_patterns

    def predict_next_day(self) -> Tuple[datetime, float]:
        """
        Predict the gold price for the next day after the last date in the dataset.
        
        Returns:
            Tuple[datetime, float]: (next day date, predicted price)
        """
        # Get the most recent sequence and last date from dataset
        last_sequence, last_date = self.dataset.get_last_sequence()
        
        # Add batch dimension for model input
        model_input = last_sequence.unsqueeze(0)
        
        # Generate prediction
        with torch.no_grad():
            normalized_prediction = self.model(model_input).item()
        
        # Denormalize the prediction
        predicted_price = self.dataset.denormalize_price(normalized_prediction)
        
        # Calculate next day date
        next_date = last_date + timedelta(days=1)
        
        return next_date, predicted_price

    def predict_future(self, days: int = 30) -> pd.DataFrame:
        """
        Predict gold prices for multiple days into the future.
        For short-term predictions (< 30 days), uses iterative prediction.
        For long-term predictions, incorporates seasonal adjustments.
        
        Args:
            days (int): Number of future days to predict
            
        Returns:
            pd.DataFrame: DataFrame with dates and predicted prices
        """
        if days <= 0:
            raise ValueError("Days must be a positive integer")
        
        if days > 365 * 5:
            print("Warning: Predictions beyond 5 years may lose accuracy. Limiting to 5 years.")
            days = 365 * 5
        
        # Get last sequence and date
        last_sequence, last_date = self.dataset.get_last_sequence()
        sequence = last_sequence.clone()
        
        # Results storage
        dates = []
        predictions = []
        
        # Long-term prediction flag
        is_long_term = days > 30
        
        # Get baseline trend from recent data
        recent_df = self.dataset.df.tail(365 if len(self.dataset.df) > 365 else len(self.dataset.df))
        recent_trend = 0.0
        if len(recent_df) >= 2:
            first_price = recent_df.iloc[0]['sell']
            last_price = recent_df.iloc[-1]['sell']
            days_diff = (recent_df.iloc[-1]['date'] - recent_df.iloc[0]['date']).days
            if days_diff > 0:  # Avoid division by zero
                recent_trend = (last_price - first_price) / (days_diff * first_price) * 100  # Daily % change
        
        # Generate predictions day by day
        for i in range(days):
            # Calculate the next date
            next_date = last_date + timedelta(days=i+1)
            dates.append(next_date)
            
            # Use model for short-term prediction
            with torch.no_grad():
                # Add batch dimension for model input
                model_input = sequence.unsqueeze(0)
                
                # Get normalized prediction
                normalized_prediction = self.model(model_input).item()
                
                # Denormalize the prediction to get actual price
                predicted_price = self.dataset.denormalize_price(normalized_prediction)
            
            # Apply seasonal adjustments for long-term predictions
            if is_long_term and i >= 30:
                # Monthly seasonal adjustment
                month = next_date.month
                month_adjustment = 0.0
                if self._monthly_patterns and month in self._monthly_patterns:
                    month_adjustment = self._monthly_patterns[month]
                
                # Day of week adjustment
                day_of_week = next_date.weekday()
                day_adjustment = 0.0
                if self._weekly_patterns and day_of_week in self._weekly_patterns:
                    day_adjustment = self._weekly_patterns[day_of_week]
                
                # Apply adjustments
                seasonal_adjustment = 1.0 + (month_adjustment + day_adjustment) / 2.0
                predicted_price *= seasonal_adjustment
                
                # Apply long-term trend influence (stronger as we go further in time)
                trend_influence = min(0.8, i / days)  # Cap at 80% influence
                trend_adjustment = 1.0 + (recent_trend * i / 100.0) * trend_influence
                predicted_price *= trend_adjustment
                
                # Add some progressive randomness to simulate market uncertainty
                noise_scale = min(0.05, i / days * 0.05)  # Up to 5% noise for farthest predictions
                noise_factor = 1.0 + np.random.normal(0, noise_scale)
                predicted_price *= max(0.9, min(1.1, noise_factor))  # Limit noise impact
            
            predictions.append(predicted_price)
            
            # Update sequence for next iteration (for short-term predictions)
            if i < days - 1 and not is_long_term:
                # Extract features for the predicted point
                new_point_features = self._generate_features_for_prediction(
                    predicted_price,
                    sequence,
                    next_date
                )
                
                # Roll the sequence forward (remove oldest, add newest)
                new_sequence = torch.cat([
                    sequence[1:],  # All but the first element
                    new_point_features.unsqueeze(0)  # New point
                ], dim=0)
                
                sequence = new_sequence
        
        # Create DataFrame with results
        return pd.DataFrame({'date': dates, 'predicted_price': predictions})
    
    def _generate_features_for_prediction(self, predicted_price: float, sequence: torch.Tensor, date: datetime) -> torch.Tensor:
        """
        Generate feature vector for a predicted price to use in subsequent predictions.
        
        Args:
            predicted_price (float): The predicted price in original scale
            sequence (torch.Tensor): The current input sequence
            date (datetime): The date for the prediction
            
        Returns:
            torch.Tensor: Feature vector for the new prediction point
        """
        # Normalize the predicted price
        sell_min = self.dataset.min_vals["sell"]
        sell_max = self.dataset.max_vals["sell"]
        norm_price = (predicted_price - sell_min) / (sell_max - sell_min) if sell_max > sell_min else 0
        
        # Previous price is needed for buy price estimate and change calculations
        prev_price_norm = sequence[-1, 0].item()  # Previous normalized sell price
        prev_price = prev_price_norm * (sell_max - sell_min) + sell_min  # Denormalized
        
        # Normalize based on the existing sequence
        features = []
        
        # Handle each feature based on its position in the sequence
        for i, feature_name in enumerate(self.dataset.features):
            if feature_name == "sell":
                features.append(norm_price)
            elif feature_name == "buy":
                # Estimate buy price based on typical sell-buy spread
                typical_ratio = 0.95  # Buy is typically around 95% of sell
                buy_value = predicted_price * typical_ratio
                buy_min = self.dataset.min_vals["buy"]
                buy_max = self.dataset.max_vals["buy"] 
                norm_buy = (buy_value - buy_min) / (buy_max - buy_min) if buy_max > buy_min else 0
                features.append(norm_buy)
            elif feature_name == "sell_ma7":
                # Update the 7-day moving average
                ma7_values = [sequence[-6:, i].mean().item(), norm_price]
                features.append(sum(ma7_values) / len(ma7_values))
            elif feature_name == "sell_ma30":
                # Update the 30-day moving average
                ma30_values = [sequence[-29:, i].mean().item(), norm_price]
                features.append(sum(ma30_values) / len(ma30_values))
            elif feature_name == "sell_ma365":
                # Update the 365-day moving average
                features.append(sequence[-1, i].item())  # Just use previous value
            elif feature_name == "price_change_pct":
                # Calculate percentage change
                if prev_price > 0:
                    pct_change = (predicted_price - prev_price) / prev_price * 100
                    pct_min = self.dataset.min_vals["price_change_pct"]
                    pct_max = self.dataset.max_vals["price_change_pct"]
                    norm_change = (pct_change - pct_min) / (pct_max - pct_min) if pct_max > pct_min else 0
                    features.append(norm_change)
                else:
                    features.append(0.0)
            elif feature_name == "sell_volatility_30":
                # Just use previous volatility as estimate
                features.append(sequence[-1, i].item())
            elif feature_name == "sin_day":
                # Calculate sine of day of week
                day_of_week = date.weekday()
                features.append(np.sin(2 * np.pi * day_of_week / 7))
            elif feature_name == "cos_day":
                # Calculate cosine of day of week
                day_of_week = date.weekday()
                features.append(np.cos(2 * np.pi * day_of_week / 7))
            elif feature_name == "sin_month":
                # Calculate sine of month
                features.append(np.sin(2 * np.pi * date.month / 12))
            elif feature_name == "cos_month":
                # Calculate cosine of month
                features.append(np.cos(2 * np.pi * date.month / 12))
            else:
                # For any other features, use the previous value
                features.append(sequence[-1, i].item())
                
        return torch.tensor(features, dtype=torch.float32)
    
    def predict_specific_date(self, target_date: datetime) -> Tuple[datetime, float]:
        """
        Predict gold price for a specific future date.
        
        Args:
            target_date (datetime): The specific date to predict for
            
        Returns:
            Tuple[datetime, float]: (target date, predicted price)
            Returns (None, None) if date is in the past
        """
        # Get the last date in our dataset
        _, last_date = self.dataset.get_last_sequence()
        
        # Check if target date is in the past
        if target_date <= last_date:
            print(f"Error: Target date {target_date.strftime('%Y-%m-%d')} is not in the future.")
            return None, None
        
        # Calculate days between last date and target date
        days_ahead = (target_date - last_date).days
        
        # Get predictions for all days up to the target date
        predictions = self.predict_future(days=days_ahead)
        
        # Find the row with the target date
        target_row = predictions[predictions['date'] == target_date]
        
        if not target_row.empty:
            return target_date, target_row.iloc[0]['predicted_price']
        else:
            print(f"Error: Could not generate prediction for {target_date.strftime('%Y-%m-%d')}")
            return None, None
    
    def predict_years_ahead(self, years: int = 1) -> pd.DataFrame:
        """
        Predict gold prices for multiple years ahead.
        
        Args:
            years (int): Number of years to predict (up to 5)
            
        Returns:
            pd.DataFrame: DataFrame with dates and predicted prices, 
                          with monthly frequency for clarity
        """
        if years <= 0:
            raise ValueError("Years must be a positive integer")
        
        if years > 5:
            print("Warning: Predictions beyond 5 years may lose accuracy. Limiting to 5 years.")
            years = 5
        
        # Calculate days to predict
        days = years * 365
        
        # Get full predictions
        full_predictions = self.predict_future(days=days)
        
        # Resample to monthly frequency for better readability
        # First ensure the date column is a datetime
        full_predictions['date'] = pd.to_datetime(full_predictions['date'])
        
        # Set date as index to enable resampling
        indexed_df = full_predictions.set_index('date')
        
        # Resample to monthly frequency, taking the mean for each month
        monthly_predictions = indexed_df.resample('M').mean().reset_index()
        
        # Also include the last day prediction for complete range
        last_day = full_predictions.iloc[-1:].copy()
        
        # Combine and sort
        result = pd.concat([monthly_predictions, last_day]).sort_values('date')
        
        return result
                
# For direct execution
if __name__ == "__main__":
    model_path = "models/gold_price_lstm.pth"
    data_path = "data/processed/gold_prices_cleaned.csv"
    
    # Check if files exist
    import os
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using train.py")
        exit(1)
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please process the data first using scripts/preprocess_data.py")
        exit(1)
    
    # Initialize predictor
    predictor = GoldPricePredictor(model_path, data_path)
    
    # Predict for next day
    next_date, next_price = predictor.predict_next_day()
    print(f"\n✅ Gold Price Prediction for {next_date.strftime('%Y-%m-%d')}: {next_price:.2f} IDR per 0.01 gram")
    
    # Predict for 1 year ahead (monthly results)
    print("\n✅ Predictions for 1 year ahead (monthly):")
    yearly_pred = predictor.predict_years_ahead(years=1)
    print(yearly_pred[['date', 'predicted_price']].head())
    
    # Predict for 5 years ahead (showing first and last months)
    print("\n✅ Predictions for 5 years ahead (showing first and last 3 months):")
    five_year_pred = predictor.predict_years_ahead(years=5)
    combined = pd.concat([five_year_pred.head(3), five_year_pred.tail(3)])
    print(combined[['date', 'predicted_price']])
    print(f"\n... and {len(five_year_pred) - 6} more monthly predictions.")
    
    print("\nDone!")
