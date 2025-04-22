import torch
import pandas as pd
from model import GoldPriceLSTM

class GoldPricePredictor:
    def __init__(self, model_path: str, data_path: str, sequence_length: int = 30):
        self.model_path = model_path
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.features = ["sell", "sell_ma7", "sell_ma30", "sell_volatility_30"]
        self.model = self.load_model()
        self.df = self.load_data()

    def load_model(self):
        model = GoldPriceLSTM(input_size=len(self.features))
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def load_data(self):
        df = pd.read_csv(self.data_path, parse_dates=["date"])
        for feature in self.features:
            df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
        return df

    def predict_next_day(self):
        recent_data = self.df[self.features].iloc[-self.sequence_length:].values
        recent_data_tensor = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            future_price = self.model(recent_data_tensor).item()

        # Retrieve original prices before normalization
        min_price = self.df["sell"].min()
        max_price = self.df["sell"].max()

        # Denormalize predicted price
        actual_price = future_price * (max_price - min_price) + min_price
        print(f"✅ Predicted Gold Price for Next Day (Denormalized): {actual_price:.2f} IDR")

        return actual_price

model_path = "models/gold_price_lstm.pth"
data_path = "data/processed/gold_prices_cleaned.csv"

predictor = GoldPricePredictor(model_path, data_path)
predicted_price = predictor.predict_next_day()
print(f"✅ Predicted Gold Price for Next Day: {predicted_price:.2f} IDR")
