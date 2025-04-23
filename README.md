# **The Next 5 Years of Gold Prices in Indonesia: AI Predictions Based on Historical Trends**

## 📌 **Project Overview**

This project uses machine learning to predict Indonesia's gold prices over the next 5 years based on historical data. The system employs LSTM (Long Short-Term Memory) neural networks that can recognize complex patterns in time series data, allowing for accurate short-term predictions and reasonable long-term forecasts.

The model is trained on 10+ years of historical gold price data and can predict future gold prices with various time horizons:
- Next day prediction
- Short-term predictions (days to months)
- Long-term forecasts (1-5 years)

The predictions are visualized through plots and can be exported to CSV files for further analysis.

## 🔍 **Objectives**

1. **Data Processing:** Clean and prepare historical gold price data for model training
2. **Feature Engineering:** Generate relevant features like moving averages and volatility metrics
3. **Model Development:** Create and train LSTM models for time series forecasting
4. **Prediction System:** Build a flexible prediction system for various time horizons
5. **Visualization:** Generate insightful plots showing prediction trends
6. **Data Export:** Save predictions to CSV files for further analysis

## 📊 **Methodology**

1. **Data Collection:** Historical gold price data from multiple sources
2. **Preprocessing:** Cleaning, normalization, and feature engineering
3. **Model Architecture:** Multi-layer LSTM neural networks with attention mechanism
4. **Training Process:** Supervised learning with historical data sequences
5. **Evaluation:** Validation against historical trends and market dynamics
6. **Deployment:** Command-line interface for making predictions

## 🛠️ **Project Structure**

```
├── data/                 # Data storage
│   ├── raw/              # Raw gold price data
│   └── processed/        # Processed datasets
├── models/               # Trained model files
├── notebooks/            # Jupyter notebooks for exploration
├── plots/                # Generated prediction plots
├── results/              # CSV prediction results  
├── scripts/              # Data processing scripts
└── src/                  # Source code
    ├── data_loader.py    # Dataset preparation
    ├── model.py          # LSTM model architecture
    ├── predict.py        # Prediction functionality
    ├── run_prediction.py # CLI for predictions
    └── train.py          # Model training
```

## 📈 **Features**

- **Multiple Prediction Modes:**
  - Next-day prediction
  - Multi-day forecasting (specific number of days)
  - Date-specific prediction
  - Date range predictions
  - 1-5 year forecasts

- **Advanced Visualizations:**
  - Historical vs predicted prices
  - Various time horizons (1 month to 5 years)
  - Clear trend indicators

- **Data Export:**
  - Save predictions to CSV files in the results folder
  - Flexible output formatting

## 🚀 **How to Use**

### 1. **Prerequisites**
```bash
pip install -r requirements.txt
```

### 2. **Train the Model** (optional, pre-trained model included)
```bash
python src/train.py
```

### 3. **Make Predictions**

#### Next Day Prediction:
```bash
python src/run_prediction.py --mode next_day --output results/next_day_prediction.csv
```

#### Predict Multiple Days:
```bash
python src/run_prediction.py --mode days --days 30 --output results/30_days_prediction.csv
```

#### Predict for Specific Date:
```bash
python src/run_prediction.py --mode specific_date --date 2025-12-31 --output results/specific_date_prediction.csv
```

#### Predict for Date Range:
```bash
python src/run_prediction.py --mode range --start_date 2025-05-01 --end_date 2025-06-30 --output results/date_range_prediction.csv
```

#### Generate Prediction Plots:
```bash
python src/run_prediction.py --mode plot
```

## 📊 **Saving Predictions**

All prediction modes support saving results to CSV files using the `--output` parameter. By default, these should be stored in the `results/` folder:

1. **Basic Example:**
```bash
python src/run_prediction.py --mode next_day --output results/gold_prediction.csv
```

2. **Including Date in Filename:**
```bash
python src/run_prediction.py --mode days --days 365 --output results/gold_prediction_1year_$(date +%Y%m%d).csv
```

3. **Long-term Predictions:**
```bash
python src/run_prediction.py --mode range --start_date 2025-05-01 --end_date 2030-05-01 --output results/five_year_forecast.csv
```

The CSV files contain two columns:
- `date`: The prediction date
- `predicted_price`: The predicted gold price in IDR per 0.01g

## 📋 **Results Interpretation**

- **Short-term predictions (1-30 days):** High accuracy and confidence
- **Medium-term (1-6 months):** Good directional accuracy with moderate price precision
- **Long-term (1-5 years):** Best used for trend analysis rather than exact price predictions

## 🔮 **Future Improvements**

1. Integration of external economic indicators
2. Ensemble models combining multiple forecasting approaches
3. Web interface for easy prediction access
4. Real-time data updates and predictions

## 📝 **License**

This project is licensed under the MIT License - see the LICENSE file for details.
