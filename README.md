# **The Next 5 Years of Gold Prices in Indonesia: AI Predictions Based on Historical Trends**

## 📌 **Project Overview**

This project aims to forecast **gold prices in the Indonesian market** for the next **1 to 5 years** using **AI-driven predictive modeling**. By leveraging **historical gold price data from the past 10 years**, we apply **deep learning techniques** to analyze trends and provide insights for investors and policymakers.

## 🔍 **Objectives**

- Develop an **AI model** for long-term gold price prediction.
- Analyze **market trends and historical price fluctuations**.
- Provide **data-driven forecasts** to assist investors in strategic decision-making.
- Utilize **time-series forecasting techniques** with **PyTorch**.

## 📂 **Project Structure**

```txt
indonesia-gold-price-prediction/
│── data/                   # Historical gold price datasets
│── notebooks/              # Jupyter notebooks for analysis & modeling
│── src/                    # Source code for PyTorch models
│── configs/                # Model & training configurations
│── reports/                # Findings, results, and evaluation metrics
│── scripts/                # Utility scripts for data processing
│── models/                 # Saved machine learning models
│── README.md               # Project documentation
│── requirements.txt        # Python dependencies
│── .gitignore              # Ignored files for version control

...

```txt
indonesia-gold-price-prediction/
│── data/                     # Raw & processed datasets
│   ├── raw/                   # Unprocessed data (CSV, JSON)
│   ├── processed/             # Cleaned and formatted data
│── notebooks/                 # Jupyter notebooks for analysis
│   ├── data_exploration.ipynb # Initial EDA (Exploratory Data Analysis)
│   ├── model_training.ipynb   # Training and evaluation notebook
│   ├── forecasting.ipynb      # Prediction and visualization
│── src/                       # Source code for training & inference
│   ├── data_loader.py         # Functions for loading/preprocessing data
│   ├── model.py               # PyTorch model implementation
│   ├── train.py               # Training pipeline
│   ├── predict.py             # Inference script for predictions
│── configs/                   # Configuration files (hyperparameters, etc.)
│   ├── config.yaml            # YAML file for model settings
│── reports/                   # Documentation & research reports
│   ├── project_summary.md     # Summary of findings & insights
│   ├── evaluation_results.md  # Model performance reports
│── tests/                     # Unit tests for robustness
│   ├── test_model.py          # Tests for model accuracy
│   ├── test_data_loader.py    # Tests for data loading functions
│── logs/                      # Logging system for experiments
│── scripts/                   # Utility scripts (automation, data fetching)
│   ├── preprocess_data.py     # Preprocessing pipeline
│   ├── run_experiment.py      # Experiment automation
│── models/                    # Saved model checkpoints
│── docs/                      # Documentation folder
│── requirements.txt           # Dependencies for Python packages
│── setup.py                   # Setup script for deployment
│── README.md                  # Project overview & instructions
│── .gitignore                 # Files to exclude from version control
```

## 📊 **Methodology**

1. **Data Collection:** Gather historical gold price data from **Indonesian sources**.
2. **Preprocessing:** Clean, normalize, and structure time-series data.
3. **Feature Engineering:** Extract **seasonality, inflation impact, and volatility trends**.
4. **Model Selection:**
   - **LSTM/GRU** for sequential predictions.
   - **Transformers for deep learning forecasting**.
   - **Hybrid models combining AI & econometric techniques**.
5. **Training & Validation:** Optimize hyperparameters for best performance.
6. **Deployment:** Host model with **Flask/Streamlit** for real-time predictions.

## ⚙️ **Installation**

To run the project, clone the repository and install dependencies:

```bash
git clone https://github.com/ridwaanhall/indonesia-gold-price-prediction.git
cd indonesia-gold-price-prediction
pip install -r requirements.txt
```

## 🛠 **Technologies Used**

- **Python** 🐍
- **PyTorch** 🔥 (Deep Learning Framework)
- **NumPy, Pandas** (Data Processing)
- **Matplotlib, Seaborn** (Visualization)
- **Flask/Streamlit** (Model Deployment)

## 📌 **Future Improvements**
