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

## To-Do

### **1️⃣ Data Preparation (First Step)**

📌 Files to work on first:

- `data/raw/` → Gather and clean historical gold price datasets.
- `scripts/preprocess_data.py` → Write a script to **clean, normalize, and structure** your raw data.
- `notebooks/data_exploration.ipynb` → Perform **Exploratory Data Analysis (EDA)** to understand trends.

👉 **Why?** Your model is only as good as your data. Cleaning and analyzing it first ensures accuracy in predictions.

### **2️⃣ Model Development**

📌 Files to focus on:

- `src/data_loader.py` → Create a **data loader** to efficiently handle your dataset.
- `src/model.py` → Define the **PyTorch model architecture** (LSTM, GRU, etc.).
- `configs/config.yaml` → Set up **hyperparameters and training settings**.

👉 **Why?** Once data is ready, designing your model ensures it learns patterns effectively.

### **3️⃣ Training & Evaluation**

📌 Next steps:

- `src/train.py` → Train your model using historical gold price data.
- `notebooks/model_training.ipynb` → **Analyze model performance** and fine-tune hyperparameters.
- `tests/test_model.py` → Write **unit tests** to validate model accuracy.

👉 **Why?** Testing ensures your predictions are reliable before deployment.

### **4️⃣ Prediction & Deployment**

📌 Final steps:

- `src/predict.py` → Implement **real-time inference** for gold price predictions.
- `notebooks/forecasting.ipynb` → Visualize price trends.
- `setup.py` → Prepare for **deployment** (Flask/Streamlit app).

👉 **Why?** Once your model is trained and tested, you can deploy it for actual use.

### **5️⃣ Documentation & Reports**

📌 Complete:

- `README.md` → Update with project progress.
- `reports/project_summary.md` → Document findings and insights.
- `logs/` → Track model training experiments.
