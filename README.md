# Personal Finance Analytics & Forecasting System

An end-to-end personal finance analysis and forecasting system that automatically processes personal spending data and generates interpretable financial insights using Python and machine learning.

The system is designed around a single-call pipeline that ingests raw financial data, cleans it, analyzes spending behavior, and predicts next-month financial indicators.

---

## 📌 Project Objectives

- Ingest personal financial data from CSV files  
- Automatically clean and validate user datasets  
- Detect common financial columns (date, income, expenses)  
- Analyze spending habits and category distributions  
- Forecast next-month total expenditure  
- Predict expense risk level using machine learning  
- Provide interpretable numerical and visual outputs  
- Ensure data privacy and reproducibility  

---

## 🧩 System Architecture

Raw CSV Data
        ↓
Automatic Cleaning & Validation
        ↓
Feature Engineering (Lag-Based)
        ↓
Spending Analysis
        ↓
Model Selection & Training
        ↓
Next-Month Forecasting
        ↓
Risk Assessment & Visualization


The system operates through a unified and modular pipeline interface.

---

## 📁 Project Structure

```text
personal-finance-analytics/
├── data/
│   ├── raw/                 # Private data (ignored)
│   └── sample/              # Public example data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_RunMe.ipynb
├── src/
│   └── finance_pipeline.py
├── reports/
│   └── screenshots/
├── requirements.txt
└── README.md
```


---

## 🛠 Technologies Used

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Git & GitHub

---

## 📊 Data Sources

- Personal monthly expense records (private)
- Public sample CSV files for demonstration

All private financial data is excluded from this repository.

---

## 🚀 Core Features

### Automatic Data Handling

- Flexible detection of date and income columns  
- Regex-based column normalization  
- Automatic expense aggregation  
- Robust validation  

### Spending Analytics

- Category contribution analysis  
- Income vs expense growth tracking  
- Expense ratio monitoring  
- Top category identification  

### Forecasting System

- Lag-based supervised learning (no data leakage)  
- Automatic model selection  
- Linear Regression and Random Forest models  
- Backtesting on unseen data  

### Risk Assessment

- Expense ratio prediction  
- Financial risk classification:
  - Excellent  
  - Stable  
  - Warning  
  - Critical  

### Visualization

- Historical spending trends  
- Forecast continuation plots  
- Expense ratio trajectory  

### One-Call Interface

- Users only need:

  - python
    - from src.finance_pipeline import run
    - results = run("data/raw/my_data.csv")

##📈 Model Evaluation

- Models are evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Visual inspection
- Time-based backtesting
- Only lag-based features are used to avoid future data leakage.

## ▶️ How to Run

### 1️⃣ Install Dependencies
- pip install -r requirements.txt

### 2️⃣ Place Your Data
- Put your CSV file in:
  - data/raw/

### 3️⃣ Run the Pipeline
- Open:
  - notebooks/04_RunMe.ipynb

## 📤 Output

### The system produces:

- Console Report
- Next-month predicted spending
- Expense ratio
- Risk level
- Plots
- Historical spending + forecast
- Expense ratio trend
- Structured Output

### Returned as a Python dictionary:

- results["next_month_forecast"]
- results["spending_habits"]
- results["models"]

## 🔒 Data Privacy

- All personal data remains local
- Raw data is ignored by Git
- No cloud upload or external APIs
- Fully offline operation

## 🔮 Future Improvements

- Multi-step forecasting
- Deep learning models (LSTM, Transformer)
- Web interface
- Automated budget recommendations
- Personalized alerts
- Mobile dashboard

## 📄 License

This project is for academic and personal learning purposes only.

## 👤 Author

### Jad Kandah
GitHub: https://github.com/jadkandah
LinkedIn: https://www.linkedin.com/in/jad-kandah-992294132
