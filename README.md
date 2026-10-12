# 📊 Data-Driven Sales Forecasting & Analysis System

A machine-learning-powered sales forecasting application built with **Streamlit**, **scikit-learn**, and **XGBoost**. It enables interactive exploratory data analysis (EDA), model training, and real-time sales predictions through a polished web interface.

---

## ✨ Features

- **Interactive EDA Dashboard** — Visualize sales trends, store-type breakdowns, top product families, and promotion impact using Plotly charts.
- **Model Training Pipeline** — Train and evaluate an XGBoost regression model on the Kaggle *Store Sales* dataset with one click.
- **Real-Time Predictions** — Input store, family, and date parameters to get instant sales forecasts.
- **Streamlit Frontend** — Fully responsive, dark-themed UI with sidebar navigation and interactive widgets.
- **Modular Codebase** — Clean separation between data loading (`src/data_loader.py`), EDA (`src/eda.py`), training (`src/model_trainer.py`), and inference (`src/predictor.py`).

---

## 🗂️ Project Structure

```
sales_forecasting_system/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
│
├── src/                    # Core Python modules
│   ├── __init__.py
│   ├── data_loader.py      # Data ingestion & feature engineering
│   ├── eda.py              # Exploratory data analysis & Plotly charts
│   ├── model_trainer.py    # XGBoost model training & evaluation
│   └── predictor.py        # Inference / prediction logic
│
├── data/                   # Dataset files (see Data Setup below)
│   ├── train.csv           # ⚠️ Not tracked by Git (large file ~116 MB)
│   ├── test.csv            # ⚠️ Not tracked by Git
│   ├── sample_submission.csv # ⚠️ Not tracked by Git
│   ├── holidays_events.csv
│   ├── oil.csv
│   ├── stores.csv
│   └── transactions.csv
│
├── models/                 # Saved model artifacts
│   └── best_model.joblib   # Trained XGBoost model
│
└── notebooks/              # Jupyter exploration notebooks
    └── 01_data_exploration.ipynb
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit ≥ 1.35 |
| Data Manipulation | Pandas ≥ 2.0, NumPy ≥ 1.26 |
| Visualisation | Plotly ≥ 5.20 |
| Machine Learning | scikit-learn ≥ 1.4, XGBoost ≥ 2.0 |
| Model Persistence | Joblib ≥ 1.3 |
| Language | Python 3.10+ |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/sales_forecasting_system.git
cd sales_forecasting_system
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Data Setup

This project uses the [**Store Sales — Time Series Forecasting**](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) dataset from Kaggle.

> **Note:** `train.csv`, `test.csv`, and `sample_submission.csv` are excluded from version control due to their large size. You must download them manually.

1. Go to the Kaggle competition page linked above.
2. Download all dataset files.
3. Place them inside the `data/` directory:

```
data/
├── train.csv
├── test.csv
├── sample_submission.csv
├── holidays_events.csv
├── oil.csv
├── stores.csv
└── transactions.csv
```

---

## 🚀 Usage

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Application Sections

| Section | Description |
|---|---|
| **EDA** | Explore sales trends, store types, top families, and promotion impact |
| **Train Model** | Trigger model training and view evaluation metrics (RMSE, R²) |
| **Predict** | Select store/family/date inputs and get a forecasted sales value |

---

## 📁 Environment Variables

This project does not require any API keys or secrets for its core functionality. If you add integrations in the future, create a `.env` file in the project root (it is already excluded by `.gitignore`).

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is for academic and educational purposes.
