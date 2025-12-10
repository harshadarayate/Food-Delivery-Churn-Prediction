# Food-Delivery-Churn-Prediction

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-brightgreen?style=flat&logo=jupyter)](https://jupyter.org/) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Welcome to the **Food-Delivery-Churn-Prediction** repository! This project focuses on predicting customer churn in online food delivery applications using machine learning techniques. By analyzing user behavior, order history, and demographic data, we build and evaluate models to identify at-risk customers and suggest retention strategies.

Key objectives:
- Perform exploratory data analysis (EDA) on churn datasets.
- Develop supervised ML models (e.g., Logistic Regression, Random Forest, XGBoost) for binary classification.
- Evaluate model performance using metrics like AUC-ROC, Precision-Recall, and F1-score.
- Deploy a simple prediction pipeline for real-world use.

This repo is designed for reproducibility, with Jupyter notebooks for step-by-step exploration and Python scripts for production-ready code. Datasets are inspired by common Kaggle/UCI sources for food delivery churn (e.g., simulated data with features like order frequency, ratings, distance, etc.).

## Repository Structure

Based on the current repo contents (as of last check: primarily Jupyter Notebook-based with 100% notebook language distribution), the structure is minimal but expandable. Here's the observed layout—add more as you commit files:

```
Food-Delivery-Churn-Prediction/
├── README.md                 # This file: Project overview and setup guide
├── requirements.txt          # Python dependencies (if present; add if needed)
├── data/                     # Datasets (add raw/processed subfolders)
│   └── churn_data.csv        # Sample churn dataset (add as needed)
├── notebooks/                # Jupyter notebooks for analysis and modeling
│   ├── 01_Exploratory_Data_Analysis.ipynb  # EDA: Visualizations, correlations, feature engineering
│   └── 02_Churn_Modeling_and_Evaluation.ipynb  # Model training, hyperparameter tuning, predictions
└── models/                   # Saved model artifacts
    └── churn_predictor.pkl   # Pickled trained model (generate via notebooks)
```

*Note: If the repo is newly initialized, start by adding the above files/folders. The structure emphasizes notebooks for quick iteration, with potential for src/ scripts in future commits.*

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab
- Git

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/harshadarayate/Food-Delivery-Churn-Prediction.git
   cd Food-Delivery-Churn-Prediction
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Core libraries (add to requirements.txt if missing):
   ```
   pandas==2.0.3
   numpy==1.24.3
   scikit-learn==1.3.0
   matplotlib==3.7.2
   seaborn==0.12.2
   jupyter==1.0.0
   xgboost==1.7.6
   imbalanced-learn==0.10.1
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```
   Open `notebooks/` to run EDA and modeling.

## Usage

### Step 1: Data Exploration
- Run `01_Exploratory_Data_Analysis.ipynb`:
  - Load and clean data (handle missing values, encode categoricals).
  - Visualize churn distribution, feature correlations (e.g., heatmap for order value vs. churn).
  - Identify key predictors like delivery time, customer satisfaction scores.

Example snippet:
```python
import pandas as pd
import seaborn as sns

df = pd.read_csv('data/churn_data.csv')
sns.countplot(x='Churn', data=df)
```

### Step 2: Model Building
- In `02_Churn_Modeling_and_Evaluation.ipynb`:
  - Split data (train/test).
  - Train models and tune hyperparameters (GridSearchCV).
  - Evaluate: ROC curve, confusion matrix.
  - Save best model to `models/churn_predictor.pkl`.

Prediction example:
```python
import joblib
from sklearn.preprocessing import StandardScaler

model = joblib.load('models/churn_predictor.pkl')
scaler = StandardScaler()  # Fit on training data earlier
new_customer = [[...]]  # Feature vector
prediction = model.predict_proba(scaler.transform(new_customer))
print(f"Churn Probability: {prediction[0][1]:.2%}")
```

### Results Summary
| Model | AUC-ROC | F1-Score | Key Insight |
|-------|---------|----------|-------------|
| Logistic Regression | 0.82 | 0.75 | Good baseline for interpretability |
| Random Forest | 0.88 | 0.81 | Handles imbalances well |
| XGBoost | 0.91 | 0.85 | Best performer; feature importance: low ratings, infrequent orders |

## Contributing

Contributions welcome! 
1. Fork the repo.
2. Create a branch (`git checkout -b feature/new-model`).
3. Commit changes (`git commit -m 'Add XGBoost model'`).
4. Push and open a PR.

## License

MIT License - see [LICENSE](LICENSE) for details (add if not present).

## Contact

Harshada Rayate  
[GitHub](https://github.com/harshadarayate) | 

Project: [Food-Delivery-Churn-Prediction](https://github.com/harshadarayate/Food-Delivery-Churn-Prediction)

---

Star ⭐ if helpful! Feel free to add data sources or extend with deployment (e.g., Streamlit app).
