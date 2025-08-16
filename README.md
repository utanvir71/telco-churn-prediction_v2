![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-ScikitLearn%2FXGBoost-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Ready-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)
# 📉 Telco Churn Prediction Dashboard

End-to-end machine learning pipeline for predicting customer churn using the Telco dataset.
Includes preprocessing, model training, evaluation, and deployment via a Streamlit dashboard.

---
## 🎥 Demo
![Streamlit Dashboard Demo](assets/demo.gif)


## 🔧 Features

- ⚡ Modular pipeline architecture (clean, reusable code)
- 🧹 Preprocessing & feature engineering:
  - Missing value handling
  - Scaling, binning, one-hot encoding
  - Polynomial & interaction features
  - Feature selection with SelectKBest
- 🤖 Model training with hyperparameter tuning:
  - LogisticRegression
  - RandomForest
  - GradientBoostingClassifer
  - AdaBoostClassifier
  - SVC
  - KNeighboursClassifier
  - DecisionTreeClassifier
  - GuassianNB
  - MLPClassifier
  - XGBoost
- 📊 Evaluation metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- 🎛️ Interactive Streamlit dashboard:
  - Exploratory Data Analysis (EDA)
  - Model evaluation & visualization (Confusion Matrix, ROC Curve)
  - Real-time churn predictions
  - CSV export of predictions

---

## 📁 Project Structure
``` 
churn-prediction-2/
├── artifact/                     # Stores intermediate artifacts
├── Chun_prediction.egg-info/     # Metadata for packaging
├── logs/                         # Log files
├── ml_venv/                      # Virtual environment (should be gitignored)
├── notebook/                     # Jupyter notebooks
│   ├── data/
│   │   └── telco_churn.csv       # Dataset
│   ├── eda.ipynb                 # Exploratory Data Analysis
│   └── MODEL TRAINING.ipynb      # Model training notebook
├── src/                          # Source code
│   ├── components/               # Core components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── __init__.py
│   ├── pipeline/                 # Training & prediction pipelines
│   │   ├── train_pipeline.py
│   │   ├── predict_pipeline.py
│   │   └── __init__.py
│   ├── exception.py              # Custom exception handling
│   ├── logger.py                 # Logging utility
│   ├── utils.py                  # Helper functions
│   └── __init__.py
├── templates/                    # Flask HTML templates
│   ├── home.html
│   └── index.html
├── .gitignore
├── .python-version
├── app.py                        # Flask entrypoint
├── streamlit_app.py              # Streamlit dashboard
├── requirements.txt              # Dependencies
├── setup.py                      # Setup for packaging
└── README.md                     # Project documentation
 ```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/churn-prediction_v2.git
cd churn-prediction_v2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```


### 3. Reword install & run for clarity
```bash
python -m src.pipeline.train_pipeline
```
### 4. Launch Streamlit Dashboard
```
streamlit run streamlit_app.py
```


## 📦 Dependencies
pandas
matplotlib
seaborn
numpy
scikit-learn==1.6.1
imblearn
xgboost
dill
Flask
streamlit


## 📊 Dataset
This project uses the Telco Customer Churn dataset available at:
Kaggle Telco Churn Dataset (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 📌 License
MIT - You’re free to use, modify, and share it.

## 💡 Future Improvements
- ☁️ Cloud deployment (Hugging Face, AWS, Heroku)
- 🔎 AutoML (Optuna / RandomizedSearch)
- 📂 Handle unseen schema in uploaded CSVs
- 📈 Real-time monitoring dashboard with Streamlit metrics

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---
🔗 **Connect with me:** [LinkedIn](https://linkedin.com/in/your-link) | [Twitter](https://twitter.com/your-handle)
