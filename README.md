# 📉 Telco Churn Prediction Dashboard

This project provides an end-to-end machine learning pipeline for predicting customer churn using the Telco dataset. It includes preprocessing, feature engineering, model training with hyperparameter tuning, evaluation, and deployment using Streamlit.

---

## 🔧 Features

- Clean modular architecture
- Extensive preprocessing (scaling, binning, encoding, polynomial features)
- Feature selection with `SelectKBest`
- Hyperparameter tuning using `GridSearchCV`
- Models used: RandomForest, XGBoost, LightGBM
- Evaluation metrics (Accuracy, Precision, Recall, F1, AUC)
- Streamlit dashboard with:
  - EDA tab
  - Model evaluation
  - Visualization (Confusion Matrix, ROC Curve)
  - Prediction results and export

---

## 📁 Project Structure
``` still don't know ```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/churn-prediction_v2.git
cd churn-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```


### 3. Run the Pipeline (preprocess, train, save model & scaler)
```bash
python run.py
```

### 4. Launch Streamlit Dashboard
```bash
cd app
streamlit run streamlit_app.py
```

## 📦 Dependencies
pandas

numpy

scikit-learn

xgboost

lightgbm

streamlit

matplotlib

seaborn

joblib

Install them with:

```bash
pip install -r requirements.txt
```

## 📊 Dataset
This project uses the Telco Customer Churn dataset available at:
Kaggle Telco Churn Dataset

## 📌 License
You’re free to use, modify, and share it.

## 💡 Future Improvements
Add cloud deployment (e.g., Hugging Face Space or Heroku)

AutoML support (Optuna or RandomizedSearch)

Handle unseen data schema in uploaded CSV

Real-time monitoring dashboard with Streamlit metrics

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.