# app.py
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            form = request.form

            def to_float(val, default=None):
                s = str(val).strip()
                if s == "" or s.lower() == "none":
                    return default
                return float(s)

            monthly_charges = to_float(form.get("MonthlyCharges", ""), default=0.0)
            total_charges   = to_float(form.get("TotalCharges", ""),  default=0.0)  # let preprocessor fill if 0.0

            data = CustomData(
                customerID=form.get("customerID","Dummy"),
                gender=form.get("gender","Male"),
                SeniorCitizen=int(form.get("SeniorCitizen",0)),
                Partner=form.get("Partner","No"),
                Dependents=form.get("Dependents","No"),
                tenure=int(form.get("tenure", 12)),
                PhoneService=form.get("PhoneService","Yes"),
                MultipleLines=form.get("MultipleLines","No"),
                InternetService=form.get("InternetService","Fiber optic"),
                OnlineSecurity=form.get("OnlineSecurity","No"),
                OnlineBackup=form.get("OnlineBackup","No"),
                DeviceProtection=form.get("DeviceProtection","No"),
                TechSupport=form.get("TechSupport","No"),
                StreamingTV=form.get("StreamingTV","No"),
                StreamingMovies=form.get("StreamingMovies","No"),
                Contract=form.get("Contract","Month-to-month"),
                PaperlessBilling=form.get("PaperlessBilling","Yes"),
                PaymentMethod=form.get("PaymentMethod","Electronic check"),
                MonthlyCharges=monthly_charges,
                TotalCharges=total_charges
            )

            df = data.get_data_as_frame()
            pipe = PredictPipeline()
            preds, probs = pipe.predict(df)

            prob_txt  = f"{float(probs[0]):.2%}" if probs is not None else "N/A"
            label_txt = int(preds[0])

            return render_template('home.html', results=prob_txt, label=label_txt)

        except Exception as e:
            # show the actual error on the page so you’re not guessing
            return render_template('home.html', error=str(e))

    

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8001)


