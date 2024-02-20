import uvicorn
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import glob

from BankCustomer import BankCustomer

app = FastAPI()

# will search automatically for pkl files and uses the latest one
pkl_files = glob.glob('*.pkl')
pkl_files.sort(reverse=True)

pipeline = joblib.load(pkl_files[0])


@app.get('/')
def index():
    return {'message': 'bank customer churn classification API'}


@app.post('/churn/predict')
def predict_customer_churn(data: BankCustomer):
    data = data.dict()
    surname = data['Surname']
    credit_score = data['CreditScore']
    geography = data['Geography']
    gender = data['Gender']
    age = data['Age']
    tenure = data['Tenure']
    balance = data['Balance']
    num_products = data['NumOfProducts']
    has_cc = data['HasCrCard']
    is_active = data['IsActiveMember']
    estimated_salary = data['EstimatedSalary']

    # Convert NumPy int64 objects to standard Python integers
    credit_score = int(credit_score)
    tenure = int(tenure)
    balance = int(balance)
    num_products = int(num_products)
    has_cc = int(has_cc)
    is_active = int(is_active)

    input_dict = {
        'Surname': [surname],
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [has_cc],
        'IsActiveMember': [is_active],
        'EstimatedSalary': [estimated_salary]
    }

    input_df = pd.DataFrame(input_dict)

    prediction = pipeline.predict(input_df)

    # Also convert output from np.int64 to standard python int
    return {
        'churned': int(prediction[0])
    }

# where swagger is at: http://127.0.0.1:8000/docs
# where openapi spec is at: http://127.0.0.1:8000/redoc

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
