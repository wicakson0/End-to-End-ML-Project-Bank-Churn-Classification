import uvicorn
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

from BankCustomer import BankCustomer

app = FastAPI()
model_path = '20240219203143.pkl'
pipeline = joblib.load(model_path)

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
    
    # input_data = [[surname,
    #                credit_score,
    #                geography,
    #                gender,
    #                age,
    #                tenure,
    #                balance,
    #                num_products,
    #                has_cc,
    #                is_active,
    #                estimated_salary]]
    
    # Convert all values to standard Python types
    # input_data = [item if not isinstance(item, np.integer) else int(item) for item in input_data[0]]
    
    # columns = ['Surname',
    #            'CreditScore',
    #            'Geography',
    #            'Gender',
    #            'Age',
    #            'Tenure',
    #            'Balance',
    #            'NumOfProducts',
    #            'HasCrCard',
    #            'IsActiveMember',
    #            'EstimatedSalary']

    # input_df = pd.DataFrame([input_data], columns=columns)
    
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

    # Also convert this to a standard python integer
    return {
        'churned': int(prediction[0])
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# where swagger is at: http://127.0.0.1:8000/docs
# where openapi spec is at: http://127.0.0.1:8000/redoc