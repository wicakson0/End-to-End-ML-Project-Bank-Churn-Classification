.\.venv\Scripts\activate
cd .\prod\
jupyter nbconvert --execute --to notebook --inplace Churn_Classification_Pipeline.ipynb
optuna-dashboard sqlite:///db.sqlite3