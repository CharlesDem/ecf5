from churnguard.train import train_model
from churnguard.data import load_data, preprocess

runs = {
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators":200, 
        "max_depth": 10, 
        "random_state": 42, 
        "n_jobs": -1
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42,
    }
}

df_to_test = load_data('data/telco_churn_test.csv')
X, y = preprocess(df_to_test)

for model_name, params in runs.items():
    train_model(X, y, model_name, params)

