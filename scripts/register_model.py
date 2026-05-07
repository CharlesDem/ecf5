from churnguard.train import train_model, promotion_to_laprod
from churnguard.data import load_data, preprocess
from download_data import download

print("start train")

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


print("Download du dataset Telco")
download()

print("Chargement des données dans data")
df_to_test = load_data('data/telco_churn.csv')

print("Preprocessing")
X, y = preprocess(df_to_test)


print("Entrainement des trois modèles")
for model_name, params in runs.items():
    train_model(X, y, model_name, params)


print("Tag du meilleur modèle")
promotion_to_laprod()


print("Done")

