from os import path
import optuna
from optuna.trial import Trial
import pandas as pd
from pycaret.anomaly import AnomalyExperiment
from sklearn.metrics import fbeta_score
import MySQLdb


transaction_data = pd.read_csv(r"../Data/credit_card_transactional_data.csv")
X_data = transaction_data.drop(["Class", "Timestamp"], axis=1)
X_data = X_data.fillna(X_data.min()*2)
Y_true = transaction_data['Class']

fraud_ses = AnomalyExperiment()
fraud_ses.exp_name_log = "FraudDetection"
fraud_ses.setup(X_data)

def objective(trial:Trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_samples = trial.suggest_float('max_samples', 0.1, 1.0)
    contamination = trial.suggest_float('contamination', 0.001, 0.05, log=True)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)
    model = fraud_ses.create_model("iforest", n_estimators = n_estimators,
                           max_samples = max_samples,
                           max_features = max_features,
                           random_state=3490,
                           fraction = contamination, # pay attention here!
                           n_jobs=-1)
    print(f"model info: {model}")
    last_best = trial.study.best_value
    pred_result = fraud_ses.assign_model(model)
    anomaly_pred = pred_result['Anomaly']
    
    curr_metric = -fbeta_score(Y_true, anomaly_pred, beta=2)
    if curr_metric < last_best:
        model_file = f'iforest{trial.number}-{curr_metric}.pkl'
        fraud_ses.save_model(model, model_file)
        print(f"save best model at {path.abspath(model_file)}")
    return curr_metric
    
study = optuna.load_study(
        study_name="project-pro", storage="mysql://root@localhost/pp_iforest"
    )
study.optimize(objective, n_trials=100)