import inspect
import numpy as np
from os import path
import optuna
from optuna.trial import Trial
from pycaret.anomaly import AnomalyExperiment
from typing import Dict, List
import MySQLdb

class Suggestion:
    def __init__(self, name:str, dtype:str, low, high, log=False):
        self.name = name
        self.dtype = dtype
        self.low = low
        self.high = high
        self.log = log

    def suggest(self, t:Trial) -> (int|float):
        if self.low == self.high:
            return self.low 
        
        if self.dtype == 'int':
            return t.suggest_int(self.name, self.low, self.high, 1, self.log)
        elif self.dtype == 'float':
            return t.suggest_float(self.name, self.low, self.high, log=self.log)
        else:
            raise RuntimeError(f"dtype {self.dtype} not supported for this job")
            

class FraudDetector:
    __default_model_params = {
        "iforest": [Suggestion('n_estimators', 'int', 50, 1000),
                    Suggestion('max_samples', 'float', 0.1, 1.0),
                    Suggestion('fraction', 'float', 0.001, 0.05, log=True),
                    Suggestion('max_features', 'float', 0.1, 1.0)],
        "lof": [Suggestion('n_neighbors', 'int', 10, 1000),
                Suggestion('leaf_size', 'int', 10, 1000),
                Suggestion('p', 'float', 0.5,  2.0),
                Suggestion('fraction', 'float', 0.001, 0.05, log=True)]
    }
    def __init__(self, study_name:str, init_study:False, random_state = None, model_params: Dict[str, List[Suggestion]] = None) -> None:
        self.study_name = study_name
        if model_params is None:
            self.model_params = FraudDetector.__default_model_params
        else:
            self.model_params = model_params
        if random_state is None:
            self.random_state = np.random.randint(10000)
        else:
            self.random_state = random_state

        if init_study:
            print(f"init study: {study_name}")
            conn = MySQLdb.connect("localhost", "root")
            cursor = conn.cursor()
            cursor.execute("DROP DATABASE IF EXISTS "+study_name)
            print(f"init study, drop table: {study_name}")
            cursor.execute("CREATE DATABASE "+study_name)
            self.study = optuna.create_study(study_name=study_name, storage="mysql://root@localhost/"+study_name)
        else:
            print(f"load study: {study_name}")
            self.study = optuna.load_study(study_name=study_name, storage="mysql://root@localhost/"+study_name)

    def set_objective(self, obj):
        self.objective = obj 

    def set_experiment_session(self, X_data, Y_true):
        self.session = AnomalyExperiment()
        self.session.exp_name_log = "FraudDetection_"+self.study_name
        self.session.setup(X_data)
        print(self.session)
        self.Y_true = Y_true

    def _inspect_sig(self, model_name):
        print(f'inspect input parameters of {model_name}')
        model = self.session.create_model(model_name)
        sig = inspect.signature(model.__init__)
        print(f'input parameters of {model_name} is {sig}')
        return sig.parameters.keys()
    
    def optimize(self, model_name, n_trials = 100):
        if model_name not in self.model_params:
            raise RuntimeError(f"model: {model_name} not a supported model! plz check the name and its spelling")
        model_params = self.model_params[model_name]
        all_params = self._inspect_sig(model_name)
        if 'random_state' in all_params:
            model_name += [Suggestion('random_state', 'int', self.random_state, self.random_state)]
        if 'n_jobs' in all_params:
            model_name += [Suggestion('n_jobs', 'int', -1, -1)]

        if self.session is None:
            raise RuntimeError(f"plz run set_experiment_session first")
        
        if self.objective is None:
            raise RuntimeError(f"plz run set_objective first")
        
        def __objective(trial:Trial):
            params = {s.name:s.suggest(trial) for s in model_params}
            model = self.session.create_model(model_name, 
                           **params)
            print(f"model info: {model}")
            try:
                last_best = trial.study.best_value
            except ValueError:
                last_best = None

            pred_result = self.session.assign_model(model)
            anomaly_pred = pred_result['Anomaly']
            
            curr_metric = self.objective(self.Y_true, anomaly_pred)
            if last_best is None or curr_metric < last_best:
                model_file = f'iforest{trial.number}-{curr_metric}'
                _, model_file = self.session.save_model(model, model_file)
                print(f"save best model at {path.abspath(model_file)}")
            return curr_metric
    
        self.study.optimize(__objective, n_trials=n_trials)


if __name__ == "__main__":
    import pandas as pd
    from sklearn.metrics import fbeta_score

    det = FraudDetector("iforest_test", init_study=True)
    det.set_objective(lambda tr, pred: -fbeta_score(tr, pred, beta=2))


    transaction_data = pd.read_csv("Data/credit_card_transactional_data.csv")
    X_data = transaction_data.drop(["Class", "Timestamp"], axis=1)
    X_data = X_data.fillna(X_data.min()*2)
    Y_true = transaction_data['Class']
    det.set_experiment_session(X_data, Y_true)

    det.optimize("iforest", 100)