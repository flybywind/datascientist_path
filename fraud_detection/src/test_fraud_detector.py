import pytest
from fraud_detector import FraudDetector
from pathlib import Path
import pandas as pd
from sklearn.metrics import fbeta_score


proj_dir = Path(__file__).parent.parent

def construct_test(model_name, init=False):
    det = FraudDetector(model_name+"_test", init_study=init)
    det.set_objective(lambda tr, pred: -fbeta_score(tr, pred, beta=2))


    transaction_data = pd.read_csv(proj_dir.joinpath("Data/credit_card_transactional_data.csv").as_posix())
    X_data = transaction_data.drop(["Class", "Timestamp"], axis=1)
    X_data = X_data.fillna(X_data.min()*2)
    Y_true = transaction_data['Class']
    det.set_experiment_session(X_data[:1000], Y_true[:1000])

    det.optimize(model_name, 1)

def test_init_iforest():
    construct_test('iforest', True)

def test_load_iforest():
    construct_test('iforest', False)

def test_init_lof():
    construct_test('lof', True)

def test_load_lof():
    construct_test('lof', False)

def test_init_sod():
    construct_test('sod', True)

def test_load_sod():
    construct_test('sod', False)