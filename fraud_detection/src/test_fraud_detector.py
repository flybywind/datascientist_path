import pytest
from fraud_detector import FraudDetector
from pathlib import Path

proj_dir = Path(__file__).parent.parent
def test_init_iforest():
    import pandas as pd
    from sklearn.metrics import fbeta_score

    det = FraudDetector("iforest_test", init_study=True)
    det.set_objective(lambda tr, pred: -fbeta_score(tr, pred, beta=2))


    transaction_data = pd.read_csv(proj_dir.joinpath("Data/credit_card_transactional_data.csv").as_posix())
    X_data = transaction_data.drop(["Class", "Timestamp"], axis=1)
    X_data = X_data.fillna(X_data.min()*2)
    Y_true = transaction_data['Class']
    det.set_experiment_session(X_data[:1000], Y_true[:1000])

    det.optimize("iforest", 1)


def test_load_iforest():
    import pandas as pd
    from sklearn.metrics import fbeta_score

    det = FraudDetector("iforest_test", init_study=False)
    det.set_objective(lambda tr, pred: -fbeta_score(tr, pred, beta=2))


    transaction_data = pd.read_csv(proj_dir.joinpath("Data/credit_card_transactional_data.csv").as_posix())
    X_data = transaction_data.drop(["Class", "Timestamp"], axis=1)
    X_data = X_data.fillna(X_data.min()*2)
    Y_true = transaction_data['Class']
    det.set_experiment_session(X_data[:1000], Y_true[:1000])

    det.optimize("iforest", 1)

def test_init_lof():
    import pandas as pd
    from sklearn.metrics import fbeta_score

    det = FraudDetector("lof_test", init_study=True)
    det.set_objective(lambda tr, pred: -fbeta_score(tr, pred, beta=2))


    transaction_data = pd.read_csv(proj_dir.joinpath("Data/credit_card_transactional_data.csv").as_posix())
    X_data = transaction_data.drop(["Class", "Timestamp"], axis=1)
    X_data = X_data.fillna(X_data.min()*2)
    Y_true = transaction_data['Class']
    det.set_experiment_session(X_data[:1000], Y_true[:1000])

    det.optimize("lof", 1)

def test_load_lof():
    import pandas as pd
    from sklearn.metrics import fbeta_score

    det = FraudDetector("lof_test", init_study=False)
    det.set_objective(lambda tr, pred: -fbeta_score(tr, pred, beta=2))


    transaction_data = pd.read_csv(proj_dir.joinpath("Data/credit_card_transactional_data.csv").as_posix())
    X_data = transaction_data.drop(["Class", "Timestamp"], axis=1)
    X_data = X_data.fillna(X_data.min()*2)
    Y_true = transaction_data['Class']
    det.set_experiment_session(X_data[:1000], Y_true[:1000])

    det.optimize("lof", 1)