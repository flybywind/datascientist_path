import argparse
import pandas as pd
from sklearn.metrics import fbeta_score

from fraud_detector import FraudDetector

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--init_study", action="store_true", dest="init_study", default=False)
parser.add_argument("-m", "--model_name", dest="model_name", type=str)
parser.add_argument("-n", "--n_trial", dest="n_trial", type=int, default=100)

args = parser.parse_args()
print(f"arguments: {args}")

det = FraudDetector(args.model_name+"_study", init_study=args.init_study)
det.set_objective(lambda tr, pred: -fbeta_score(tr, pred, beta=2))

transaction_data = pd.read_csv("Data/credit_card_transactional_data.csv")
X_data = transaction_data.drop(["Class", "Timestamp"], axis=1)
Y_true = transaction_data['Class']

if args.model_name == 'iforest':
    X_data = X_data.fillna(X_data.min()*2)
    det.set_experiment_session(X_data, Y_true)
else:
    # normalize data:
    norm_data = (X_data - X_data.mean()) /2/X_data.std()
    X_data = norm_data.fillna(norm_data.min()*2)

det.set_experiment_session(X_data, Y_true)
det.optimize(args.model_name, args.n_trial)