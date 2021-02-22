print(__doc__)

import sys, getopt

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from supervised import decision_tree, boosting, knn, neural_network, svm

def get_inputs():
    """
    Extract inputs from command line
    Returns:
        args (argparse.NameSpace): Argparse namespace object
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-g", "--grid-search", help="Perform grid search", action="store_true",
    )
    parser.add_argument(
        "-l", "--learner", choices=['dt', 'boosting', 'knn', 'neural', 'svm'], help="Learner to run",
    )
    parser.add_argument(
        "-t", "--task", choices=['grid_search', 'plot_best', 'mca'], help="Task to perform with specified learner"
    )
    return parser.parse_args()

def preprocess_inputs(df, task='classification'):
    df = df.copy()
    scaler = StandardScaler()
    #normalizer = Normalizer()

    if task == 'classification':
        df = df.drop('stab', axis=1)

        y = df['stabf'].copy()
        X = df.drop('stabf', axis=1).copy()

    elif task == 'regression':
        df = df.drop('stabf', axis=1)

        y = df['stab'].copy()
        X = df.drop('stab', axis=1).copy()


    # Derived from https://stackoverflow.com/questions/29438265/stratified-train-test-split-in-scikit-learn
    #splitter = StratifiedShuffleSplit(test_size=0.2, random_state=27)
    #for train_index, test_index in splitter.split(X, y):
    #    X_train, X_test = X[train_index], X[test_index]
    #    y_train, y_test = y[train_index], y[test_index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=27, stratify=y)


    #train_scaler = StandardScaler()
    train_scaler = RobustScaler()
    X_train_scaler = train_scaler.fit(X_train)
    X_train = X_train_scaler.transform(X_train)

    X_test = X_train_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def main():
    args = get_inputs()
    data = pd.read_csv('smart_grid_2.csv')
    params_file = 'smart_grid_params.json'
    X_train, X_test, y_train, y_test = preprocess_inputs(data)
    print('asdf')
    if args.learner == 'dt':
        print('asdf')
        decision_tree.decision_tree_learner(X_train, X_test, y_train, y_test, args.task, params_file)
    elif args.learner == 'boosting':
        boosting.boosting_learner(X_train, X_test, y_train, y_test, args.task, params_file)
    elif args.learner == 'knn':
        knn.knn_learner(X_train, X_test, y_train, y_test, args.task, params_file)
    elif args.learner == 'neural':
        neural_network.neural_learner(X_train, X_test, y_train, y_test, args.task, params_file)
    elif args.learner == 'svm':
        svm.svm_learner(X_train, X_test, y_train, y_test, args.task, params_file)
    else:
        print('Invalid learner input, run with -h flag for options')
    return

if __name__ == "__main__":
    main()