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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from supervised import decision_tree_learner

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Supervised Learning - Main script to run pipeline"""
import argparse
import datetime
import json
import logging
import os
import shutil

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

def preprocess_inputs(survey, task='classification'):
    #Extracting the feature attributes from dataset
    X = survey.drop(columns = ['Contraceptive Method Used'])
    #Extracting the target(label) attributes from dataset
    y = survey['Contraceptive Method Used']

    #columnTransformer = ColumnTransformer([('Standardizer', StandardScaler(), ['Age','Number of Children'])],
    #                                    remainder='passthrough')
    #X = columnTransformer.fit_transform(X)
    # Standardize and
    #scaler = StandardScaler()
    #X_fit_scaler = scaler.fit(X)
    #X = X_fit_scaler.transform(X)

    #X_fit_norm = normalize(X, axis=0)
    #StratifiedShuffleSplit.split(X_fit_norm, y, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    train_scaler = StandardScaler()
    X_train_scaler = train_scaler.fit(X_train)
    X_train = X_train_scaler.transform(X_train)

    X_test = X_train_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def svc_learner(data):
    X_train, X_test, y_train, y_test = preprocess_inputs(data, task='classification')
    svc_learner = SVC(decision_function_shape='ovo', class_weight='balanced', kernel='linear')
    svc_learner_plot = plot_learning_curve(svc_learner, "SVC Learner", X_train, y_train, n_jobs=4)
    svc_learner_plot.show()
    svc_learner.fit(X_train, y_train)
    svc_score = svc_learner.score(X_test, y_test)
    print('SVC learner score = ' + str(svc_score))
    gridsearch = True
    if gridsearch:
        #param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']}
        param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'tol': [1e-4, 1e-3, 1e-2, 1e-1], 'kernel': ['linear']}
        #param_grid = {'C': [10, 100, 1000], 'gamma': [1, 10, 100, 1000],'kernel': ['rbf', 'poly']}
        #parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], 'C': [10, 100, 1000, 10000, 100000]},
        #               {'kernel': ['poly'], 'degree': [1, 2, 3, 4, 5], 'C': [10, 100, 1000, 10000]}]
        #parameters = [{'kernel': ['rbf'], 'gamma': [1e-5], 'C': [1, 10]}]
        grid_search = GridSearchCV(svc_learner, param_grid=param_grid, scoring='accuracy', n_jobs=4, cv=10, verbose=2)
        grid_search.fit(X_train, y_train)
        print('Best estimator chosen by grid search....\n')
        print(grid_search.best_estimator_)
        grid_predictions = grid_search.predict(X_test)
        print(confusion_matrix(y_test,grid_predictions))
        print(classification_report(y_test,grid_predictions))
        #tuned_params = {'C': 1, 'kernel': 'rbf', 'gamma': 1e-5}
        svc_learner_tuned = grid_search.best_estimator_.fit(X_train, y_train)
        svc_learner_plot_tuned = plot_learning_curve(svc_learner_tuned, "SVC Learner (Grid Search)", X_train, y_train, n_jobs=4)
        svc_learner_plot_tuned.show()
        svc_tuned_score = svc_learner_tuned.score(X_test, y_test)
        print('SVC Tuned learner score = ' + str(svc_tuned_score))
        return
    else:
        #tuned_params = {'C': 1, 'kernel': 'rbf', 'gamma': 0.1}
        #tuned_params = {'C': 0.4, 'kernel': 'rbf', 'gamma': 0.1}
        #tuned_params = {'C': 0.4, 'kernel': 'rbf', 'gamma': 0.02}
        #tuned_params = {'C': 0.2, 'kernel': 'rbf', 'gamma': 0.02}
        #tuned_params = {'C': 0.2, 'kernel': 'rbf', 'gamma': 0.2}
        #tuned_params = {'C': 1.3, 'kernel': 'rbf', 'gamma': 0.2}
        #tuned_params = {'C': 5, 'kernel': 'rbf', 'gamma': 0.001}
        #tuned_params = {'C': 20, 'kernel': 'rbf', 'gamma': 0.001}
        tuned_params = {'C': 1.2, 'kernel': 'rbf', 'gamma': 0.03}

        #tuned_params = {'C': 100, 'kernel': 'rbf', 'gamma': 0.1}
        svc_learner_tuned = SVC(C=tuned_params.get('C'), kernel=tuned_params.get('kernel'), gamma=tuned_params.get('gamma'))
        svc_learner_tuned.fit(X_train, y_train)

    svc_learner_plot_tuned = plot_learning_curve(svc_learner_tuned, "SVC Learner", X_train, y_train, n_jobs=4, cv=10)
    svc_learner_plot_tuned.show()
    svc_tuned_score = svc_learner_tuned.score(X_test, y_test)
    print('SVC Tuned learner score = ' + str(svc_tuned_score))

    tune_C = True

    if tune_C:
        #hyper_param = np.linspace(0.1, 5, 100)
        #hyper_param = np.linspace(0.05, 3, 100)
        #hyper_param = np.linspace(0.1, 5, 100)
        #hyper_param = np.linspace(1, 20)
        hyper_param = np.linspace(1, 10, 100)
        train_score, test_score = validation_curve(svc_learner_tuned, X_train, y_train,
                                        param_name='C', param_range=hyper_param, scoring='accuracy', n_jobs=4)
        label = 'C'
    else:
        #hyper_param = np.linspace(0.05, 1)
        #hyper_param = np.linspace(0.01, 5)
        #hyper_param = np.linspace(0.01, 2)
        hyper_param = np.linspace(0.001, 1, 100)
        train_score, test_score = validation_curve(svc_learner_tuned, X_train, y_train,
                                                param_name='gamma', param_range=hyper_param, scoring='accuracy', n_jobs=4)
        label = 'gamma'

    fig = plot.plot_validation_curve(
        svc_learner_tuned,
        'SVC Tuned Learner MCC',
        X_train,
        self.data.y_train,
        param_name=param_name,
        param_range=param_range,
        n_jobs=self.num_cores,
    )
    plt.plot(hyper_param, np.median(train_score, 1), color='blue', label='training score')
    plt.plot(hyper_param, np.median(test_score, 1), color='red', label='testing score')
    plt.legend(loc='best')
    #plt.xscale('log')
    plt.xlabel(label)
    plt.ylabel('score')
    plt.show()

    # for degree in range(1,8):
    #     print('SVM learning.... degree = ' + str(degree))
    #     print('\n')
    #     svc_learner = SVC(gamma='auto', degree=degree)
    #     start = timer()
    #     svc_learner = svc_learner.fit(X_train, y_train)
    #     end = timer()
    #     fit_time = end - start
    #     print('SVC fit time = ' + str(fit_time) + ' seconds')
    #     test_accuracy = svc_learner.score(X_test, y_test)
    #     print('SVC Test set accuracy = '+ str(test_accuracy))
    #     crossvalscore = cross_val_score(svc_learner, X_train, y_train, fit_params={gamma: 'auto', degree: degree})
    #     print('Cross val score = ')
    #     print(crossvalscore)
    #     print('\n')
    #     print('\n')

def main():
    args = get_inputs()
    data = pd.read_csv('contraceptive.csv')
    X_train, X_test, y_train, y_test = preprocess_inputs(data)
    print('asdf')
    if args.learner == 'dt':
        print('asdf')
        decision_tree_learner.decision_tree_learner(X_train, X_test, y_train, y_test, args.task)
    elif args.learner == 'boosting':
        boosting.boosting_learner(X_train, X_test, y_train, y_test, args.task)
    elif args.learner == 'knn':
        svc_learner(data)
    elif args.learner == 'nn':
        svc_learner(data)
    elif args.learner == 'svc':
        svc_learner(data)

if __name__ == "__main__":
    main()