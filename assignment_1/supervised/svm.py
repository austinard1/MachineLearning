import json
import numpy as np

from yellowbrick.classifier import ClassificationReport
from sklearn.svm import SVC
from supervised import plotting
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def svm_learner(X_train, X_test, y_train, y_test, task, params_file):
    svm_learner = SVC()
    svm_learner_plot = plotting.plot_learning_curve(svm_learner, "svm Learner", X_train, y_train, n_jobs=4)
    svm_learner_plot.show()
    svm_learner.fit(X_train, y_train)
    svm_learner_score = svm_learner.score(X_test, y_test)
    print('svm learner score = ' + str(svm_learner_score))

    with open(params_file) as f:
        params = json.load(f)
    tuning_params = params['SVMLearner']['tuning']
    best_params = params['SVMLearner']['best']

    if task == 'grid_search':
        param_grid = {"C": tuning_params['C'],
                "gamma": tuning_params['gamma'],
                "kernel": tuning_params['kernel']}
              #"criterion": ["gini", "entropy"]}
        grid_search = GridSearchCV(svm_learner, param_grid=param_grid, scoring='accuracy', n_jobs=4, cv=10)
        grid_search.fit(X_train, y_train)
        print('Best params chosen by grid search....\n')
        print(grid_search.best_params_)
        grid_best = grid_search.best_estimator_
        predictions = grid_best.predict(X_test)
        class_report = classification_report(y_test, predictions)
        print(class_report)
        svm_learner_plot = plotting.plot_learning_curve(grid_best, "svm Learner (Grid Search)", X_train, y_train, n_jobs=4)
        svm_learner_plot.show()
        #class_plot = plotting.plot_classification_report(class_report)
        svm_learner_tuned = grid_search.best_estimator_.fit(X_train, y_train)
        svm_learner_score = svm_learner_tuned.score(X_test, y_test)
        print('svm learner score (Grid Search) = ' + str(svm_learner_score))

    elif task == 'mca':
        svm_learner_tuned = SVC(n_neighbors=best_params['n_neighbors'], metric=best_params['metric'])
        if len(tuning_params['n_neighbors']) != 1:
            tuning_param = 'n_neighbors'
            tuning_range = range(tuning_params[tuning_param][0], tuning_params[tuning_param][1])
        elif len(tuning_params['metric']) != 1:
            tuning_param = 'metric'
            tuning_range = tuning_params['metric']
        else:
            print('Error, no params have multiple tuning values... exiting')
            return
        title = 'Model Complexity Curve, varying ' + str(tuning_param)
        mcc = plotting.plot_validation_curve(svm_learner_tuned, title, X_train, y_train, param_name=tuning_param, param_range=tuning_range, n_jobs=None)
        mcc.show()

    elif task == 'plot_best':
        svm_learner_tuned = SVC(n_neighbors=best_params['n_neighbors'], metric=best_params['metric'])
        svm_learner_tuned.fit(X_train, y_train)
        predictions = svm_learner_tuned.predict(X_test)
        class_report = classification_report(y_test,predictions)
        print(class_report)
        #class_plot = plot_classification_report(class_report)
        #class_plot.show()
        svm_learner_score = svm_learner_tuned.score(X_test, y_test)
        print('svm learner best score = ' + str(svm_learner_score))
        svm_learner_plot_tuned = plotting.plot_learning_curve(svm_learner_tuned, "svm Learner (tuned)", X_train, y_train, n_jobs=4)
        svm_learner_plot_tuned.show()

    else:
        print('Invalid task')
        return

    #plotting.plot_confusion_matrix(svm_learner_tuned, X_test, y_test)
    #svm_learner_plot_tuned = plotting.plot_learning_curve(svm_learner_tuned, "svm Learner (tuned)", X_train, y_train, n_jobs=4)
    #svm_learner_plot_tuned.show()

    return