import json
import numpy as np

from time import process_time
from yellowbrick.classifier import ClassificationReport
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from supervised import plotting
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.evaluate import bias_variance_decomp

def boosting_learner(X_train, X_test, y_train, y_test, task, params_file):
    boosting_learner = AdaBoostClassifier(learning_rate=1, n_estimators=1)
    boosting_learner_plot = plotting.plot_learning_curve(boosting_learner, "Boosting Learner (default parameters)", X_train, y_train, n_jobs=4)
    boosting_learner_plot.show()
    boosting_learner.fit(X_train, y_train)
    boosting_learner_score = boosting_learner.score(X_test, y_test)
    print('Boosting learner score = ' + str(boosting_learner_score))
    conf_mat = plotting.plot_conf_mat(boosting_learner, X_test, y_test, "Boosting Learner Confusion Matrix (default)")
    conf_mat.show()

    with open(params_file) as f:
        params = json.load(f)
    tuning_params = params['BoostingLearner']['tuning']
    best_params = params['BoostingLearner']['best']

    if task == 'grid_search':
        param_grid = {"n_estimators": tuning_params['n_estimators'],
                "learning_rate": tuning_params['learning_rate']}
        grid_search = GridSearchCV(boosting_learner, param_grid=param_grid, scoring='accuracy', n_jobs=4, cv=10)
        grid_search.fit(X_train, y_train)
        print('Best params chosen by grid search....\n')
        print(grid_search.best_params_)
        grid_best = grid_search.best_estimator_
        predictions = grid_best.predict(X_test)
        class_report = classification_report(y_test, predictions)
        print(class_report)
        boosting_learner_plot = plotting.plot_learning_curve(grid_best, "Boosting Learner (Grid Search) " + str(grid_search.best_params_), X_train, y_train, n_jobs=4)
        boosting_learner_plot.show()
        #class_plot = plotting.plot_classification_report(class_report)
        boosting_learner_tuned = grid_search.best_estimator_.fit(X_train, y_train)
        boosting_learner_score = boosting_learner_tuned.score(X_test, y_test)
        print('Boosting learner score (Grid Search) = ' + str(boosting_learner_score))

        conf_mat = plotting.plot_conf_mat(boosting_learner_tuned, X_test, y_test, "Boosting Learner Confusion Matrix (Grid Search)")
        conf_mat.show()

    elif task == 'mca':
        boosting_learner_tuned = AdaBoostClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'])
        if len(tuning_params['n_estimators']) != 1:
            tuning_param = 'n_estimators'
            tuning_range = range(tuning_params[tuning_param][0], tuning_params[tuning_param][1])
            set_param = 'learning_rate'
        elif len(tuning_params['learning_rate']) != 1:
            tuning_param = 'learning_rate'
            tuning_range = np.linspace(tuning_params[tuning_param][0], tuning_params[tuning_param][1])
            set_param = 'n_estimators'
        else:
            print('Error, no params have multiple tuning values... exiting')
            return
        title = 'Model Complexity Curve, ' + str(set_param) + '=' + str(best_params[set_param]) + ' varying ' + str(tuning_param)
        mcc = plotting.plot_validation_curve(boosting_learner_tuned, title, X_train, y_train, param_name=tuning_param, param_range=tuning_range, n_jobs=None)
        mcc.show()

    elif task == 'plot_best':
        boosting_learner_tuned = AdaBoostClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'])
        fit_time_start = process_time()
        boosting_learner_tuned.fit(X_train, y_train)
        fit_time_end = process_time()
        print(' Neural Network Learner (tuned) fit time: ' + str(fit_time_end-fit_time_start))

        predict_time_start = process_time()
        predictions = boosting_learner_tuned.predict(X_test)
        predict_time_end = process_time()
        print(' Boosting Learner (tuned) predict time: ' + str(predict_time_end-predict_time_start))

        class_report = classification_report(y_test,predictions)
        print(class_report)
        #class_plot = plot_classification_report(class_report)
        #class_plot.show()
        boosting_learner_tuned_score = boosting_learner_tuned.score(X_test, y_test)
        print('Boosting learner best score = ' + str(boosting_learner_tuned_score))
        boosting_learner_plot_tuned = plotting.plot_learning_curve(boosting_learner_tuned, "Boosting Learner (tuned) " + str(best_params), X_train, y_train, n_jobs=4)
        boosting_learner_plot_tuned.show()

        conf_mat = plotting.plot_conf_mat(boosting_learner_tuned, X_test, y_test, "Boosting Learner Confusion Matrix (tuned)")
        conf_mat.show()

        # Taken from https://machinelearningmastery.com/calculate-the-bias-variance-trade-off/
        # estimate bias and variance
        #mse_tuned_default, bias_default, var_default = bias_variance_decomp(boosting_learner, np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), loss='mse', num_rounds=200, random_seed=27)
        #mse_tuned, bias_tuned, var_tuned = bias_variance_decomp(boosting_learner_tuned, np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), loss='mse', num_rounds=200, random_seed=27)
        # summarize results
        #print('MSE: %.3f' % mse)
        #print('Bias: %.3f' % bias)
        #print('Variance: %.3f' % var)

    else:
        print('Invalid task')
        return

    #plotting.plot_confusion_matrix(boosting_learner_tuned, X_test, y_test)
    #boosting_learner_plot_tuned = plotting.plot_learning_curve(boosting_learner_tuned, "boosting Learner (tuned)", X_train, y_train, n_jobs=4)
    #boosting_learner_plot_tuned.show()

    return