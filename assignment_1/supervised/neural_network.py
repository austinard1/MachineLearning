import json
import numpy as np

import matplotlib.pyplot as plt
from time import process_time
from yellowbrick.classifier import ClassificationReport
from sklearn.neural_network import MLPClassifier
from supervised import plotting
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def neural_learner(X_train, X_test, y_train, y_test, task, params_file):
    neural_learner = MLPClassifier(hidden_layer_sizes=[10], max_iter = 100, random_state=27)
    neural_learner_plot = plotting.plot_learning_curve(neural_learner, "Neural Network Learner (default parameters)", X_train, y_train, n_jobs=4)
    neural_learner_plot.show()
    neural_learner.fit(X_train, y_train)
    neural_learner_score = neural_learner.score(X_test, y_test)
    print('neural learner score = ' + str(neural_learner_score))
    conf_mat = plotting.plot_conf_mat(neural_learner, X_test, y_test, "Neural Network Learner Confusion Matrix (default)")
    conf_mat.show()

    with open(params_file) as f:
        params = json.load(f)
    tuning_params = params['NeuralLearner']['tuning']
    best_params = params['NeuralLearner']['best']

    if task == 'grid_search':
        param_grid = {"max_iter": tuning_params['max_iter'],
                "hidden_layer_sizes": tuning_params['hidden_layer_sizes']}

        grid_search = GridSearchCV(neural_learner, param_grid=param_grid, scoring='accuracy', n_jobs=4, cv=10)
        grid_search.fit(X_train, y_train)
        print('Best params chosen by grid search....\n')
        print(grid_search.best_params_)
        grid_best = grid_search.best_estimator_
        predictions = grid_best.predict(X_test)
        class_report = classification_report(y_test, predictions)
        print(class_report)
        neural_learner_plot = plotting.plot_learning_curve(grid_best, "Neural Network Learner (Grid Search)" + str(grid_search.best_params_), X_train, y_train, n_jobs=4)
        neural_learner_plot.show()
        #class_plot = plotting.plot_classification_report(class_report)
        neural_learner_tuned = grid_search.best_estimator_.fit(X_train, y_train)
        neural_learner_score = neural_learner_tuned.score(X_test, y_test)
        print('Neural Network learner score (Grid Search) = ' + str(neural_learner_score))

        conf_mat = plotting.plot_conf_mat(neural_learner_tuned, X_test, y_test, "Neural Network Learner Confusion Matrix (Grid Search)")
        conf_mat.show()

    elif task == 'mca':
        neural_learner_tuned = MLPClassifier(hidden_layer_sizes=best_params['hidden_layer_sizes'], max_iter=best_params['max_iter'], random_state=27)
        neural_learner_tuned.fit(X_train, y_train)
        if len(tuning_params['hidden_layer_sizes']) != 1:
            tuning_param = 'hidden_layer_sizes'
            tuning_range = tuning_params[tuning_param]
            set_param = 'max_iter'
        elif len(tuning_params['max_iter']) != 1:
            tuning_param = 'max_iter'
            tuning_range = np.arange(tuning_params[tuning_param][0], tuning_params[tuning_param][1], 10)
            set_param = 'hidden_layer_sizes'
        else:
            print('Error, no params have multiple tuning values... exiting')
            return
        title = 'Model Complexity Curve, ' + str(set_param) + '=' + str(best_params[set_param]) + ' varying ' + str(tuning_param)
        mcc = plotting.plot_validation_curve(neural_learner_tuned, title, X_train, y_train, param_name=tuning_param, param_range=tuning_range, n_jobs=4)
        mcc.show()

    elif task == 'plot_best':
        neural_learner_tuned = MLPClassifier(hidden_layer_sizes=best_params['hidden_layer_sizes'], max_iter=best_params['max_iter'], random_state=27)
        fit_time_start = process_time()
        neural_learner_tuned.fit(X_train, y_train)
        fit_time_end = process_time()
        print(' Neural Network Learner (tuned) fit time: ' + str(fit_time_end-fit_time_start))

        predict_time_start = process_time()
        predictions = neural_learner_tuned.predict(X_test)
        predict_time_end = process_time()

        print(' Neural Network Learner (tuned) predict time: ' + str(predict_time_end-predict_time_start))
        predictions = neural_learner_tuned.predict(X_test)
        class_report = classification_report(y_test,predictions)
        print(class_report)
        #class_plot = plot_classification_report(class_report)
        #class_plot.show()
        neural_learner_score = neural_learner_tuned.score(X_test, y_test)
        print('Neural Network learner best score = ' + str(neural_learner_score))
        neural_learner_plot_tuned = plotting.plot_learning_curve(neural_learner_tuned, "Neural Network Learner (tuned)", X_train, y_train, n_jobs=4)
        neural_learner_plot_tuned.show()

        conf_mat = plotting.plot_conf_mat(neural_learner_tuned, X_test, y_test, "Neural Network Learner Confusion Matrix (tuned)")
        conf_mat.show()

        plt.plot(neural_learner_tuned.loss_curve_)
        plt.title('Neural Network Loss Curve (tuned)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    else:
        print('Invalid task')
        return

    return