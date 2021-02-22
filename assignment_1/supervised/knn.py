import json
import numpy as np

from time import process_time
from yellowbrick.classifier import ClassificationReport
from sklearn.neighbors import KNeighborsClassifier
from supervised import plotting
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def knn_learner(X_train, X_test, y_train, y_test, task, params_file):
    knn_learner = KNeighborsClassifier(n_neighbors=1, weights="uniform")
    knn_learner_plot = plotting.plot_learning_curve(knn_learner, "KNN Learner (default parameters)", X_train, y_train, n_jobs=4)
    knn_learner_plot.show()
    knn_learner.fit(X_train, y_train)
    knn_learner_score = knn_learner.score(X_test, y_test)
    print('KNN learner score = ' + str(knn_learner_score))
    conf_mat = plotting.plot_conf_mat(knn_learner, X_test, y_test, "KNN Learner Confusion Matrix (default)")
    conf_mat.show()

    with open(params_file) as f:
        params = json.load(f)
    tuning_params = params['KNNLearner']['tuning']
    best_params = params['KNNLearner']['best']

    if task == 'grid_search':
        param_grid = {"n_neighbors": tuning_params['n_neighbors'],
                "weights": tuning_params['weights']}
        grid_search = GridSearchCV(knn_learner, param_grid=param_grid, scoring='accuracy', n_jobs=4, cv=10)
        grid_search.fit(X_train, y_train)
        print('Best params chosen by grid search....\n')
        print(grid_search.best_params_)
        grid_best = grid_search.best_estimator_
        predictions = grid_best.predict(X_test)
        class_report = classification_report(y_test, predictions)
        print(class_report)
        knn_learner_plot = plotting.plot_learning_curve(grid_best, "KNN Learner (Grid Search) "+ str(grid_search.best_params_), X_train, y_train, n_jobs=4, cv=10)
        knn_learner_plot.show()
        #class_plot = plotting.plot_classification_report(class_report)
        knn_learner_tuned = grid_search.best_estimator_.fit(X_train, y_train)
        knn_learner_score = knn_learner_tuned.score(X_test, y_test)
        print('knn learner score (Grid Search) = ' + str(knn_learner_score))

        conf_mat = plotting.plot_conf_mat(knn_learner_tuned, X_test, y_test, "KNN Learner Confusion Matrix (Grid Search)")
        conf_mat.show()

    elif task == 'mca':
        knn_learner_tuned = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
        if len(tuning_params['n_neighbors']) != 1:
            tuning_param = 'n_neighbors'
            tuning_range = range(tuning_params[tuning_param][0], tuning_params[tuning_param][1])
            set_param = 'weights'
        elif len(tuning_params['weights']) != 1:
            tuning_param = 'weights'
            tuning_range = tuning_params['weights']
            set_param = 'n_neighbors'
        else:
            print('Error, no params have multiple tuning values... exiting')
            return
        title = 'Model Complexity Curve, ' + str(set_param) + '=' + str(best_params[set_param]) + ' varying ' + str(tuning_param)
        mcc = plotting.plot_validation_curve(knn_learner_tuned, title, X_train, y_train, param_name=tuning_param, param_range=tuning_range, n_jobs=None)
        mcc.show()

    elif task == 'plot_best':
        knn_learner_tuned = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
        fit_time_start = process_time()
        knn_learner_tuned.fit(X_train, y_train)
        fit_time_end = process_time()
        print(' KNN Learner (tuned) fit time: ' + str(fit_time_end-fit_time_start))

        predict_time_start = process_time()
        predictions = knn_learner_tuned.predict(X_test)
        predict_time_end = process_time()

        print(' KNN Learner (tuned) predict time: ' + str(predict_time_end-predict_time_start))

        class_report = classification_report(y_test,predictions)
        print(class_report)
        #class_plot = plot_classification_report(class_report)
        #class_plot.show()
        knn_learner_score = knn_learner_tuned.score(X_test, y_test)
        print('KNN learner best score = ' + str(knn_learner_score))
        knn_learner_plot_tuned = plotting.plot_learning_curve(knn_learner_tuned, "KNN Learner (tuned) " + str(best_params), X_train, y_train, n_jobs=4, cv=10)
        knn_learner_plot_tuned.show()

        conf_mat = plotting.plot_conf_mat(knn_learner_tuned, X_test, y_test, "KNN Learner Confusion Matrix (tuned)")
        conf_mat.show()

    else:
        print('Invalid task')
        return

    return