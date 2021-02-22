import json

from time import process_time
from yellowbrick.classifier import ClassificationReport
from sklearn.tree import DecisionTreeClassifier
from supervised import plotting
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def decision_tree_learner(X_train, X_test, y_train, y_test, task, params_file):
    dt_learner = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2, random_state=27)
    dt_learner_plot = plotting.plot_learning_curve(dt_learner, "DT Learner (default parameters)", X_train, y_train, n_jobs=4)
    dt_learner_plot.show()
    dt_learner.fit(X_train, y_train)
    dt_learner_score = dt_learner.score(X_test, y_test)
    print('DT learner score = ' + str(dt_learner_score))
    conf_mat = plotting.plot_conf_mat(dt_learner, X_test, y_test, "DT Learner Confusion Matrix (default)")
    conf_mat.show()

    with open(params_file) as f:
        params = json.load(f)
    tuning_params = params['DTLearner']['tuning']
    best_params = params['DTLearner']['best']

    if task == 'grid_search':
        param_grid = {"max_depth": tuning_params['max_depth'],
                "max_leaf_nodes": tuning_params['max_leaf_nodes']}
        grid_search = GridSearchCV(dt_learner, param_grid=param_grid, scoring='accuracy', n_jobs=4, cv=10)
        grid_search.fit(X_train, y_train)
        print('Best params chosen by grid search....\n')
        print(grid_search.best_params_)
        grid_best = grid_search.best_estimator_
        predictions = grid_best.predict(X_test)
        class_report = classification_report(y_test, predictions)
        print(class_report)
        dt_learner_plot = plotting.plot_learning_curve(grid_best, "DT Learner (Grid Search)" + str(grid_search.best_params_), X_train, y_train, n_jobs=4)
        dt_learner_plot.show()
        #class_plot = plotting.plot_classification_report(class_report)
        dt_learner_tuned = grid_search.best_estimator_.fit(X_train, y_train)
        dt_learner_score = dt_learner_tuned.score(X_test, y_test)
        print('DT learner score (Grid Search) = ' + str(dt_learner_score))

        conf_mat = plotting.plot_conf_mat(dt_learner_tuned, X_test, y_test, "DT Learner Confusion Matrix (Grid Search)")
        conf_mat.show()

    elif task == 'mca':
        dt_learner_tuned = DecisionTreeClassifier(max_depth=best_params['max_depth'], max_leaf_nodes=best_params['max_leaf_nodes'], random_state=27)
        if len(tuning_params['max_depth']) != 1:
            tuning_param = 'max_depth'
            tuning_range = tuning_range = range(tuning_params[tuning_param][0], tuning_params[tuning_param][1])
            set_param = 'max_leaf_nodes'
        elif len(tuning_params['max_leaf_nodes']) != 1:
            tuning_param = 'max_leaf_nodes'
            tuning_range = tuning_range = range(tuning_params[tuning_param][0], tuning_params[tuning_param][1])
            set_param = 'max_depth'
        else:
            print('Error, no params have multiple tuning values... exiting')
            return
        title = 'Model Complexity Curve, ' + str(set_param) + '=' + str(best_params[set_param]) + ' varying ' + str(tuning_param)
        mcc = plotting.plot_validation_curve(dt_learner_tuned, title, X_train, y_train, param_name=tuning_param, param_range=tuning_range, n_jobs=None)
        mcc.show()

    elif task == 'plot_best':
        dt_learner_tuned = DecisionTreeClassifier(max_depth=best_params['max_depth'], max_leaf_nodes=best_params['max_leaf_nodes'], random_state=27)
        fit_time_start = process_time()
        dt_learner_tuned.fit(X_train, y_train)
        fit_time_end = process_time()
        print(' DT Learner (tuned) fit time: ' + str(fit_time_end-fit_time_start))

        predict_time_start = process_time()
        predictions = dt_learner_tuned.predict(X_test)
        predict_time_end = process_time()

        print(' DT Learner (tuned) predict time: ' + str(predict_time_end-predict_time_start))

        class_report = classification_report(y_test,predictions)
        print(class_report)
        #class_plot = plot_classification_report(class_report)
        #class_plot.show()
        dt_learner_score = dt_learner_tuned.score(X_test, y_test)
        print('DT learner best score = ' + str(dt_learner_score))
        dt_learner_plot_tuned = plotting.plot_learning_curve(dt_learner_tuned, "DT Learner (tuned) " + str(best_params), X_train, y_train, n_jobs=4)
        dt_learner_plot_tuned.show()

        conf_mat = plotting.plot_conf_mat(dt_learner_tuned, X_test, y_test, "DT Learner Confusion Matrix (tuned)")
        conf_mat.show()

    else:
        print('Invalid task')
        return

    return