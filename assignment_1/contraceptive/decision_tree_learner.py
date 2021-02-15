import json

from yellowbrick.classifier import ClassificationReport
from sklearn.tree import DecisionTreeClassifier
from supervised import plotting
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def decision_tree_learner(X_train, X_test, y_train, y_test, task):
    dt_learner = DecisionTreeClassifier()
    dt_learner_plot = plotting.plot_learning_curve(dt_learner, "dt Learner", X_train, y_train, n_jobs=4)
    dt_learner_plot.show()
    dt_learner.fit(X_train, y_train)
    dt_learner_score = dt_learner.score(X_test, y_test)
    print('DT learner score = ' + str(dt_learner_score))

    with open('params.json') as f:
        params = json.load(f)
    tuning_params = params['DTLearner']['tuning']
    best_params = params['DTLearner']['best']

    if task == 'grid_search':
        param_grid = {"max_depth": tuning_params['max_depth'],
                "max_leaf_nodes": tuning_params['max_leaf_nodes']}
              #"criterion": ["gini", "entropy"]}
        grid_search = GridSearchCV(dt_learner, param_grid=param_grid, scoring='accuracy', n_jobs=4, cv=10)
        grid_search.fit(X_train, y_train)
        print('Best params chosen by grid search....\n')
        print(grid_search.best_params_)
        grid_best = grid_search.best_estimator_
        predictions = grid_best.predict(X_test)
        class_report = classification_report(y_test, predictions)
        print(class_report)
        dt_learner_plot = plotting.plot_learning_curve(grid_best, "DT Learner (Grid Search)", X_train, y_train, n_jobs=4)
        dt_learner_plot.show()
        #class_plot = plotting.plot_classification_report(class_report)
        dt_learner_tuned = grid_search.best_estimator_.fit(X_train, y_train)
        dt_learner_score = dt_learner_tuned.score(X_test, y_test)
        print('DT learner score (Grid Search) = ' + str(dt_learner_score))

    elif task == 'mca':
        dt_learner_tuned = DecisionTreeClassifier(max_depth=best_params['max_depth'], max_leaf_nodes=best_params['max_leaf_nodes'], criterion=best_params['criterion'])
        if len(tuning_params['max_depth']) != 1:
            tuning_param = 'max_depth'
            tuning_range = tuning_params['max_depth']
        elif len(tuning_params['max_leaf_nodes']) != 1:
            tuning_param = 'max_leaf_nodes'
            tuning_range = tuning_params['max_leaf_nodes']
        else:
            print('Error, no params have multiple tuning values... exiting')
            return
        title = 'asdf'
        mcc = plotting.plot_validation_curve(dt_learner_tuned, title, X_train, y_train, param_name=tuning_param, param_range=tuning_range, n_jobs=None)
        mcc.show()

    elif task == 'plot_best':
        dt_learner_tuned = DecisionTreeClassifier(max_depth=best_params['max_depth'], max_leaf_nodes=best_params['max_leaf_nodes'], criterion=best_params['criterion'])
        dt_learner_tuned.fit(X_train, y_train)
        predictions = dt_learner_tuned.predict(X_test)
        class_report = classification_report(y_test,predictions)
        print(class_report)
        #class_plot = plot_classification_report(class_report)
        #class_plot.show()
        dt_learner_score = dt_learner_tuned.score(X_test, y_test)
        print('DT learner best score = ' + str(dt_learner_score))
        dt_learner_plot_tuned = plotting.plot_learning_curve(dt_learner_tuned, "DT Learner (tuned)", X_train, y_train, n_jobs=4)
        dt_learner_plot_tuned.show()

    else:
        print('Invalid task')
        return

    #plotting.plot_confusion_matrix(dt_learner_tuned, X_test, y_test)
    #dt_learner_plot_tuned = plotting.plot_learning_curve(dt_learner_tuned, "DT Learner (tuned)", X_train, y_train, n_jobs=4)
    #dt_learner_plot_tuned.show()

    return