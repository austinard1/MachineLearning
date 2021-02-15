import json

from yellowbrick.classifier import ClassificationReport
from sklearn.ensemble import AdaBoostClassifier
from supervised import plotting
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def boosting_learner(X_train, X_test, y_train, y_test, task, params_file):
    boosting_learner = AdaBoostClassifier(base_estimator= DecisionTreeClassifier())
    boosting_learner_plot = plotting.plot_learning_curve(boosting_learner, "Boosting Learner", X_train, y_train, n_jobs=4)
    boosting_learner_plot.show()
    boosting_learner.fit(X_train, y_train)
    boosting_learner_score = boosting_learner.score(X_test, y_test)
    print('Boosting learner score = ' + str(boosting_learner_score))

    with open(params_file) as f:
        params = json.load(f)
    tuning_params = params['BoostingLearner']['tuning']
    best_params = params['BoostingLearner']['best']

    if task == 'grid_search':
        param_grid = {"n_estimators": tuning_params['n_estimators'],
                "learning_rate": tuning_params['learning_rate'],
                "max_depth": tuning_params['max_depth']}
              #"criterion": ["gini", "entropy"]}
        grid_search = GridSearchCV(boosting_learner, param_grid=param_grid, scoring='accuracy', n_jobs=4, cv=10)
        grid_search.fit(X_train, y_train)
        print('Best params chosen by grid search....\n')
        print(grid_search.best_params_)
        grid_best = grid_search.best_estimator_
        predictions = grid_best.predict(X_test)
        class_report = classification_report(y_test, predictions)
        print(class_report)
        boosting_learner_plot = plotting.plot_learning_curve(grid_best, "Boosting Learner (Grid Search)", X_train, y_train, n_jobs=4)
        boosting_learner_plot.show()
        #class_plot = plotting.plot_classification_report(class_report)
        boosting_learner_tuned = grid_search.best_estimator_.fit(X_train, y_train)
        boosting_learner_score = boosting_learner_tuned.score(X_test, y_test)
        print('Boosting learner score (Grid Search) = ' + str(boosting_learner_score))

    elif task == 'mca':
        boosting_learner_tuned = AdaBoostClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], base_estimator=best_params['base_estimator'])
        if len(tuning_params['n_estimators']) != 1:
            tuning_param = 'n_estimators'
            tuning_range = range(tuning_params[tuning_param][0], tuning_params[tuning_param][1])
        elif len(tuning_params['learning_rate']) != 1:
            tuning_param = 'learning_rate'
            tuning_range = np.linalg(tuning_params[tuning_param][0], tuning_params[tuning_param][1])
        elif len(tuning_params['max_depth']) != 1:
            tuning_param = 'max_depth'
            tuning_range = range(tuning_params[tuning_param][0], tuning_params[tuning_param][1])
        else:
            print('Error, no params have multiple tuning values... exiting')
            return
        title = 'asdf'
        mcc = plotting.plot_validation_curve(boosting_learner_tuned, title, X_train, y_train, param_name=tuning_param, param_range=tuning_range, n_jobs=None)
        mcc.show()

    elif task == 'plot_best':
        boosting_learner_tuned = AdaBoostClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'])
        boosting_learner_tuned.fit(X_train, y_train)
        predictions = boosting_learner_tuned.predict(X_test)
        class_report = classification_report(y_test,predictions)
        print(class_report)
        #class_plot = plot_classification_report(class_report)
        #class_plot.show()
        boosting_learner_score = boosting_learner_tuned.score(X_test, y_test)
        print('Boosting learner best score = ' + str(boosting_learner_score))
        boosting_learner_plot_tuned = plotting.plot_learning_curve(boosting_learner_tuned, "Boosting Learner (tuned)", X_train, y_train, n_jobs=4)
        boosting_learner_plot_tuned.show()

    else:
        print('Invalid task')
        return

    #plotting.plot_confusion_matrix(boosting_learner_tuned, X_test, y_test)
    #boosting_learner_plot_tuned = plotting.plot_learning_curve(boosting_learner_tuned, "boosting Learner (tuned)", X_train, y_train, n_jobs=4)
    #boosting_learner_plot_tuned.show()

    return