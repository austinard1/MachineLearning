import json
import numpy as np

from time import process_time
from yellowbrick.classifier import ClassificationReport
from sklearn.svm import SVC
from supervised import plotting
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.evaluate import bias_variance_decomp

def svm_learner(X_train, X_test, y_train, y_test, task, params_file):
    svm_learner = SVC(kernel='linear')
    svm_learner_plot = plotting.plot_learning_curve(svm_learner, "SVM Learner (default parameters)", X_train, y_train, n_jobs=4)
    svm_learner_plot.show()
    svm_learner.fit(X_train, y_train)
    svm_learner_score = svm_learner.score(X_test, y_test)
    print('svm learner score = ' + str(svm_learner_score))
    conf_mat = plotting.plot_conf_mat(svm_learner, X_test, y_test, "SVM Learner Confusion Matrix (default)")
    conf_mat.show()

    with open(params_file) as f:
        params = json.load(f)
    tuning_params = params['SVMLearner']['tuning']
    best_params = params['SVMLearner']['best']

    if task == 'grid_search':
        print(tuning_params['kernel'])
        if tuning_params['kernel'] == ['rbf']:
            param_grid = {"C": tuning_params['C'],
                    "gamma": tuning_params['gamma'],
                    "kernel": tuning_params['kernel']}
        elif tuning_params['kernel'] == ['linear']:
                param_grid = {"C": tuning_params['C'],
                    "tol" : tuning_params['tol'],
                    "kernel": tuning_params['kernel']}
        else:
            print('Invalid SVM kernel')
            return
        grid_search = GridSearchCV(svm_learner, param_grid=param_grid, scoring='accuracy', n_jobs=4, cv=10)
        grid_search.fit(X_train, y_train)
        print('Best params chosen by grid search....\n')
        print(grid_search.best_params_)
        grid_best = grid_search.best_estimator_
        predictions = grid_best.predict(X_test)
        class_report = classification_report(y_test, predictions)
        print(class_report)
        svm_learner_plot = plotting.plot_learning_curve(grid_best, "SVM Learner (Grid Search), " + str(grid_search.best_params_), X_train, y_train, n_jobs=4, cv=10)
        svm_learner_plot.show()
        #class_plot = plotting.plot_classification_report(class_report)
        svm_learner_tuned = grid_search.best_estimator_.fit(X_train, y_train)
        svm_learner_score = svm_learner_tuned.score(X_test, y_test)
        print('SVM Learner score (Grid Search) = ' + str(svm_learner_score))

        conf_mat = plotting.plot_conf_mat(svm_learner_tuned, X_test, y_test, "SVM Learner Confusion Matrix (Grid Search)")
        conf_mat.show()

    elif task == 'mca':
        if tuning_params['kernel'] == ['rbf']:
            svm_learner_tuned = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
            if len(tuning_params['C']) != 1:
                tuning_param = 'C'
                tuning_range = np.linspace(tuning_params[tuning_param][0], tuning_params[tuning_param][1], 100)
                set_param = 'gamma'
            elif len(tuning_params['gamma']) != 1:
                tuning_param = 'gamma'
                tuning_range = np.linspace(tuning_params[tuning_param][0], tuning_params[tuning_param][1], 100)
                set_param = 'C'
            else:
                print('Error, no params have multiple tuning values... exiting')
                return
        elif tuning_params['kernel'] == ['linear']:
            svm_learner_tuned = SVC(C=best_params['C'], tol=best_params['tol'], kernel=best_params['kernel'])
            if len(tuning_params['C']) != 1:
                tuning_param = 'C'
                tuning_range = np.linspace(tuning_params[tuning_param][0], tuning_params[tuning_param][1], 100)
                set_param = 'tol'
            elif len(tuning_params['tol']) != 1:
                tuning_param = 'tol'
                tuning_range = np.linspace(tuning_params[tuning_param][0], tuning_params[tuning_param][1], 100)
                set_param = 'C'
            else:
                print('Error, no params have multiple tuning values... exiting')
                return
        title = 'Model Complexity Curve, ' + str(set_param) + '=' + str(best_params[set_param]) + ' varying ' + str(tuning_param)
        mcc = plotting.plot_validation_curve(svm_learner_tuned, title, X_train, y_train, param_name=tuning_param, param_range=tuning_range, n_jobs=None)
        mcc.show()

    elif task == 'plot_best':
        if best_params['kernel'] == 'rbf':
            svm_learner_tuned = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
        elif best_params['kernel'] == 'linear':
            svm_learner_tuned = SVC(C=best_params['C'], tol=best_params['tol'], kernel=best_params['kernel'])
        else:
            print('Invalid SVM kernel')
            return
        fit_time_start = process_time()
        svm_learner_tuned.fit(X_train, y_train)
        fit_time_end = process_time()
        print(' SVM Learner (tuned) fit time: ' + str(fit_time_end-fit_time_start))

        predict_time_start = process_time()
        predictions = svm_learner_tuned.predict(X_test)
        predict_time_end = process_time()

        print(' SVM Learner (tuned) predict time: ' + str(predict_time_end-predict_time_start))

        predictions = svm_learner_tuned.predict(X_test)
        class_report = classification_report(y_test,predictions)
        print(class_report)
        #class_plot = plot_classification_report(class_report)
        #class_plot.show()
        svm_learner_score = svm_learner_tuned.score(X_test, y_test)
        print('SVM learner best score = ' + str(svm_learner_score))
        if best_params['kernel'] == 'rbf':
            svm_learner_plot_tuned = plotting.plot_learning_curve(svm_learner_tuned, "SVM Learner (tuned), " + \
            "C=" + str(best_params['C']) + " gamma=" + str(best_params['gamma']) + " kernel=" + str(best_params['kernel']), X_train, y_train, n_jobs=4, cv=10)
        elif best_params['kernel'] == 'linear':
            svm_learner_plot_tuned = plotting.plot_learning_curve(svm_learner_tuned, "SVM Learner (tuned), " + \
            "C=" + str(best_params['C']) + " tol=" + str(best_params['tol']) + " kernel=" + str(best_params['kernel']), X_train, y_train, n_jobs=4, cv=10)
        else:
            print('Invalid SVM kernel')
            return
        svm_learner_plot_tuned.show()

        conf_mat = plotting.plot_conf_mat(svm_learner_tuned, X_test, y_test, "SVM Learner Confusion Matrix (tuned)")
        conf_mat.show()

        # Taken from https://machinelearningmastery.com/calculate-the-bias-variance-trade-off/
        # estimate bias and variance
        #mse, bias, var = bias_variance_decomp(svm_learner_tuned, np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), loss='mse', num_rounds=200, random_seed=27)
        # summarize results
        #print('MSE: %.3f' % mse)
        #print('Bias: %.3f' % bias)
        #print('Variance: %.3f' % var)

    else:
        print('Invalid task')
        return

    return