print(__doc__)

import sys, getopt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def preprocess_inputs(kidneyData, task='classification'):
    #Extracting the feature attributes from dataset
    X = kidneyData.drop(columns = ['Class'])
    #Extracting the target(label) attributes from dataset
    y = kidneyData['Class']

    #Preprocess the out of range Hypertension(Htn) data
    X.loc[(X['Htn'] != 0) & (X['Htn'] != 1),'Htn'] = 0

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(categories='auto'), ['Rbc','Htn']),
                                        ('Standardizer', StandardScaler(), ['Bp','Sg','Al','Su','Bu','Sc','Sod','Pot','Hemo','Wbcc','Rbcc'])],
                                        remainder='passthrough')
    X = columnTransformer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 200)

    return X_train, X_test, y_train, y_test

def svc_learner(data):
    X_train, X_test, y_train, y_test = preprocess_inputs(data, task='classification')
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-5, 1e-4, 1e-3], 'C': [1, 10, 100]},
                    {'kernel': ['poly'], 'gamma': [1e-5, 1e-4, 1e-3], 'C': [1, 10, 100]}]
    #parameters = [{'kernel': ['rbf'], 'gamma': [1e-5], 'C': [1, 10]}]
    svc_learner = SVC(probability=True)
    svc_learner_plot = plot_learning_curve(svc_learner, "SVC Learner", X_train, y_train, n_jobs=4)
    grid_search = GridSearchCV(svc_learner, param_grid=parameters, scoring='balanced_accuracy', n_jobs=4)
    grid_search.fit(X_train, y_train)
    print('Best params chosen by grid search....\n')
    print(grid_search.best_params_)
    grid_search_df = pd.DataFrame(data=grid_search.cv_results_)
    print(grid_search_df)
    tuned_params = grid_search.best_params_
    #tuned_params = {'C': 1, 'kernel': 'rbf', 'gamma': 1e-5}
    svc_learner_tuned = SVC(kernel=tuned_params.get('kernel'), C=tuned_params.get('C'), gamma=tuned_params.get('gamma'), probability=True)
    svc_learner_plot_tuned = plot_learning_curve(svc_learner_tuned, "SVC Learner", X_train, y_train, n_jobs=4)
    svc_learner_plot_tuned.show()

    # Best so far was {'C': 1, 'gamma': 1e-05, 'kernel': 'rbf'} (gives really high bias (underfit))
    #hyper_param = np.array([1, 10, 100, 1000, 10000])
    #hyper_param = np.linspace(1, 100, 20)
    hyper_param = np.linspace(0.0005, 0.0015)
    #hyper_param = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1])

    train_score, test_score = validation_curve(svc_learner_tuned, X_train, y_train,
                                            param_name='gamma', param_range=hyper_param, scoring='balanced_accuracy', n_jobs=4)

    plt.plot(hyper_param, np.median(train_score, 1), color='blue', label='training score')
    plt.plot(hyper_param, np.median(test_score, 1), color='red', label='testing score')
    plt.legend(loc='best')
    #plt.xscale('log')
    plt.xlabel('gamm')
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

# Taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    #plt.show()
    return plt

def main(argv):
    data_file = str(argv)
    data = pd.read_csv(data_file)
    #decision_tree_learner(data)
    svc_learner(data)

if __name__ == "__main__":
    main(sys.argv[1])