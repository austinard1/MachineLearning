import os
import json
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.random_projection import SparseRandomProjection
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import validation_curve

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def main(X_train_smart, X_test_smart, y_train_smart, y_test_smart, X_train_bank, X_test_bank, y_train_bank, y_test_bank, args):
    os.environ["PYTHONWARNINGS"] = "ignore"
    cv=5
    n_jobs=1
    train_sizes=np.linspace(.2, 1.0, 5)
    if args.dimensionality is not None:
        with open('experiment_best.json') as f:
            params = json.load(f)

        # Baseline (same as A1)
        base_nn = MLPClassifier(hidden_layer_sizes=[30, 30], max_iter = 100, random_state=27)
        base_fit_t = time()
        base_nn.fit(X_train_smart, y_train_smart)
        base_fit_time = time() - base_fit_t
        print('Baseline fit time: ' + str(base_fit_time))
        base_pred_t = time()
        base_nn_pred = base_nn.predict(X_test_smart)
        base_pred_time = time() - base_pred_t
        print('Baseline predict time: ' + str(base_pred_time))
        base_score = base_nn.score(X_test_smart, y_test_smart)
        print('Baseline accuracy = ' + str(base_score))
        base_nn_loss = base_nn.loss_
        print('Baseline Loss: ' + str(base_nn_loss))

        base_loss_curve = base_nn.loss_curve_

        print('Generating learning curve...')
        base_train_sizes, base_train_scores, base_test_scores, base_fit_times, _ = \
        learning_curve(base_nn, X_train_smart, y_train_smart, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        base_train_scores_mean = np.mean(base_train_scores, axis=1)
        base_train_scores_std = np.std(base_train_scores, axis=1)
        base_test_scores_mean = np.mean(base_test_scores, axis=1)
        base_test_scores_std = np.std(base_test_scores, axis=1)
        base_fit_times_mean = np.mean(base_fit_times, axis=1)
        base_fit_times_std = np.std(base_fit_times, axis=1)

        # PCA
        num_comps_smart = params['pca']['smart']
        num_comps_bank = params['pca']['bank']
        pca = PCA(n_components=num_comps_smart, random_state=27)
        pca.fit(X_train_smart)
        transformed_train = pca.transform(X_train_smart)
        transformed_test = pca.transform(X_test_smart)

        pca_nn = MLPClassifier(hidden_layer_sizes=[30, 30], max_iter = 100, random_state=27)

        pca_fit_t = time()
        pca_nn.fit(transformed_train, y_train_smart)
        pca_fit_time = time() - pca_fit_t
        print('PCA fit time: ' + str(pca_fit_time))

        # Tuning
        tuning_param = 'max_iter'
        tuning_range = np.arange(10, 200, 10)
        set_param = 'hidden_layer_sizes'
        title = 'Model Complexity Curve (PCA), varying ' + str(tuning_param)
        mcc = plot_validation_curve(pca_nn, title, transformed_train, y_train_smart, param_name=tuning_param, param_range=tuning_range, n_jobs=4)
        mcc.show()

        pca_pred_t = time()
        pca_nn_pred = pca_nn.predict(transformed_test)
        pca_pred_time = time() - pca_pred_t
        print('PCA predict time: ' + str(pca_pred_time))
        pca_score = pca_nn.score(transformed_test, y_test_smart)
        print('PCA accuracy = ' + str(pca_score))
        pca_nn_loss = pca_nn.loss_
        print('PCA Loss: ' + str(pca_nn_loss))

        pca_loss_curve = pca_nn.loss_curve_

        print('Generating learning curve...')
        pca_train_sizes, pca_train_scores, pca_test_scores, pca_fit_times, _ = \
        learning_curve(pca_nn, transformed_train, y_train_smart, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        pca_train_scores_mean = np.mean(pca_train_scores, axis=1)
        pca_train_scores_std = np.std(pca_train_scores, axis=1)
        pca_test_scores_mean = np.mean(pca_test_scores, axis=1)
        pca_test_scores_std = np.std(pca_test_scores, axis=1)
        pca_fit_times_mean = np.mean(pca_fit_times, axis=1)
        pca_fit_times_std = np.std(pca_fit_times, axis=1)

        # ICA
        num_comps_smart = params['ica']['smart']
        num_comps_bank = params['ica']['bank']
        ica = FastICA(n_components=num_comps_smart, random_state=27)
        ica.fit(X_train_smart)
        transformed_train = ica.transform(X_train_smart)
        transformed_test = ica.transform(X_test_smart)

        ica_nn = MLPClassifier(hidden_layer_sizes=[30, 30], max_iter = 40, random_state=27)

        ica_fit_t = time()
        ica_nn.fit(transformed_train, y_train_smart)
        ica_fit_time = time() - ica_fit_t
        print('ICA fit time: ' + str(ica_fit_time))

        # Hyperparam tuning
        tuning_param = 'max_iter'
        tuning_range = np.arange(10, 200, 10)
        set_param = 'hidden_layer_sizes'
        title = 'Model Complexity Curve (ICA), varying ' + str(tuning_param)
        mcc = plot_validation_curve(ica_nn, title, transformed_train, y_train_smart, param_name=tuning_param, param_range=tuning_range, n_jobs=4)
        mcc.show()

        ica_pred_t = time()
        ica_nn_pred = ica_nn.predict(transformed_test)
        ica_pred_time = time() - ica_pred_t
        print('ICA predict time: ' + str(ica_pred_time))
        ica_score = ica_nn.score(transformed_test, y_test_smart)
        print('ICA accuracy = ' + str(ica_score))
        ica_nn_loss = ica_nn.loss_
        print('ICA Loss: ' + str(ica_nn_loss))

        ica_loss_curve = ica_nn.loss_curve_

        print('Generating learning curve...')
        ica_train_sizes, ica_train_scores, ica_test_scores, ica_fit_times, _ = \
        learning_curve(ica_nn, transformed_train, y_train_smart, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        ica_train_scores_mean = np.mean(ica_train_scores, axis=1)
        ica_train_scores_std = np.std(ica_train_scores, axis=1)
        ica_test_scores_mean = np.mean(ica_test_scores, axis=1)
        ica_test_scores_std = np.std(ica_test_scores, axis=1)
        ica_fit_times_mean = np.mean(ica_fit_times, axis=1)
        ica_fit_times_std = np.std(ica_fit_times, axis=1)

        # RP
        num_comps_smart = params['rp']['smart']
        num_comps_bank = params['rp']['bank']
        best_seed_smart = params['rp']['smart_seed']
        best_seed_bank = params['rp']['bank_seed']
        rp = SparseRandomProjection(n_components=num_comps_smart, random_state=best_seed_smart)
        rp.fit(X_train_smart)
        transformed_train = rp.transform(X_train_smart)
        transformed_test = rp.transform(X_test_smart)

        rp_nn = MLPClassifier(hidden_layer_sizes=[30, 30], max_iter = 90, random_state=27)
        rp_fit_t = time()
        rp_nn.fit(transformed_train, y_train_smart)
        rp_fit_time = time() - rp_fit_t
        print('Random Projection fit time: ' + str(rp_fit_time))

        # Tuning
        tuning_param = 'max_iter'
        tuning_range = np.arange(10, 200, 10)
        set_param = 'hidden_layer_sizes'
        title = 'Model Complexity Curve (RP), varying ' + str(tuning_param)
        mcc = plot_validation_curve(rp_nn, title, transformed_train, y_train_smart, param_name=tuning_param, param_range=tuning_range, n_jobs=4)
        mcc.show()

        rp_pred_t = time()
        rp_nn_pred = rp_nn.predict(transformed_test)
        rp_pred_time = time() - rp_pred_t
        print('Random Projection predict time: ' + str(rp_pred_time))
        rp_score = rp_nn.score(transformed_test, y_test_smart)
        print('Random Projection accuracy = ' + str(rp_score))
        rp_nn_loss = rp_nn.loss_
        print('Random Projection Loss: ' + str(rp_nn_loss))

        rp_loss_curve = rp_nn.loss_curve_

        print('Generating learning curve...')
        rp_train_sizes, rp_train_scores, rp_test_scores, rp_fit_times, _ = \
        learning_curve(rp_nn, transformed_train, y_train_smart, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        rp_train_scores_mean = np.mean(rp_train_scores, axis=1)
        rp_train_scores_std = np.std(rp_train_scores, axis=1)
        rp_test_scores_mean = np.mean(rp_test_scores, axis=1)
        rp_test_scores_std = np.std(rp_test_scores, axis=1)
        rp_fit_times_mean = np.mean(rp_fit_times, axis=1)
        rp_fit_times_std = np.std(rp_fit_times, axis=1)

        # FA
        num_comps_smart = params['fa']['smart']
        num_comps_bank = params['fa']['bank']
        fa = FactorAnalysis(n_components=num_comps_smart, random_state=27)
        fa.fit(X_train_smart)
        transformed_train = fa.transform(X_train_smart)
        transformed_test = fa.transform(X_test_smart)

        fa_nn = MLPClassifier(hidden_layer_sizes=[30, 30], max_iter = 60, random_state=27)
        fa_fit_t = time()
        fa_nn.fit(transformed_train, y_train_smart)
        fa_fit_time = time() - fa_fit_t
        print('Factor Analysis fit time: ' + str(fa_fit_time))

        # Tuning
        tuning_param = 'max_iter'
        tuning_range = np.arange(10, 200, 10)
        set_param = 'hidden_layer_sizes'
        title = 'Model Complexity Curve (FA)), varying ' + str(tuning_param)
        mcc = plot_validation_curve(fa_nn, title, transformed_train, y_train_smart, param_name=tuning_param, param_range=tuning_range, n_jobs=4)
        mcc.show()

        fa_pred_t = time()
        fa_nn_pred = fa_nn.predict(transformed_test)
        fa_pred_time = time() - fa_pred_t
        print('Factor Analysis predict time: ' + str(fa_pred_time))
        fa_score = fa_nn.score(transformed_test, y_test_smart)
        print('Factor Analysis accuracy = ' + str(fa_score))
        fa_nn_loss = fa_nn.loss_
        print('Factor Analysis Loss: ' + str(fa_nn_loss))

        fa_loss_curve = fa_nn.loss_curve_

        print('Generating learning curve...')
        fa_train_sizes, fa_train_scores, fa_test_scores, fa_fit_times, _ = \
        learning_curve(fa_nn, transformed_train, y_train_smart, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        fa_train_scores_mean = np.mean(fa_train_scores, axis=1)
        fa_train_scores_std = np.std(fa_train_scores, axis=1)
        fa_test_scores_mean = np.mean(fa_test_scores, axis=1)
        fa_test_scores_std = np.std(fa_test_scores, axis=1)
        fa_fit_times_mean = np.mean(fa_fit_times, axis=1)
        fa_fit_times_std = np.std(fa_fit_times, axis=1)

        # Plot learning curve
        plt.rc("font", size=8)
        plt.rc("axes", titlesize=10)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", fontsize=8)
        plt.rc("figure", titlesize=11)

        fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
        fig.suptitle('Neural Network Learning Curves', fontsize=14)

        axes[0].set(xlabel='Training Examples', ylabel = 'Accuracy', title='Baseline (A1 Learner)')
        axes[0].grid()
        axes[0].fill_between(base_train_sizes, base_train_scores_mean - base_train_scores_std,
                            base_train_scores_mean + base_train_scores_std, alpha=0.1,
                            color="r")
        axes[0].fill_between(base_train_sizes, base_test_scores_mean - base_test_scores_std,
                            base_test_scores_mean + base_test_scores_std, alpha=0.1,
                            color="g")
        axes[0].plot(base_train_sizes, base_train_scores_mean, 'o-', color="r",
                    label="Training")
        axes[0].plot(base_train_sizes, base_test_scores_mean, 'o-', color="g",
                    label="Cross-validation")
        axes[0].legend(loc="best")

        axes[1].set(xlabel='Training Examples', title='PCA')
        axes[1].grid()
        axes[1].fill_between(pca_train_sizes, pca_train_scores_mean - pca_train_scores_std,
                            pca_train_scores_mean + pca_train_scores_std, alpha=0.1,
                            color="r")
        axes[1].fill_between(pca_train_sizes, pca_test_scores_mean - pca_test_scores_std,
                            pca_test_scores_mean + pca_test_scores_std, alpha=0.1,
                            color="g")
        axes[1].plot(pca_train_sizes, pca_train_scores_mean, 'o-', color="r",
                    label="Training")
        axes[1].plot(pca_train_sizes, pca_test_scores_mean, 'o-', color="g",
                    label="Cross-validation")
        axes[1].legend(loc="best")

        axes[2].set(xlabel='Training Examples', title='ICA')
        axes[2].grid()
        axes[2].fill_between(ica_train_sizes, ica_train_scores_mean - ica_train_scores_std,
                            ica_train_scores_mean + ica_train_scores_std, alpha=0.1,
                            color="r")
        axes[2].fill_between(ica_train_sizes, ica_test_scores_mean - ica_test_scores_std,
                            ica_test_scores_mean + ica_test_scores_std, alpha=0.1,
                            color="g")
        axes[2].plot(ica_train_sizes, ica_train_scores_mean, 'o-', color="r",
                    label="Training")
        axes[2].plot(ica_train_sizes, ica_test_scores_mean, 'o-', color="g",
                    label="Cross-validation")
        axes[2].legend(loc="best")

        axes[3].set(xlabel='Training Examples', title='Random Projection')
        axes[3].grid()
        axes[3].fill_between(rp_train_sizes, rp_train_scores_mean - rp_train_scores_std,
                            rp_train_scores_mean + rp_train_scores_std, alpha=0.1,
                            color="r")
        axes[3].fill_between(rp_train_sizes, rp_test_scores_mean - rp_test_scores_std,
                            rp_test_scores_mean + rp_test_scores_std, alpha=0.1,
                            color="g")
        axes[3].plot(rp_train_sizes, rp_train_scores_mean, 'o-', color="r",
                    label="Training")
        axes[3].plot(rp_train_sizes, rp_test_scores_mean, 'o-', color="g",
                    label="Cross-validation")
        axes[3].legend(loc="best")

        axes[4].set(xlabel='Training Examples', title='Factor Analysis')
        axes[4].grid()
        axes[4].fill_between(fa_train_sizes, fa_train_scores_mean - fa_train_scores_std,
                            fa_train_scores_mean + fa_train_scores_std, alpha=0.1,
                            color="r")
        axes[4].fill_between(fa_train_sizes, fa_test_scores_mean - fa_test_scores_std,
                            fa_test_scores_mean + fa_test_scores_std, alpha=0.1,
                            color="g")
        axes[4].plot(fa_train_sizes, fa_train_scores_mean, 'o-', color="r",
                    label="Training")
        axes[4].plot(fa_train_sizes, fa_test_scores_mean, 'o-', color="g",
                    label="Cross-validation")
        axes[4].legend(loc="best")


        plt.show()
        # plt.savefig('nn_performance_graph', format='png')

        fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
        fig.suptitle('Neural Network Loss Curves', fontsize=14)

        axes[0].plot(base_loss_curve)
        axes[0].set(xlabel='Iteration', ylabel = 'Loss', title='Baseline (A1 Learner)')

        axes[1].plot(pca_loss_curve)
        axes[1].set(xlabel='Iteration', title='PCA')

        axes[2].plot(ica_loss_curve)
        axes[2].set(xlabel='Iteration', title='ICA')

        axes[3].plot(rp_loss_curve)
        axes[3].set(xlabel='Iteration', title='Random Projection')

        axes[4].plot(fa_loss_curve)
        axes[4].set(xlabel='Iteration', title='Factor Analysis')

        plt.show()
        # plt.savefig('nn_loss_curves', format='png')

    if args.clustering is not None:
        with open('experiment_best.json') as f:
            params = json.load(f)

        num_features_reduced = 6
        # Baseline (same as A1)
        base_nn = MLPClassifier(hidden_layer_sizes=[30, 30], max_iter = 100, random_state=27)
        base_fit_t = time()
        base_nn.fit(X_train_smart, y_train_smart)
        base_fit_time = time() - base_fit_t
        print('Baseline fit time: ' + str(base_fit_time))
        base_pred_t = time()
        base_nn_pred = base_nn.predict(X_test_smart)
        base_pred_time = time() - base_pred_t
        print('Baseline predict time: ' + str(base_pred_time))
        base_score = base_nn.score(X_test_smart, y_test_smart)
        print('Baseline accuracy = ' + str(base_score))
        base_nn_loss = base_nn.loss_
        print('Baseline Loss: ' + str(base_nn_loss))

        base_loss_curve = base_nn.loss_curve_

        print('Generating learning curve...')
        base_train_sizes, base_train_scores, base_test_scores, base_fit_times, _ = \
        learning_curve(base_nn, X_train_smart, y_train_smart, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        base_train_scores_mean = np.mean(base_train_scores, axis=1)
        base_train_scores_std = np.std(base_train_scores, axis=1)
        base_test_scores_mean = np.mean(base_test_scores, axis=1)
        base_test_scores_std = np.std(base_test_scores, axis=1)
        base_fit_times_mean = np.mean(base_fit_times, axis=1)
        base_fit_times_std = np.std(base_fit_times, axis=1)

        # K-Means
        num_clusters_smart = params['k_means']['smart']
        k_means_train = pd.DataFrame(X_train_smart)
        k_means_test = pd.DataFrame(X_test_smart)
        k_means_train = k_means_train.iloc[:, :-num_features_reduced]
        k_means_test = k_means_test.iloc[:, :-num_features_reduced]

        k_means = KMeans(n_clusters=num_clusters_smart, random_state=27)
        k_means.fit(X_train_smart)
        prediction_train = k_means.predict(X_train_smart)
        prediction_test = k_means.predict(X_test_smart)

        # k_means_train.assign(str(num_clusters) = prediction_train)
        # k_means_test.assign(str(num_clusters) = prediction_test)
        k_means_train[12 - num_features_reduced] = prediction_train
        k_means_test[12 - num_features_reduced] = prediction_test
        # k_means_train.append(pd.Series(prediction_train, name='cluster_labels'))
        # k_means_test.append(pd.Series(prediction_test, name='cluster_labels'))
        print(k_means_train)
        # print(k_means_train)
        # print(prediction)
        # sil_score_list_smart.append(silhouette_score(X_train_smart, prediction))
        # cal_har_score_list_smart.append(calinski_harabasz_score(X_train_smart, prediction))
        # davies_bouldin_score_list_smart.append(davies_bouldin_score(X_train_smart, prediction))
        # exit()
        km_nn = MLPClassifier(hidden_layer_sizes=[30, 30], max_iter = 60, random_state=27)
        km_fit_t = time()
        km_nn.fit(k_means_train, y_train_smart)
        km_fit_time = time() - km_fit_t
        print('K-Means fit time: ' + str(km_fit_time))

        # Tuning
        tuning_param = 'max_iter'
        tuning_range = np.arange(10, 200, 10)
        set_param = 'hidden_layer_sizes'
        title = 'Model Complexity Curve (K-Means), varying ' + str(tuning_param)
        mcc = plot_validation_curve(km_nn, title, k_means_train, y_train_smart, param_name=tuning_param, param_range=tuning_range, n_jobs=4)
        mcc.show()

        km_pred_t = time()
        km_nn_pred = km_nn.predict(k_means_test)
        km_pred_time = time() - km_pred_t
        print('K-Means predict time: ' + str(km_pred_time))
        km_score = km_nn.score(k_means_test, y_test_smart)
        print('K-Means accuracy = ' + str(km_score))
        km_nn_loss = km_nn.loss_
        print('K-Means Loss: ' + str(km_nn_loss))

        km_loss_curve = km_nn.loss_curve_

        print('Generating learning curve...')
        km_train_sizes, km_train_scores, km_test_scores, km_fit_times, _ = \
        learning_curve(km_nn, k_means_train, y_train_smart, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        km_train_scores_mean = np.mean(km_train_scores, axis=1)
        km_train_scores_std = np.std(km_train_scores, axis=1)
        km_test_scores_mean = np.mean(km_test_scores, axis=1)
        km_test_scores_std = np.std(km_test_scores, axis=1)
        km_fit_times_mean = np.mean(km_fit_times, axis=1)
        km_fit_times_std = np.std(km_fit_times, axis=1)

        # EM
        num_clusters_smart = params['em']['smart']
        em_train = pd.DataFrame(X_train_smart)
        em_test = pd.DataFrame(X_test_smart)
        em_train = em_train.iloc[:, :-num_features_reduced]
        em_test = em_test.iloc[:, :-num_features_reduced]
        em = GaussianMixture(n_components=num_clusters_smart, random_state=27)
        em.fit(X_train_smart)
        prediction_train = em.predict(X_train_smart)
        prediction_test = em.predict(X_test_smart)

        em_train[12 - num_features_reduced] = prediction_train
        em_test[12 - num_features_reduced] = prediction_test
        print(em_train)
        # sil_score_list_smart.append(silhouette_score(X_train_smart, prediction))
        # cal_har_score_list_smart.append(calinski_harabasz_score(X_train_smart, prediction))
        # davies_bouldin_score_list_smart.append(davies_bouldin_score(X_train_smart, prediction))

        em_nn = MLPClassifier(hidden_layer_sizes=[30, 30], max_iter = 60, random_state=27)
        em_fit_t = time()
        em_nn.fit(em_train, y_train_smart)
        em_fit_time = time() - em_fit_t
        print('EM fit time: ' + str(em_fit_time))

        # Tuning
        tuning_param = 'max_iter'
        tuning_range = np.arange(10, 200, 10)
        set_param = 'hidden_layer_sizes'
        title = 'Model Complexity Curve (EM), varying ' + str(tuning_param)
        mcc = plot_validation_curve(em_nn, title, em_train, y_train_smart, param_name=tuning_param, param_range=tuning_range, n_jobs=4)
        mcc.show()

        em_pred_t = time()
        em_nn_pred = em_nn.predict(em_test)
        em_pred_time = time() - em_pred_t
        print('EM predict time: ' + str(em_pred_time))
        em_score = em_nn.score(em_test, y_test_smart)
        print('EM accuracy = ' + str(em_score))
        em_nn_loss = em_nn.loss_
        print('EM Loss: ' + str(em_nn_loss))

        em_loss_curve = em_nn.loss_curve_

        print('Generating learning curve...')
        em_train_sizes, em_train_scores, em_test_scores, em_fit_times, _ = \
        learning_curve(em_nn, em_train, y_train_smart, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        em_train_scores_mean = np.mean(em_train_scores, axis=1)
        em_train_scores_std = np.std(em_train_scores, axis=1)
        em_test_scores_mean = np.mean(em_test_scores, axis=1)
        em_test_scores_std = np.std(em_test_scores, axis=1)
        em_fit_times_mean = np.mean(em_fit_times, axis=1)
        em_fit_times_std = np.std(em_fit_times, axis=1)

        # Plot learning curve
        plt.rc("font", size=8)
        plt.rc("axes", titlesize=10)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", fontsize=8)
        plt.rc("figure", titlesize=11)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        fig.suptitle('Neural Network Learning Curves', fontsize=14)

        axes[0].set(xlabel='Training Examples', ylabel = 'Accuracy', title='Baseline (A1 Learner)')
        axes[0].grid()
        axes[0].fill_between(base_train_sizes, base_train_scores_mean - base_train_scores_std,
                            base_train_scores_mean + base_train_scores_std, alpha=0.1,
                            color="r")
        axes[0].fill_between(base_train_sizes, base_test_scores_mean - base_test_scores_std,
                            base_test_scores_mean + base_test_scores_std, alpha=0.1,
                            color="g")
        axes[0].plot(base_train_sizes, base_train_scores_mean, 'o-', color="r",
                    label="Training")
        axes[0].plot(base_train_sizes, base_test_scores_mean, 'o-', color="g",
                    label="Cross-validation")
        axes[0].legend(loc="best")

        axes[1].set(xlabel='Training Examples', title='K-Means')
        axes[1].grid()
        axes[1].fill_between(km_train_sizes, km_train_scores_mean - km_train_scores_std,
                            km_train_scores_mean + km_train_scores_std, alpha=0.1,
                            color="r")
        axes[1].fill_between(km_train_sizes, km_test_scores_mean - km_test_scores_std,
                            km_test_scores_mean + km_test_scores_std, alpha=0.1,
                            color="g")
        axes[1].plot(km_train_sizes, km_train_scores_mean, 'o-', color="r",
                    label="Training")
        axes[1].plot(km_train_sizes, km_test_scores_mean, 'o-', color="g",
                    label="Cross-validation")
        axes[1].legend(loc="best")

        axes[2].set(xlabel='Training Examples', title='EM')
        axes[2].grid()
        axes[2].fill_between(em_train_sizes, em_train_scores_mean - em_train_scores_std,
                            em_train_scores_mean + em_train_scores_std, alpha=0.1,
                            color="r")
        axes[2].fill_between(em_train_sizes, em_test_scores_mean - em_test_scores_std,
                            em_test_scores_mean + em_test_scores_std, alpha=0.1,
                            color="g")
        axes[2].plot(em_train_sizes, em_train_scores_mean, 'o-', color="r",
                    label="Training")
        axes[2].plot(em_train_sizes, em_test_scores_mean, 'o-', color="g",
                    label="Cross-validation")
        axes[2].legend(loc="best")

        plt.show()
        # plt.savefig('nn_performance_graph', format='png')

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        fig.suptitle('Neural Network Loss Curves', fontsize=14)

        axes[0].plot(base_loss_curve)
        axes[0].set(xlabel='Iteration', ylabel = 'Loss', title='Baseline (A1 Learner)')

        axes[1].plot(km_loss_curve)
        axes[1].set(xlabel='Iteration', title='PCA')

        axes[2].plot(em_loss_curve)
        axes[2].set(xlabel='Iteration', title='ICA')

        plt.show()

    return

def plot_validation_curve(
    estimator,
    title,
    X,
    y,
    param_name,
    param_range,
    n_jobs=None,
):
    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        scoring="accuracy",
        n_jobs=n_jobs,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.4, 1.1)
    lw = 2

    # Turn params into strings for hidden_layer_sizes
    if param_name == 'hidden_layer_sizes':
        param_range = [str(param) for param in param_range]

    plt.plot(
        param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
    )
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    plt.plot(
        param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    plt.legend(loc="best")

    return plt