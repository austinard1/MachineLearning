import json

import math
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.classifier import ClassificationReport
from sklearn.decomposition import FastICA
# from supervised import plotting
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, mean_squared_error

from yellowbrick.features import RadViz, ParallelCoordinates, Rank1D, PCA

from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from yellowbrick.text import TSNEVisualizer

from sklearn.ensemble import AdaBoostClassifier

def main(X_train_smart, X_test_smart, y_train_smart, y_test_smart, X_train_bank, X_test_bank, y_train_bank, y_test_bank, args):
    with open('experiment_best.json') as f:
        params = json.load(f)
    num_comps_smart = params['ica']['smart']
    num_comps_bank = params['ica']['bank']

    # VISUALIZE NEW CLUSTERS AND THATS GOOD ENOUGH
    # SMART GRID
    kurtosis_list_smart = []
    # print(X_train)
    # exit()
    # cal_har_score_list = []
    # davies_bouldin_score_list = []
    num_clusters_list_smart = np.arange(2, X_train_smart.shape[1] + 1)
    orig_feature_list_smart = np.arange(0, X_train_smart.shape[1])
    # # rad_viz = RadViz()
    # print(y_train)
    # rad_viz.fit_transform(X_train, y_train)
    # # rad_viz.transform(X_train)
    # rad_viz.show()
    for num_clusters in num_clusters_list_smart:
        ica = FastICA(n_components=num_clusters, random_state=27)
        ica.fit(X_train_smart)
        transformed = ica.transform(X_train_smart)
        kurtosis = pd.DataFrame(transformed).kurt(axis=0).abs().mean()
        kurtosis_list_smart.append(kurtosis)


    print(kurtosis_list_smart)
    max_kurt_idx = kurtosis_list_smart.index(max(kurtosis_list_smart))
    print(max_kurt_idx)
    # num_comps_smart = num_clusters_list_smart[max_kurt_idx]
    # num_comps_smart = 11
    new_feature_list_smart = np.arange(0, num_comps_smart)
    print(new_feature_list_smart)
    ica = FastICA(n_components=num_comps_smart, random_state=27)
    ica.fit(X_train_smart)
    transformed = ica.transform(X_train_smart)
    new_kurt_smart = pd.DataFrame(transformed).kurt(axis=0).abs()
    original_kurt_smart = pd.DataFrame(X_train_smart).kurt(axis=0).abs()

    # VISUALIZE NEW CLUSTERS AND THATS GOOD ENOUGH
    # BANK LOAN
    kurtosis_list_bank = []
    # print(X_train)
    # exit()
    # cal_har_score_list = []
    # davies_bouldin_score_list = []
    num_clusters_list_bank = np.arange(2, X_train_bank.shape[1] + 1)
    orig_feature_list_bank = np.arange(0, X_train_bank.shape[1])
    # # rad_viz = RadViz()
    # print(y_train)
    # rad_viz.fit_transform(X_train, y_train)
    # # rad_viz.transform(X_train)
    # rad_viz.show()
    for num_clusters in num_clusters_list_bank:
        ica = FastICA(n_components=num_clusters, random_state=27)
        ica.fit(X_train_bank)
        transformed = ica.transform(X_train_bank)
        kurtosis = pd.DataFrame(transformed).kurt(axis=0).abs().mean()
        kurtosis_list_bank.append(kurtosis)


    print(kurtosis_list_bank)
    max_kurt_idx = kurtosis_list_bank.index(max(kurtosis_list_bank))
    print(max_kurt_idx)
    # num_comps_bank = num_clusters_list_bank[max_kurt_idx]
    # num_comps_bank = 4
    new_feature_list_bank = np.arange(0, num_comps_bank)
    print(new_feature_list_bank)
    ica = FastICA(n_components=num_comps_bank, random_state=27)
    ica.fit(X_train_bank)
    transformed = ica.transform(X_train_bank)
    new_kurt_bank = pd.DataFrame(transformed).kurt(axis=0).abs()
    original_kurt_bank = pd.DataFrame(X_train_bank).kurt(axis=0).abs()
    plt.rc("font", size=8)
    plt.rc("axes", titlesize=12)
    plt.rc("axes", labelsize=10)
    plt.rc("xtick", labelsize=8)
    plt.rc("ytick", labelsize=8)
    plt.rc("legend", fontsize=8)
    plt.rc("figure", titlesize=11)
    #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
    fig, ax = plt.subplots(1,4,figsize=(14,3.5))
    fig.suptitle('ICA Kurtosis Analysis (Left: Smart Grid, Right: Bank Loan)', fontsize=14)
    # ax[0].plot(problem_size_list, sa_fitness_list, 'b-', label='Simulated Annealing', linewidth=1)
    # ax[0].plot(problem_size_list, ga_fitness_list, 'g:', label='Genetic', linewidth=1)
    w = 0.4
    ax[0].plot(num_clusters_list_smart, kurtosis_list_smart, 'b-', linewidth=1)
    ax[0].set(xlabel='# of ICA components', ylabel = 'Mean Kurtosis')
    ax[0].set_ylim([0, 1.2*max(kurtosis_list_smart)])
    ax[0].legend()

    ax[1].bar(orig_feature_list_smart - 0.5*w, original_kurt_smart, width=w, color='red', label='Original Features ({0})'.format(X_train_smart.shape[1]))
    ax[1].bar(new_feature_list_smart + 0.5*w, new_kurt_smart, width=w, color='green', label='ICA Features ({0})'.format(num_comps_smart))
    ax[1].set(xlabel='Feature #', ylabel = 'Kurtosis')
    ax[1].set_ylim([0, 1.2*max(kurtosis_list_smart)])
    ax[1].set_xticks(orig_feature_list_smart)
    ax[1].legend()

    ax[2].plot(num_clusters_list_bank, kurtosis_list_bank, 'b-', linewidth=1)
    ax[2].set(xlabel='# of ICA components', ylabel = 'Mean Kurtosis')
    ax[2].set_ylim([0, 1.2*max(kurtosis_list_bank)])
    ax[2].legend()

    ax[3].bar(orig_feature_list_bank - 0.5*w, original_kurt_bank, width=w, color='red', label='Original Features ({0})'.format(X_train_bank.shape[1]))
    ax[3].bar(new_feature_list_bank + 0.5*w, new_kurt_bank, width=w, color='green', label='ICA Features ({0})'.format(num_comps_bank))
    ax[3].set(xlabel='Feature #', ylabel = 'Kurtosis')
    ax[3].set_ylim([0, 1.2*max(new_kurt_bank)])
    ax[3].set_xticks(orig_feature_list_bank)
    ax[3].legend()

    plt.show()

    # Return best
    ica = FastICA(n_components=num_comps_smart, random_state=27)
    ica_fit_t = time()
    ica.fit(X_train_smart)
    ica_fit_time = time() - ica_fit_t
    print('ICA fit time (smart): ' + str(ica_fit_time))
    smart_transformed = ica.transform(X_train_smart)
    smart_transformed_test = ica.transform(X_test_smart)

    # Smart grid
    boosting_learner = AdaBoostClassifier(learning_rate=1, n_estimators=100)
    boost_fit_t = time()
    boosting_learner.fit(X_train_smart, y_train_smart)
    boost_fit_time = time() - boost_fit_t
    print('Boosting baseline fit time (smart): ' + str(boost_fit_time))
    boost_pred_t = time()
    boost_pred = boosting_learner.predict(X_test_smart)
    boost_pred_time = time() - boost_pred_t
    print('Boosting baseline predict time (smart): ' + str(boost_pred_time))
    boost_score = cross_val_score(boosting_learner, X_train_smart, y_train_smart, cv=10)
    print('Boosting baseline cross validation score (smart): ' + str(np.mean(boost_score)))

    boosting_learner = AdaBoostClassifier(learning_rate=1, n_estimators=100)
    boost_fit_t = time()
    boosting_learner.fit(smart_transformed, y_train_smart)
    boost_fit_time = time() - boost_fit_t
    print('Boosting ICA fit time (smart): ' + str(boost_fit_time))
    boost_pred_t = time()
    boost_pred = boosting_learner.predict(smart_transformed)
    boost_pred_time = time() - boost_pred_t
    print('Boosting ICA predict time (smart): ' + str(boost_pred_time))
    boost_score = cross_val_score(boosting_learner, smart_transformed, y_train_smart, cv=10)
    print('Boosting ICA cross validation score (smart): ' + str(np.mean(boost_score)))

    ica = FastICA(n_components=num_comps_bank, random_state=27)
    ica_fit_t = time()
    ica.fit(X_train_bank)
    ica_fit_time = time() - ica_fit_t
    print('ICA fit time (bank): ' + str(ica_fit_time))
    bank_transformed = ica.transform(X_train_bank)
    bank_transformed_test = ica.transform(X_test_bank)

    # Bank loan
    boosting_learner = AdaBoostClassifier(learning_rate=1, n_estimators=100)
    boost_fit_t = time()
    boosting_learner.fit(X_train_bank, y_train_bank)
    boost_fit_time = time() - boost_fit_t
    print('Boosting baseline fit time (bank): ' + str(boost_fit_time))
    boost_pred_t = time()
    boost_pred = boosting_learner.predict(X_test_bank)
    boost_pred_time = time() - boost_pred_t
    print('Boosting baseline predict time (bank): ' + str(boost_pred_time))
    boost_score = cross_val_score(boosting_learner, X_train_bank, y_train_bank, cv=10)
    print('Boosting baseline cross validation score (bank): ' + str(np.mean(boost_score)))

    boosting_learner = AdaBoostClassifier(learning_rate=1, n_estimators=100)
    boost_fit_t = time()
    boosting_learner.fit(bank_transformed, y_train_bank)
    boost_fit_time = time() - boost_fit_t
    print('Boosting ICA fit time (bank): ' + str(boost_fit_time))
    boost_pred_t = time()
    boost_pred = boosting_learner.predict(bank_transformed)
    boost_pred_time = time() - boost_pred_t
    print('Boosting ICA predict time (bank): ' + str(boost_pred_time))
    boost_score = cross_val_score(boosting_learner, bank_transformed, y_train_bank, cv=10)
    print('Boosting ICA cross validation score (bank): ' + str(np.mean(boost_score)))

    return smart_transformed, smart_transformed_test, bank_transformed, bank_transformed_test

    # rad_viz = PCA(scale=True, proj_features=True)
    # rad_viz.fit_transform(transformed, y_train)
    # # rad_viz.transform(transformed)
    # rad_viz.show()
    # tsne = TSNEVisualizer(decompose_by=num_clusters_list[max_kurt_idx])
    # tsne.fit(X_train)
    # tsne.show()
    # exit()
    # plt.rc("font", size=8)
    # plt.rc("axes", titlesize=12)
    # plt.rc("axes", labelsize=10)
    # plt.rc("xtick", labelsize=8)
    # plt.rc("ytick", labelsize=8)
    # plt.rc("legend", fontsize=8)
    # plt.rc("figure", titlesize=11)
    # #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
    # fig, ax = plt.subplots(1,3,figsize=(12,3.5))
    # fig.suptitle('ExpectiMax - # of components Analysis', fontsize=14)
    # # ax[0].plot(problem_size_list, sa_fitness_list, 'b-', label='Simulated Annealing', linewidth=1)
    # # ax[0].plot(problem_size_list, ga_fitness_list, 'g:', label='Genetic', linewidth=1)
    # ax[0].plot(num_clusters_list, sil_score_list, 'r--', label='Random Hill Climb', linewidth=1)
    # ax[0].set(xlabel='K', ylabel = 'Silhouette')
    # #ax[0].set_title('Fitness vs. Iteration')
    # ax[1].plot(num_clusters_list, cal_har_score_list, 'r--', label='Random Hill Climb', linewidth=1)
    # ax[1].set(xlabel='K', ylabel = 'Calinksi Harabasz')

    # ax[2].plot(num_clusters_list, davies_bouldin_score_list, 'r--', label='Random Hill Climb', linewidth=1)
    # ax[2].set(xlabel='K', ylabel = 'Davies Bouldin')

    # plt.show()
    # # score = em.score(X_test)

    # visualizer = KElbowVisualizer(em, k=(2,20), random_state=27)

    # visualizer.fit(X_train)        # Fit the data to the visualizer
    # visualizer.show()        # Finalize and render the figure

    # elbow = visualizer.elbow_value_
    # em = KMeans(n_clusters=elbow)
    # visualizer = SilhouetteVisualizer(em, colors='yellowbrick', random_state=27)

    # visualizer.fit(X_train)        # Fit the data to the visualizer
    # visualizer.show()

    # visualizer = InterclusterDistance(em, random_state=27)

    # visualizer.fit(X_train)        # Fit the data to the visualizer
    # visualizer.show()        # Finalize and render the figure

    # # Create the visualizer and draw the vectors
    # tsne = TSNEVisualizer()
    # tsne.fit(X)
    # tsne.show()