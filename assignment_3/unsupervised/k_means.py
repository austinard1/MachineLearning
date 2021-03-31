import json
import numpy as np
from time import time
import matplotlib.pyplot as plt
from yellowbrick.classifier import ClassificationReport
from sklearn.cluster import KMeans
# from supervised import plotting
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, accuracy_score

from yellowbrick.features import RadViz
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from yellowbrick.text import TSNEVisualizer, UMAPVisualizer

from sklearn.ensemble import AdaBoostClassifier

def main(X_train_smart, X_test_smart, y_train_smart, y_test_smart, X_train_bank, X_test_bank, y_train_bank, y_test_bank, args):

    # em = KMeans(n_clusters=4, random_state=27)
    # em.fit(X_train_smart)
    # prediction = em.predict(X_train_smart)

    # viz = RadViz()
    # viz.fit_transform(X_train_smart, prediction)
    # viz.show()

    # umap = UMAPVisualizer()
    # umap.fit(X_train_smart, ["c{}".format(c) for c in prediction])
    # umap.show()

    # tsne = TSNEVisualizer(decompose_by=4)
    # tsne.fit(X_train_smart, ["c{}".format(c) for c in prediction])
    # tsne.show()
    # exit()
    sil_score_list_smart = []
    cal_har_score_list_smart = []
    davies_bouldin_score_list_smart = []
    sil_score_list_bank = []
    cal_har_score_list_bank = []
    davies_bouldin_score_list_bank = []
    num_clusters_list = np.arange(2, 25)
    for num_clusters in num_clusters_list:
        k_means = KMeans(n_clusters=num_clusters, random_state=27)
        k_means.fit(X_train_smart)
        prediction = k_means.predict(X_train_smart)
        # print(prediction)
        sil_score_list_smart.append(silhouette_score(X_train_smart, prediction))
        cal_har_score_list_smart.append(calinski_harabasz_score(X_train_smart, prediction))
        davies_bouldin_score_list_smart.append(davies_bouldin_score(X_train_smart, prediction))

    for num_clusters in num_clusters_list:
        k_means = KMeans(n_clusters=num_clusters, random_state=27)
        k_means.fit(X_train_bank)
        prediction = k_means.predict(X_train_bank)
        # print(prediction)
        sil_score_list_bank.append(silhouette_score(X_train_bank, prediction))
        cal_har_score_list_bank.append(calinski_harabasz_score(X_train_bank, prediction))
        davies_bouldin_score_list_bank.append(davies_bouldin_score(X_train_bank, prediction))

    with open('experiment_best.json') as f:
        params = json.load(f)
    if args.dimensionality is None:
        num_clusters_smart = params['k_means']['smart']
        num_clusters_bank = params['k_means']['bank']
    else:
        num_clusters_smart = params[args.dimensionality[0]]['k_means']['smart']
        num_clusters_bank = params[args.dimensionality[0]]['k_means']['bank']

    # Scale these for plotting
    cal_har_score_list_smart = [x / 500 for x in cal_har_score_list_smart]
    cal_har_score_list_bank = [x / 500 for x in cal_har_score_list_bank]
    davies_bouldin_score_list_smart = [x / 5 for x in davies_bouldin_score_list_smart]
    davies_bouldin_score_list_bank = [x / 5 for x in davies_bouldin_score_list_bank]


    plt.rc("font", size=8)
    plt.rc("axes", titlesize=12)
    plt.rc("axes", labelsize=10)
    plt.rc("xtick", labelsize=8)
    plt.rc("ytick", labelsize=8)
    plt.rc("legend", fontsize=8)
    plt.rc("figure", titlesize=11)
    #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
    fig, ax = plt.subplots(1,4,figsize=(15,4))
    fig.suptitle('K-Means Clusters - # of clusters Analysis (Left: Smart Grid, Right: Bank Loan)', fontsize=14)
    # ax[0].plot(problem_size_list, sa_fitness_list, 'b-', label='Simulated Annealing', linewidth=1)
    # ax[0].plot(problem_size_list, ga_fitness_list, 'g:', label='Genetic', linewidth=1)
    ax[0].plot(num_clusters_list, sil_score_list_smart, 'b-', label='Silhouette', linewidth=1)
    ax[0].plot(num_clusters_list, cal_har_score_list_smart, 'r--', label='Calinksi-Harabasz / 500', linewidth=1)
    ax[0].plot(num_clusters_list, davies_bouldin_score_list_smart, 'g-.', label='Davies-Bouldin / 5', linewidth=1)
    ax[0].set(xlabel='K (# of clusters)', ylabel = 'Scores')
    ax[0].set_title('Clustering Scores')
    ax[0].legend()

    k_means = KMeans(n_clusters=num_clusters_smart, random_state=27)
    k_means.fit(X_train_smart)
    prediction_smart = k_means.predict(X_train_smart)
    tsne = TSNEVisualizer(decompose_by=X_train_smart.shape[1] - 1, ax=ax[1], random_state=27)
    tsne.fit(X_train_smart, ["c{}".format(c) for c in prediction_smart])
    ax[1].set_title('tSNE Projection (clusters = {0})'.format(num_clusters_smart))
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])

    ax[2].plot(num_clusters_list, sil_score_list_bank, 'b-', label='Silhouette', linewidth=1)
    ax[2].plot(num_clusters_list, cal_har_score_list_bank, 'r--', label='Calinksi-Harabasz / 5d00', linewidth=1)
    ax[2].plot(num_clusters_list, davies_bouldin_score_list_bank, 'g-.', label='Davies-Bouldin / 5', linewidth=1)
    ax[2].set(xlabel='K (# of clusters)', ylabel = 'Scores')
    ax[2].set_title('Clustering Scores')
    ax[2].legend()

    k_means = KMeans(n_clusters=num_clusters_bank, random_state=27)
    k_means.fit(X_train_bank)
    prediction_bank = k_means.predict(X_train_bank)
    tsne_bank = TSNEVisualizer(decompose_by=X_train_bank.shape[1] - 1, ax=ax[3], random_state=27)
    tsne_bank.fit(X_train_bank, ["c{}".format(c) for c in prediction_bank])
    ax[3].set_title('tSNE Projection (clusters = {0})'.format(num_clusters_bank))
    ax[3].set_xticklabels([])
    ax[3].set_yticklabels([])

    plt.show()

    # Boosting validation
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
    # boost_accuracy = accuracy(boosting_learner, y_test, boost_pred)
    # print('Boosting baseline test set predict accuracy: ' + str(boost_accuracy))

    boosting_learner = AdaBoostClassifier(learning_rate=1, n_estimators=100)
    boost_fit_t = time()
    boosting_learner.fit(X_train_smart, prediction_smart)
    boost_fit_time = time() - boost_fit_t
    print('Boosting DR + cluster fit time (smart): ' + str(boost_fit_time))
    boost_pred_t = time()
    boost_pred = boosting_learner.predict(X_test_smart)
    boost_pred_time = time() - boost_pred_t
    print('Boosting DR + cluster predict time (smart): ' + str(boost_pred_time))
    boost_score = cross_val_score(boosting_learner, X_train_smart, prediction_smart, cv=10)
    print('Boosting DR + cluster cross validation score (smart): ' + str(np.mean(boost_score)))

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
    boosting_learner.fit(X_train_bank, prediction_bank)
    boost_fit_time = time() - boost_fit_t
    print('Boosting DR + cluster fit time (bank): ' + str(boost_fit_time))
    boost_pred_t = time()
    boost_pred = boosting_learner.predict(X_test_bank)
    boost_pred_time = time() - boost_pred_t
    print('Boosting DR + cluster predict time (bank): ' + str(boost_pred_time))
    boost_score = cross_val_score(boosting_learner, X_train_bank, prediction_bank, cv=10)
    print('Boosting DR + cluster cross validation score (bank): ' + str(np.mean(boost_score)))

    return

    # SIL SCORE -> HIGH, CH SCORE -> HIGH, DB SCORE -> LOW
    # print(sil_score_list)
    # print('Sil Score = {0}'.format(sil_score))
    # score = k_means.score(X_test)
    # k_means = KMeans(random_state=27)
    # fig, ax = plt.subplots(1,1,figsize=(12,3.5))
    # visualizer = KElbowVisualizer(k_means, ax=ax[0], k=(2,20), random_state=27)
    # ax[0].legend()
    # ax[0].set_title('Boner soup')
    # visualizer.fit(X_train)        # Fit the data to the visualizer
    # #visualizer.show()        # Finalize and render the figure

    # elbow = visualizer.elbow_value_
    # k_means = KMeans(n_clusters=elbow)
    # visualizer = SilhouetteVisualizer(k_means, ax=ax[1], colors='yellowbrick', random_state=27)
    # ax[1].legend()
    # ax[1].set_title('fuck yeah')
    # visualizer.fit(X_train)        # Fit the data to the visualizer

    # plt.show()

    # visualizer = InterclusterDistance(k_means, random_state=27)

    # visualizer.fit(X_train)        # Fit the data to the visualizer
    # visualizer.show()        # Finalize and render the figure

    # Kurtosis
    # pd.DataFrame(experiment.dataset.labels).kurt(axis=0).abs().mean()
    # plot over n_components for all 4 DR

    # FOR NEURAL NETS, USE CLUSTER LABELS AS A NEW FEATURE
    # FOR NEURAL NETS, USE MULTIPLE NUMBERS OF CLUSTER SIZES TO PRODUCE NEW FEATURES AND RUN