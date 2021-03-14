from time import time
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
from mlrose_hiive import neural
from mlrose_hiive import runners
from mlrose_hiive import algorithms
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier

# For part 2, neural network, freeze your architecture from ass1
#  - Can be tiny differences, but make sure activation and hyperparameters
#  - mlrose is bad at this, they didn't implement hyperparameters well or completely
#  - If performance is significantly worse, retune or swap APIs
#  - Can use new data sets, retune same as ass1 but dont talk about what you tuned
#  - mlrose will almost always need retuning for any data set, retune with mlrose
#  - You need LCA from Ass1 and then LCAs for different optimizers to compare them
#  - Compare each optimizer vs eachother and backprop results (assignment 1)
#  - Compare performance with time and accuracy

def main(task):

    # Smart Grid
    data = pd.read_csv('smart_grid_2.csv')

    data = data.drop('stab', axis=1)

    y = data['stabf'].copy()
    y= y.astype('category')
    y = y.cat.codes
    X = data.drop('stabf', axis=1).copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=27, stratify=y)


    train_scaler = RobustScaler()
    X_train_scaler = train_scaler.fit(X_train)
    X_train = X_train_scaler.transform(X_train)

    X_test = X_train_scaler.transform(X_test)

    if 'tuning_plots' in task:
        # # Tune RHC
        # restart_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # # restart_list = [0]
        # rhc_accuracy_tuning_list = []
        # rhc_accuracy_test_list = []
        # for restarts in restart_list:
        #     rhc_nn = neural.NeuralNetwork( hidden_nodes=[30, 30],
        #                     activation='relu',
        #                     algorithm='random_hill_climb',
        #                     max_iters=10000,
        #                     bias=True,
        #                     is_classifier=True,
        #                     learning_rate=0.1,
        #                     early_stopping=True,
        #                     clip_max=1e+10,
        #                     restarts=restarts,
        #                     schedule=mlrose.GeomDecay(1),
        #                     pop_size=200,
        #                     mutation_prob=0.1,
        #                     max_attempts=100,
        #                     random_state=27,
        #                     curve=True)
        #     # print(y_train)
        #     rhc_nn.fit(X_train, y_train)

        #     y_train_pred = rhc_nn.predict(X_train)
        #     # print(y_train_pred)
        #     y_train_accuracy = accuracy_score(y_train, y_train_pred)
        #     print(y_train_accuracy)
        #     rhc_accuracy_tuning_list.append(y_train_accuracy)
        #     y_test_pred = rhc_nn.predict(X_test)
        #     y_test_accuracy = accuracy_score(y_test, y_test_pred)
        #     rhc_accuracy_test_list.append(y_test_accuracy)
        # print(rhc_accuracy_tuning_list)

        # temperature_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # #temperature_list = [1, 5, 10]
        # sa_accuracy_tuning_list = []
        # sa_accuracy_test_list = []
        # for temp in temperature_list:
        #     sa_nn = neural.NeuralNetwork( hidden_nodes=[30, 30],
        #                 activation='relu',
        #                 algorithm='simulated_annealing',
        #                 max_iters=10000,
        #                 bias=True,
        #                 is_classifier=True,
        #                 learning_rate=0.1,
        #                 early_stopping=True,
        #                 clip_max=1e+10,
        #                 restarts=1,
        #                 schedule=mlrose.GeomDecay(temp),
        #                 pop_size=200,
        #                 mutation_prob=0.1,
        #                 max_attempts=500,
        #                 random_state=27,
        #                 curve=True)
        #     sa_nn.fit(X_train, y_train)

        #     y_train_pred = sa_nn.predict(X_train)
        #     y_train_accuracy = accuracy_score(y_train, y_train_pred)
        #     print(y_train_accuracy)
        #     sa_accuracy_tuning_list.append(y_train_accuracy)
        #     y_test_pred = sa_nn.predict(X_test)
        #     y_test_accuracy = accuracy_score(y_test, y_test_pred)
        #     sa_accuracy_test_list.append(y_test_accuracy)
        # print(sa_accuracy_tuning_list)

        # population_size_list = [25, 50, 75, 100, 125, 150, 175, 200]
        population_size_list = [100]
        ga_accuracy_tuning_list = []
        ga_accuracy_test_list = []
        for population in population_size_list:
            ga_nn = neural.NeuralNetwork( hidden_nodes=[30, 30],
                        activation='relu',
                        algorithm='genetic_alg',
                        max_iters=5000,
                        bias=True,
                        is_classifier=True,
                        learning_rate=0.001,
                        early_stopping=True,
                        clip_max=1e+10,
                        restarts=0,
                        schedule=mlrose.GeomDecay(1),
                        pop_size=population,
                        mutation_prob=0.1,
                        max_attempts=200,
                        random_state=27,
                        curve=True)
            ga_nn.fit(X_train, y_train)

            y_train_pred = ga_nn.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
            print(y_train_accuracy)
            ga_accuracy_tuning_list.append(y_train_accuracy)
            y_test_pred = ga_nn.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            ga_accuracy_test_list.append(y_test_accuracy)
        print(ga_accuracy_tuning_list)

        plt.rc("font", size=8)
        plt.rc("axes", titlesize=10)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", fontsize=8)
        plt.rc("figure", titlesize=11)
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        plt.ylim([0.7, 1.0])
        fig.suptitle('Neural Networks Optimizer Tuning', fontsize=14)

        ax[0].scatter(restart_list, rhc_accuracy_tuning_list, label='Training', c='r', marker='x', s=10)
        ax[0].scatter(restart_list, rhc_accuracy_test_list, label='Test', c='g', marker='o', s=10)
        ax[0].set(xlabel='Restarts', ylabel = 'Accuracy', title='RHC Restarts')
        ax[0].legend()

        ax[1].scatter(temperature_list, sa_accuracy_tuning_list, label='Training', c='r', marker='x', s=10)
        ax[1].scatter(temperature_list, sa_accuracy_test_list, label='Test', c='g', marker='o', s=10)
        ax[1].set(xlabel='Temperature', ylabel = 'Accuracy', title='SA Temperature')
        ax[1].legend()

        ax[2].scatter(population_size_list, ga_accuracy_tuning_list, label='Training', c='r', marker='x', s=10)
        ax[2].scatter(population_size_list, ga_accuracy_test_list, label='Test', c='g', marker='o', s=10)
        ax[2].set(xlabel='Population Size', ylabel = 'Accuracy', title='GA Population Size')
        ax[2].legend()
        ax[2].yaxis.tick_right()


        plt.show()

    if 'performance_graph' in task:
        cv=5
        n_jobs=1
        train_sizes=np.linspace(.2, 1.0, 5)

        # Assignment 1 learner (back propagation)
        bp_nn = MLPClassifier(hidden_layer_sizes=[30, 30], max_iter=100, random_state=27)
        bp_fit_t = time()
        bp_nn.fit(X_train, y_train)
        bp_fit_time = time() - bp_fit_t
        print('Back Propagation fit time: ' + str(bp_fit_time))
        bp_pred_t = time()
        bp_nn_pred = bp_nn.predict(X_test)
        bp_pred_time = time() - bp_pred_t
        print('Back Propagation predict time: ' + str(bp_pred_time))
        bp_nn_loss = bp_nn.loss_
        print('Back Propagation Loss: ' + str(bp_nn_loss))
        bp_loss_curve = bp_nn.loss_curve_

        print('Generating learning curve...')
        bp_train_sizes, bp_train_scores, bp_test_scores, bp_fit_times, _ = \
        learning_curve(bp_nn, X_train, y_train, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        bp_train_scores_mean = np.mean(bp_train_scores, axis=1)
        bp_train_scores_std = np.std(bp_train_scores, axis=1)
        bp_test_scores_mean = np.mean(bp_test_scores, axis=1)
        bp_test_scores_std = np.std(bp_test_scores, axis=1)
        bp_fit_times_mean = np.mean(bp_fit_times, axis=1)
        bp_fit_times_std = np.std(bp_fit_times, axis=1)

        # Random Hill Climb
        print('Random Hill Climb...')
        rhc_nn = neural.NeuralNetwork( hidden_nodes=[30, 30],
                    activation='relu',
                    algorithm='random_hill_climb',
                    max_iters=10000,
                    bias=True,
                    is_classifier=True,
                    learning_rate=0.5,
                    early_stopping=True,
                    clip_max=1e+10,
                    restarts=2,
                    schedule=mlrose.GeomDecay(1),
                    pop_size=200,
                    mutation_prob=0.1,
                    max_attempts=200,
                    random_state=27,
                    curve=True)
        rhc_fit_t = time()
        rhc_nn.fit(X_train, y_train)
        rhc_fit_time = time() - rhc_fit_t
        print('Random Hill Climb fit time: ' + str(rhc_fit_time))
        rhc_pred_t = time()
        rhc_nn_pred = rhc_nn.predict(X_test)
        rhc_pred_time = time() - rhc_pred_t
        print('Random Hill Climb predict time: ' + str(rhc_pred_time))
        rhc_nn_loss = rhc_nn.loss
        print('Random Hill Climb Loss: ' + str(rhc_nn_loss))

        rhc_loss_curve = rhc_nn.fitness_curve

        print('Generating learning curve...')
        rhc_train_sizes, rhc_train_scores, rhc_test_scores, rhc_fit_times, _ = \
        learning_curve(rhc_nn, X_train, y_train, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        rhc_train_scores_mean = np.mean(rhc_train_scores, axis=1)
        rhc_train_scores_std = np.std(rhc_train_scores, axis=1)
        rhc_test_scores_mean = np.mean(rhc_test_scores, axis=1)
        rhc_test_scores_std = np.std(rhc_test_scores, axis=1)
        rhc_fit_times_mean = np.mean(rhc_fit_times, axis=1)
        rhc_fit_times_std = np.std(rhc_fit_times, axis=1)

        print('Simulated Annealing...')
        # Simluated Annealing
        sa_nn = neural.NeuralNetwork( hidden_nodes=[30, 30],
                    activation='relu',
                    algorithm='simulated_annealing',
                    max_iters=15000,
                    bias=True,
                    is_classifier=True,
                    learning_rate=0.5,
                    early_stopping=True,
                    clip_max=1e+10,
                    restarts=0,
                    schedule=mlrose.GeomDecay(2),
                    pop_size=200,
                    mutation_prob=0.1,
                    max_attempts=200,
                    random_state=27,
                    curve=True)
        sa_fit_t = time()
        sa_nn.fit(X_train, y_train)
        sa_fit_time = time() - sa_fit_t
        print('Simulated Annealing fit time: ' + str(sa_fit_time))
        sa_pred_t = time()
        sa_nn_pred = sa_nn.predict(X_test)
        sa_pred_time = time() - sa_pred_t
        print('Simulated Annealing predict time: ' + str(sa_pred_time))
        sa_nn_loss = sa_nn.loss
        print('Simulated Annealing Loss: ' + str(sa_nn_loss))
        sa_loss_curve = sa_nn.fitness_curve

        print('Generating learning curve...')
        sa_train_sizes, sa_train_scores, sa_test_scores, sa_fit_times, _ = \
        learning_curve(sa_nn, X_train, y_train, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        sa_train_scores_mean = np.mean(sa_train_scores, axis=1)
        sa_train_scores_std = np.std(sa_train_scores, axis=1)
        sa_test_scores_mean = np.mean(sa_test_scores, axis=1)
        sa_test_scores_std = np.std(sa_test_scores, axis=1)
        sa_fit_times_mean = np.mean(sa_fit_times, axis=1)
        sa_fit_times_std = np.std(sa_fit_times, axis=1)

        # Genetic Algorithm
        print('Genetic Algorithm')
        ga_nn = neural.NeuralNetwork( hidden_nodes=[30, 30],
                    activation='relu',
                    algorithm='genetic_alg',
                    max_iters=10000,
                    bias=True,
                    is_classifier=True,
                    learning_rate=0.05,
                    early_stopping=True,
                    clip_max=1e+10,
                    restarts=0,
                    schedule=mlrose.GeomDecay(1),
                    pop_size=100,
                    mutation_prob=0.1,
                    max_attempts=200,
                    random_state=27,
                    curve=True)
        ga_fit_t = time()
        ga_nn.fit(X_train, y_train)
        ga_fit_time = time() - ga_fit_t
        print('Genetic Algorithm fit time: ' + str(ga_fit_time))
        ga_pred_t = time()
        ga_nn_pred = ga_nn.predict(X_test)
        ga_pred_time = time() - ga_pred_t
        print('Genetic Algorithm predict time: ' + str(ga_pred_time))
        ga_nn_loss = ga_nn.loss
        print('Genetic Algorithm Loss: ' + str(ga_nn_loss))
        ga_loss_curve = ga_nn.fitness_curve

        print('Generating learning curve...')
        ga_train_sizes, ga_train_scores, ga_test_scores, ga_fit_times, _ = \
        learning_curve(ga_nn, X_train, y_train, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        ga_train_scores_mean = np.mean(ga_train_scores, axis=1)
        ga_train_scores_std = np.std(ga_train_scores, axis=1)
        ga_test_scores_mean = np.mean(ga_test_scores, axis=1)
        ga_test_scores_std = np.std(ga_test_scores, axis=1)
        ga_fit_times_mean = np.mean(ga_fit_times, axis=1)
        ga_fit_times_std = np.std(ga_fit_times, axis=1)


        # Plot learning curve
        plt.rc("font", size=8)
        plt.rc("axes", titlesize=10)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", fontsize=8)
        plt.rc("figure", titlesize=11)

        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        fig.suptitle('Neural Network Learning Curves', fontsize=14)

        axes[0].set(xlabel='Training Examples', ylabel = 'Accuracy', title='Back Propagation')
        axes[0].grid()
        axes[0].fill_between(bp_train_sizes, bp_train_scores_mean - bp_train_scores_std,
                            bp_train_scores_mean + bp_train_scores_std, alpha=0.1,
                            color="r")
        axes[0].fill_between(bp_train_sizes, bp_test_scores_mean - bp_test_scores_std,
                            bp_test_scores_mean + bp_test_scores_std, alpha=0.1,
                            color="g")
        axes[0].plot(bp_train_sizes, bp_train_scores_mean, 'o-', color="r",
                    label="Training")
        axes[0].plot(bp_train_sizes, bp_test_scores_mean, 'o-', color="g",
                    label="Cross-validation")
        axes[0].legend(loc="best")

        axes[1].set(xlabel='Training Examples', title='Random Hill Climb')
        axes[1].grid()
        axes[1].fill_between(rhc_train_sizes, rhc_train_scores_mean - rhc_train_scores_std,
                            rhc_train_scores_mean + rhc_train_scores_std, alpha=0.1,
                            color="r")
        axes[1].fill_between(rhc_train_sizes, rhc_test_scores_mean - rhc_test_scores_std,
                            rhc_test_scores_mean + rhc_test_scores_std, alpha=0.1,
                            color="g")
        axes[1].plot(rhc_train_sizes, rhc_train_scores_mean, 'o-', color="r",
                    label="Training")
        axes[1].plot(rhc_train_sizes, rhc_test_scores_mean, 'o-', color="g",
                    label="Cross-validation")
        axes[1].legend(loc="best")

        axes[2].set(xlabel='Training Examples', title='Simulated Annealing')
        axes[2].grid()
        axes[2].fill_between(sa_train_sizes, sa_train_scores_mean - sa_train_scores_std,
                            sa_train_scores_mean + sa_train_scores_std, alpha=0.1,
                            color="r")
        axes[2].fill_between(sa_train_sizes, sa_test_scores_mean - sa_test_scores_std,
                            sa_test_scores_mean + sa_test_scores_std, alpha=0.1,
                            color="g")
        axes[2].plot(sa_train_sizes, sa_train_scores_mean, 'o-', color="r",
                    label="Training")
        axes[2].plot(sa_train_sizes, sa_test_scores_mean, 'o-', color="g",
                    label="Cross-validation")
        axes[2].legend(loc="best")

        axes[3].set(xlabel='Training Examples', title='Genetic Algorithm')
        axes[3].grid()
        axes[3].fill_between(ga_train_sizes, ga_train_scores_mean - ga_train_scores_std,
                            ga_train_scores_mean + ga_train_scores_std, alpha=0.1,
                            color="r")
        axes[3].fill_between(ga_train_sizes, ga_test_scores_mean - ga_test_scores_std,
                            ga_test_scores_mean + ga_test_scores_std, alpha=0.1,
                            color="g")
        axes[3].plot(ga_train_sizes, ga_train_scores_mean, 'o-', color="r",
                    label="Training")
        axes[3].plot(ga_train_sizes, ga_test_scores_mean, 'o-', color="g",
                    label="Cross-validation")
        axes[3].legend(loc="best")


        plt.show()
        plt.savefig('nn_performance_graph', format='png')

        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        fig.suptitle('Neural Network Loss Curves', fontsize=14)

        axes[0].plot(bp_loss_curve)
        axes[0].set(xlabel='Iteration', ylabel = 'Loss', title='Back Propagation')

        axes[1].plot(rhc_loss_curve)
        axes[1].set(xlabel='Iteration', title='Random Hill Climb')

        axes[2].plot(sa_loss_curve)
        axes[2].set(xlabel='Iteration', title='Simulated Annealing')

        axes[3].plot(ga_loss_curve)
        axes[3].set(xlabel='Iteration', title='Genetic Algorithm')

        plt.show()
        plt.savefig('nn_performance_graph', format='png')
    return

