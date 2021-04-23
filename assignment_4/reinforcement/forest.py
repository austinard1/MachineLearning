import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
# from hiivemdptoolbox.hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
from reinforcement.mdp import ValueIteration, PolicyIteration, QLearning
from hiivemdptoolbox.hiive.visualization.mdpviz import mdp_env, mdp_spec, utils
import mdptoolbox.example

np.random.seed(27)

def main(task):
    #alpha: is the learning rate, set generally between 0 and 1. Setting the alpha value to 0 means that the Q-values are never updated, thereby nothing is learned. If we set the alpha to a high value such as 0.9, it means that the learning can occur quickly.

    # gamma: It is the discount factor that is set between 0 and 1. This model the fact that future rewards are worth less than immediate rewards.
    if 'tuning_plots' in task:
        problem_size_list = [100, 2500]
        for problem_size in problem_size_list:
            P, R = mdptoolbox.example.forest(S=problem_size)

            # Value Iteration
            vi_max_value_tuning_list = []
            vi_mean_value_tuning_list = []
            vi_time_tuning_list = []
            vi_iteration_tuning_list = []
            epsilon_list = np.arange(0.0001, 0.1, 0.0005)
            for epsilon in epsilon_list:
            # Epsilon here neeeds to be smaller for larger problem sizes (stricter stopping criteria)
                vi = ValueIteration(P, R, gamma=0.9, epsilon=epsilon, max_iter=10000)
                vi_run_stats = vi.run()
                vi_df = pd.DataFrame(vi_run_stats)
                print(vi_df)
                vi_max_value_tuning_list.append(vi_df.loc[vi_df['Max V'].idxmax()]['Max V'])
                vi_mean_value_tuning_list.append(vi_df.loc[vi_df['Mean V'].idxmax()]['Mean V'])
                vi_time_tuning_list.append(vi_df.loc[vi_df['Time'].idxmax()]['Time'])
                vi_iteration_tuning_list.append(vi_df.loc[vi_df['Iteration'].idxmax()]['Iteration'])

            plt.rc("font", size=8)
            plt.rc("axes", titlesize=12)
            plt.rc("axes", labelsize=10)
            plt.rc("xtick", labelsize=8)
            plt.rc("ytick", labelsize=8)
            plt.rc("legend", fontsize=8)
            plt.rc("figure", titlesize=11)
            #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
            fig, ax = plt.subplots(1,4,figsize=(16,3.5))
            fig.suptitle('Value Iteration Epsilon/Gamma Tuning, problem_size = ' + str(problem_size))
            ax[0].scatter(epsilon_list, vi_max_value_tuning_list, c='g', marker='o', s=10, label='Max Value')
            ax[0].scatter(epsilon_list, vi_mean_value_tuning_list, c='orange', marker='o', s=10, label='Mean Value')
            ax[0].set(xlabel='Epsilon', ylabel = 'Value')
            ax[0].legend()

            # ax[1].scatter(epsilon_list, vi_time_tuning_list, c='r', marker='x', s=10)
            # ax[1].set(xlabel='Epsilon', ylabel = 'Time')

            ax[1].scatter(epsilon_list, vi_iteration_tuning_list, c='b', marker='o', s=10)
            ax[1].set(xlabel='Epsilon', ylabel = 'Iterations')
            # ax[1].yaxis.tick_right()

            # plt.show()

            vi_max_value_tuning_list = []
            vi_mean_value_tuning_list = []
            vi_time_tuning_list = []
            vi_iteration_tuning_list = []
            gamma_list = np.arange(0.01, 1.0, 0.01)
            for gamma in gamma_list:
            # Epsilon here neeeds to be smaller for larger problem sizes (stricter stopping criteria)
                vi = ValueIteration(P, R, gamma=gamma, epsilon=0.04, max_iter=10000)
                vi_run_stats = vi.run()
                vi_df = pd.DataFrame(vi_run_stats)
                print(vi_df)
                vi_max_value_tuning_list.append(vi_df.loc[vi_df['Max V'].idxmax()]['Max V'])
                vi_mean_value_tuning_list.append(vi_df.loc[vi_df['Mean V'].idxmax()]['Mean V'])
                vi_time_tuning_list.append(vi_df.loc[vi_df['Time'].idxmax()]['Time'])
                vi_iteration_tuning_list.append(vi_df.loc[vi_df['Iteration'].idxmax()]['Iteration'])

            plt.rc("font", size=8)
            plt.rc("axes", titlesize=12)
            plt.rc("axes", labelsize=10)
            plt.rc("xtick", labelsize=8)
            plt.rc("ytick", labelsize=8)
            plt.rc("legend", fontsize=8)
            plt.rc("figure", titlesize=11)
            #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
            # fig, ax = plt.subplots(1,3,figsize=(14,3.5))
            # fig.suptitle('Value Iteration Gamma Tuning, problem_size = ' + str(problem_size))
            ax[2].scatter(gamma_list, vi_max_value_tuning_list, c='g', marker='o', s=10, label='Max Value')
            ax[2].scatter(gamma_list, vi_mean_value_tuning_list, c='orange', marker='o', s=10, label='Mean Value')
            ax[2].set(xlabel='Gamma', ylabel = 'Value')
            ax[2].legend()

            # ax[1].scatter(gamma_list, vi_time_tuning_list, c='r', marker='x', s=10)
            # ax[1].set(xlabel='Gamma', ylabel = 'Time')

            ax[3].scatter(gamma_list, vi_iteration_tuning_list, c='b', marker='o', s=10)
            ax[3].set(xlabel='Gamma', ylabel = 'Iterations')
            # ax[3].yaxis.tick_right()

            plt.show()
            # # exit()
            # print(vi.run_stats)
            # print(vi.run_stats['Time'])
            # print(vi.policy)
            # print(vi.V)
            # print(vi.iter)

            # Policy Iteration
            pi_max_value_tuning_list = []
            pi_mean_value_tuning_list = []
            pi_time_tuning_list = []
            pi_iteration_tuning_list = []
            epsilon_list = np.arange(0.0001, 0.1, 0.0005)
            for epsilon in epsilon_list:
            # Epsilon here neeeds to be smaller for larger problem sizes (stricter stopping criteria)
                pi = PolicyIteration(P, R, gamma=0.9, epsilon=epsilon, max_iter=10000)
                pi_run_stats = pi.run()
                pi_df = pd.DataFrame(pi_run_stats)
                print(pi_df)
                pi_max_value_tuning_list.append(pi_df.loc[pi_df['Max V'].idxmax()]['Max V'])
                pi_mean_value_tuning_list.append(pi_df.loc[pi_df['Mean V'].idxmax()]['Mean V'])
                pi_time_tuning_list.append(pi_df.loc[pi_df['Time'].idxmax()]['Time'])
                pi_iteration_tuning_list.append(pi_df.loc[pi_df['Iteration'].idxmax()]['Iteration'])

            plt.rc("font", size=8)
            plt.rc("axes", titlesize=12)
            plt.rc("axes", labelsize=10)
            plt.rc("xtick", labelsize=8)
            plt.rc("ytick", labelsize=8)
            plt.rc("legend", fontsize=8)
            plt.rc("figure", titlesize=11)
            #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
            fig, ax = plt.subplots(1,4,figsize=(16,3.5))
            fig.suptitle('Policy Iteration Epsilon/Gamma Tuning, problem_size = ' + str(problem_size))

            ax[0].scatter(epsilon_list, pi_max_value_tuning_list, c='g', marker='o', s=10, label='Max Value')
            ax[0].scatter(epsilon_list, pi_mean_value_tuning_list, c='orange', marker='o', s=10, label='Mean Value')
            ax[0].set(xlabel='Epsilon', ylabel = 'Value')
            ax[0].legend()

            # ax[1].scatter(epsilon_list, pi_time_tuning_list, c='r', marker='x', s=10)
            # ax[1].set(xlabel='Epsilon', ylabel = 'Time')

            ax[1].scatter(epsilon_list, pi_iteration_tuning_list, c='b', marker='o', s=10)
            ax[1].set(xlabel='Epsilon', ylabel = 'Iterations')
            # ax[1].yaxis.tick_right()

            # plt.show()

            pi_max_value_tuning_list = []
            pi_mean_value_tuning_list = []
            pi_time_tuning_list = []
            pi_iteration_tuning_list = []
            gamma_list = np.arange(0.01, 1.0, 0.01)
            for gamma in gamma_list:
            # Epsilon here neeeds to be smaller for larger problem sizes (stricter stopping criteria)
                pi = PolicyIteration(P, R, gamma=gamma, epsilon=0.04, max_iter=10000)
                pi_run_stats = pi.run()
                pi_df = pd.DataFrame(pi_run_stats)
                print(pi_df)
                pi_max_value_tuning_list.append(pi_df.loc[pi_df['Max V'].idxmax()]['Max V'])
                pi_mean_value_tuning_list.append(pi_df.loc[pi_df['Mean V'].idxmax()]['Mean V'])
                pi_time_tuning_list.append(pi_df.loc[pi_df['Time'].idxmax()]['Time'])
                pi_iteration_tuning_list.append(pi_df.loc[pi_df['Iteration'].idxmax()]['Iteration'])

            plt.rc("font", size=8)
            plt.rc("axes", titlesize=12)
            plt.rc("axes", labelsize=10)
            plt.rc("xtick", labelsize=8)
            plt.rc("ytick", labelsize=8)
            plt.rc("legend", fontsize=8)
            plt.rc("figure", titlesize=11)
            #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
            # fig, ax = plt.subplots(1,3,figsize=(14,3.5))
            # fig.suptitle('Policy Iteration Gamma Tuning, problem_size = ' + str(problem_size))
            ax[2].scatter(gamma_list, pi_max_value_tuning_list, c='g', marker='o', s=10, label='Max Value')
            ax[2].scatter(gamma_list, pi_mean_value_tuning_list, c='orange', marker='o', s=10, label='Mean Value')
            ax[2].set(xlabel='Gamma', ylabel = 'Value')
            ax[2].legend()

            # ax[1].scatter(gamma_list, pi_time_tuning_list, c='r', marker='x', s=10)
            # ax[1].set(xlabel='Gamma', ylabel = 'Time')

            ax[3].scatter(gamma_list, pi_iteration_tuning_list, c='b', marker='o', s=10)
            ax[3].set(xlabel='Gamma', ylabel = 'Iterations')
            # ax[3].yaxis.tick_right()

            plt.show()
            # exit()

            # Q-Learning
            ql_max_value_tuning_list = []
            ql_mean_value_tuning_list = []
            ql_time_tuning_list = []
            ql_iteration_tuning_list = []
            epsilon_decay_list = np.arange(0.999995, 1.0, 0.0000005)
            # epsilon_decay_list = np.linspace(0.9999, 0.99999, 20)
            # epsilon_decay_list = np.linspace(0.9999, 0.99999, 50)
            for epsilon_decay in epsilon_decay_list:
                # Epsilon is exploration vs. exploitation, similar to temperature for simulated annealing
                ql = QLearning(P, R, gamma=0.9, n_iter=1000000, alpha=1.0, alpha_decay=1.0, alpha_min=0.001, epsilon=1.0, epsilon_min=0.1, epsilon_decay=epsilon_decay, run_stat_frequency=1000, stop_criteria=0.000001)
                ql_run_stats = ql.run()
                ql_df = pd.DataFrame(ql_run_stats)
                print(ql_df)
                ql_max_value_tuning_list.append(ql_df.loc[ql_df['Max V'].idxmax()]['Max V'])
                ql_mean_value_tuning_list.append(ql_df.loc[ql_df['Mean V'].idxmax()]['Mean V'])
                ql_time_tuning_list.append(ql_df.loc[ql_df['Time'].idxmax()]['Time'])
                ql_iteration_tuning_list.append(ql_df.loc[ql_df['Iteration'].idxmax()]['Iteration'])

            plt.rc("font", size=8)
            plt.rc("axes", titlesize=12)
            plt.rc("axes", labelsize=10)
            plt.rc("xtick", labelsize=8)
            plt.rc("ytick", labelsize=8)
            plt.rc("legend", fontsize=8)
            plt.rc("figure", titlesize=11)
            #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
            fig, ax = plt.subplots(1,4,figsize=(16,3.5))
            fig.suptitle('Q-Learning Epsilon Decay/Gamma Tuning, problem_size = ' + str(problem_size))

            ax[0].scatter(epsilon_decay_list, ql_max_value_tuning_list, c='g', marker='o', s=10, label='Max Value')
            ax[0].scatter(epsilon_decay_list, ql_mean_value_tuning_list, c='orange', marker='o', s=10, label='Mean Value')
            ax[0].set(xlabel='Epsilon Decay', ylabel = 'Value')
            ax[0].legend()

            # ax[1].scatter(epsilon_decay_list, ql_time_tuning_list, c='r', marker='x', s=10)
            # ax[1].set(xlabel='Epsilon Decay', ylabel = 'Time')

            ax[1].scatter(epsilon_decay_list, ql_iteration_tuning_list, c='b', marker='o', s=10)
            ax[1].set(xlabel='Epsilon Decay', ylabel = 'Iterations')
            # ax[1].yaxis.tick_right()

            # plt.show()

            # ql_max_value_tuning_list = []
            # ql_mean_value_tuning_list = []
            # ql_time_tuning_list = []
            # ql_iteration_tuning_list = []
            # alpha_list = np.linspace(0.001, 1.0, 20)
            # for alpha in alpha_list:
            #     # Epsilon is exploration vs. exploitation, similar to temperature for simulated annealing
            #     ql = QLearning(P, R, gamma=1.0, n_iter=10000000, alpha=alpha, alpha_decay=1, alpha_min=0.001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9999, run_stat_frequency=1000, stop_criteria=0.000001)
            #     ql_run_stats = ql.run()
            #     ql_df = pd.DataFrame(ql_run_stats)
            #     #     print(ql_df)
            #     ql_max_value_tuning_list.append(ql_df.loc[ql_df['Max V'].idxmax()]['Max V'])
            #     ql_mean_value_tuning_list.append(ql_df.loc[ql_df['Mean V'].idxmax()]['Mean V'])
            #     ql_time_tuning_list.append(ql_df.loc[ql_df['Time'].idxmax()]['Time'])
            #     ql_iteration_tuning_list.append(ql_df.loc[ql_df['Iteration'].idxmax()]['Iteration'])

            # plt.rc("font", size=8)
            # plt.rc("axes", titlesize=12)
            # plt.rc("axes", labelsize=10)
            # plt.rc("xtick", labelsize=8)
            # plt.rc("ytick", labelsize=8)
            # plt.rc("legend", fontsize=8)
            # plt.rc("figure", titlesize=11)
            # #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
            # fig, ax = plt.subplots(1,3,figsize=(14,3.5))
            # fig.suptitle('Q-Learning Alpha Tuning, problem_size = ' + str(problem_size))

            # ax[0].scatter(alpha_list, ql_max_value_tuning_list, c='g', marker='o', s=10, label='Max Value')
            # ax[0].scatter(alpha_list, ql_mean_value_tuning_list, c='orange', marker='o', s=10, label='Mean Value')
            # ax[0].set(xlabel='Alpha', ylabel = 'Value')
            # ax[0].legend()

            # ax[1].scatter(alpha_list, ql_time_tuning_list, c='r', marker='x', s=10)
            # ax[1].set(xlabel='Alpha', ylabel = 'Time')

            # ax[2].scatter(alpha_list, ql_iteration_tuning_list, c='b', marker='o', s=10)
            # ax[2].set(xlabel='Alpha', ylabel = 'Iterations')
            # ax[2].yaxis.tick_right()

            # plt.show()

            ql_max_value_tuning_list = []
            ql_mean_value_tuning_list = []
            ql_time_tuning_list = []
            ql_iteration_tuning_list = []
            gamma_list = np.linspace(0.01, 0.99, 20)
            for gamma in gamma_list:
                # Epsilon is exploration vs. exploitation, similar to temperature for simulated annealing
                ql = QLearning(P, R, gamma=gamma, n_iter=1000000, alpha=1.0, alpha_decay=1, alpha_min=0.001, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999999, run_stat_frequency=1000, stop_criteria=0.000001)
                ql_run_stats = ql.run()
                ql_df = pd.DataFrame(ql_run_stats)
                #     print(ql_df)
                ql_max_value_tuning_list.append(ql_df.loc[ql_df['Max V'].idxmax()]['Max V'])
                ql_mean_value_tuning_list.append(ql_df.loc[ql_df['Mean V'].idxmax()]['Mean V'])
                ql_time_tuning_list.append(ql_df.loc[ql_df['Time'].idxmax()]['Time'])
                ql_iteration_tuning_list.append(ql_df.loc[ql_df['Iteration'].idxmax()]['Iteration'])

            plt.rc("font", size=8)
            plt.rc("axes", titlesize=12)
            plt.rc("axes", labelsize=10)
            plt.rc("xtick", labelsize=8)
            plt.rc("ytick", labelsize=8)
            plt.rc("legend", fontsize=8)
            plt.rc("figure", titlesize=11)
            #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
            # fig, ax = plt.subplots(1,3,figsize=(14,3.5))
            # fig.suptitle('Q-Learning Gamma Tuning, problem_size = ' + str(problem_size))

            ax[2].scatter(gamma_list, ql_max_value_tuning_list, c='g', marker='o', s=10, label='Max Value')
            ax[2].scatter(gamma_list, ql_mean_value_tuning_list, c='orange', marker='o', s=10, label='Mean Value')
            ax[2].set(xlabel='Gamma', ylabel = 'Value')
            ax[2].legend()

            # ax[1].scatter(gamma_list, ql_time_tuning_list, c='r', marker='x', s=10)
            # ax[1].set(xlabel='Gamma', ylabel = 'Time (s)')

            ax[3].scatter(gamma_list, ql_iteration_tuning_list, c='b', marker='o', s=10)
            ax[3].set(xlabel='Gamma', ylabel = 'Iterations')
            # ax[3].yaxis.tick_right()

            plt.show()
        exit()

    if 'complexity_graph' in task:
        problem_size_list = np.arange(100, 2600, 100)
        vi_time_list = []
        vi_max_value_list = []
        vi_mean_value_list = []
        vi_reward_list = []
        vi_iteration_list = []
        pi_time_list = []
        pi_max_value_list = []
        pi_mean_value_list = []
        pi_reward_list = []
        pi_iteration_list = []
        ql_time_list = []
        ql_max_value_list = []
        ql_mean_value_list = []
        ql_reward_list = []
        ql_iteration_list = []

        for problem_size in problem_size_list:
            P, R = mdptoolbox.example.forest(S=problem_size)

            # Value iteration
            vi = ValueIteration(P, R, gamma=0.9, epsilon=0.04, max_iter=10000)
            vi_run_stats = vi.run()
            vi_df = pd.DataFrame(vi_run_stats)
            print(vi_df)
            vi_max_value_list.append(vi_df.loc[vi_df['Max V'].idxmax()]['Max V'])
            vi_mean_value_list.append(vi_df.loc[vi_df['Mean V'].idxmax()]['Mean V'])
            # vi_reward_list.append(vi_df.loc[vi_df['Mean V'].idxmax()]['Mean V'])
            vi_time_list.append(vi_df.loc[vi_df['Time'].idxmax()]['Time'])
            vi_iteration_list.append(vi_df.loc[vi_df['Iteration'].idxmax()]['Iteration'])

            # Policy iteration
            pi = PolicyIteration(P, R, gamma=0.9, epsilon=0.04, max_iter=10000)
            pi_run_stats = pi.run()
            pi_df = pd.DataFrame(pi_run_stats)
            print(pi_df)
            pi_max_value_list.append(pi_df.loc[pi_df['Max V'].idxmax()]['Max V'])
            pi_mean_value_list.append(pi_df.loc[pi_df['Mean V'].idxmax()]['Mean V'])
            pi_reward_list.append(pi_df.loc[pi_df['Mean V'].idxmax()]['Mean V'])
            pi_time_list.append(pi_df.loc[pi_df['Time'].idxmax()]['Time'])
            pi_iteration_list.append(pi_df.loc[pi_df['Iteration'].idxmax()]['Iteration'])

            # Q-Learning
            ql = QLearning(P, R, gamma=0.9, n_iter=1000000, alpha=1.0, alpha_decay=1, alpha_min=0.001, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999999, run_stat_frequency=1000, stop_criteria=0.000001)
            ql_run_stats = ql.run()
            ql_df = pd.DataFrame(ql_run_stats)
            print(ql_df)
            ql_max_value_list.append(ql_df.loc[ql_df['Max V'].idxmax()]['Max V'])
            ql_mean_value_list.append(ql_df.loc[ql_df['Mean V'].idxmax()]['Mean V'])
            # ql_reward_list.append(ql_df.loc[ql_df['Mean V'].idxmax()]['Mean V'])
            ql_time_list.append(ql_df.loc[ql_df['Time'].idxmax()]['Time'])
            ql_iteration_list.append(ql_df.loc[ql_df['Iteration'].idxmax()]['Iteration'])


        plt.rc("font", size=8)
        plt.rc("axes", titlesize=12)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", fontsize=8)
        plt.rc("figure", titlesize=11)
        #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
        fig, ax = plt.subplots(1,3,figsize=(12,3.5))
        fig.suptitle('Forest Complexity Analysis (VI/PI)', fontsize=14)
        # ax[0].plot(problem_size_list, sa_fitness_list, 'b-', label='Simulated Annealing', linewidth=1)
        # ax[0].plot(problem_size_list, ga_fitness_list, 'g:', label='Genetic', linewidth=1)
        w = 40
        ax[0].bar(problem_size_list - 0.5*w, vi_max_value_list, width=w, color='red', label='VI (Max Value)')
        ax[0].bar(problem_size_list - 0.5*w, vi_mean_value_list, width=w, color='blue', label='VI (Mean Value)')
        ax[0].bar(problem_size_list + 0.5*w, pi_max_value_list, width=w, color='green', label='PI (Max Value)')
        ax[0].bar(problem_size_list + 0.5*w, pi_mean_value_list, width=w, color='orange', label='PI (Mean Value)')
        ax[0].set(xlabel='# of states', ylabel = 'Value')
        ax[0].legend()

        ax[1].plot(problem_size_list, vi_iteration_list, 'g:', label='Value Iteration', linewidth=1)
        ax[1].plot(problem_size_list, pi_iteration_list, 'r--', label='Policy Iteration', linewidth=1)
        ax[1].set(xlabel='# of states', ylabel = 'Iterations')
        ax[1].legend()

        ax[2].plot(problem_size_list, vi_time_list, 'g:', label='Value Iteration', linewidth=1)
        ax[2].plot(problem_size_list, pi_time_list, 'r--', label='Policy Iteration', linewidth=1)
        ax[2].set(xlabel='# of states', ylabel = 'Runtime (s)')
        ax[2].legend()
        ax[2].yaxis.tick_right()
        plt.show()

        fig, ax = plt.subplots(1,3,figsize=(12,3.5))
        fig.suptitle('Forest Complexity Analysis (Q-Learning)', fontsize=14)
        # ax[0].plot(problem_size_list, sa_fitness_list, 'b-', label='Simulated Annealing', linewidth=1)
        # ax[0].plot(problem_size_list, ga_fitness_list, 'g:', label='Genetic', linewidth=1)
        w = 40
        ax[0].bar(problem_size_list, ql_max_value_list, width=w, color='red', label='Max Value')
        ax[0].bar(problem_size_list, ql_mean_value_list, width=w, color='green', label='Mean Value')
        ax[0].set(xlabel='# of states', ylabel = 'Value')
        ax[0].legend()

        ax[1].plot(problem_size_list, ql_iteration_list, 'r--', linewidth=1)
        ax[1].set(xlabel='# of states', ylabel = 'Iterations')
        ax[1].legend()

        ax[2].plot(problem_size_list, ql_time_list, 'r--', linewidth=1)
        ax[2].set(xlabel='# of states', ylabel = 'Runtime (s)')
        ax[2].legend()
        ax[2].yaxis.tick_right()
        plt.show()
    if 'performance_graph' in task:
        problem_size_list = [100, 2500]
        vi_gamma_list = [0.9, 0.9]
        vi_epsilon_list = [0.04, 0.04]
        pi_gamma_list = [0.9, 0.9]
        pi_epsilon_list = [0.04, 0.04]
        ql_epsilon_decay_list = [0.999999, 0.999999]
        ql_alpha_list = [1.0, 1.0]
        for idx in range(2):
            P, R = mdptoolbox.example.forest(S=problem_size_list[idx])

            # Value iteration
            vi = ValueIteration(P, R, gamma=vi_gamma_list[idx], epsilon=vi_epsilon_list[idx], max_iter=10000)
            vi_run_stats = vi.run()
            vi_df = pd.DataFrame(vi_run_stats)
            vi_time =vi_df['Time']
            vi_reward = vi_df['Reward']
            vi_iteration = vi_df['Iteration']
            vi_max_V = vi_df['Max V']
            vi_mean_V = vi_df['Mean V']
            vi_error = vi_df['Error']
            vi_policy = vi_df['Policy']

            # Policy iteration
            pi = PolicyIteration(P, R, gamma=pi_gamma_list[idx], epsilon=pi_epsilon_list[idx], max_iter=10000)
            pi_run_stats = pi.run()
            pi_df = pd.DataFrame(pi_run_stats)
            pi_time =pi_df['Time']
            pi_reward = pi_df['Reward']
            pi_iteration = pi_df['Iteration']
            pi_max_V = pi_df['Max V']
            pi_mean_V = pi_df['Mean V']
            pi_error = pi_df['Error']
            pi_policy = pi_df['Policy']

            # Q-Learning
            ql = QLearning(P, R, gamma=0.9, n_iter=1000000, alpha=ql_alpha_list[idx], alpha_decay=1, alpha_min=0.001, epsilon=1.0, epsilon_min=0.1, epsilon_decay=ql_epsilon_decay_list[idx], run_stat_frequency=1000, stop_criteria=0.000001)
            ql_run_stats = ql.run()
            ql_df = pd.DataFrame(ql_run_stats)
            ql_time =ql_df['Time']
            ql_reward = ql_df['Reward']
            ql_iteration = ql_df['Iteration']
            ql_max_V = ql_df['Max V']
            ql_mean_V = ql_df['Mean V']
            ql_error = ql_df['Error']
            ql_policy = ql_df['Policy']

            ql_vi_policy_diff = np.absolute(ql_policy.apply(lambda x: np.linalg.norm(x)) - np.linalg.norm(vi_policy.values[-1]))
            # print(ql_vi_policy_diff)
            # exit()
            ql_pi_policy_diff = np.absolute(ql_policy.apply(lambda x: np.linalg.norm(x)) - np.linalg.norm(pi_policy.values[-1]))

            plt.rc("font", size=8)
            plt.rc("axes", titlesize=12)
            plt.rc("axes", labelsize=10)
            plt.rc("xtick", labelsize=8)
            plt.rc("ytick", labelsize=8)
            plt.rc("legend", fontsize=8)
            plt.rc("figure", titlesize=11)
            #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
            fig, ax = plt.subplots(1,4,figsize=(16,3.5))
            fig.suptitle('Forest Performance Analysis (VI/PI), problem size = ' + str(problem_size_list[idx]), fontsize=14)
            # ax[0].plot(problem_size_list, sa_fitness_list, 'b-', label='Simulated Annealing', linewidth=1)
            # ax[0].plot(problem_size_list, ga_fitness_list, 'g:', label='Genetic', linewidth=1)
            w = 1
            ax[0].plot(vi_iteration, vi_max_V, 'r--', label='VI (Max Value)', linewidth=1)
            ax[0].plot(vi_iteration, vi_mean_V, 'b--', label='VI (Mean Value)', linewidth=1)
            ax[0].plot(pi_iteration, pi_max_V, 'g:', label='PI (Max Value)', linewidth=1)
            ax[0].plot(pi_iteration, pi_mean_V, ':', color='orange', label='PI (Mean Value)', linewidth=1)
            ax[0].set(xlabel='Iteration', ylabel = 'Value',)
            ax[0].legend()
            #ax[0].set_title('Fitness vs. Iteration')

            ax[1].plot(vi_time, vi_max_V, 'r--', label='VI (Max Value)', linewidth=1)
            ax[1].plot(vi_time, vi_mean_V, 'b--', label='VI (Mean Value)', linewidth=1)
            ax[1].plot(pi_time, pi_max_V, 'g:', label='PI (Max Value)', linewidth=1)
            ax[1].plot(pi_time, pi_mean_V, ':', color='orange', label='PI (Mean Value)', linewidth=1)
            ax[1].set(xlabel='Time (s)')
            ax[1].legend()

            ax[2].plot(vi_iteration, vi_error, 'r--', label='VI', linewidth=1)
            ax[2].plot(pi_iteration, pi_error, 'g:', label='PI', linewidth=1)
            ax[2].set(xlabel='Iteration')
            ax[2].yaxis.tick_right()
            ax[2].legend()
            #ax[1].set_title('Fitness vs. Runtime')

            ax[3].plot(vi_time, vi_error, 'r--', label='VI', linewidth=1)
            ax[3].plot(pi_time, pi_error, 'g:', label='PI', linewidth=1)
            ax[3].set(xlabel='Time (s)', ylabel = 'Error')
            ax[3].yaxis.tick_right()
            ax[3].legend()
            plt.show()

            fig, ax = plt.subplots(1, 4,figsize=(16,3.5))
            fig.suptitle('Forest Performance Analysis (Q-Learning), problem size = ' + str(problem_size_list[idx]), fontsize=14)
            # ax[0].plot(problem_size_list, sa_fitness_list, 'b-', label='Simulated Annealing', linewidth=1)
            # ax[0].plot(problem_size_list, ga_fitness_list, 'g:', label='Genetic', linewidth=1)
            w = 1
            ax[0].plot(ql_iteration, ql_max_V, 'r--', label='Max Value', linewidth=1)
            ax[0].plot(ql_iteration, ql_mean_V, 'g:', label='Mean Value', linewidth=1)
            ax[0].set(xlabel='Iteration', ylabel = 'Value')
            ax[0].legend()
            #ax[0].set_title('Fitness vs. Iteration')

            ax[1].plot(ql_time, ql_max_V, 'r--', label='Max Value', linewidth=1)
            ax[1].plot(ql_time, ql_mean_V, 'g:', label='Mean Value', linewidth=1)
            ax[1].set(xlabel='Time (s)')
            ax[1].legend()

            ax[2].plot(ql_iteration, ql_vi_policy_diff, 'r--', label='diff from VI optimal policy', linewidth=1)
            ax[2].set(xlabel='Iteration')
            ax[2].yaxis.tick_right()
            ax[2].legend()
            #ax[1].set_title('Fitness vs. Runtime')

            ax[3].plot(ql_iteration, ql_pi_policy_diff, 'r--', label='diff from PI optimal policy', linewidth=1)
            ax[3].set(xlabel='Iteration', ylabel = 'Diff of Norms')
            ax[3].yaxis.set_label_position("right")
            ax[3].yaxis.tick_right()
            ax[3].legend()
            plt.show()

    if 'policy_plots' in task:
        color_map = {
            1: 'red',
            0: 'green',
        }
        # direction_map={
        #         3: '⬆',
        #         2: '➡',
        #         1: '⬇',
        #         0: '⬅'
        #     }

        action_map={
                1: 'C',
                0: 'W'
            }
        problem_size_list = [100, 2500]
        # for problem_size in problem_size_list:
        #     P, R = mdptoolbox.example.forest(S=problem_size)

        #     # Value iteration
        #     vi = ValueIteration(P, R, gamma=0.9, epsilon=0.05, max_iter=10000)
        #     vi_run_stats = vi.run()
        #     vi_df = pd.DataFrame(vi_run_stats)
        #     print(vi_df['Mean V'])

        #     # Policy iteration
        #     pi = PolicyIteration(P, R, gamma=0.9, epsilon=0.05, max_iter=10000)
        #     pi_run_stats = pi.run()
        #     pi_df = pd.DataFrame(pi_run_stats)
        #     print(pi_df['Mean V'])
        #     policy = pi.policy
        #     # map_desc = np.asarray(map_lake, dtype='c')
        #     # map_desc = [[c.decode('utf-8') for c in line] for line in map_desc]
        #     plt.rc("font", size=8)
        #     plt.rc("axes", titlesize=12)
        #     plt.rc("axes", labelsize=10)
        #     plt.rc("xtick", labelsize=8)
        #     plt.rc("ytick", labelsize=8)
        #     plt.rc("legend", fontsize=8)
        #     plt.rc("figure", titlesize=11)
        #     fig, ax = plt.subplots(1,2,figsize=(15,4.5))
        #     fig.suptitle('Forest Optimal Policy Visualization (VI/PI), problem_size = ' + str(problem_size), fontsize=14)

        #     if len(policy) > 200:
        #         font_size = 'xx-small'
        #     else:
        #         font_size = 'large'

        #     policy = pi.policy
        #     V = pi.V
        #     length = np.sqrt(len(policy)).astype(int)
        #     ax[0].set(title='Policy Iteration', xlim=(0, length), ylim=(0, length))
        #     ax[0].axis('off')
        #     for i in range(length):
        #         for j in range(length):
        #             x = j
        #             y = length - i - 1
        #             p = plt.Rectangle([x, y], 1, 1, alpha=1)
        #             p.set_facecolor(color_map[policy[length*i + j]])
        #             p.set_edgecolor('black')
        #             ax[0].add_patch(p)

        #             text = ax[0].text(x+0.5, y+0.5, action_map[policy[length*i + j]], size=font_size,
        #                             horizontalalignment='center', verticalalignment='center', color='black')
        #             # text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
        #             #                         path_effects.Normal()])
        #             if len(policy) < 200:
        #                 text = ax[0].text(x+0.5, y+0.2, str(np.around(V[length*i + j], 2)), size=font_size,
        #                             horizontalalignment='center', verticalalignment='center', color='black')
        #             # text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
        #             #                        path_effects.Normal()])

        #     policy = vi.policy
        #     V = vi.V
        #     length = np.sqrt(len(policy)).astype(int)
        #     ax[1].set(title='Value Iteration', xlim=(0, length), ylim=(0, length))
        #     ax[1].axis('off')
        #     for i in range(length):
        #         for j in range(length):
        #             x = j
        #             y = length - i - 1
        #             p = plt.Rectangle([x, y], 1, 1, alpha=1)
        #             p.set_facecolor(color_map[policy[length*i + j]])
        #             p.set_edgecolor('black')
        #             ax[1].add_patch(p)

        #             text = ax[1].text(x+0.5, y+0.5, action_map[policy[length*i + j]], size=font_size,
        #                             horizontalalignment='center', verticalalignment='center', color='black')
        #             # text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
        #             #                         path_effects.Normal()])
        #             if len(policy) < 200:
        #                 text = ax[1].text(x+0.5, y+0.2, str(np.around(V[length*i + j], 2)), size=font_size,
        #                             horizontalalignment='center', verticalalignment='center', color='black')

        #     # policy = ql.policy
        #     # length = np.sqrt(len(policy)).astype(int)
        #     # ax[2].set(title='Q-Learning', xlim=(0, length), ylim=(0, length))
        #     # ax[2].axis('off')
        #     # for i in range(length):
        #     #     for j in range(length):
        #     #         x = j
        #     #         y = length - i - 1
        #     #         p = plt.Rectangle([x, y], 1, 1, alpha=1)
        #     #         p.set_facecolor(color_map[map_desc[i][j]])
        #     #         p.set_edgecolor('black')
        #     #         ax[2].add_patch(p)
        #     #         if map_desc[i][j] == 'H' or map_desc[i][j] == 'G':
        #     #             continue

        #     #         text = ax[2].text(x+0.5, y+0.5, direction_map[policy[length*i + j]], weight='bold', size=font_size,
        #     #                         horizontalalignment='center', verticalalignment='center', color='black')
        #     #         text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
        #     #                                 path_effects.Normal()])

        #     plt.show()
            # exit()
        plt.rc("font", size=8)
        plt.rc("axes", titlesize=12)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", fontsize=8)
        plt.rc("figure", titlesize=11)
        fig, ax = plt.subplots(1,2,figsize=(15,4.5))
        fig.suptitle('Forest Optimal Policy Visualization (Q-Learning)', fontsize=14)

        # Small
        P, R = mdptoolbox.example.forest(S=problem_size_list[0])
        ql = QLearning(P, R, gamma=0.9, n_iter=1000000, alpha=1.0, alpha_decay=1.0, alpha_min=0.001, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999999, run_stat_frequency=1000, stop_criteria=0.000001)
        ql.run()

        policy = ql.policy
        V = ql.V
        if len(policy) > 200:
            font_size = 'xx-small'
        else:
            font_size = 'large'
        # if policy.shape[1] > 16:
            # font_size = 'small'
        length = np.sqrt(len(policy)).astype(int)
        ax[0].set(title='problem_size = ' + str(length), xlim=(0, length), ylim=(0, length))
        ax[0].axis('off')
        for i in range(length):
            for j in range(length):
                x = j
                y = length - i - 1
                p = plt.Rectangle([x, y], 1, 1, alpha=1)
                p.set_facecolor(color_map[policy[length*i + j]])
                p.set_edgecolor('black')
                ax[0].add_patch(p)

                text = ax[0].text(x+0.5, y+0.5, action_map[policy[length*i + j]], size=font_size,
                                horizontalalignment='center', verticalalignment='center', color='black')
                # text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                #                         path_effects.Normal()])
                if len(policy) < 200:
                    text = ax[0].text(x+0.5, y+0.2, str(np.around(V[length*i + j], 2)), size=font_size,
                                horizontalalignment='center', verticalalignment='center', color='black')

        # Large
        P, R = mdptoolbox.example.forest(S=problem_size_list[1])
        ql = QLearning(P, R, gamma=0.9, n_iter=1000000, alpha=1.0, alpha_decay=1.0, alpha_min=0.001, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999999, run_stat_frequency=1000, stop_criteria=0.000001)
        ql.run()

        policy = ql.policy
        V = ql.V
        if len(policy) > 200:
            font_size = 'xx-small'
        else:
            font_size = 'large'
        # if policy.shape[1] > 16:
            # font_size = 'small'

        length = np.sqrt(len(policy)).astype(int)
        ax[1].set(title='problem_size = ' + str(length*length), xlim=(0, length), ylim=(0, length))
        ax[1].axis('off')
        for i in range(length):
            for j in range(length):
                x = j
                y = length - i - 1
                p = plt.Rectangle([x, y], 1, 1, alpha=1)
                p.set_facecolor(color_map[policy[length*i + j]])
                p.set_edgecolor('black')
                ax[1].add_patch(p)

                text = ax[1].text(x+0.5, y+0.5, action_map[policy[length*i + j]], size=font_size,
                                horizontalalignment='center', verticalalignment='center', color='black')
                # text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                #                         path_effects.Normal()])
                if len(policy) < 200:
                    text = ax[1].text(x+0.5, y+0.2, str(np.around(V[length*i + j], 2)), size=font_size,
                                horizontalalignment='center', verticalalignment='center', color='black')
        plt.show()
    return
