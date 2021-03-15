import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
from mlrose_hiive import runners
from mlrose_hiive import algorithms
from mlrose_hiive import generators
import networkx as nx

def main(task):
    if 'tune_problem' in task:
        # Tune Knapsack problem
        problem_size = 50
        num_connections_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        k_colors_tuning_fitness = []
        k_colors_tuning_time = []
        k_colors_tuning_fevals = []
        for num_connections in num_connections_list:

            # Generate edges (TAKEN DIRECTLY FROM mlrose-hiive)
            number_of_nodes = problem_size
            np.random.seed(27)
            # all nodes have to be connected, somehow.
            node_connection_counts = 1 + np.random.randint(num_connections, size=number_of_nodes)

            node_connections = {}
            nodes = range(number_of_nodes)
            for n in nodes:
                all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or
                                                                        n not in node_connections[o]))]
                count = min(node_connection_counts[n], len(all_other_valid_nodes))
                other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
                node_connections[n] = [(n, o) for o in other_nodes]

            # check connectivity
            g = nx.Graph()
            g.add_edges_from([x for y in node_connections.values() for x in y])

            for n in nodes:
                cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
                for s, f in cannot_reach:
                    g.add_edge(s, f)
                    check_reach = len([(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()])
                    if check_reach == 0:
                        break

            edges = [(s, f) for (s, f) in g.edges()]

            k_colors_fitness = mlrose.MaxKColor(edges)
            problem = mlrose.DiscreteOpt(problem_size, k_colors_fitness, maximize=True, max_val=2)
            experiment_name = 'k_colors_connections_' + str(num_connections)
            population_sizes_list= [200]
            keep_percent_list=np.arange(0.05, 1.0, 0.1)
            k_colors = runners.MIMICRunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory='k_colors',
                    seed=27,
                    iteration_list=[100],
                    population_sizes=population_sizes_list,
                    keep_percent_list=keep_percent_list,
                    max_attempts=5,
                    use_fast_mimic=True)

                # the two data frames will contain the results
            k_colors_run_stats, k_colors_run_curves = k_colors.run()
            k_colors_tuning_fitness.append(k_colors_run_curves.loc[k_colors_run_curves['Fitness'].idxmax()]['Fitness'])
            k_colors_tuning_time.append(k_colors_run_curves.loc[k_colors_run_curves['Time'].idxmax()]['Time'])
            k_colors_tuning_fevals.append(population_sizes_list[0] * k_colors_run_curves.loc[k_colors_run_curves['Iteration'].idxmax()]['Iteration'])

        plt.rc("font", size=8)
        plt.rc("axes", titlesize=12)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", fontsize=11)
        plt.rc("figure", titlesize=11)
        fig, ax = plt.subplots(1, 3, figsize=(10,3.5))
        fig.suptitle('Max K-Colors Tuning w/ MIMIC Optimizer')
        ax[0].scatter(num_connections_list, k_colors_tuning_fitness, c='r', marker='x', s=10)
        ax[0].set(xlabel='Number of Node Connections', ylabel = 'Max Fitness')
        ax[1].scatter(num_connections_list, k_colors_tuning_time, c='g', marker='o', s=10)
        ax[1].set(xlabel='Number of Node Connections', ylabel = 'Max Runtime (s)')
        ax[2].scatter(num_connections_list, k_colors_tuning_fevals, c='b', marker='+')
        ax[2].set(xlabel='Number of Node Connections', ylabel = 'Max Function Evaluations')
        ax[2].yaxis.tick_right()
        plt.show()

        return

    if 'tuning_plots' in task:
    # FOUR PEAKS GOOD FOR GENETIC
        problem_size = 50
        num_connections = 50
        # Generate edges (TAKEN DIRECTLY FROM mlrose-hiive)
        number_of_nodes = problem_size
        np.random.seed(27)
        # all nodes have to be connected, somehow.
        node_connection_counts = 1 + np.random.randint(num_connections, size=number_of_nodes)

        node_connections = {}
        nodes = range(number_of_nodes)
        for n in nodes:
            all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or
                                                                    n not in node_connections[o]))]
            count = min(node_connection_counts[n], len(all_other_valid_nodes))
            other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
            node_connections[n] = [(n, o) for o in other_nodes]

        # check connectivity
        g = nx.Graph()
        g.add_edges_from([x for y in node_connections.values() for x in y])

        for n in nodes:
            cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
            for s, f in cannot_reach:
                g.add_edge(s, f)
                check_reach = len([(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()])
                if check_reach == 0:
                    break

        edges = [(s, f) for (s, f) in g.edges()]

        k_colors_fitness = mlrose.MaxKColor(edges)
        problem = mlrose.DiscreteOpt(problem_size, k_colors_fitness, maximize=True, max_val=2)

        rhc_fitness_tuning_list = []
        rhc_param_tuning_list = []
        time_tuning_list = []
        rhc_feval_tuning_list = []
        asdf_list = []
        fdsa_list = []
        experiment_name = 'rhc_k_colors_tuning_size_' + str(problem_size)
        #restart_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        restart_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        rhc = runners.RHCRunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory='k_colors',
                    seed=27,
                    iteration_list=[5000],
                    max_attempts=50,
                    restart_list=restart_list)
        # the two data frames will contain the results
        rhc_run_stats, rhc_run_curves = rhc.run()
        for restart in restart_list:
            this_temp_df = rhc_run_curves.loc[rhc_run_curves['Restarts'] == restart]
            this_temp_df['Iteration'] = this_temp_df['Iteration'] - this_temp_df.loc[this_temp_df['Iteration'].idxmin()]['Iteration'] + 1
            rhc_fitness_tuning_list.append(this_temp_df.loc[this_temp_df['Fitness'].idxmax()]['Fitness'])
            rhc_param_tuning_list.append(restart)
            rhc_feval_tuning_list.append(3 * this_temp_df.loc[this_temp_df['Iteration'].idxmax()]['Iteration'])
            time_tuning_list.append(this_temp_df.loc[this_temp_df['Time'].idxmax()]['Time'])
            asdf_list.append(this_temp_df['Fitness'])
            fdsa_list.append(this_temp_df['Iteration'])
        plt.rc("font", size=8)
        plt.rc("axes", titlesize=12)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", fontsize=8)
        plt.rc("figure", titlesize=11)
        # #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
        # fig, ax = plt.subplots(1,3,figsize=(12,3.5))
        # fig.suptitle('RHC Restarts Tuning, problem_size = ' + str(problem_size))
        # ax[0].scatter(param_tuning_list, time_tuning_list, c='r', marker='x', s=10)
        # ax[0].set(xlabel='Restarts', ylabel = 'Time')

        # ax[1].scatter(param_tuning_list, fitness_tuning_list, c='g', marker='o', s=10)
        # ax[1].set(xlabel='Restarts', ylabel = 'Fitness')

        # ax[2].scatter(param_tuning_list, feval_tuning_list, c='g', marker='o', s=10)
        # ax[2].set(xlabel='Restarts', ylabel = 'Function Evaluations')
        # ax[2].yaxis.tick_right()

        # plt.show()

        # fig, ax = plt.subplots()
        # ax.scatter(fdsa_list[7], asdf_list[7])
        # ax.set(xlabel='Iteration', ylabel = 'Fitness')
        # plt.show()

        sa_fitness_tuning_list = []
        sa_param_tuning_list = []
        time_tuning_list = []
        sa_feval_tuning_list = []
        asdf_list = []
        fdsa_list = []
        experiment_name = 'sa_k_colors_tuning_size_' + str(problem_size)
        temperature_list = np.arange(1, 50, 0.5)
        sa = runners.SARunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory='k_colors',
                    seed=27,
                    iteration_list=[5000],
                    max_attempts=50,
                    temperature_list=temperature_list)
                    #decay_list=mlrose.GeomDecay(init_temp=1.1))
                    #temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
                    #temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
        # the two data frames will contain the results
        df_run_stats, df_run_curves = sa.run()
        df_run_curves['Temperature'] = pd.to_numeric(df_run_curves['Temperature'].astype(str).astype(float))
        for temp in temperature_list:
            this_temp_df = df_run_curves.loc[df_run_curves['Temperature'] == temp]
            this_temp_df['Iteration'] = this_temp_df['Iteration'] - this_temp_df.loc[this_temp_df['Iteration'].idxmin()]['Iteration'] + 1
            sa_fitness_tuning_list.append(this_temp_df.loc[this_temp_df['Fitness'].idxmax()]['Fitness'])
            sa_param_tuning_list.append(temp)
            sa_feval_tuning_list.append(2 * this_temp_df.loc[this_temp_df['Iteration'].idxmax()]['Iteration'])
            time_tuning_list.append(this_temp_df.loc[this_temp_df['Time'].idxmax()]['Time'])
            asdf_list.append(this_temp_df['Fitness'])
            fdsa_list.append(this_temp_df['Iteration'])
        plt.rc("font", size=8)
        plt.rc("axes", titlesize=12)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", fontsize=8)
        plt.rc("figure", titlesize=11)
        # #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
        # fig, ax = plt.subplots(1,3,figsize=(12,3.5))
        # fig.suptitle('SA Temperature Tuning, problem_size = ' + str(problem_size))
        # ax[0].scatter(param_tuning_list, time_tuning_list, c='r', marker='x', s=10)
        # ax[0].set(xlabel='Temperature', ylabel = 'Time')

        # ax[1].scatter(param_tuning_list, fitness_tuning_list, c='g', marker='o', s=10)
        # ax[1].set(xlabel='Temperature', ylabel = 'Fitness')

        # ax[2].scatter(param_tuning_list, feval_tuning_list, c='g', marker='o', s=10)
        # ax[2].set(xlabel='Temperature', ylabel = 'Function Evaluations')
        # ax[2].yaxis.tick_right()

        # plt.show()

        # fig, ax = plt.subplots()
        # ax.scatter(fdsa_list[17], asdf_list[17])
        # ax.set(xlabel='Iteration', ylabel = 'Fitness')
        # plt.show()

        # ga_fitness_tuning_list = []
        # ga_param_tuning_list = []
        # time_tuning_list = []
        # feval_tuning_list = []
        # asdf_list = []
        # fdsa_list = []
        # experiment_name = 'ga_k_colors_tuning_size_' + str(problem_size)
        # population_sizes_list=100,
        # mutation_rates_list=np.arange(0.05, 1.0, 0.05)
        # ga = runners.GARunner(problem=problem,
        #             experiment_name=experiment_name,
        #             output_directory='k_colors',
        #             seed=27,
        #             iteration_list=[5000],
        #             population_sizes=population_sizes_list,
        #             mutation_rates=mutation_rates_list,
        #             max_attempts=100)

        # # the two data frames will contain the results
        # df_run_stats, df_run_curves = ga.run()

        # for rate in mutation_rates_list:
        #     this_temp_df = df_run_curves.loc[df_run_curves['Mutation Rate'] == rate]
        #     this_temp_df['Iteration'] = this_temp_df['Iteration'] - this_temp_df.loc[this_temp_df['Iteration'].idxmin()]['Iteration'] + 1
        #     ga_fitness_tuning_list.append(this_temp_df.loc[this_temp_df['Fitness'].idxmax()]['Fitness'])
        #     ga_param_tuning_list.append(rate)
        #     feval_tuning_list.append(population_sizes_list[0] * this_temp_df.loc[this_temp_df['Iteration'].idxmax()]['Iteration'])
        #     time_tuning_list.append(this_temp_df.loc[this_temp_df['Time'].idxmax()]['Time'])
        #     asdf_list.append(this_temp_df['Fitness'])
        #     fdsa_list.append(this_temp_df['Iteration'])
        # plt.rc("font", size=8)
        # plt.rc("axes", titlesize=12)
        # plt.rc("axes", labelsize=10)
        # plt.rc("xtick", labelsize=8)
        # plt.rc("ytick", labelsize=8)
        # plt.rc("legend", fontsize=8)
        # plt.rc("figure", titlesize=11)
        # #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
        # fig, ax = plt.subplots(1,3,figsize=(12,3.5))
        # fig.suptitle('GA Mutation Rate Tuning, problem_size = ' + str(problem_size))
        # ax[0].scatter(param_tuning_list, time_tuning_list, c='r', marker='x', s=10)
        # ax[0].set(xlabel='Mutation Rate', ylabel = 'Time (s)')

        # ax[1].scatter(param_tuning_list, fitness_tuning_list, c='g', marker='o', s=10)
        # ax[1].set(xlabel='Mutation Rate', ylabel = 'Fitness')

        # ax[2].scatter(param_tuning_list, feval_tuning_list, c='g', marker='o', s=10)
        # ax[2].set(xlabel='Mutation Rate', ylabel = 'Function Evaluations')
        # ax[2].yaxis.tick_right()

        # plt.show()

        # fig, ax = plt.subplots()
        # ax.scatter(fdsa_list[17], asdf_list[17])
        # ax.set(xlabel='Iteration', ylabel = 'Fitness')
        # plt.show()

        # Tune population size
        ga_population_tuning_fitness = []
        ga_population_tuning_time = []
        ga_population_tuning_feval = []
        population_sizes_list= np.arange(10, 500, 10)
        for population_size in population_sizes_list:
            experiment_name = 'ga_k_colors_tuning_population_size_' + str(problem_size)
            mutation_rates_list=[0.15]
            ga = runners.GARunner(problem=problem,
                        experiment_name=experiment_name,
                        output_directory='k_colors',
                        seed=27,
                        iteration_list=[5000],
                        population_sizes=[int(population_size)],
                        mutation_rates=mutation_rates_list,
                        max_attempts=100)

            # the two data frames will contain the results
            ga_run_stats, ga_run_curves = ga.run()
            ga_population_tuning_fitness.append(ga_run_curves.loc[ga_run_curves['Fitness'].idxmax()]['Fitness'])
            ga_population_tuning_time.append(ga_run_curves.loc[ga_run_curves['Time'].idxmax()]['Time'])
            ga_population_tuning_feval.append(population_size * ga_run_curves.loc[ga_run_curves['Iteration'].idxmax()]['Iteration'])

        # plt.rc("font", size=8)
        # plt.rc("axes", titlesize=12)
        # plt.rc("axes", labelsize=10)
        # plt.rc("xtick", labelsize=8)
        # plt.rc("ytick", labelsize=8)
        # plt.rc("legend", fontsize=8)
        # plt.rc("figure", titlesize=11)
        # #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
        # fig, ax = plt.subplots(1,3,figsize=(12,3.5))
        # fig.suptitle('GA Population Size Tuning, problem_size = ' + str(problem_size))
        # ax[0].scatter(population_sizes_list, ga_population_tuning_time, c='r', marker='x', s=10)
        # ax[0].set(xlabel='Population Size', ylabel = 'Time')

        # ax[1].scatter(population_sizes_list, ga_population_tuning_fitness, c='g', marker='x', s=10)
        # ax[1].set(xlabel='Population Size', ylabel = 'Fitness')

        # ax[2].scatter(population_sizes_list, ga_population_tuning_feval, c='g', marker='o', s=10)
        # ax[2].set(xlabel='Population Size', ylabel = 'Function Evaluations')
        # ax[2].yaxis.tick_right()

        # plt.show()

        # mimic_fitness_tuning_list = []
        # mimic_param_tuning_list = []
        # time_tuning_list = []
        # mimic_feval_tuning_list = []
        # asdf_list = []
        # fdsa_list = []
        # experiment_name = 'mimic_k_colors_tuning_size_' + str(problem_size)
        # population_sizes_list=100,
        # keep_percent_list=np.arange(0.05, 1.0, 0.05)
        # mimic = runners.MIMICRunner(problem=problem,
        #             experiment_name=experiment_name,
        #             output_directory='k_colors',
        #             seed=27,
        #             iteration_list=[100],
        #             population_sizes=population_sizes_list,
        #             keep_percent_list=keep_percent_list,
        #             max_attempts=5,
        #             use_fast_mimic=True)

        # # the two data frames will contain the results
        # df_run_stats, df_run_curves = mimic.run()

        # # print(df_run_curves.dtypes)
        # # print(df_run_curves)
        # # #df_run_curves['Temperature'] = pd.to_numeric(df_run_curves['Temperature'].astype(str).astype(float))
        # # print(df_run_curves)
        # for percent in keep_percent_list:
        #     this_temp_df = df_run_curves.loc[df_run_curves['Keep Percent'] == percent]
        #     this_temp_df['Iteration'] = this_temp_df['Iteration'] - this_temp_df.loc[this_temp_df['Iteration'].idxmin()]['Iteration'] + 1
        #     mimic_fitness_tuning_list.append(this_temp_df.loc[this_temp_df['Fitness'].idxmax()]['Fitness'])
        #     mimic_param_tuning_list.append(percent)
        #     mimic_feval_tuning_list.append(population_sizes_list[0] * this_temp_df.loc[this_temp_df['Iteration'].idxmax()]['Iteration'])
        #     time_tuning_list.append(this_temp_df.loc[this_temp_df['Time'].idxmax()]['Time'])
        #     asdf_list.append(this_temp_df['Fitness'])
        #     fdsa_list.append(this_temp_df['Iteration'])

        # plt.rc("font", size=8)
        # plt.rc("axes", titlesize=12)
        # plt.rc("axes", labelsize=10)
        # plt.rc("xtick", labelsize=8)
        # plt.rc("ytick", labelsize=8)
        # plt.rc("legend", fontsize=8)
        # plt.rc("figure", titlesize=11)
        # #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
        # fig, ax = plt.subplots(1,3,figsize=(12,3.5))
        # fig.suptitle('MIMIC Keep Percent Tuning, problem_size = ' + str(problem_size))
        # ax[0].scatter(param_tuning_list, time_tuning_list, c='r', marker='x', s=10)
        # ax[0].set(xlabel='Keep Percent (decimal)', ylabel = 'Time (s)')

        # ax[1].scatter(param_tuning_list, fitness_tuning_list, c='g', marker='o', s=10)
        # ax[1].set(xlabel='Keep Percent (decimal)', ylabel = 'Fitness')

        # ax[2].scatter(param_tuning_list, feval_tuning_list, c='g', marker='o', s=10)
        # ax[2].set(xlabel='Keep Percent (decimal)', ylabel = 'Function Evaluations')
        # ax[2].yaxis.tick_right()

        # plt.show()

        # fig, ax = plt.subplots()
        # ax.scatter(fdsa_list[17], asdf_list[17])
        # ax.set(xlabel='Iteration', ylabel = 'Fitness')
        # plt.show()

        # # Tune population size
        mimic_population_tuning_fitness = []
        mimic_population_tuning_time = []
        mimic_population_tuning_feval = []
        population_sizes_list= np.arange(10, 500, 10)
        for population_size in population_sizes_list:
            experiment_name = 'mimic_k_colors_tuning_population_size_' + str(problem_size)
            keep_percent_list=[0.3]
            mimic = runners.MIMICRunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory='k_colors',
                    seed=27,
                    iteration_list=[100],
                    population_sizes=[int(population_size)],
                    keep_percent_list=keep_percent_list,
                    max_attempts=10,
                        use_fast_mimic=True)

            # the two data frames will contain the results
            mimic_run_stats, mimic_run_curves = mimic.run()
            mimic_population_tuning_fitness.append(mimic_run_curves.loc[mimic_run_curves['Fitness'].idxmax()]['Fitness'])
            mimic_population_tuning_time.append(mimic_run_curves.loc[mimic_run_curves['Time'].idxmax()]['Time'])
            mimic_population_tuning_feval.append(population_size * mimic_run_curves.loc[mimic_run_curves['Iteration'].idxmax()]['Iteration'])

        # plt.rc("font", size=8)
        # plt.rc("axes", titlesize=14)
        # plt.rc("axes", labelsize=10)
        # plt.rc("xtick", labelsize=8)
        # plt.rc("ytick", labelsize=8)
        # plt.rc("legend", fontsize=11)
        # plt.rc("figure", titlesize=11)
        # #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
        # fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
        # fig.suptitle('MIMIC Population Size Tuning, problem_size = ' + str(problem_size))

        # ax[0].scatter(population_sizes_list, mimic_population_tuning_time, c='r', marker='x', s=10)
        # ax[0].set(xlabel='Population Size', ylabel = 'Time (s)')

        # ax[1].scatter(population_sizes_list, mimic_population_tuning_fitness, c='g', marker='o', s=10)
        # ax[1].set(xlabel='Population Size', ylabel = 'Fitness')

        # ax[2].scatter(population_sizes_list, mimic_population_tuning_feval, c='g', marker='o', s=10)
        # ax[2].set(xlabel='Population Size', ylabel = 'Function Evaluations')
        # ax[2].yaxis.tick_right()

        # plt.show()

        plt.rc("font", size=8)
        plt.rc("axes", titlesize=14)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", fontsize=11)
        plt.rc("figure", titlesize=11)
        fig, ax = plt.subplots(2, 4, figsize=(12, 7))
        fig.suptitle('Max K-Colors Algorithm Tuning, problem size = ' + str(problem_size))

        ax[0, 0].scatter(rhc_param_tuning_list, rhc_fitness_tuning_list, c='r', marker='x', s=10)
        ax[0, 0].set(xlabel='Restarts', ylabel = 'Fitness', title='RHC Restarts')

        ax[0, 1].scatter(sa_param_tuning_list, sa_fitness_tuning_list, c='g', marker='o', s=10)
        ax[0, 1].set(xlabel='Temperature', title='SA Temperature')

        ax[0, 2].scatter(population_sizes_list, ga_population_tuning_fitness, c='g', marker='o', s=10)
        ax[0, 2].set(xlabel='Population Size', title='GA Population Size')
        ax[0, 2].yaxis.tick_right()

        ax[0, 3].scatter(population_sizes_list, mimic_population_tuning_fitness, c='g', marker='o', s=10)
        ax[0, 3].set(xlabel='Population Size', title='MIMIC Population Size')
        ax[0, 3].yaxis.tick_right()

        ax[1, 0].scatter(rhc_param_tuning_list, rhc_feval_tuning_list, c='r', marker='x', s=10)
        ax[1, 0].set(xlabel='Restarts', ylabel = 'Function Evaluations')

        ax[1, 1].scatter(sa_param_tuning_list, sa_feval_tuning_list, c='g', marker='o', s=10)
        ax[1, 1].set(xlabel='Temperature')

        ax[1, 2].scatter(population_sizes_list, ga_population_tuning_feval, c='g', marker='o', s=10)
        ax[1, 2].set(xlabel='Population Size')
        ax[1, 2].yaxis.tick_right()

        ax[1, 3].scatter(population_sizes_list, mimic_population_tuning_feval, c='g', marker='o', s=10)
        ax[1, 3].set(xlabel='Population Size')
        ax[1, 3].yaxis.tick_right()

        plt.show()

    if 'complexity_graph' in task:
        problem_size_list = np.arange(5, 85, 5)
        sa_time_list = []
        sa_fitness_list = []
        sa_feval_list = []
        rhc_time_list = []
        rhc_fitness_list = []
        rhc_feval_list = []
        ga_time_list = []
        ga_fitness_list = []
        ga_feval_list = []
        mimic_time_list = []
        mimic_fitness_list = []
        mimic_feval_list = []
        for problem_size in problem_size_list:
            # Generate edges (TAKEN DIRECTLY FROM mlrose-hiive)
            number_of_nodes = problem_size
            num_connections = problem_size
            np.random.seed(27)
            # all nodes have to be connected, somehow.
            node_connection_counts = 1 + np.random.randint(num_connections, size=number_of_nodes)

            node_connections = {}
            nodes = range(number_of_nodes)
            for n in nodes:
                all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or
                                                                        n not in node_connections[o]))]
                count = min(node_connection_counts[n], len(all_other_valid_nodes))
                other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
                node_connections[n] = [(n, o) for o in other_nodes]

            # check connectivity
            g = nx.Graph()
            g.add_edges_from([x for y in node_connections.values() for x in y])

            for n in nodes:
                cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
                for s, f in cannot_reach:
                    g.add_edge(s, f)
                    check_reach = len([(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()])
                    if check_reach == 0:
                        break

            edges = [(s, f) for (s, f) in g.edges()]

            k_colors_fitness = mlrose.MaxKColor(edges)
            problem = mlrose.DiscreteOpt(int(problem_size), k_colors_fitness, maximize=True, max_val=2)
            # RHC
            experiment_name = 'rhc_k_colors_complexity_size_' + str(problem_size)
            restart_list = [10]
            rhc = runners.RHCRunner(problem=problem,
                        experiment_name=experiment_name,
                        output_directory='k_colors',
                        seed=27,
                        iteration_list=[5000],
                        max_attempts=50,
                        restart_list=restart_list)
            # the two data frames will contain the results
            rhc_run_stats, rhc_run_curves = rhc.run()
            rhc_time = rhc_run_curves['Time']
            rhc_fitness = rhc_run_curves['Fitness']
            rhc_iteration = rhc_run_curves['Iteration']
            rhc_fitness_list.append(rhc_run_curves.loc[rhc_run_curves['Fitness'].idxmax()]['Fitness'])
            rhc_time_list.append(rhc_run_curves.loc[rhc_run_curves['Time'].idxmax()]['Time'])
            rhc_feval_list.append(3 * rhc_run_curves.loc[rhc_run_curves['Iteration'].idxmax()]['Iteration'])

            # SA
            experiment_name = 'sa_k_colors_complexity_size_' + str(problem_size)
            temperature_list = [5]
            sa = runners.SARunner(problem=problem,
                        experiment_name=experiment_name,
                        output_directory='k_colors',
                        seed=27,
                        iteration_list=[2000],
                        max_attempts=50,
                        temperature_list=temperature_list)
            # the two data frames will contain the results
            sa_run_stats, sa_run_curves = sa.run()
            # print(sa_run_curves.dtypes)
            # print(sa_run_curves)
            sa_run_curves['Temperature'] = pd.to_numeric(sa_run_curves['Temperature'].astype(str).astype(float))
            # print(df_run_curves)
            sa_time = sa_run_curves['Time']
            sa_fitness = sa_run_curves['Fitness']
            sa_iteration = sa_run_curves['Iteration']
            sa_fitness_list.append(sa_run_curves.loc[sa_run_curves['Fitness'].idxmax()]['Fitness'])
            sa_time_list.append(sa_run_curves.loc[sa_run_curves['Time'].idxmax()]['Time'])
            sa_feval_list.append(2 * sa_run_curves.loc[sa_run_curves['Iteration'].idxmax()]['Iteration'])

            # GA
            experiment_name = 'ga_k_colors_complexity_size_' + str(problem_size)
            population_sizes_list=150,
            mutation_rates_list= [0.05]
            ga = runners.GARunner(problem=problem,
                        experiment_name=experiment_name,
                        output_directory='k_colors',
                        seed=27,
                        iteration_list=[1000],
                        population_sizes=population_sizes_list,
                        mutation_rates=mutation_rates_list,
                        max_attempts=50)
            # the two data frames will contain the results
            ga_run_stats, ga_run_curves = ga.run()
            # print(ga_run_curves.dtypes)
            # print(ga_run_curves)
            # print(df_run_curves)
            ga_time = ga_run_curves['Time']
            ga_fitness = ga_run_curves['Fitness']
            ga_iteration = ga_run_curves['Iteration']
            ga_fitness_list.append(ga_run_curves.loc[ga_run_curves['Fitness'].idxmax()]['Fitness'])
            ga_time_list.append(ga_run_curves.loc[ga_run_curves['Time'].idxmax()]['Time'])
            ga_feval_list.append(population_sizes_list[0] * ga_run_curves.loc[ga_run_curves['Iteration'].idxmax()]['Iteration'])

            # MIMC
            experiment_name = 'mimic_k_colors_complexity_size_' + str(problem_size)
            population_sizes_list=250,
            keep_percent_list=[0.35]
            mimic = runners.MIMICRunner(problem=problem,
                        experiment_name=experiment_name,
                        output_directory='k_colors',
                        seed=27,
                        iteration_list=[100],
                        population_sizes=population_sizes_list,
                        keep_percent_list=keep_percent_list,
                        max_attempts=5,
                        use_fast_mimic=True)
            # the two data frames will contain the results
            mimic_run_stats, mimic_run_curves = mimic.run()
            # print(mimic_run_curves.dtypes)
            # print(mimic_run_curves)
            # print(df_run_curves)
            mimic_time = mimic_run_curves['Time']
            mimic_fitness = mimic_run_curves['Fitness']
            mimic_iteration = mimic_run_curves['Iteration']
            mimic_fitness_list.append(mimic_run_curves.loc[mimic_run_curves['Fitness'].idxmax()]['Fitness'])
            mimic_time_list.append(mimic_run_curves.loc[mimic_run_curves['Time'].idxmax()]['Time'])
            mimic_feval_list.append(population_sizes_list[0] * mimic_run_curves.loc[mimic_run_curves['Iteration'].idxmax()]['Iteration'])

        plt.rc("font", size=8)
        plt.rc("axes", titlesize=12)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", fontsize=8)
        plt.rc("figure", titlesize=11)
        #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
        fig, ax = plt.subplots(1,3,figsize=(12,3.5))
        fig.suptitle('Max K-Colors Complexity Analysis', fontsize=14)
        # ax[0].plot(problem_size_list, sa_fitness_list, 'b-', label='Simulated Annealing', linewidth=1)
        # ax[0].plot(problem_size_list, ga_fitness_list, 'g:', label='Genetic', linewidth=1)
        w = 1
        ax[0].bar(problem_size_list - w, sa_fitness_list, width=w, color='blue', label='Simulated Annealing')
        ax[0].bar(problem_size_list, ga_fitness_list, width=w, color='green', label='Genetic')
        ax[0].bar(problem_size_list - 2*w, rhc_fitness_list, width=w, color='red', label='Random Hill Climb')
        ax[0].bar(problem_size_list + w, mimic_fitness_list, width=w, color='orange', label='MIMIC')
        ax[0].set(xlabel='Number of Nodes', ylabel = 'Fitness')
        ax[0].legend()

        ax[1].plot(problem_size_list, sa_time_list, 'b-', label='Simulated Annealing', linewidth=1)
        ax[1].plot(problem_size_list, ga_time_list, 'g:', label='Genetic', linewidth=1)
        ax[1].plot(problem_size_list, rhc_time_list, 'r--', label='Random Hill Climb', linewidth=1)
        ax[1].plot(problem_size_list, mimic_time_list, '-.', color='orange', label='MIMIC', linewidth=1)
        ax[1].set(xlabel='Number of Nodes', ylabel = 'Time (s)')
        ax[1].legend()

        ax[2].plot(problem_size_list, sa_feval_list, 'b-', label='Simulated Annealing', linewidth=1)
        ax[2].plot(problem_size_list, ga_feval_list, 'g:', label='Genetic', linewidth=1)
        ax[2].plot(problem_size_list, rhc_feval_list, 'r--', label='Random Hill Climb', linewidth=1)
        ax[2].plot(problem_size_list, mimic_feval_list, '-.', color='orange', label='MIMIC', linewidth=1)
        ax[2].set(xlabel='Number of Nodes', ylabel='Function Evaluations')
        ax[2].yaxis.tick_right()
        plt.show()

    if 'performance_graph' in task:
        problem_size = 80
        sa_time_list = []
        sa_fitness_list = []
        rhc_time_list = []
        rhc_fitness_list = []
        ga_time_list = []
        ga_fitness_list = []
        mimic_time_list = []
        mimic_fitness_list = []

        # Generate edges (TAKEN DIRECTLY FROM mlrose-hiive)
        number_of_nodes = problem_size
        num_connections = problem_size
        np.random.seed(27)
        # all nodes have to be connected, somehow.
        node_connection_counts = 1 + np.random.randint(num_connections, size=number_of_nodes)

        node_connections = {}
        nodes = range(number_of_nodes)
        for n in nodes:
            all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or
                                                                    n not in node_connections[o]))]
            count = min(node_connection_counts[n], len(all_other_valid_nodes))
            other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
            node_connections[n] = [(n, o) for o in other_nodes]

        # check connectivity
        g = nx.Graph()
        g.add_edges_from([x for y in node_connections.values() for x in y])

        for n in nodes:
            cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
            for s, f in cannot_reach:
                g.add_edge(s, f)
                check_reach = len([(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()])
                if check_reach == 0:
                    break

        edges = [(s, f) for (s, f) in g.edges()]

        # K-Colors
        k_colors_fitness = mlrose.MaxKColor(edges)
        problem = mlrose.DiscreteOpt(int(problem_size), k_colors_fitness, maximize=True, max_val=2)

        # RHC
        experiment_name = 'rhc_k_colors_performance_size_' + str(problem_size)
        restart_list = [10]
        rhc = runners.RHCRunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory='k_colors',
                    seed=27,
                    iteration_list=[5000],
                    max_attempts=50,
                    restart_list=restart_list)
        # the two data frames will contain the results
        rhc_run_stats, rhc_run_curves = rhc.run()
        # print(rhc_run_curves.dtypes)
        # print(rhc_run_curves)
        # print(df_run_curves)
        rhc_time = rhc_run_curves['Time']
        rhc_fitness = rhc_run_curves['Fitness']
        rhc_iteration = rhc_run_curves['Iteration']
        rhc_feval = rhc_run_curves['Iteration'] * 2

        # SA
        experiment_name = 'sa_k_colors_performance_size_' + str(problem_size)
        temperature_list = [5]
        sa = runners.SARunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory='k_colors',
                    seed=27,
                    iteration_list=[2000],
                    max_attempts=50,
                    temperature_list=temperature_list)
        # the two data frames will contain the results
        sa_run_stats, sa_run_curves = sa.run()
        # print(sa_run_curves.dtypes)
        # print(sa_run_curves)
        sa_run_curves['Temperature'] = pd.to_numeric(sa_run_curves['Temperature'].astype(str).astype(float))
        # print(df_run_curves)
        sa_time = sa_run_curves['Time']
        sa_fitness = sa_run_curves['Fitness']
        sa_iteration = sa_run_curves['Iteration']
        sa_feval = sa_run_curves['Iteration'] * 2

        # GA
        experiment_name = 'ga_k_colors_performance_size_' + str(problem_size)
        population_sizes_list=150,
        mutation_rates_list= [0.05]
        ga = runners.GARunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory='k_colors',
                    seed=27,
                    iteration_list=[2000],
                    population_sizes=population_sizes_list,
                    mutation_rates=mutation_rates_list,
                    max_attempts=50)
        # the two data frames will contain the results
        ga_run_stats, ga_run_curves = ga.run()
        # print(ga_run_curves.dtypes)
        # print(ga_run_curves)
        # print(df_run_curves)
        ga_time = ga_run_curves['Time']
        ga_fitness = ga_run_curves['Fitness']
        ga_iteration = ga_run_curves['Iteration']
        ga_feval = ga_run_curves['Iteration'] * population_sizes_list

        # MIMC
        experiment_name = 'mimic_k_colors_performance_size_' + str(problem_size)
        population_sizes_list=250,
        keep_percent_list=[0.35]
        mimic = runners.MIMICRunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory='k_colors',
                    seed=27,
                    iteration_list=[100],
                    population_sizes=population_sizes_list,
                    keep_percent_list=keep_percent_list,
                    max_attempts=5,
                    use_fast_mimic=True)
        # the two data frames will contain the results
        mimic_run_stats, mimic_run_curves = mimic.run()
        # print(mimic_run_curves.dtypes)
        # print(mimic_run_curves)
        # print(df_run_curves)
        mimic_time = mimic_run_curves['Time']
        mimic_fitness = mimic_run_curves['Fitness']
        mimic_iteration = mimic_run_curves['Iteration']
        mimic_feval = mimic_run_curves['Iteration'] * population_sizes_list

        plt.rc("font", size=8)
        plt.rc("axes", titlesize=12)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("legend", fontsize=8)
        plt.rc("figure", titlesize=11)
        #fig, ax = plt.subplots(2, 1, dpi=100, sharex=True, figsize=(5,4))
        fig, ax = plt.subplots(1,3,figsize=(12,3.5))
        fig.suptitle('Max K-Colors Algorithm Performance Analysis, problem size = ' + str(problem_size), fontsize=14)
        # ax[0].plot(problem_size_list, sa_fitness_list, 'b-', label='Simulated Annealing', linewidth=1)
        # ax[0].plot(problem_size_list, ga_fitness_list, 'g:', label='Genetic', linewidth=1)
        w = 1
        ax[0].plot(rhc_iteration, rhc_fitness, 'r--', label='Random Hill Climb', linewidth=1)
        ax[0].plot(sa_iteration, sa_fitness, 'b:', label='Simulated Annealing', linewidth=1)
        ax[0].plot(ga_iteration, ga_fitness, 'g-', label='Genetic', linewidth=1)
        ax[0].plot(mimic_iteration, mimic_fitness, '-.', color='orange', label='MIMIC', linewidth=1)
        ax[0].set(xlabel='Iteration', ylabel = 'Fitness')
        ax[0].legend()
        #ax[0].set_title('Fitness vs. Iteration')

        ax[1].plot(rhc_time, rhc_fitness, 'r--', label='Random Hill Climb', linewidth=1)
        ax[1].plot(sa_time, sa_fitness, 'b:', label='Simulated Annealing', linewidth=1)
        ax[1].plot(ga_time, ga_fitness, 'g-', label='Genetic', linewidth=1)
        ax[1].plot(mimic_time, mimic_fitness, '-.', color='orange', label='MIMIC', linewidth=1)
        ax[1].set(xlabel='Time (s)')
        #ax[1].set_title('Fitness vs. Runtime')

        ax[2].plot(rhc_feval, rhc_fitness, 'r--', label='Random Hill Climb', linewidth=1)
        ax[2].plot(sa_feval, sa_fitness, 'b:', label='Simulated Annealing', linewidth=1)
        ax[2].plot(ga_feval, ga_fitness, 'g-', label='Genetic', linewidth=1)
        ax[2].plot(mimic_feval, mimic_fitness, '-.', color='orange', label='MIMIC', linewidth=1)
        ax[2].set(xlabel='Function Evaluations')
        plt.show()

    return