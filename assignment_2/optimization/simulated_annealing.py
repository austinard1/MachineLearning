import json
import numpy as np

from time import process_time
import mlrose_hiive as mlrose
from mlrose_hiive import runners
from optimization import four_peaks

def main():

    four_peaks_fitness = four_peaks.get_four_peaks()
    four_peaks_tuning_list = four_peaks.get_four_peaks_tuning
    #state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    problem = mlrose.DiscreteOpt(10, four_peaks_fitness, maximize=True, max_val=2)
    problem_size_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #problem_size_list = [10, 20]
    best_fitness_list = []
    for size in problem_size_list:
        problem = mlrose.DiscreteOpt(size, four_peaks_fitness, maximize=True, max_val=2)
        experiment_name = 'sa_tuning_size_' + str(size)
        sa = runners.SARunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory='./',
                    seed=27,
                    iteration_list=[10000],
                    max_attempts=500,
                    temperature_list=[1])
                    #decay_list=mlrose.GeomDecay(init_temp=1.1))
                    #temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
                    #temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
        # the two data frames will contain the results
        df_run_stats, df_run_curves = sa.run()

        #print(df_run_curves.loc[df_run_curves['Fitness'].idxmax()])
        #curr_best_fitness = df_run_curves.loc[df_run_curves['Fitness'].idxmax()]['Fitness']
        best_fitness_list.append(df_run_curves.loc[df_run_curves['Fitness'].idxmax()])
    print(best_fitness_list)