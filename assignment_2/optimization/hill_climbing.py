import json
import numpy as np

from time import process_time
import mlrose_hiive as mlrose
from mlrose_hiive import runners
from optimization import four_peaks

def main():

    four_peaks_fitness = four_peaks.get_four_peaks()
    four_peaks_list = four_peaks.get_four_peaks_tuning
    #state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    problem = mlrose.DiscreteOpt(100, four_peaks_fitness, maximize=True, max_val=2)
    problem_list = []
    for item in four_peaks_list:
        problem_list.append(mlrose.DiscreteOpt(100, item, maximize=True, max_val=2))
    # Random hill climb
    print('Random Hill Climb')
    print(mlrose.random_hill_climb(problem, max_attempts=100, max_iters=1000, restarts=100, init_state=None, curve=True, random_state=27))

    for item in range(len(four_peaks_list)):
        experiment_name = 'rhc_tuning_' + str(item)
        rhc = runners.RHCRunner(problem=problem,
                experiment_name=experiment_name,
                output_directory='./',
                seed=27,
                iteration_list=2 ** np.arange(10),
                max_attempts=5000,
                restart_list=[25],
                max_iters=10)
        # the two data frames will contain the results
        df_run_stats, df_run_curves = rhc.run()
        print(df_run_curves.loc[df_run_curves['Fitness'].idxmax()])
    return
# the two data frames will contain the results
#df_run_stats, df_run_curves = sa.run()
# four peaks, knapsack and k-colors
#four_peaks_fitness = mlrose.FourPeaks(t_pct=0.15)
#state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
#fitness.evaluate(state)

# Knapsack
#weights = [10, 5, 2, 8, 15]
#weights = np.ones(100)
#values = [1, 2, 3, 4, 5]
# values = np.arange(1, 101)
# max_weight_pct = 0.6
# knapsack_fitness = mlrose.Knapsack(weights, values, max_weight_pct)
# state = np.array([1, 0, 2, 1, 0])
#fitness.evaluate(state)

# K-colors
# edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
# k_color_fitness = mlrose.MaxKColor(edges)
# state = np.array([0, 1, 0, 1, 1])
#fitness.evaluate(state)

# Two aspects of optimization:
#  - Performance
#  - Runtime
#  Make sure to have numbers and analysis for both these numbers for all parts

# For part 1 (requires lots of tuning again, yay)
#  - grab some problems: Salesman, knapsack
#  - Make sure to highlight (one algorithm outperforms all the others significantly) each of the 3
#  - Lose major points if one is missing. Important part is highlighting here
#  - Vary problem size/difficulties, some different stuff. Should change the behavior
#  - If there is a change, see when one starts beating the others and explain why
#  - People are having problems getting MIMIC to outperforms (K-coloring should work)
#  - Play with hyper parameters if it doesnt outperform when should. Also TIMING >= Accuracy here
#  - mlrose is bad for MIMIC, takes a lot of tuning, previous semesters people have re-written internal mlrose code
#  - max iteration should be tuning independantly for each separate algorithm, some algorithms take way more iterations to converge
#  - Dont retune for every different problem size, tune once for one size, find the results you want, use the same HPs for other sizes and show how it changes
#  - For varying problem size, vary in smaller increments, some 10 -> 50 to see a difference, some 20 -> 30
#  - Some are quadratic/exponential, watch out for enormous times for problem size varying.

# ALL THESE PLOTS HAVE TO BE IN THIS: (called this summary of whole assignment)
# 1. F-evals vs fitness on single plot with Iterations vs fitness
# 2. Hyper parameter tuning for both problem and algorithms
# 3. Complexity vs Fitness on single plot with Complexity vs fevals

# max iteration should be tuning independantly for each separate algorithm, some algorithms take way more iterations to converge

# For varying problem size, vary length (LEAVE MAX_VAL AS 2 SO THEY ARE BIT STRINGS)
#problem = mlrose.DiscreteOpt(100, four_peaks_fitness, maximize=True, max_val=2)

# Random hill climb
# print('Random Hill Climb')
# print(mlrose.random_hill_climb(problem, max_attempts=100, max_iters=1000, restarts=100, init_state=None, curve=True, random_state=27))

# # Simulated_annealing
# print('Simulated annealing')
# #print(mlrose.simulated_annealing(problem, schedule=mlrose.GeomDecay(), max_attempts=100, max_iters=1000, init_state=None, curve=True, random_state=27))
# experiment_name = 'example_experiment'

# sa = runners.SARunner(problem=problem,
#                 experiment_name=experiment_name,
#                 output_directory='./',
#                 seed=27,
#                 iteration_list=2 ** np.arange(14),
#                 max_attempts=5000,
#                 temperature_list=[1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
# # the two data frames will contain the results
# df_run_stats, df_run_curves = sa.run()

# # Genetic algorithm (best with four peaks)
# print('Genetic Algorithm')
# print(mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=100, max_iters=1000, curve=True, random_state=27))

# # MIMIC
# print('MIMIC')
# print(mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=10, curve=True, random_state=27))