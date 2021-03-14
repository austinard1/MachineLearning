import sys, getopt
import argparse
from optimization import hill_climbing, simulated_annealing, four_peaks, knapsack, k_colors, neural_network

def get_inputs():
    """
    Extract inputs from command line
    Returns:
        args (argparse.NameSpace): Argparse namespace object
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-t", "--task", nargs=1, choices=['tune_problem', 'tuning_plots', 'complexity_graph', 'performance_graph'], help="What task to perform", required=True
    )
    parser.add_argument(
        "-o", "--optimizer", nargs="+", choices=['rhc', 'sa', 'ga', 'mimic'], help="Optimizer to run", required=True
    )
    parser.add_argument(
        "-p", "--optimization-problem", nargs=1, choices=['knapsack', 'four_peaks', 'k_colors', 'neural_network'], help="Optimization problem to run", required=True
    )
    return parser.parse_args()

def main():
    # Extract inputs
    args = get_inputs()
    if 'knapsack' in args.optimization_problem:
        knapsack.main(args.optimizer, args.task)
    if 'four_peaks' in args.optimization_problem:
        four_peaks.main(args.optimizer, args.task)
    if 'k_colors' in args.optimization_problem:
        k_colors.main(args.optimizer, args.task)
    if 'neural_network' in args.optimization_problem:
        neural_network.main(args.task)
    #hill_climbing.main()
    #simulated_annealing.main()
    #four_peaks.main()
    return

if __name__ == "__main__":
    main()