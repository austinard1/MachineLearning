import sys, getopt
import argparse
from reinforcement import frozen_lake, forest

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
        "-t", "--task", nargs=1, choices=['tune_problem', 'tuning_plots', 'complexity_graph', 'performance_graph', 'policy_plots'], help="What task to perform", required=True
    )
    # parser.add_argument(
    #     "-o", "--optimizer", nargs="+", choices=['rhc', 'sa', 'ga', 'mimic'], help="Optimizer to run", required=True
    # )
    parser.add_argument(
        "-p", "--optimization-problem", nargs=1, choices=['frozen_lake', 'forest'], help="Optimization problem to run", required=True
    )
    return parser.parse_args()

def main():
    # Extract inputs
    args = get_inputs()
    if 'frozen_lake' in args.optimization_problem:
        frozen_lake.main(args.task)
    if 'forest' in args.optimization_problem:
        forest.main(args.task)
    #hill_climbing.main()
    #simulated_annealing.main()
    #four_peaks.main()
    return

if __name__ == "__main__":
    main()