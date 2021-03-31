import os
import sys, getopt
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from unsupervised import k_means, em, pca, ica, random_proj, factor_analysis, neural_network
os.environ["PYTHONWARNINGS"] = "ignore"
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
        "-c", "--clustering", nargs=1, choices=['k_means', 'em'], help="What clustering task to perform", required=False
    )
    parser.add_argument(
        "-dr", "--dimensionality", nargs=1, choices=['pca', 'ica', 'rp', 'fa'], help="What dimensionality reduction task to perform", required=False
    )
    parser.add_argument("-nn", "--neural-network", help="Flag to run through neural network", action="store_true", default=False)
    # parser.add_argument(
    #     "-d", "--dataset", nargs=1, choices=['bank_loan', 'smart_grid'], help="What dataset to use", required=True
    # )
    # parser.add_argument(
    #     "-o", "--optimizer", nargs="+", choices=['rhc', 'sa', 'ga', 'mimic'], help="Optimizer to run", required=True
    # )
    # parser.add_argument(
    #     "-p", "--optimization-problem", nargs=1, choices=['knapsack', 'four_peaks', 'k_colors', 'neural_network'], help="Optimization problem to run", required=True
    # )
    return parser.parse_args()

def preprocess_inputs(task='classification'):
    # SmartGrid
    df = pd.read_csv('smart_grid_2.csv')

    if task == 'classification':
        df = df.drop('stab', axis=1)

        y = df['stabf'].copy()
        X = df.drop('stabf', axis=1).copy()

    elif task == 'regression':
        df = df.drop('stabf', axis=1)

        y = df['stab'].copy()
        X = df.drop('stab', axis=1).copy()

    X_train_smart, X_test_smart, y_train_smart, y_test_smart = train_test_split(X, y, test_size = 0.2, random_state=27, stratify=y)


    train_scaler = RobustScaler()
    X_train_scaler = train_scaler.fit(X_train_smart)
    X_train_smart = X_train_scaler.transform(X_train_smart)

    X_test_smart = X_train_scaler.transform(X_test_smart)

    # BAnk loan
    survey = pd.read_csv('bank_loan.csv')
    X = survey.drop(columns = ['Education'])
    y = survey['Education']

    X_train_bank, X_test_bank, y_train_bank, y_test_bank = train_test_split(X, y, test_size = 0.2, random_state=27, stratify=y)

    train_scaler = RobustScaler()
    X_train_scaler = train_scaler.fit(X_train_bank)
    X_train_bank = X_train_scaler.transform(X_train_bank)

    X_test_bank = X_train_scaler.transform(X_test_bank)

    return X_train_smart, X_test_smart, y_train_smart, y_test_smart, X_train_bank, X_test_bank, y_train_bank, y_test_bank


def main():
    os.environ["PYTHONWARNINGS"] = "ignore"
    # Extract inputs
    args = get_inputs()
    X_train_smart, X_test_smart, y_train_smart, y_test_smart, X_train_bank, X_test_bank, y_train_bank, y_test_bank = preprocess_inputs()
    if args.neural_network:
        neural_network.main(X_train_smart, X_test_smart, y_train_smart, y_test_smart, X_train_bank, X_test_bank, y_train_bank, y_test_bank, args)
        return
    if args.dimensionality is not None:
        if 'pca' in args.dimensionality:
            X_train_smart, X_test_smart, X_train_bank, X_test_bank = pca.main(X_train_smart, X_test_smart, y_train_smart, y_test_smart, X_train_bank, X_test_bank, y_train_bank, y_test_bank, args)
        if 'ica' in args.dimensionality:
            X_train_smart, X_test_smart, X_train_bank, X_test_bank = ica.main(X_train_smart, X_test_smart, y_train_smart, y_test_smart, X_train_bank, X_test_bank, y_train_bank, y_test_bank, args)
        if 'rp' in args.dimensionality:
            X_train_smart, X_test_smart, X_train_bank, X_test_bank = random_proj.main(X_train_smart, X_test_smart, y_train_smart, y_test_smart, X_train_bank, X_test_bank, y_train_bank, y_test_bank, args)
        if 'fa' in args.dimensionality:
            X_train_smart, X_test_smart, X_train_bank, X_test_bank = factor_analysis.main(X_train_smart, X_test_smart, y_train_smart, y_test_smart, X_train_bank, X_test_bank, y_train_bank, y_test_bank, args)
    if args.clustering is not None:
        if 'k_means' in args.clustering:
            k_means.main(X_train_smart, X_test_smart, y_train_smart, y_test_smart, X_train_bank, X_test_bank, y_train_bank, y_test_bank, args)
        if 'em' in args.clustering:
            em.main(X_train_smart, X_test_smart, y_train_smart, y_test_smart, X_train_bank, X_test_bank, y_train_bank, y_test_bank, args)
    # if 'k_colors' in args.optimization_problem:
    #     k_colors.main(args.task)
    # if 'neural_network' in args.optimization_problem:
    #     neural_network.main(args.task)
    #hill_climbing.main()
    #simulated_annealing.main()
    #four_peaks.main()
    return

if __name__ == "__main__":
    main()