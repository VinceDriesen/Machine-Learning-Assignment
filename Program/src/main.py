#!/usr/bin/env python3
"""
Module Docstring
"""
from src.models.support_vector_machine import support_vector_machine
from src.file_load import create_test_train_data
from src.models.multilayer_perceptron import multilayer_perceptron_regressor
from src.models.long_short_term_memory import lstm_regressor
from src.models.recurrent_neural_network import run_grid_search, recurrent_neural_network_regressor

__author__ = "Thibo De Belie, Vince Driesen, Daan Hollands"
__version__ = "0.1.0"
__license__ = "GPLv3"

import argparse
import os

def main(args):
    try:
        training_file = args.training_file if args.training_file else "../resources/TrainingData.csv"
        testing_file = args.testing_file if args.testing_file else "../resources/testingData.csv"
        if not os.path.exists(training_file):
            raise FileNotFoundError(f"Training data file not found: {training_file}")
        if not os.path.exists(testing_file):
            raise FileNotFoundError(f"Testing data file not found: {testing_file}")
        print(f"Using training file: {training_file}")
        print(f"Using testing file: {testing_file}")
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler = create_test_train_data(
            train_file=training_file, test_file=testing_file)

        best_kernel, mapeSVM = support_vector_machine(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
        mapeMLP = multilayer_perceptron_regressor(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
        mapeLSTM = lstm_regressor(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
        """ Manier om na te gaan welke data input het beste is voor de RNN """
        # run_grid_search(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
        mapeRNN = recurrent_neural_network_regressor(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 5, 50, 16, 0.001, 50, 1)

        # Resultaten printen
        print(f"---------------------------------")
        print(f"The best kernel is: {best_kernel} with a MAPE: {mapeSVM * 100:.2f}%")
        print(f"Multilayer Perceptron Regressor MAPE: {mapeMLP * 100:.2f}%")
        print(f"LSTM Regressor MAPE: {mapeLSTM * 100:.2f}%")
        print(f"RNN Regressor MAPE: {mapeRNN * 100:.2f}%")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Optional positional arguments
    parser.add_argument("-t", "--training_file", help="Training data file")
    parser.add_argument("-e", "--testing_file", help="Testing data file")
    parser.add_argument("-f", "--full_file", help="Full data file for both training and testing")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)
