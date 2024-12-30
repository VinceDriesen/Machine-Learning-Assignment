#!/usr/bin/env python3
"""
Module Docstring
"""
from models.support_vector_machine import support_vector_machine
from file_load import create_test_train_data
from models.multilayer_perceptron import multilayer_perceptron_regressor
from models.long_short_term_memory import calculate_lstm_regressor
from models.recurrent_neural_network import recurrent_neural_network_regressor

__author__ = "Thibo De Belie, Vince Driesen, Daan Hollands"
__version__ = "0.1.0"
__license__ = "GPLv3"

import argparse
import os

def main(args):
    try:
        training_file = args.training_file
        testing_file = args.testing_file
        if not training_file:
            raise FileNotFoundError(f"Training file has not been provided")
        if not testing_file:
            raise FileNotFoundError(f"Testing file has not been provided")
        if not os.path.exists(training_file):
            raise FileNotFoundError(f"Training data file not found: {training_file}")
        if not os.path.exists(testing_file):
            raise FileNotFoundError(f"Testing data file not found: {testing_file}")
        print(f"Using training file: {training_file}")
        print(f"Using testing file: {testing_file}")
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler = create_test_train_data(
            train_file=training_file, test_file=testing_file)

        best_kernel, mape_svm = support_vector_machine(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
        mape_mlp = multilayer_perceptron_regressor(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, (20,), 'relu', 'lbfgs', 'constant')
        mape_lstm = calculate_lstm_regressor(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 5, 150, 16, 0.001, 100, 1)
        mape_rnn = recurrent_neural_network_regressor(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 5, 50, 16, 0.001, 100, 1)

        print(f"---------------------------------")
        print(f"The best kernel is: {best_kernel} with a MAPE: {mape_svm * 100:.2f}%")
        print(f"Multilayer Perceptron Regressor MAPE: {mape_mlp * 100:.2f}%")
        print(f"LSTM Regressor MAPE: {mape_lstm * 100:.2f}%")
        print(f"RNN Regressor MAPE: {mape_rnn * 100:.2f}%")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Optional positional arguments
    parser.add_argument("training_file", help="Training data file")
    parser.add_argument("testing_file", help="Testing data file")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)
