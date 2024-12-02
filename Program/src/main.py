#!/usr/bin/env python3
"""
Module Docstring
"""
from src.models.support_vector_machine import support_vector_machine
from src.file_load import create_test_train_data

__author__ = "Thibo De Belie, Vince Driesen, Daan Hollands"
__version__ = "0.1.0"
__license__ = "GPLv3"

import argparse
import os

def main(args):
    if args.full_file or (not args.training_file and not args.testing_file):
        full_file = args.full_file if args.full_file else "../resources/historicalData_IE00B5BMR087_clean.csv"
        print(f"Using full data file: {full_file}")
        X_train, y_train, X_test, y_test = create_test_train_data(full_file=full_file)
    else:
        training_file = args.training_file
        testing_file = args.testing_file
        print(f"Using training file: {training_file}")
        print(f"Using testing file: {testing_file}")
        X_train, y_train, X_test, y_test = create_test_train_data(train_file=training_file, test_file=testing_file)

    best_kernel = support_vector_machine(X_train, y_train, X_test, y_test)
    print(f'De beste kernel is: {best_kernel}')

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
