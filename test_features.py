from util.VisualizeDataset import VisualizeDataset
from util.dim_reduction import PrincipalComponentAnalysis, IndependentComponentAnalysis
from util.temporalAbstraction import NumericalAbstraction
from util.frequencyAbstraction import FourierTransformation
from util.VisualizeDataset import VisualizeDataset

from util.filters import Filters
import os
import sys
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Set up file names and locations.

FOLDER_PATH = Path("./intermediate_datafiles/motor_imagery/test2_result")
RESULT_PATH = Path("./intermediate_datafiles/motor_imagery/test3_result")


def main(n):
    # We'll create an instance of our visualization class to plot results.
    # DataViz = VisualizeDataset(__file__)

    # initialize outlier classes
    RESULT_PATH.mkdir(exist_ok=True, parents=True)
    # initialize feature engineering classes
    PCA = PrincipalComponentAnalysis()
    ICA = IndependentComponentAnalysis()
    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()
    # initialize example dataset stuff for experimenting
    # get the first csv inside the folder_path, check if it is a .csv file
    example_filename = next(
        (f for f in os.scandir(FOLDER_PATH) if f.is_file() and f.name.endswith(".csv")),
        None,
    )
    example_dataset = pd.read_csv(example_filename, index_col=0)
    example_dataset.index = pd.to_datetime(example_dataset.index)

    # ms per instance is used for the freq and time features
    milliseconds_per_instance = (
        example_dataset.index[1] - example_dataset.index[0]
    ).microseconds / 1000

    for instance in os.scandir(FOLDER_PATH):  # go through all instances of experiments
        instance_path = instance.path
        print(f"Going through pipeline for file {instance_path}.")
        dataset = pd.read_csv(instance_path, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        selected_cols = [c for c in dataset.columns if not "label" in c]

        # check nan values
        print(dataset[selected_cols].isnull().sum())

        # PCA with n_pcs of 4 as found in experiment above
        n_pcs = 4
        dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_cols, n_pcs)

        # DataViz.plot_dataset(dataset, ['Delta_TP9', 'Theta_AF7', 'Alpha_AF8', 'Beta_TP10', 'Gamma_AF7', 'pca_1'],
        # ['like', 'like', 'like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'line', 'line', 'line'], algo='finalPCA')

        # ICA: we apply FastICA for all components (all cols)
        dataset = ICA.apply_ica(copy.deepcopy(dataset), selected_cols)

        # DataViz.plot_dataset(dataset, ['Delta_TP9', 'Theta_AF7', 'Alpha_AF8', 'Beta_TP10', 'Gamma_AF7', 'FastICA_1'],
        # ['like', 'like', 'like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'line', 'line', 'line'], algo='finalICA')

        # Freq and time domain features for ws of 1 sec, 2 sec, and 3 sec
        window_sizes = [
            int(float(1000) / milliseconds_per_instance),
            int(float(2000) / milliseconds_per_instance),
            int(float(3000) / milliseconds_per_instance),
        ]
        fs = 100  # sample frequency

        for ws in window_sizes:
            dataset = NumAbs.abstract_numerical(
                dataset,
                selected_cols,
                ws,
                ["mean", "std", "max", "min", "median", "slope"],
            )

        # we only do fourier transformation for smallest ws [1 sec]
        dataset = FreqAbs.abstract_frequency(
            dataset, selected_cols, window_sizes[0], fs
        )

        # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.
        # The percentage of overlap we allow:
        window_overlap = 0.5
        # we do this for the biggest ws
        skip_points = int((1 - window_overlap) * window_sizes[-1])
        dataset = dataset.iloc[::skip_points, :]

        # apparently the first two rows are NaNs so delete those.
        dataset = dataset.iloc[2:]

        # save data
        dataset.to_csv(Path(str(RESULT_PATH) + "/" + instance.name))
        print(dataset.shape)


if __name__ == "__main__":
    # Command line arguments

    mode = "mixture"
    c = 2
    n = 3
    K = 5
    dmin = 0.10
    fmin = 0.99

    main(n)
