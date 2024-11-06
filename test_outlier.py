from util.VisualizeDataset import VisualizeDataset
from util.outlier import (
    DistributionBasedOutlierDetection,
    DistanceBasedOutlierDetection,
)
from util.filters import Filters
import os
import sys
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Set up file names and locations.

FOLDER_PATH = Path("./intermediate_datafiles/motor_imagery/test1_result")
RESULT_PATH = Path("./intermediate_datafiles/motor_imagery/test2_result")


def main(n):
    # We'll create an instance of our visualization class to plot results.
    # DataViz = VisualizeDataset(__file__)

    # initialize outlier classes
    RESULT_PATH.mkdir(exist_ok=True, parents=True)

    OutlierDistr = DistributionBasedOutlierDetection()
    OutlierDist = DistanceBasedOutlierDetection()

    for instance in os.scandir(FOLDER_PATH):  # go through all instances of experiments
        instance_path = instance.path
        print(f"Going through pipeline for file {instance_path}.")
        dataset = pd.read_csv(instance_path, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)

        for col in [c for c in dataset.columns if not "label" in c]:
            print(f"Measurement is now: {col}")
            # print('Step 1: Outlier detection')

            # we use mixture model as it is used in one paper with n=3. Number of outliers is very low
            # but measurements are short so this is explainable, also we use brain wave data now
            # in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7728142/pdf/sensors-20-06730.pdf
            # they actually check sort of manually what data is noisy and discard that
            # I did that as well, by looking at the figure I made in step 1 for creating the dataset

            dataset = OutlierDistr.mixture_model(dataset, col, 3)
            # print('Number of outliers for points with prob < 5e-5 for feature ' + col + ': ' + str(dataset[col+'_mixture'][dataset[col+'_mixture'] < 0.0005].count()))

            dataset.loc[dataset[f"{col}_mixture"] < 0.0005, col] = np.nan
            del dataset[col + "_mixture"]

            # print('Step 2: Imputation')
            # print('Before interpolation, number of nans left should be > 0: ' + str(dataset[col].isna().sum()))
            # print('Also count amount of zeroes:' + str((dataset[col] == 0).sum()))

            dataset[col] = dataset[col].interpolate()  # interpolating missing values
            dataset[col] = dataset[col].fillna(
                method="bfill"
            )  # And fill the initial data points if needed

            # check if all nan are filled in
            print(
                "Check, number of nans left should be 0: "
                + str(dataset[col].isna().sum())
            )

            # Step 3: lowpass filtering of periodic measurements. As all our features are brain waves and thus periodic,
            # we do this for all features expect the labels
            # Note that the brain wave values are already filtered as per https://mind-monitor.com/Technical_Manual.php#help_graph_absolute
            # but if I later want to work with Raw EEG data, filtering is abosutely necessary
            # I would NOT use a High pass filter (https://sapienlabs.org/pitfalls-of-filtering-the-eeg-signal/)
            # which IS currently used by the mind monitor / muse app as the delta freqs are 1-4Hz
            # dataset = Filters.low_pass_filter(dataset, col, fs, cutoff, order=10)
            # dataset[col] = dataset[col + '_lowpass']
            # del dataset[col + '_lowpass']

        dataset.to_csv(Path(str(RESULT_PATH) + "/" + instance.name))


if __name__ == "__main__":
    # Command line arguments

    mode = "mixture"
    c = 2
    n = 3
    K = 5
    dmin = 0.10
    fmin = 0.99

    main(n)
