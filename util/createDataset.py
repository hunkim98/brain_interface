##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

import pandas as pd
import numpy as np
import re
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plot
import matplotlib.dates as md
from scipy import stats


class CreateDataset:
    base_dir = ""
    granularity = 0
    labels = []

    def __init__(self, base_dir, granularity):
        self.base_dir = base_dir
        self.granularity = granularity

    # Add left and right labels to the dataset
    # Once there is a trigger, we keep adding 1 to left until we get a trigger for right
    def add_left_right(self, dataset):
        left = []
        right = []
        # the label is set as a modifiable variable to be able to change it
        label = 0
        for i in range(0, len(dataset.index)):
            if type(dataset.loc[i, "Elements"]) == str:
                # if it is str it menas it is a marker line
                # if a value in elements is a string, we have a marker line OR a eye blink line (no event gives nan)
                if dataset.loc[i, "Elements"].startswith(
                    "/Marker/"
                ):  # only when it is a marker line, we change label, else pass
                    label = int(
                        dataset.loc[i, "Elements"][-1]
                    )  # marker is 1 (for left) or 2 (for right) or 3 (relaxing)
            if label == 1:  # keep adding 1 to left if label stays 1
                left.append(1)
                right.append(0)
            elif label == 2:  # add right when label is 2 / stays 2
                left.append(0)
                right.append(1)
            else:
                # this means label is 3 or nan, so we add 0 to both
                left.append(0)
                right.append(0)
        dataset["label_left"] = left
        dataset["label_right"] = right
        dataset["label_on"] = np.logical_or(left, right)
        return dataset

    def add_label_activate(self, dataset):
        label_ons = []

    def create_dataset(self, start_time, end_time, cols):
        # we will create a dataset with granularity as the time step
        timestamps = pd.date_range(
            start_time, end_time, freq=str(self.granularity) + "ms"
        )
        data_table = pd.DataFrame(index=timestamps, columns=cols)
        for col in cols:
            data_table[str(col)] = np.nan  # initialize the columns
        return data_table

    def num_sampling(self, dataset, data_table, value_cols, aggregation="avg"):
        for i in range(0, len(data_table.index)):
            relevant_rows = dataset[
                (dataset["TimeStamp"] >= data_table.index[i])
                & (
                    dataset["TimeStamp"]
                    < (data_table.index[i] + timedelta(milliseconds=self.granularity))
                )
            ]
            for col in value_cols:
                # numerical cols which for the EEG data are the brain waves
                # We take the average value
                if len(relevant_rows) > 0:
                    data_table.loc[data_table.index[i], str(col)] = np.average(
                        relevant_rows[col]
                    )
                else:
                    data_table.loc[data_table.index[i], str(col)] = np.nan
        return data_table

    # this is categorical sampling, we take the mode of the labels
    def cat_sampling(self, dataset, data_table, label_cols):
        for i in range(0, len(data_table.index)):
            relevant_rows = dataset[
                (dataset["TimeStamp"] >= data_table.index[i])
                & (
                    dataset["TimeStamp"]
                    < (data_table.index[i] + timedelta(milliseconds=self.granularity))
                )
            ]
            for col in label_cols:
                # We put 1 when most value of the labels in relevant rows are 1, else 0
                if len(relevant_rows) > 0:
                    # stats.mode prints out mode as well as counts
                    data_table.loc[data_table.index[i], str(col)] = stats.mode(
                        relevant_rows[col]
                    )[
                        0
                    ]  # so only select the mode, not counts
                else:
                    data_table.loc[data_table.index[i], str(col)] = np.nan
        return data_table

    # Add numerical data, we assume timestamps in the form of nanoseconds from the epoch
    def add_data(self, file, value_cols, label_cols, aggregation="avg"):
        dataset = pd.read_csv(file, skipinitialspace=True)

        marker_dict = {}
        filtered = dataset[dataset["Elements"].str.startswith("/Marker")][
            "Elements"
        ].unique()
        for marker in filtered:
            label = marker.split("/")[-1]
            if marker in marker_dict:
                marker_dict[label] += 1
            else:
                marker_dict[label] = 1

        # we update the labels with keys
        self.labels = list(marker_dict.keys())

        dataset["TimeStamp"] = pd.to_datetime(dataset["TimeStamp"])
        dataset = self.add_left_right(
            dataset
        )  # add features for left and right motor imagery labels

        threshold = dataset.shape[1] - 10
        dataset.dropna(
            thresh=threshold, axis=0, inplace=True
        )  # delete the rows of logs of markers (rows with > col-10 nans)

        # now we initialize the sampled dataset with our granularity
        all_columns = value_cols + label_cols

        data_table = self.create_dataset(
            min(dataset["TimeStamp"]), max(dataset["TimeStamp"]), all_columns
        )  # this creates a df named data_table

        data_table = self.num_sampling(
            dataset, data_table, value_cols
        )  # add numerical data
        data_table = self.cat_sampling(
            dataset, data_table, label_cols
        )  # add label data
        return data_table
