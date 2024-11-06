from sklearn.model_selection import train_test_split
import numpy as np
import random
import copy
import pandas as pd


# This class creates datasets that can be used by the learning algorithms. Up till now we have
# assumed binary columns for each class, we will for instance introduce approaches to create
# a single categorical attribute.
class PrepareDatasetForLearning:
    default_label = "undefined"
    class_col = "class"
    person_col = "person"

    # This function creates a single class column based on a set of binary class columns.
    # it essentially merges them. It removes the old label columns.
    def assign_label(self, dataset, class_labels):
        # Find which columns are relevant based on the possibly partial class_label
        # specification.

        labels = []
        for i in range(0, len(class_labels)):
            labels.extend(
                [
                    name
                    for name in list(dataset.columns)
                    if class_labels[i] == name[0 : len(class_labels[i])]
                ]
            )

        # Determine how many class values are label as 'true' in our class columns.
        sum_values = dataset[labels].sum(axis=1)
        # Create a new 'class' column and set the value to the default class.
        dataset["class"] = self.default_label
        for i in range(0, len(dataset.index)):
            # If we have exactly one true class column, we can assign that value,
            # otherwise we keep the default class ('none').
            if sum_values[i] == 1:
                dataset.iloc[i, dataset.columns.get_loc(self.class_col)] = (
                    dataset[labels].iloc[i].idxmax(axis=0)
                )

        # And remove our old binary columns.
        dataset = dataset.drop(labels, axis=1)
        return dataset

    # If 'like' is specified in matching, we will merge the columns that contain the class_labels into a single
    # column. We can select a filter for rows where we are unable to identifty a unique
    # class and we can select whether we have a temporal dataset or not. In the former, we will select the first
    # training_frac of the data for training and the last 1-training_frac for testing. Otherwise, we select points randomly.
    # We return a training set, the labels of the training set, and the same for a test set. We can set the random seed
    # to make the split reproducible.

    def split_single_dataset_classification(
        self,
        dataset,
        class_labels,
        matching,
        split_fracs,
        filter=True,
        temporal=False,
        random_state=0,
    ):
        # Create a single class column if we have the 'like' option.

        if matching == "like":
            dataset = self.assign_label(dataset, class_labels)
            class_labels = self.class_col
        elif len(class_labels) == 1:
            class_labels = class_labels[0]

        # Filer NaN is desired and those for which we cannot determine the class should be removed.
        if filter:
            dataset = dataset.dropna()
            dataset = dataset[dataset["class"] != self.default_label]

        # The features are the ones not in the class label.
        features = [
            dataset.columns.get_loc(x) for x in dataset.columns if x not in class_labels
        ]
        class_label_indices = [
            dataset.columns.get_loc(x) for x in dataset.columns if x in class_labels
        ]

        # For temporal data, we select the desired fraction of training data from the first part
        # and use the rest as test set.
        if temporal:
            end_training_set = int(split_fracs[0] * len(dataset.index))
            training_set_X = dataset.iloc[0:end_training_set, features]
            training_set_y = dataset.iloc[0:end_training_set, class_label_indices]
            test_set_X = dataset.iloc[end_training_set : len(dataset.index), features]
            test_set_y = dataset.iloc[
                end_training_set : len(dataset.index), class_label_indices
            ]

            # TODO: add validation set code for temporal part

        # For non temporal data we use a standard function to randomly split the dataset.
        else:
            training_set_X, test_set_X, training_set_y, test_set_y = train_test_split(
                dataset.iloc[:, features],
                dataset.iloc[:, class_label_indices],
                test_size=split_fracs[0],
                stratify=dataset.iloc[:, class_label_indices],
                random_state=random_state,
            )
            training_set_X, val_set_X, training_set_y, val_set_y = train_test_split(
                training_set_X,
                training_set_y,
                test_size=split_fracs[1],
                stratify=training_set_y,
                random_state=random_state,
            )
        return (
            training_set_X,
            val_set_X,
            test_set_X,
            training_set_y,
            val_set_y,
            test_set_y,
        )

    # If we have multiple overlapping indices (e.g. user 1 and user 2 have the same time stamps) our
    # series cannot me merged properly, therefore we can create a new index.
    def update_set(self, source_set, addition):
        if source_set is None:
            return addition
        else:
            # Check if the index is unique. If not, create a new index.
            if len(set(source_set.index) & set(addition.index)) > 0:
                return source_set._append(addition).reset_index(drop=True)
            else:
                return source_set._append(addition)

    def split_multiple_datasets_classification(
        self,
        datasets,
        class_labels,
        matching,
        split_fracs,
        filter=False,
        temporal=False,
        unknown_users=False,
        random_state=0,
    ):
        training_set_X = None
        training_set_y = None
        val_set_X = None
        val_set_y = None
        test_set_X = None
        test_set_y = None

        init = True
        # If we want to predict for each experiment
        # we split each dataset individually in a training and test set and then add all data together randomly.
        for i in range(0, len(datasets)):
            (
                training_set_X_person,
                val_set_X_person,
                test_set_X_person,
                training_set_y_person,
                val_set_y_person,
                test_set_y_person,
            ) = self.split_single_dataset_classification(
                datasets[i],
                class_labels,
                matching,
                split_fracs,
                filter=filter,
                temporal=temporal,
                random_state=random_state,
            )

            training_set_X = self.update_set(training_set_X, training_set_X_person)
            training_set_y = self.update_set(training_set_y, training_set_y_person)

            val_set_X = self.update_set(val_set_X, val_set_X_person)
            val_set_y = self.update_set(val_set_y, val_set_y_person)

            test_set_X = self.update_set(test_set_X, test_set_X_person)
            test_set_y = self.update_set(test_set_y, test_set_y_person)

        return (
            training_set_X,
            val_set_X,
            test_set_X,
            training_set_y,
            val_set_y,
            test_set_y,
        )
