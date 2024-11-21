import os
import pandas as pd
from pathlib import Path
from util.prepareDatasetForLearning import PrepareDatasetForLearning
from util.learningAlgorithms import ClassificationAlgorithms
from util.evaluation import ClassificationEvaluation
from util.featureSelection import FeatureSelectionClassification
from util import util
from util.VisualizeDataset import VisualizeDataset

# Set up file names and locations.
FOLDER_PATH = Path("./intermediate_datafiles/motor_imagery/test3_result")
RESULT_PATH = Path("./intermediate_datafiles/motor_imagery/test4_result")


def main():

    RESULT_PATH.mkdir(exist_ok=True, parents=True)
    # for this script, we want to first load in all datasets
    # since the Prepare dataset function accepts a list of pd dataframes
    prepare = PrepareDatasetForLearning()
    all_datasets = []

    for instance in os.scandir(FOLDER_PATH):  # go through all instances of experiments
        instance_path = instance.path
        dataset = pd.read_csv(instance_path, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        all_datasets.append(dataset)

    # now all dataframes are added to the list all_datasets
    # print(all_datasets)

    """
    the classification of the motor imagery can be seen as a non-temporal task, as we want to predict imagery based on a window of e.g. 2 sec,
    without taking into account previous windows.
    We first create 1 column representing our classes, and then create a train val test split of 60 20 20
    In order to do this, we first create a train test split of 80 20, and then for the train set we split again in 75 25
    For each dataset instance. we split trainvaltest split individually.
    Then later we add all train data together, all val data together, and all test data together.
    This way we sample randomly across all users to get a result for the whole 'population' of subjects.
    """
    # we set filter is false so also the data besides left and right are taken with us
    train_X, val_X, test_X, train_y, val_y, test_y = (
        prepare.split_multiple_datasets_classification(
            all_datasets,
            ["label_left", "label_right"],
            "like",
            [0.2, 0.25],
            filter=False,
            temporal=False,
        )
    )
    print("Training set length is: ", len(train_X.index))
    print("Validation set length is: ", len(val_X.index))
    print("Test set length is: ", len(test_X.index))

    # select subsets of features which we will consider:
    pca_features = ["pca_1", "pca_2", "pca_3", "pca_4"]
    ica_features = [
        "FastICA_1",
        "FastICA_2",
        "FastICA_3",
        "FastICA_4",
        "FastICA_5",
        "FastICA_6",
        "FastICA_7",
        "FastICA_8",
        "FastICA_9",
        "FastICA_10",
        "FastICA_11",
        "FastICA_12",
        "FastICA_13",
        "FastICA_14",
        "FastICA_15",
        "FastICA_16",
        "FastICA_17",
        "FastICA_18",
        "FastICA_19",
        "FastICA_20",
    ]
    time_features = [name for name in dataset.columns if "_temp_" in name]
    freq_features = [
        name for name in dataset.columns if (("_freq" in name) or ("_pse" in name))
    ]

    # feature selection below we will use as input for our models:
    basic_features = [
        "Delta_TP9",
        "Delta_AF7",
        "Delta_AF8",
        "Delta_TP10",
        "Theta_TP9",
        "Theta_AF7",
        "Theta_AF8",
        "Theta_TP10",
        "Alpha_TP9",
        "Alpha_AF7",
        "Alpha_AF8",
        "Alpha_TP10",
        "Beta_TP9",
        "Beta_AF7",
        "Beta_AF8",
        "Beta_TP10",
        "Gamma_TP9",
        "Gamma_AF7",
        "Gamma_AF8",
        "Gamma_TP10",
    ]
    basic_w_PCA = list(set().union(basic_features, pca_features))
    basic_w_ICA = list(set().union(basic_features, ica_features))
    all_features = list(
        set().union(basic_features, ica_features, time_features, freq_features)
    )

    fs = FeatureSelectionClassification()
    num_features = 20

    # we will select the top 20 features based on an experiment with a deciscion tree which we will use as input for our models as well
    # this is already been run, see below

    selected_features, ordered_features, ordered_scores = fs.forward_selection(
        num_features,
        train_X[all_features],
        test_X[all_features],
        train_y,
        test_y,
        gridsearch=False,
    )
    print(selected_features)

    # then here, we run each model
    learner = ClassificationAlgorithms()
    eval = ClassificationEvaluation()

    # and then we chose the 1 or 2 best ones to apply gridsearch etc
    # from my initial results, RF with all features seems to perform best!
    # lets try it with the validation set and gridsearch = True.
    # eventually if we are happy with the best one, and we save that by setting save_model=True for later use with the real time predictions part!
    # print(test_y)
    # print(train_X)
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = (
        learner.random_forest(
            train_X,
            train_y,
            test_X,
            gridsearch=True,
            print_model_details=True,
            save_model=True,
        )
    )
    performance_training_rf_final = eval.f1(train_y, class_train_y)
    performance_test_rf_final = eval.f1(test_y, class_test_y)
    confusionmatrix_rf_final = eval.confusion_matrix(
        test_y, class_test_y, ["label_left", "label_right", "undefined"]
    )
    print(performance_test_rf_final)  # test performance is reasonable!
    print(confusionmatrix_rf_final)


if __name__ == "__main__":
    main()
