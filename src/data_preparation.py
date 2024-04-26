"""
Module docstring: This module provides functions for data preparation and preprocessing.
"""

import os
import pandas as pd
import numpy as np
from loguru import logger

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

import const as c


def load_data():
    """
    Loads the data from a CSV file and returns a DataFrame.
    """
    data = pd.read_csv(os.path.join("..", "data", "customer_data.csv"))
    return data


def data_prep(data):
    """
    Prepares the data by generating additional features to be used in predictive modelling
    """
    data["account_tenure_months"] = np.random.randint(1, 49, size=len(data))
    data["num_comms_msgs"] = np.random.randint(0, 10, size=len(data))
    data["product_A_purchase"] = np.random.choice(
        [False, True], size=len(data), p=c.CLASS_DISTRIBUTIONS
    )
    data["product_B_purchase"] = np.random.choice(
        [False, True], size=len(data), p=c.CLASS_DISTRIBUTIONS
    )
    data["product_C_purchase"] = np.random.choice(
        [False, True], size=len(data), p=c.CLASS_DISTRIBUTIONS
    )
    data = data.drop(["name"], axis=1)
    logger.info(f"Missing Data Values: {data.isnull().sum()}")
    logger.info(f"Data Types: {data.info()}")
    logger.info(
        f"Country Variable Distribution: {data.country.value_counts(normalize=True)}"
    )
    logger.info(
        f"Education Variable Distribution: {data.education.value_counts(normalize=True)}"
    )
    logger.info(
        f"Gender Variable Distribution: {data.gender.value_counts(normalize=True)}"
    )
    return data


def label_encode(data):
    """
    Encodes categorical variables using LabelEncoder.
    """
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = label_encoder.fit_transform(data[column])
    return data


def select_independent_dependent_variables(data):
    """
    Selects the independent variables (modelling_df) and target variables
    (product_A, product_B, product_C).
    """
    target_variable_product_a = data["product_A_purchase"]
    target_variable_product_b = data["product_B_purchase"]
    target_variable_product_c = data["product_C_purchase"]
    modelling_df = data.drop(
        [
            "product_A_purchase",
            "product_B_purchase",
            "product_C_purchase",
        ],
        axis=1,
    )
    return (
        target_variable_product_a,
        target_variable_product_b,
        target_variable_product_c,
        modelling_df,
    )


def stratified_split(modelling_df, target_variable, target_variable_name, test_size):
    """
    Performs stratified splitting of the data into train and test sets.
    """
    split = StratifiedShuffleSplit(
        n_splits=c.NUM_SPLITS, test_size=test_size, random_state=c.RANDOM_STATE
    )
    for train_index, test_index in split.split(modelling_df, target_variable):
        x_train = modelling_df.loc[train_index]  # Train set
        y_train = pd.DataFrame(target_variable.loc[train_index])  # Train targets
        x_test = modelling_df.loc[test_index]  # Test set
        y_test = target_variable.loc[test_index]  # Test targets

    print(f"Train set of {target_variable_name} has {x_train.shape[0]} samples.")
    print(f"Testing set of {target_variable_name} has {x_test.shape[0]} samples.")
    return x_train, y_train, x_test, y_test


def apply_smote(final_train, y_train, target_variable_name):
    """
    Applies SMOTE to balance the class distribution.
    """
    class_0_samples = np.sum(y_train == 0)
    desired_samples_class_1 = int(c.SMOTE_DESIRED_RATIO * class_0_samples)
    smote = SMOTE(sampling_strategy={1: desired_samples_class_1})
    final_train, y_train = smote.fit_resample(final_train, y_train)
    print(
        f"Train set of {target_variable_name} after SMOTE  has {final_train.shape[0]} samples."
    )
    return final_train, y_train


def final_data_prep():
    """
    Performs final data preparation including loading data, preprocessing, and splitting.
    """
    data = load_data()
    final_data = data_prep(data)
    final_data = label_encode(final_data)
    (
        target_variable_product_a,
        target_variable_product_b,
        target_variable_product_c,
        modelling_df,
    ) = select_independent_dependent_variables(final_data)
    (
        x_train_product_a,
        y_train_product_a,
        x_test_product_a,
        y_test_product_a,
    ) = stratified_split(
        modelling_df, target_variable_product_a, "product_A", c.TEST_SIZE
    )
    (
        x_train_product_b,
        y_train_product_b,
        x_test_product_b,
        y_test_product_b,
    ) = stratified_split(
        modelling_df, target_variable_product_b, "product_B", c.TEST_SIZE
    )
    (
        x_train_product_c,
        y_train_product_c,
        x_test_product_c,
        y_test_product_c,
    ) = stratified_split(
        modelling_df, target_variable_product_c, "product_C", c.TEST_SIZE
    )

    x_train_product_a, y_train_product_a = apply_smote(
        x_train_product_a, y_train_product_a, "product_A"
    )
    x_train_product_b, y_train_product_b = apply_smote(
        x_train_product_b, y_train_product_b, "product_B"
    )
    x_train_product_c, y_train_product_c = apply_smote(
        x_train_product_c, y_train_product_c, "product_C"
    )
    return (
        x_train_product_a,
        y_train_product_a,
        x_test_product_a,
        y_test_product_a,
        x_train_product_b,
        y_train_product_b,
        x_test_product_b,
        y_test_product_b,
        x_train_product_c,
        y_train_product_c,
        x_test_product_c,
        y_test_product_c,
    )
