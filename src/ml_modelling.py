"""
This module provides functions for data transformation and ml modelling.
"""
import time
import pandas as pd
import numpy as np
from loguru import logger

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
)

import const as c


def build_ml_pipeline():
    """
    This function builds an ML pipleine.
    """
    # Define preprocessing steps for numerical and categorical features
    numerical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),  # Impute missing numeric values using median strategy
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),  # Impute missing categorical values with most frequent strategy
        ]
    )
    # Combine preprocessing steps for numerical and categorical features using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, c.NUMERIC_FEATURES),
            ("cat", categorical_transformer, c.CATEGORICAL_FEATURES),
        ]
    )
    # Create a pipeline which combines the data preprocesser and random forest model
    ml_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    criterion="gini",
                    max_depth=5,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=c.RANDOM_STATE,
                ),
            ),
        ]
    )

    return ml_pipeline


def train_classification_model(x_train, y_train, x_test, y_test, product_name):
    """
    this function trains the ML model on each product separately.
    """
    start_time = time.time()  # Record the start time
    model = build_ml_pipeline()
    model.fit(x_train, y_train.values.ravel())
    # Get predicted probabilities
    proba = model.predict_proba(x_test)[:, 1]
    # Convert predicted probabilities to binary class labels based on threshold
    y_predict = (proba >= c.PROB_THRESHOLD).astype(int)
    confusion_matrix_result = confusion_matrix(y_test, y_predict)
    classification_report_result = classification_report(y_predict, y_test)
    logger.info(
        f"Classification Report of {product_name} Model on Test-set: {classification_report_result}"
    )
    logger.info(
        f"Confusion Matrix of {product_name} Model on Test-set: {confusion_matrix_result}"
    )
    f1_score_val = round(f1_score(y_test, y_predict, average="micro"), 2)
    # Log the model F-1 score
    logger.info(f"F1 Score of {product_name} Model on Validation-set: {f1_score_val}")
    end_time = time.time()  # Record the end time
    run_time = end_time - start_time  # Calculate the elapsed time in secs
    # Log the model run time
    logger.info(f"Model Run Time for {product_name} Model is: {run_time:.2f} secs")
    return model


def final_inference(
    x_test_product_a,
    x_test_product_b,
    x_test_product_c,
    model_product_a,
    model_product_b,
    model_product_c,
):

    """
    this function generates predictions onreal dataset and get the reccomended product at account level
    """
    inference_df = pd.concat(
        [x_test_product_a, x_test_product_b, x_test_product_c]
    ).sample(frac=0.4, random_state=c.RANDOM_STATE)
    # get predictions of each product separately
    y_pred_proba_product_a = model_product_a.predict_proba(inference_df)[:, 1]
    y_pred_proba_product_b = model_product_b.predict_proba(inference_df)[:, 1]
    y_pred_proba_product_c = model_product_c.predict_proba(inference_df)[:, 1]
    # create propensities columns with rounded probabaility
    inference_df["predicted_propensity_product_a"] = np.round(y_pred_proba_product_a, 3)
    inference_df["predicted_propensity_product_b"] = np.round(y_pred_proba_product_b, 3)
    inference_df["predicted_propensity_product_c"] = np.round(y_pred_proba_product_c, 3)
    # create deciles from 1-10; 1 is the lowest and 10 is the highest
    inference_df["predicted_propensity_product_a_decile"] = (
        pd.qcut(inference_df["predicted_propensity_product_a"], q=10, labels=False) + 1
    )
    inference_df["predicted_propensity_product_b_decile"] = (
        pd.qcut(inference_df["predicted_propensity_product_b"], q=10, labels=False) + 1
    )
    inference_df["predicted_propensity_product_c_decile"] = (
        pd.qcut(inference_df["predicted_propensity_product_c"], q=10, labels=False) + 1
    )
    # Create reccomended product column
    # Find the maximum value among the three columns
    max_value = inference_df[
        [
            "predicted_propensity_product_a_decile",
            "predicted_propensity_product_b_decile",
            "predicted_propensity_product_c_decile",
        ]
    ].max(axis=1)
    # Define conditions as boolean ndarrays
    condition_product_a = (
        inference_df["predicted_propensity_product_a_decile"] == max_value
    ).astype(bool)
    condition_product_b = (
        inference_df["predicted_propensity_product_b_decile"] == max_value
    ).astype(bool)
    condition_product_c = (
        inference_df["predicted_propensity_product_c_decile"] == max_value
    ).astype(bool)
    # Define corresponding values
    values = ["Product A", "Product B", "Product C"]

    # Use numpy.select to determine the recommended product
    recommended_product = np.select(
        [condition_product_a, condition_product_b, condition_product_c], values
    )
    inference_df["recommended_product"] = recommended_product
    recommended_product_counts = (
        inference_df["recommended_product"].value_counts(normalize=True).round(2)
    )
    logger.info(
        f"The breakdown of recommended products to customers is: {recommended_product_counts}"
    )
