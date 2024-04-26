"""A module that includes basic unit tests for the main functions"""
import sys

sys.path.append("../src/")

from data_preparation import *
from ml_modelling import *

from sklearn.model_selection import train_test_split

import pytest


@pytest.fixture
def sample_data():
    # Create sample data for testing
    data = pd.DataFrame(
        {
            "feature1": ["A", "B", "C", "A", "B"],
            "feature2": [1, 2, 3, 4, 5],
            "product_A_purchase": [0, 1, 1, 0, 1],
            "product_B_purchase": [1, 0, 1, 0, 1],
            "product_C_purchase": [1, 0, 0, 1, 0],
        }
    )
    return data


def test_label_encode(sample_data):
    # Test if label encoding works correctly
    encoded_data = label_encode(sample_data.copy())
    for column in encoded_data.columns:
        if encoded_data[column].dtype == "object":
            assert encoded_data[column].dtype == "int64"


def test_select_independent_dependent_variables(sample_data):
    # Test if independent and dependent variables are selected correctly
    target_a, target_b, target_c, modelling_df = select_independent_dependent_variables(
        sample_data.copy()
    )
    assert list(target_a) == [0, 1, 1, 0, 1]
    assert list(target_b) == [1, 0, 1, 0, 1]
    assert list(target_c) == [1, 0, 0, 1, 0]
    assert list(modelling_df.columns) == ["feature1", "feature2"]


def test_stratified_split(sample_data):
    # Test if stratified split works correctly
    target_a, _, _, modelling_df = select_independent_dependent_variables(
        sample_data.copy()
    )
    x_train, y_train, x_test, y_test = stratified_split(
        modelling_df, target_a, "Product A", test_size=0.5
    )  # Adjusted test_size
    assert x_train.shape[0] + x_test.shape[0] == modelling_df.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == target_a.shape[0]


@pytest.fixture
def sample_data_1():
    # Generate sample data for ml training test
    data = {
        "age": ["5", "6", "6", "9", "0"],
        "income": [1, 2, 3, 4, 5],
        "purchase_frequency": np.random.uniform(0, 1, size=5),
        "spending": np.random.uniform(10, 200, size=5),
        "account_tenure_months": np.random.randint(6, 60, size=5),
        "num_comms_msgs": np.random.randint(0, 100, size=5),
        "gender": np.random.choice(["1", "0"], size=5),
        "education": np.random.choice(["1", "2", "3", "4"], size=5),
        "country": np.random.choice(["1", "5", "9", "1"], size=5),
        "product_A_purchase": [1, 0, 0, 0, 1],
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Separate features and target variables
    X = df.drop(["product_A_purchase"], axis=1)
    y = df["product_A_purchase"]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return x_train, y_train, x_test, y_test


def test_train_classification_model(sample_data_1):
    x_train, y_train, x_test, y_test = sample_data_1
    model = train_classification_model(x_train, y_train, x_test, y_test, "Test Product")
    assert model is not None
