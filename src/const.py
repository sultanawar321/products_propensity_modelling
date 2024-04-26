"""
This module defines constants used in the project.
"""

# Class Distributions to create the dummy dependent variables
CLASS_DISTRIBUTIONS = [
    0.95,
    0.05,
]

# Number of Splits to be applied in Stratified Sampling
NUM_SPLITS = 3

# Random state
RANDOM_STATE = 42

# SMOTE Oversampling ratio for class 1
SMOTE_DESIRED_RATIO = 0.7

# Proportion of data to be set for validation and test purposes
TEST_SIZE = 0.2

# Numeric features
NUMERIC_FEATURES = [
    "age",
    "income",
    "purchase_frequency",
    "spending",
    "account_tenure_months",
    "num_comms_msgs",
]

# Categorical features
CATEGORICAL_FEATURES = ["gender", "education", "country"]

# Probability threshold
PROB_THRESHOLD = 0.5
