"""
Module docstring: This module includes a class that process all functions.
"""

from data_preparation import final_data_prep
from ml_modelling import train_classification_model, final_inference


class ProductPropensities:
    """
    This class runs all the processing and training steps and generates results
    """

    def __init__(self):
        self.final_data_prep = final_data_prep
        self.train_classification_model = train_classification_model
        self.final_inference = final_inference

    def train_evaluate_ml(self):
        """
        this function combines all previous processing, training, and inference functions
        """
        (
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
        ) = self.final_data_prep()

        # Train model for product A
        (model_product_a) = self.train_classification_model(
            x_train_product_a,
            y_train_product_a,
            x_test_product_a,
            y_test_product_a,
            "<product A>",
        )

        # Train model for product B
        (model_product_b) = self.train_classification_model(
            x_train_product_b,
            y_train_product_b,
            x_test_product_b,
            y_test_product_b,
            "<product B>",
        )

        # Train model for product C
        (model_product_c) = self.train_classification_model(
            x_train_product_c,
            y_train_product_c,
            x_test_product_c,
            y_test_product_c,
            "<product C>",
        )

        # Perform final inference
        self.final_inference(
            x_test_product_a,
            x_test_product_b,
            x_test_product_c,
            model_product_a,
            model_product_b,
            model_product_c,
        )


# Create an instance of the class and call the public fnt train_evaluate_ml
models_product_propensities_engagement = ProductPropensities()
if __name__ == "__main__":
    models_product_propensities_engagement.train_evaluate_ml()
