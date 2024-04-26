# products_propensity_modelling
This project includes a ML project to estimate the propensity (likelihood) of a customer purchasing a product. 

- Consider a scenario where you aim to send campaign communication messages to customers regarding three different products, and you want to identift the most suitable product for each customer. This project uses machine learning to predict the engagement propensity of each product for every customer, subsequently selecting the recommended product based on the highest propensity. The dataset is taken from [Kaggle](https://www.kaggle.com/datasets/goyaladi/customer-spending-dataset) with some additional dummy variables created for the purpose of this excercise. 

# The repo structure is as follows:

1) data
- customer_data.csv : includes the dataset

2) src
- const.py: a Python module defines constants used in the project
- data_preparation.py: a Python module provides functions for data preparation and preprocessing
- ml_modelling.py: a Python module provides functions for data transformation, ml training and inference
- main_app.py: a Python module includes a class that process the whole functions from data prep to ml training and inference

3) utils
- unit_tests.py: a Python module includes basic unit tests for the main functions

# Model Execution:
- python main_app.py

# Unit tests Execution:
- pytest unit_test.py

# Python Packaging:
- Poetry installed



