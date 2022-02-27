from catboost import CatBoostClassifier
import pandas as pd
import pytest
import yaml
from ml.data import process_data

with open('starter/starter/tests/tests_config.yaml') as stream:
    config = yaml.safe_load(stream)
sample_data_path = config["model_training"]["training_data_path"]
sample_model_path = config["model_training"]["trained_model_path"]
train_iterations = config["model_training"]["training_iterations"]
grid_search_params = config["model_training"]["model_grid_search_hparams"]
cat_features = config["data"]["categorical_features"]
label = config["data"]["label"]

@pytest.fixture()
def fix_X():
    X = pd.read_csv(sample_data_path, dtype={col: "category" for col in cat_features})
    return X

@pytest.fixture
def fix_cat_features():
    return cat_features

@pytest.fixture()
def fix_label():
    return label

@pytest.fixture()
def fix_iterations():
    return train_iterations

@pytest.fixture()
def fix_grid_params():
    return grid_search_params

@pytest.fixture
def fix_processed_data(fix_X, fix_cat_features, fix_label):
    X_train, y_train, _ = process_data(
    fix_X, categorical_features=fix_cat_features, label=fix_label, training=True
    )
    return [X_train, y_train]

@pytest.fixture
def fix_model(fix_cat_features):
    model = CatBoostClassifier()
    model.load_model(sample_model_path)
    return model