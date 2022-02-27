# Script to train machine learning model.
import json
from pkg_resources import evaluate_marker
import os
import yaml

from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import evaluate_slice_metrics, inference, train_model

with open('starter/starter/config.yaml') as stream:
    config = yaml.safe_load(stream)
data_path = config["model_training"]["training_data_path"]
model_path = config["model_training"]["trained_model_path"]
metrics_path = config["model_training"]["metrics_folder_path"]
train_iterations = config["model_training"]["training_iterations"]
grid_search_params = config["model_training"]["model_grid_search_hparams"]
cat_features = config["data"]["categorical_features"]
label = config["data"]["label"]

# Add code to load in the data.
data = pd.read_csv(data_path, dtype={col: "category" for col in cat_features})

# Optional enhancement, use K-fold cross validation instead of a train-test split.
# train, test = train_test_split(data, test_size=0.20)

# No need to split data, the model will train with k-fold cross validation
train = data

X_train, y_train, _ = process_data(
    train, categorical_features=cat_features, label=label, training=True
)

# Train and save a model.
model, _ = train_model(X_train, y_train, grid_search_params, cat_features=cat_features, iterations=train_iterations)
model.save_model(model_path)

# Save slices metrics
metrics_dict = {}
for col in cat_features:
    metrics_dict.update(evaluate_slice_metrics(model, X_train, y_train, col))

metrics_txt_path = os.path.join(metrics_path, 'slice_output.txt')
metrics_json_path = os.path.join(metrics_path, 'slice_output.json')
with open(metrics_txt_path, 'w') as f:
    json.dump(metrics_dict, f, indent=4)
with open(metrics_json_path, 'w') as f:
    json.dump(metrics_dict, f, indent=4)