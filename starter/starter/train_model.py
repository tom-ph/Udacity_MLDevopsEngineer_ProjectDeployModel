# Script to train machine learning model.
import json
from pkg_resources import evaluate_marker
import os
import pickle
import yaml

from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import compute_model_metrics, evaluate_slice_metrics, train_model, inference

with open('starter/starter/config.yaml') as stream:
    config = yaml.safe_load(stream)
data_path = config["model_training"]["training_data_path"]
model_path = config["model_training"]["trained_model_path"]
label_encoder_path = config["model_training"]["label_encoder"]
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

X_train, y_train, lb = process_data(
    train, categorical_features=cat_features, label=label, training=True
)

# save the label encoder to use it during inference
with open(label_encoder_path, 'wb') as output:
    pickle.dump(lb, output)

print(y_train.value_counts())

# Train and save a model.
model, best_params = train_model(X_train, y_train, grid_search_params, cat_features=cat_features, iterations=train_iterations)
model.save_model(model_path)

# Evaluate the metrics over the entire dataset since cross-validation was performed
preds = inference(model, X_train)
precision, recall, fbeta = compute_model_metrics(y_train, preds)
tot_metrics_dict = {"precision": precision, "recall": recall, "fbeta": fbeta}
# Save slices metrics
slice_metrics_dict = {}
for col in cat_features:
    slice_metrics_dict.update(evaluate_slice_metrics(model, X_train, y_train, col))

tot_metrics_path = os.path.join(metrics_path, 'metrics.txt')
slice_metrics_path = os.path.join(metrics_path, 'slice_output.txt')
best_params_path = os.path.join(metrics_path, 'best_params.json')
with open(tot_metrics_path, 'w') as f:
    json.dump(tot_metrics_dict, f, indent=4)
with open(slice_metrics_path, 'w') as f:
    json.dump(slice_metrics_dict, f, indent=4)
with open(best_params_path, 'w') as f:
    json.dump(best_params, f, indent=4)