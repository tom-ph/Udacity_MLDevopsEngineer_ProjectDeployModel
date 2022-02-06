# Script to train machine learning model.
import yaml

import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import inference, train_model

import os
print(os.getcwd())
with open('starter/starter/config.yaml') as stream:
    config = yaml.safe_load(stream)
data_path = config["model_training"]["training_data_path"]
grid_search_params = config["model_training"]["model_grid_search_hparams"]

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Add code to load in the data.
data = pd.read_csv(data_path, dtype={col: "category" for col in cat_features})

# Optional enhancement, use K-fold cross validation instead of a train-test split.
# train, test = train_test_split(data, test_size=0.20)

# No need to split data, the model will train with k-fold cross validation
train = data

X_train, y_train, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
model, _ = train_model(X_train, y_train, grid_search_params, cat_features=cat_features, iterations=10)

preds = inference(model, X_train)

print(preds)