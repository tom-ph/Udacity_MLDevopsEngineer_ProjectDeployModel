from catboost import CatBoostClassifier
import os
import sys

sys.path.append(os.getcwd())

from starter.ml.model import train_model, compute_model_metrics, evaluate_slice_metrics, inference


def test_train_model(fix_processed_data, fix_grid_params, fix_cat_features, fix_iterations):
    X, y = fix_processed_data[0], fix_processed_data[1]
    model, _ = train_model(X, y, fix_grid_params, fix_cat_features, fix_iterations)
    assert isinstance(model, CatBoostClassifier), "the model is not a CatBoostClassifier"

def test_compute_model_metrics(fix_processed_data):
    fake_y = fix_processed_data[1]
    fake_preds = fake_y
    precision, recall, fbeta = compute_model_metrics(fake_y, fake_preds)
    assert precision==1 and recall==1 and fbeta==1, "testing y with itself needs to return perfect results"

def test_inference(fix_processed_data, fix_model):
    X = fix_processed_data[0]
    preds = inference(fix_model, X)
    assert not preds is None, "predictions are empty"
    assert X.shape[0]==preds.shape[0], "predictions are not the same length as the data"