from catboost import CatBoostClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, grid_params, cat_features=None, iterations=None):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    grid_params : dict
        The grid search parameters
    cat_features : list
        The list of categorical features
    iterations : int
        Number of iterations during training
    Returns
    -------
    model : catboost.CatBoostClassifier
        Trained machine learning model.
    grid_search_result : dict
        A dict with the best parameters and the test results
    """
    model = CatBoostClassifier(cat_features=cat_features, iterations=iterations)

    grid_search_result = model.grid_search(grid_params,
                                        X=X_train,
                                        y=y_train,
                                        stratified=True)

    return model, grid_search_result


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : catboost.CatBoostClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def evaluate_slice_metrics(model, X, y, slice_column, round_digits=3):
    slice_vals_metrics = {}
    slice_values = X[slice_column].unique()

    for val in slice_values:
        slice_indexes = X[slice_column]==val
        X_slice = X[slice_indexes]
        y_slice = y[slice_indexes]
        preds_slice = model.predict(X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
        if not round_digits is None:
            precision, recall, fbeta = round(precision, round_digits), round(recall, round_digits), round(fbeta, round_digits)
        slice_vals_metrics.update({val: {"precision": precision, "recall": recall, "fbeta": fbeta}})
    
    slice_metrics = {slice_column: slice_vals_metrics}
    return slice_metrics