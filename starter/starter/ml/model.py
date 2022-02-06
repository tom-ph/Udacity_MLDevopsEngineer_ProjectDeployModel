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


def evaluate_slice_metrics(model, X, y, slice_column):
    pass


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
