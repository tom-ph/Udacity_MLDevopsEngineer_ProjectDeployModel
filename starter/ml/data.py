import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=[], label=None, training=True, lb=None
):
    """
    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    lb : sklearn.preprocessing._label.LabelBinarizer
#         Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : pd.DataFrame
        Processed data.
    y : pd.Series
        Processed labels if labeled=True, otherwise empty pd.Series.
    lb : sklearn.preprocessing._label.LabelBinarizer
#         Trained LabelBinarizer if training is True, otherwise returns the binarizer
#         passed in."""
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features]
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        lb = LabelBinarizer()
        y = lb.fit_transform(y.values).ravel()
    else:
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass
    y = pd.Series(y)

    X = pd.concat([X_continuous, X_categorical], axis=1)
    return X, y, lb