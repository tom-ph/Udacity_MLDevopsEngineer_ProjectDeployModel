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


# def process_data(
#     X, categorical_features=[], label=None, training=True, one_hot_encode=True, encoder=None, lb=None
# ):
#     """ Process the data used in the machine learning pipeline.

#     Processes the data using one hot encoding for the categorical features and a
#     label binarizer for the labels. This can be used in either training or
#     inference/validation.

#     Note: depending on the type of model used, you may want to add in functionality that
#     scales the continuous data.

#     Inputs
#     ------
#     X : pd.DataFrame
#         Dataframe containing the features and label. Columns in `categorical_features`
#     categorical_features: list[str]
#         List containing the names of the categorical features (default=[])
#     label : str
#         Name of the label column in `X`. If None, then an empty array will be returned
#         for y (default=None)
#     training : bool
#         Indicator if training mode or inference/validation mode.
#     one_hot_encode : bool
#         Whether to do one hot encoding over categorical features or leave them as-is
#     encoder : sklearn.preprocessing._encoders.OneHotEncoder
#         Trained sklearn OneHotEncoder, only used if training=False.
#     lb : sklearn.preprocessing._label.LabelBinarizer
#         Trained sklearn LabelBinarizer, only used if training=False.

#     Returns
#     -------
#     X : np.array
#         Processed data.
#     y : np.array
#         Processed labels if labeled=True, otherwise empty np.array.
#     encoder : sklearn.preprocessing._encoders.OneHotEncoder
#         Trained OneHotEncoder if training is True, otherwise returns the encoder passed
#         in.
#     lb : sklearn.preprocessing._label.LabelBinarizer
#         Trained LabelBinarizer if training is True, otherwise returns the binarizer
#         passed in.
#     """

#     if label is not None:
#         y = X[label]
#         X = X.drop([label], axis=1)
#     else:
#         y = np.array([])

#     X_categorical = X[categorical_features].values
#     X_continuous = X.drop(*[categorical_features], axis=1)

#     if training is True:
#         if one_hot_encode:
#             encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
#             X_categorical = encoder.fit_transform(X_categorical)
#         lb = LabelBinarizer()
#         y = lb.fit_transform(y.values).ravel()
#     else:
#         if one_hot_encode:
#             X_categorical = encoder.transform(X_categorical)
#         else:
#             X
#         try:
#             y = lb.transform(y.values).ravel()
#         # Catch the case where y is None because we're doing inference.
#         except AttributeError:
#             pass

#     X = np.concatenate([X_continuous, X_categorical], axis=1)
#     return X, y, encoder, lb
