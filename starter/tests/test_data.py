import pytest

def test_process_data(fix_processed_data, fix_label):
    X_train, y_train = fix_processed_data[0], fix_processed_data[1]
    assert not X_train is None, "X_train is None"
    assert not y_train is None, "y_train is None"
    assert X_train.shape[0]==y_train.shape[0], "X and y don't have the same length"
    assert fix_label not in X_train.columns, "target cannot be used as a feature"