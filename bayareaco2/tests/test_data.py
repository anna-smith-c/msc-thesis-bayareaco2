import pandas as pd
from sklearn.preprocessing import StandardScaler

from bayareaco2.models.data import load_Xy
from bayareaco2.models.data import fit_scaler
from bayareaco2.models.data import scale_features


def test_load_Xy_balanced():
    X, y = load_Xy(balanced=True, return_node_id=False)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[1] == 105

def test_load_Xy_unbalanced():
    X, y = load_Xy(balanced=False, return_node_id=False)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[1] == 106

def test_fit_scaler():
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    scaler = fit_scaler(X_train)
    assert isinstance(scaler, StandardScaler)
    assert scaler.mean_.tolist() == [2, 5]

def test_scale_features():
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    scaler = fit_scaler(X_train)
    X_scaled = scale_features(scaler, X_train)
    assert X_scaled.mean().round().tolist() == [0, 0]
    assert X_scaled.std().round().tolist() == [1, 1]