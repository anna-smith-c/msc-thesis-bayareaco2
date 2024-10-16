import pytest
import pandas as pd
import numpy as np
import xgboost as xgb
from bayareaco2.models.xgboost import XGBoost_Model

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
    })
    y = pd.Series(np.random.rand(100))
    return X, y

def test_initialization():
    model = XGBoost_Model(n_estimators=100, max_depth=3)
    assert isinstance(model.model, xgb.XGBRegressor)
    assert model.model.get_params()['n_estimators'] == 100
    assert model.model.get_params()['max_depth'] == 3

def test_fit(sample_data):
    X, y = sample_data
    model = XGBoost_Model()
    model.fit(X, y)
    assert model.model is not None

def test_predict(sample_data):
    X, y = sample_data
    model = XGBoost_Model()
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert isinstance(predictions, np.ndarray)

def test_calculate_metrics(sample_data):
    X, y = sample_data
    model = XGBoost_Model()
    model.fit(X, y)
    predictions = model.predict(X)
    metrics = model.calculate_metrics(y, predictions)
    assert 'R²' in metrics
    assert 'MSE' in metrics
    assert 'RMSE' in metrics
    assert 'MAE' in metrics
    assert isinstance(metrics, dict)
