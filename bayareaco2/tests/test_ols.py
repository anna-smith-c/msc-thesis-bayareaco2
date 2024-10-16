import pytest
import pandas as pd
import numpy as np
from bayareaco2.models.ols import OLS_Model

@pytest.fixture
def sample_data():
    # Generate sample data for testing
    X = pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50),
        'feature3': np.random.randn(50)
    })
    y = pd.Series(np.random.randn(50))
    
    X_train, X_test = X.iloc[:40], X.iloc[40:]
    y_train, y_test = y.iloc[:40], y.iloc[40:]
    
    return X_train, y_train, X_test, y_test

def test_fit(sample_data):
    ols_model = OLS_Model()
    assert ols_model.model is None
    X_train, y_train, _, _ = sample_data
    ols_model.fit(X_train, y_train)
    assert ols_model.model is not None

def test_predict(sample_data):
    ols_model = OLS_Model()
    X_train, y_train, X_test, _ = sample_data
    ols_model.fit(X_train, y_train)
    predictions = ols_model.predict(X_test)
    assert len(predictions) == len(X_test)