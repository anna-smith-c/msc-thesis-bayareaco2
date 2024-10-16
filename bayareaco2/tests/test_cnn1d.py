import pytest
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from bayareaco2.models.cnn1d import CNN_1D, reshape_1D_X


@pytest.fixture
def setup_data():
    np.random.seed(0)
    X_train = np.random.rand(100, 10, 1)
    y_train = np.random.rand(100, 1)
    X_test = np.random.rand(20, 10, 1)
    y_test = np.random.rand(20, 1)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def cnn_model():
    input_shape = (10, 1)
    model = CNN_1D(input_shape, verbose=False)
    model.compile(learning_rate=0.001)
    return model


def test_reshape_1D_X():
    df = pd.DataFrame(np.random.rand(100, 5))
    reshaped = reshape_1D_X(df)
    assert reshaped.shape == (100, 5, 1)


def test_model_initialization(cnn_model):
    assert isinstance(cnn_model.model, Sequential)
    assert cnn_model.model.input_shape[1:] == (10, 1)


def test_predict(cnn_model, setup_data):
    _, _, X_test, _ = setup_data
    y_pred = cnn_model.predict(X_test)
    assert y_pred.shape == (20, 1)
