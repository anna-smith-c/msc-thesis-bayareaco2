import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from bayareaco2.preprocessing.features import create_buffers
from bayareaco2.preprocessing.features import preprocess_gdf

@pytest.fixture
def sample_gdf():
    """Fixture to create a sample GeoDataFrame."""
    data = {
        'node_id': [1, 2],
        'lat': [37.6, 37.7],
        'lng': [-122.4, -122.5],
        'label': ['training node', 'central test node'],
        'geometry': [Point(-122.4, 37.6), Point(-122.5, 37.7)]
    }
    return gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:4326")


@pytest.fixture
def sample_buffers_gdf(sample_gdf):
    """Fixture to create sample buffers around nodes."""
    buffer_sizes = [50, 100]
    return create_buffers(sample_gdf, buffer_sizes)


def test_create_buffers(sample_gdf):
    """Test create_buffers function."""
    buffer_gdf = create_buffers(sample_gdf, [50, 100])
    assert 'buffer_50m' in buffer_gdf.columns
    assert 'buffer_100m' in buffer_gdf.columns
    assert len(buffer_gdf) == len(sample_gdf)


def test_preprocess_gdf(sample_gdf):
    """Test preprocess_gdf function."""
    preprocessed_gdf = preprocess_gdf(sample_gdf)
    assert preprocessed_gdf.crs == "EPSG:4326"
    assert len(preprocessed_gdf) == len(sample_gdf)