import pytest
from fatigueset.features import FeatureExtractor

def test_extract_features():
    # Arrange
    sample_data = {
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    }
    extractor = FeatureExtractor()

    # Act
    features = extractor.extract_features(sample_data)

    # Assert
    assert 'feature1_mean' in features
    assert 'feature2_mean' in features
    assert features['feature1_mean'] == 2.0
    assert features['feature2_mean'] == 5.0

def test_compute_statistics():
    # Arrange
    sample_data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 6, 7, 8, 9]
    }
    extractor = FeatureExtractor()

    # Act
    statistics = extractor.compute_statistics(sample_data)

    # Assert
    assert statistics['feature1']['mean'] == 3.0
    assert statistics['feature2']['std'] == pytest.approx(1.414, rel=1e-2)  # Allow for slight floating point error

def test_invalid_data():
    # Arrange
    invalid_data = {
        'feature1': [1, 2, None],
        'feature2': [5, 6, 7]
    }
    extractor = FeatureExtractor()

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid data: contains None values"):
        extractor.extract_features(invalid_data)