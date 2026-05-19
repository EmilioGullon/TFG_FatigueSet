import pytest
from fatigueset.processor import DataProcessor

def test_process_data():
    # Arrange
    processor = DataProcessor()
    sample_data = {
        'column1': [1, 2, 3],
        'column2': [4, 5, 6]
    }
    
    # Act
    processed_data = processor.process_data(sample_data)
    
    # Assert
    assert processed_data is not None
    assert 'processed_column' in processed_data.columns

def test_filter_data():
    # Arrange
    processor = DataProcessor()
    sample_data = {
        'column1': [1, 2, 3],
        'column2': [4, 5, 6]
    }
    
    # Act
    filtered_data = processor.filter_data(sample_data, threshold=2)
    
    # Assert
    assert len(filtered_data) == 2  # Assuming filtering removes one row

def test_aggregate_data():
    # Arrange
    processor = DataProcessor()
    sample_data = {
        'column1': [1, 2, 3, 4],
        'column2': [4, 5, 6, 7]
    }
    
    # Act
    aggregated_data = processor.aggregate_data(sample_data)
    
    # Assert
    assert aggregated_data['column1'].sum() == 10  # Check sum of column1
    assert aggregated_data['column2'].mean() == 5.5  # Check mean of column2