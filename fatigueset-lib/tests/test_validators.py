from fatigueset.validators import DataValidator
import pandas as pd
import pytest

def test_validate_schema():
    validator = DataValidator()
    valid_data = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': ['a', 'b', 'c']
    })
    invalid_data = pd.DataFrame({
        'column1': [1, 2, 3],
        'column3': ['a', 'b', 'c']
    })
    
    assert validator.validate_schema(valid_data, ['column1', 'column2']) == True
    assert validator.validate_schema(invalid_data, ['column1', 'column2']) == False

def test_check_missing_values():
    validator = DataValidator()
    data_with_nans = pd.DataFrame({
        'column1': [1, 2, None],
        'column2': ['a', 'b', 'c']
    })
    data_without_nans = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': ['a', 'b', 'c']
    })
    
    assert validator.check_missing_values(data_with_nans) == True
    assert validator.check_missing_values(data_without_nans) == False

def test_validate_data_quality():
    validator = DataValidator()
    data = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': ['a', 'b', 'c']
    })
    
    assert validator.validate_data_quality(data) == True  # Assuming this method checks for basic quality criteria

    bad_data = pd.DataFrame({
        'column1': [1, 2, 'bad_data'],
        'column2': ['a', 'b', 'c']
    })
    
    assert validator.validate_data_quality(bad_data) == False  # Assuming this method checks for data type integrity