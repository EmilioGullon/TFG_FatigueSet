import pytest
from fatigueset.loader import DataLoader

def test_load_csv_valid():
    loader = DataLoader()
    df = loader.load_csv('data/sample/metadata.csv')
    assert df is not None
    assert not df.empty
    assert 'participant_id' in df.columns

def test_load_csv_invalid():
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_csv('data/sample/non_existent_file.csv')

def test_load_all():
    loader = DataLoader()
    dfs = loader.load_all('data/sample/')
    assert len(dfs) > 0
    for df in dfs:
        assert df is not None
        assert not df.empty

def test_load_csv_with_invalid_format():
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.load_csv('data/sample/invalid_file.txt')  # Assuming this is not a valid CSV file