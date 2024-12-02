import pytest
import pandas as pd
from unittest.mock import patch
from src.file_load import get_column_names, create_test_train_data, convert_date_to_datetime, plot_data

@pytest.fixture
def mock_data():
    data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Open': [100, 102, 105],
        'High': [110, 112, 115],
        'Low': [90, 92, 95],
        'Last Close': [95, 100, 105]
    })
    return data

def test_get_column_names(mock_data):
    assert get_column_names(mock_data) == True

    incomplete_data = mock_data.drop(columns=['Open'])
    with pytest.raises(ValueError, match=r"Missing column\(s\): Open"):
        get_column_names(incomplete_data)

def test_convert_date_to_datetime(mock_data):
    convert_date_to_datetime(mock_data)
    assert pd.api.types.is_datetime64_any_dtype(mock_data['Date']), "Date column was not converted to datetime"
    assert 'Year' in mock_data.columns, "Year column not created"
    assert 'Month' in mock_data.columns, "Month column not created"
    assert 'Day' in mock_data.columns, "Day column not created"

def test_create_test_train_data(tmp_path, mock_data):
    file_path = tmp_path / "test_data.csv"
    mock_data.to_csv(file_path, index=False)

    X_train, y_train, X_test, y_test = create_test_train_data(full_file=file_path)

    assert len(X_train) > 0, "Training set is empty"
    assert len(X_test) > 0, "Testing set is empty"
    assert len(y_train) > 0, "Training targets are empty"
    assert len(y_test) > 0, "Testing targets are empty"

def test_plot_data(mock_data):
    with patch("matplotlib.pyplot.show") as mock_show:
        plot_data(mock_data)
        mock_show.assert_called_once()
