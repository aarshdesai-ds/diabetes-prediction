"""Data loading and schema validation tests."""
from src import config, data


def test_load_raw_has_expected_schema():
    df = data.load_raw()
    for col in config.RAW_COLUMNS:
        assert col in df.columns
    assert len(df) == 768  # Pima dataset size


def test_target_is_binary():
    df = data.load_raw()
    assert set(df[config.TARGET].unique()) <= {0, 1}


def test_split_X_y_shapes():
    df = data.load_raw()
    X, y = data.split_X_y(df)
    assert list(X.columns) == config.FEATURES
    assert len(X) == len(y) == len(df)
