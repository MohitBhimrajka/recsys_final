# tests/test_utils.py

import pandas as pd
import pytest
import numpy as np

# Make sure pytest can find the src modules (helped by pytest.ini)
from src.data import utils

# --- Tests for create_presentation_id ---

def test_create_presentation_id_success():
    """Tests successful creation of presentation_id."""
    data = {'code_module': ['AAA', 'BBB', 'AAA'],
            'code_presentation': ['2013J', '2014B', '2014J'],
            'other_col': [1, 2, 3]}
    df = pd.DataFrame(data)
    df_result = utils.create_presentation_id(df.copy()) # Use copy to avoid modifying original
    assert 'presentation_id' in df_result.columns
    pd.testing.assert_series_equal(
        df_result['presentation_id'],
        pd.Series(['AAA_2013J', 'BBB_2014B', 'AAA_2014J'], name='presentation_id'),
        check_dtype=True
    )
    # Check other columns remain
    assert 'other_col' in df_result.columns

def test_create_presentation_id_missing_cols():
    """Tests behavior when input columns are missing."""
    data = {'code_module': ['AAA', 'BBB'], 'other_col': [1, 2]}
    df = pd.DataFrame(data)
    df_result = utils.create_presentation_id(df.copy())
    assert 'presentation_id' not in df_result.columns # Should not create the column

# --- Tests for Mapping Functions ---

@pytest.mark.parametrize("input_str, expected", [
    ('0-10%', 1), ('10-20%', 2), ('20-30%', 3), ('30-40%', 4), ('40-50%', 5),
    ('50-60%', 6), ('60-70%', 7), ('70-80%', 8), ('80-90%', 9), ('90-100%', 10),
    ('Missing', 0), (None, 0), ('Unknown', 0), ('InvalidFormat', 0), (np.nan, 0)
])
def test_map_imd_band(input_str, expected):
    assert utils.map_imd_band(input_str) == expected

@pytest.mark.parametrize("input_str, expected", [
    ('0-35', 0), ('35-55', 1), ('55<=', 2), (None, -1), ('Other', -1)
])
def test_map_age_band(input_str, expected):
    assert utils.map_age_band(input_str) == expected

@pytest.mark.parametrize("input_str, expected", [
    ('No Formal quals', 0), ('Lower Than A Level', 1), ('A Level or Equivalent', 2),
    ('HE Qualification', 3), ('Post Graduate Qualification', 4), (None, -1), ('Unknown', -1)
])
def test_map_highest_education(input_str, expected):
    assert utils.map_highest_education(input_str) == expected

@pytest.mark.parametrize("input_str, expected", [
    ('M', 0), ('F', 1), (None, -1), ('Other', -1)
])
def test_map_gender(input_str, expected):
    assert utils.map_gender(input_str) == expected

@pytest.mark.parametrize("input_str, expected", [
    ('Y', 1), ('N', 0), (None, -1), ('Maybe', -1)
])
def test_map_disability(input_str, expected):
    assert utils.map_disability(input_str) == expected