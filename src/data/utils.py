# src/data/utils.py

import pandas as pd
import numpy as np

def create_presentation_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a composite 'presentation_id' column from 'code_module' and 'code_presentation'.

    Args:
        df (pd.DataFrame): DataFrame containing 'code_module' and 'code_presentation' columns.

    Returns:
        pd.DataFrame: DataFrame with the added 'presentation_id' column.
    """
    if 'code_module' in df.columns and 'code_presentation' in df.columns:
        print("Creating 'presentation_id' column.")
        df['presentation_id'] = df['code_module'] + '_' + df['code_presentation']
    else:
        print("Warning: 'code_module' or 'code_presentation' not found. Cannot create 'presentation_id'.")
    return df

def map_imd_band(imd_band_str: str) -> int:
    """Maps IMD band string (e.g., '0-10%') to an integer representation (e.g., 1). Handles 'Missing'."""
    if pd.isnull(imd_band_str) or imd_band_str == 'Missing':
        return 0 # Assign 0 to missing/unknown
    # Assumes format like 'X-Y%' or '90-100%'
    try:
        lower_bound = int(imd_band_str.split('-')[0])
        # Map to bands 1-10 based on lower bound
        return (lower_bound // 10) + 1
    except:
        return 0 # Handle unexpected formats

def map_age_band(age_band_str: str) -> int:
    """Maps age band string (e.g., '0-35') to an integer (0, 1, 2)."""
    if age_band_str == '0-35':
        return 0
    elif age_band_str == '35-55':
        return 1
    elif age_band_str == '55<=':
        return 2
    else:
        return -1 # Should not happen based on EDA

def map_highest_education(edu_str: str) -> int:
    """Maps highest education string to an ordered integer."""
    # Ordered mapping based on assumed level
    edu_map = {
        'No Formal quals': 0,
        'Lower Than A Level': 1,
        'A Level or Equivalent': 2,
        'HE Qualification': 3,
        'Post Graduate Qualification': 4
    }
    return edu_map.get(edu_str, -1) # Return -1 for unknown

def map_gender(gender_str: str) -> int:
    """Maps gender M/F to 0/1."""
    if gender_str == 'M':
        return 0
    elif gender_str == 'F':
        return 1
    else:
        return -1

def map_disability(disability_str: str) -> int:
    """Maps disability Y/N to 1/0."""
    if disability_str == 'Y':
        return 1
    elif disability_str == 'N':
        return 0
    else:
        return -1

# Add other utility functions as needed during development