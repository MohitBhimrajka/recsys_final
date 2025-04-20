# src/data/load_raw.py

import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path to import config
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config # Import paths from config

def load_assessments(file_path: Path = config.ASSESSMENTS_CSV) -> pd.DataFrame:
    """Loads the assessments.csv file."""
    print(f"Loading assessments data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded assessments data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

def load_courses(file_path: Path = config.COURSES_CSV) -> pd.DataFrame:
    """Loads the courses.csv file."""
    print(f"Loading courses data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded courses data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

def load_student_assessment(file_path: Path = config.STUDENT_ASSESSMENT_CSV) -> pd.DataFrame:
    """Loads the studentAssessment.csv file."""
    print(f"Loading student assessment data from: {file_path}")
    try:
        # Specify dtype for score to handle potential non-numeric entries gracefully if any exist, though EDA showed mostly numbers
        df = pd.read_csv(file_path, dtype={'score': 'float64'})
        print(f"Loaded student assessment data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

def load_student_info(file_path: Path = config.STUDENT_INFO_CSV) -> pd.DataFrame:
    """Loads the studentInfo.csv file."""
    print(f"Loading student info data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded student info data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

def load_student_registration(file_path: Path = config.STUDENT_REGISTRATION_CSV) -> pd.DataFrame:
    """Loads the studentRegistration.csv file."""
    print(f"Loading student registration data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded student registration data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

def load_student_vle(file_path: Path = config.STUDENT_VLE_CSV) -> pd.DataFrame:
    """Loads the studentVle.csv file."""
    print(f"Loading student VLE interaction data from: {file_path}")
    try:
        # Specify types for potentially large integer columns if memory becomes an issue later
        # dtype={'id_student': 'int32', 'id_site': 'int32', 'date': 'int16', 'sum_click': 'int16'}
        # For now, default inference is usually fine.
        df = pd.read_csv(file_path)
        print(f"Loaded student VLE data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

def load_vle(file_path: Path = config.VLE_CSV) -> pd.DataFrame:
    """Loads the vle.csv file."""
    print(f"Loading VLE metadata from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded VLE data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

def load_all_raw_data() -> dict[str, pd.DataFrame]:
    """Loads all raw OULAD CSV files into a dictionary of DataFrames."""
    print("\n--- Loading All Raw Data ---")
    if not config.check_raw_data_exists():
        sys.exit("Aborting: Not all raw data files were found.")

    dataframes = {
        "assessments": load_assessments(),
        "courses": load_courses(),
        "student_assessment": load_student_assessment(),
        "student_info": load_student_info(),
        "student_registration": load_student_registration(),
        "student_vle": load_student_vle(),
        "vle": load_vle(),
    }
    print("--- Finished Loading All Raw Data ---\n")
    return dataframes

if __name__ == "__main__":
    # Example of loading all data
    raw_data = load_all_raw_data()
    # Print shapes of loaded dataframes
    for name, df in raw_data.items():
        print(f"{name}: {df.shape}")