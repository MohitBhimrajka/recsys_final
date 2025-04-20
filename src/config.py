# src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Assumes .env file is in the project root directory (RECSYS_FINAL)
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
print(f"Loading .env from: {dotenv_path}") # Debug print

# --- Database Configuration ---
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432") # Default port if not set
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    print("Warning: One or more database environment variables are not set.")
    # Optionally raise an error or set defaults if appropriate
    # raise ValueError("Missing database configuration in .env file")
    DATABASE_URI = None
else:
    DATABASE_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print(f"Database URI configured: {'Yes' if DATABASE_URI else 'No'}") # Debug print

# --- Project Structure Paths ---
# Assumes this config.py is in RECSYS_FINAL/src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure processed dir exists

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True) # Ensure saved_models dir exists

SRC_DIR = PROJECT_ROOT / "src"

# --- Data File Names (Raw) ---
# Adjust filenames if they differ in your data/raw directory
ASSESSMENTS_CSV = RAW_DATA_DIR / "assessments.csv"
COURSES_CSV = RAW_DATA_DIR / "courses.csv"
STUDENT_ASSESSMENT_CSV = RAW_DATA_DIR / "studentAssessment.csv"
STUDENT_INFO_CSV = RAW_DATA_DIR / "studentInfo.csv"
STUDENT_REGISTRATION_CSV = RAW_DATA_DIR / "studentRegistration.csv"
STUDENT_VLE_CSV = RAW_DATA_DIR / "studentVle.csv"
VLE_CSV = RAW_DATA_DIR / "vle.csv"

# --- Data File Names (Processed - Example) ---
# Define names for files generated during preprocessing
PROCESSED_INTERACTIONS = PROCESSED_DATA_DIR / "interactions_final.parquet"
PROCESSED_USERS = PROCESSED_DATA_DIR / "users_final.parquet"
PROCESSED_ITEMS = PROCESSED_DATA_DIR / "items_final.parquet" # e.g., Presentations
# Add others as needed

# --- Modeling Parameters (Example - Can be expanded later) ---
RANDOM_SEED = 42
TEST_SPLIT_DATE = '2014-09-01' # Example cutoff date for time-based split
MIN_INTERACTIONS_PER_USER = 5
MIN_USERS_PER_ITEM = 5

# --- Evaluation Parameters ---
TOP_K = 10 # For evaluation metrics like P@K, R@K, NDCG@K

# --- Helper Function to check if raw data exists ---
def check_raw_data_exists():
    """Checks if all expected raw CSV files exist."""
    required_files = [
        ASSESSMENTS_CSV, COURSES_CSV, STUDENT_ASSESSMENT_CSV,
        STUDENT_INFO_CSV, STUDENT_REGISTRATION_CSV, STUDENT_VLE_CSV, VLE_CSV
    ]
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("Error: The following raw data files are missing:")
        for f in missing_files:
            print(f"- {f}")
        return False
    print("All raw data files found.")
    return True

if __name__ == "__main__":
    # Example of how to use the config paths and check data
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Data Dir: {RAW_DATA_DIR}")
    print(f"Database URI: {DATABASE_URI}") # Be cautious printing URI if password is sensitive
    check_raw_data_exists()