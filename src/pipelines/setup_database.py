# src/pipelines/setup_database.py

import sys
from pathlib import Path
import time

# Add project root to sys.path to import necessary modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.database.db_utils import check_db_connection, create_all_tables, drop_all_tables
from src.config import DATABASE_URI # To show which DB we are connecting to

def main(drop_first=False):
    """
    Sets up the database by checking the connection and creating tables.
    Optionally drops existing tables first.
    """
    print("--- Database Setup Script ---")
    print(f"Target Database URI (from config): {'*****'.join(DATABASE_URI.split(':')[-2:]) if DATABASE_URI else 'Not Configured'}") # Obfuscate password slightly

    if not DATABASE_URI:
        print("Error: DATABASE_URI is not configured in src/config.py / .env file.")
        sys.exit(1)

    print("\nChecking database connection...")
    if not check_db_connection():
        print("Aborting setup due to connection failure.")
        sys.exit(1)

    if drop_first:
        print("\n--- Dropping existing tables (if any) ---")
        # drop_all_tables() function includes confirmation prompt
        drop_all_tables()
        print("Waiting a moment before creating tables...")
        time.sleep(2) # Give DB a moment

    print("\n--- Creating tables ---")
    create_all_tables()

    print("\n--- Database setup script finished ---")

if __name__ == "__main__":
    # --- Configuration ---
    # Set to True if you want to drop all existing tables before creating new ones.
    # BE VERY CAREFUL WITH THIS IN PRODUCTION OR WITH VALUABLE DATA.
    DROP_EXISTING_TABLES = True
    # --- End Configuration ---

    main(drop_first=DROP_EXISTING_TABLES)