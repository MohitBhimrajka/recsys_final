# src/pipelines/run_preprocessing.py

import pandas as pd
import sys
from pathlib import Path
import time

# Add project root to sys.path to import necessary modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data import preprocess
from src import config

def main():
    """
    Runs the full data preprocessing pipeline and saves the output DataFrames.
    """
    start_time = time.time()
    print("--- Starting Data Preprocessing Pipeline ---")

    try:
        processed_data = preprocess.preprocess_all_data()

        # --- Save Processed Data ---
        print("\n--- Saving Processed Data ---")

        # Define output paths from config
        users_path = config.PROCESSED_DATA_DIR / "users_final.parquet"
        items_path = config.PROCESSED_DATA_DIR / "items_final.parquet"
        interactions_path = config.PROCESSED_DATA_DIR / "interactions_final.parquet"

        # Save users features
        print(f"Saving users data to: {users_path}")
        processed_data['users'].reset_index().to_parquet(users_path, index=False) # Reset index before saving

        # Save items features
        print(f"Saving items data to: {items_path}")
        processed_data['items'].reset_index().to_parquet(items_path, index=False) # Reset index before saving

        # Save interactions data
        print(f"Saving interactions data to: {interactions_path}")
        processed_data['interactions'].to_parquet(interactions_path, index=False)

        print("--- Processed data saved successfully ---")

    except Exception as e:
        print(f"\n--- An error occurred during preprocessing ---")
        print(e)
        # Consider adding more detailed error logging here
        sys.exit(1)

    end_time = time.time()
    print(f"\n--- Preprocessing Pipeline Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()