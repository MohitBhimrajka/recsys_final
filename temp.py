import pandas as pd
from src import config # Assuming you run this from project root or have src in path

users_df = pd.read_parquet(config.PROCESSED_USERS)
print(users_df['id_student'].unique()) # See all known user IDs
test_ids = [4, 10, 13, 19, 20323, 22323, 6516, 8462, 29639] # Add some known good ones
for uid in test_ids:
    is_present = uid in users_df['id_student'].values
    print(f"User {uid} present in processed users file: {is_present}")