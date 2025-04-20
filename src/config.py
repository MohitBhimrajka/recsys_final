import os

# Paths
RAW_DIR       = os.path.join(os.getcwd(), "data", "raw")
PROCESSED_DIR = os.path.join(os.getcwd(), "data", "processed")

# DB
DB_USER = "mohit"
DB_PASS = "Apple#01109004$$"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "recsys"

SQLALCHEMY_DATABASE_URI = (
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
