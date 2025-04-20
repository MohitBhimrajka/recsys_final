# src/db_setup.py
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    MetaData,
    Table,
    create_engine
)
import config

engine = create_engine(config.SQLALCHEMY_DATABASE_URI)
metadata = MetaData()

# Define tables
users = Table(
    "users", metadata,
    Column("student_id", Integer, primary_key=True),
    Column("gender", String),
    Column("region", String),
    Column("highest_education", String),
    Column("age_band", String)
    # add more columns as needed from studentInfo.csv
)

courses = Table(
    "courses", metadata,
    Column("course_id", String, primary_key=True),
    Column("module_id", String),
    Column("presentation_id", String),
    Column("course_title", String)
    # add other metadata columns if desired
)

interactions = Table(
    "interactions", metadata,
    Column("interaction_id", Integer, primary_key=True, autoincrement=True),
    Column("student_id", Integer),
    Column("course_id", String),
    Column("interaction_value", Float),
    Column("timestamp", DateTime)
)

if __name__ == "__main__":
    metadata.drop_all(engine)   # WARNING: drops existing tables
    metadata.create_all(engine)
    print("✅ Tables created successfully.")
