# src/data_ingestion.py
import os
import pandas as pd
from datetime import datetime
import config

RAW = config.RAW_DIR
PROCESSED = config.PROCESSED_DIR

def load_raw():
    students = pd.read_csv(os.path.join(RAW, "studentInfo.csv"))
    courses  = pd.read_csv(os.path.join(RAW, "courses.csv"))
    svle     = pd.read_csv(os.path.join(RAW, "studentVle.csv"))
    return students, courses, svle

def initial_process(students, courses, svle):
    # Clean students
    students = students.dropna(subset=["region"])
    
    # Build course_id
    courses["course_id"] = (
        courses["module_id"].astype(str) + "_" +
        courses["presentation_id"].astype(str)
    )
    
    # Merge studentVle with course metadata
    merged = svle.merge(
        courses[["module_id", "presentation_id", "course_id"]],
        on=["module_id", "presentation_id"], how="left"
    )
    
    # Rename and select columns
    interactions = merged[[
        "id_student", "course_id", "sum_click", "date"
    ]].rename(columns={
        "id_student": "student_id",
        "sum_click": "interaction_value",
        "date": "timestamp"
    })
    
    # Convert date to datetime
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    
    return students, courses, interactions

def save_processed(students, courses, interactions):
    os.makedirs(PROCESSED, exist_ok=True)
    students.to_csv(os.path.join(PROCESSED, "users.csv"), index=False)
    courses[["course_id","module_id","presentation_id","course_title"]].to_csv(
        os.path.join(PROCESSED, "courses_processed.csv"), index=False
    )
    interactions.to_csv(os.path.join(PROCESSED, "interactions.csv"), index=False)

if __name__ == "__main__":
    students_df, courses_df, interactions_df = load_raw()
    st, crs, intr = initial_process(students_df, courses_df, interactions_df)
    save_processed(st, crs, intr)
    print("✅ Raw data ingested and initial CSVs written to data/processed/")
