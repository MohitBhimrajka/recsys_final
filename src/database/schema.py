# src/database/schema.py

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Float,
    Boolean, Date, DateTime, ForeignKey, UniqueConstraint, Index, Text,
    ForeignKeyConstraint # Make sure ForeignKeyConstraint is imported
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func # For potential server-side defaults like timestamps

import sys
from pathlib import Path

# Add project root to sys.path to import config
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config # Import DATABASE_URI from config

# Define the base class for our ORM models
Base = declarative_base()

# --- User Table Definition ---
class User(Base):
    __tablename__ = 'users'
    student_id = Column(Integer, primary_key=True, comment='Unique student identifier')

    # Store the processed/mapped features
    gender_mapped = Column(Integer, comment='Mapped gender (e.g., M=0, F=1)')
    highest_education_mapped = Column(Integer, comment='Mapped highest education (ordinal)')
    imd_band_mapped = Column(Integer, nullable=True, comment='Mapped IMD band (ordinal, 0=Missing)')
    age_band_mapped = Column(Integer, comment='Mapped age band (ordinal)')
    disability_mapped = Column(Integer, comment='Mapped disability (Y=1, N=0)')
    num_prev_attempts = Column(Integer, comment='Number of previous attempts') # Use this name in schema
    studied_credits = Column(Integer, comment='Total credits studied')

    # Keep original region if needed, or map it too
    region = Column(String(255)) # Keep original region string

    # Relationships (Only to tables we actually load)
    aggregated_interactions = relationship("AggregatedInteraction", back_populates="user")

    def __repr__(self):
        return f"<User(student_id={self.student_id}, region='{self.region}')>"

# --- Course Table Definition ---
class Course(Base):
    __tablename__ = 'courses'
    module_id = Column(String(50), primary_key=True, comment='Course module code (e.g., AAA, BBB)')
    # Relationships (Only to tables we actually load)
    presentations = relationship("Presentation", back_populates="course_module")
    def __repr__(self): return f"<Course(module_id='{self.module_id}')>"

# --- Presentation Table Definition ---
class Presentation(Base):
    __tablename__ = 'presentations'
    module_id = Column(String(50), ForeignKey('courses.module_id'), primary_key=True)
    presentation_code = Column(String(50), primary_key=True, comment='Presentation identifier (e.g., 2013J, 2014B)')
    module_presentation_length = Column(Integer, comment='Duration in days')
    # Relationships (Only to tables we actually load)
    course_module = relationship("Course", back_populates="presentations")
    aggregated_interactions = relationship("AggregatedInteraction", back_populates="presentation")
    def __repr__(self): return f"<Presentation(module_id='{self.module_id}', presentation_code='{self.presentation_code}')>"

# --- Aggregated Interaction Table Definition ---
class AggregatedInteraction(Base):
    __tablename__ = 'aggregated_interactions'

    agg_interaction_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey('users.student_id'), index=True)
    module_id = Column(String(50)) # Part of FK to Presentation
    presentation_code = Column(String(50)) # Part of FK to Presentation

    # Features from aggregation
    total_clicks = Column(Integer)
    interaction_days = Column(Integer)
    first_interaction_date = Column(Integer)
    last_interaction_date = Column(Integer)
    implicit_feedback = Column(Float, index=True, comment='e.g., log1p(total_clicks)')

    # Relationships
    user = relationship("User", back_populates="aggregated_interactions")
    presentation = relationship("Presentation", back_populates="aggregated_interactions",
                                foreign_keys=[module_id, presentation_code],
                                primaryjoin="and_(AggregatedInteraction.module_id==Presentation.module_id, AggregatedInteraction.presentation_code==Presentation.presentation_code)")

    __table_args__ = (ForeignKeyConstraint(['module_id', 'presentation_code'],
                                           ['presentations.module_id', 'presentations.presentation_code']),
                      UniqueConstraint('student_id', 'module_id', 'presentation_code', name='uq_agg_interaction'),
                      Index('ix_agg_interactions_user_presentation', 'student_id', 'module_id', 'presentation_code'))

    def __repr__(self):
        return f"<AggregatedInteraction(student_id={self.student_id}, presentation='{self.module_id}_{self.presentation_code}', feedback={self.implicit_feedback:.2f})>"


# --- Engine and Metadata Setup ---
engine = None
SessionLocal = None
if config.DATABASE_URI:
    try:
        engine = create_engine(config.DATABASE_URI, echo=False) # Set echo=True for debugging SQL
        metadata = Base.metadata
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        print("Database engine created successfully.")
    except Exception as e:
        print(f"Error creating database engine: {e}")
        engine = None
        SessionLocal = None
else:
    print("DATABASE_URI not found in config. Database engine not created.")
    engine = None
    SessionLocal = None
    metadata = Base.metadata # Still define metadata even if engine fails


# --- Utility Functions ---
def create_tables(engine_to_use=engine):
    """Creates all tables defined in the metadata."""
    if not engine_to_use:
        print("Database engine not available. Cannot create tables.")
        return
    try:
        print("Attempting to create tables...")
        metadata.create_all(bind=engine_to_use)
        print("Tables created successfully (or already exist).")
    except Exception as e:
        print(f"Error creating tables: {e}")

def drop_tables(engine_to_use=engine):
    """Drops all tables defined in the metadata."""
    if not engine_to_use:
        print("Database engine not available. Cannot drop tables.")
        return
    try:
        # Add confirmation prompt here if desired, or rely on setup_database.py prompt
        print("Attempting to drop tables...")
        metadata.drop_all(bind=engine_to_use)
        print("Tables dropped successfully.")
    except Exception as e:
        print(f"Error dropping tables: {e}")

if __name__ == "__main__":
    if engine:
        print("\nRunning schema utility functions...")
        # print("Dropping tables first (if enabled)...")
        # drop_tables(engine) # Uncomment carefully to drop ALL tables first
        print("Creating tables...")
        create_tables(engine)
        print("Schema utility functions finished.")
    else:
        print("\nDatabase engine not configured. Cannot run schema utilities.")