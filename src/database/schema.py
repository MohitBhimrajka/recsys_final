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

# --- MODIFIED User Table Definition ---
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

    # Relationships (unchanged)
    registrations = relationship("Registration", back_populates="user")
    assessments_submitted = relationship("StudentAssessment", back_populates="user")
    interactions = relationship("Interaction", back_populates="user")
    aggregated_interactions = relationship("AggregatedInteraction", back_populates="user")

    def __repr__(self):
        return f"<User(student_id={self.student_id}, region='{self.region}')>"

# --- Other Table Definitions (Unchanged) ---

class Course(Base):
    __tablename__ = 'courses'
    module_id = Column(String(50), primary_key=True, comment='Course module code (e.g., AAA, BBB)')
    presentations = relationship("Presentation", back_populates="course_module")
    vle_items = relationship("VleItem", back_populates="course_module")
    def __repr__(self): return f"<Course(module_id='{self.module_id}')>"

class Presentation(Base):
    __tablename__ = 'presentations'
    module_id = Column(String(50), ForeignKey('courses.module_id'), primary_key=True)
    presentation_code = Column(String(50), primary_key=True, comment='Presentation identifier (e.g., 2013J, 2014B)')
    module_presentation_length = Column(Integer, comment='Duration in days')
    course_module = relationship("Course", back_populates="presentations")
    registrations = relationship("Registration", back_populates="presentation")
    vle_items = relationship("VleItem", back_populates="presentation")
    assessments = relationship("Assessment", back_populates="presentation")
    interactions = relationship("Interaction", back_populates="presentation")
    aggregated_interactions = relationship("AggregatedInteraction", back_populates="presentation")
    def __repr__(self): return f"<Presentation(module_id='{self.module_id}', presentation_code='{self.presentation_code}')>"

class VleItem(Base):
    __tablename__ = 'vle_items'
    vle_item_id = Column(Integer, primary_key=True, comment='Unique identifier for the VLE item (id_site)')
    module_id = Column(String(50), ForeignKey('courses.module_id'))
    presentation_code = Column(String(50))
    activity_type = Column(String(100))
    week_from = Column(Integer, nullable=True)
    week_to = Column(Integer, nullable=True)
    course_module = relationship("Course", back_populates="vle_items")
    presentation = relationship("Presentation", back_populates="vle_items",
                                foreign_keys=[module_id, presentation_code],
                                primaryjoin="and_(VleItem.module_id==Presentation.module_id, VleItem.presentation_code==Presentation.presentation_code)")
    interactions = relationship("Interaction", back_populates="vle_item")
    __table_args__ = (ForeignKeyConstraint(['module_id', 'presentation_code'],
                                           ['presentations.module_id', 'presentations.presentation_code']),
                      Index('ix_vle_items_presentation', 'module_id', 'presentation_code'))
    def __repr__(self): return f"<VleItem(vle_item_id={self.vle_item_id}, type='{self.activity_type}')>"

class Registration(Base):
    # Assumes we load the raw registration data separately if needed
    # This schema isn't populated by the current load_to_db.py focusing on processed data
    __tablename__ = 'registrations'
    registration_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey('users.student_id'), index=True)
    module_id = Column(String(50))
    presentation_code = Column(String(50))
    date_registration = Column(Integer, nullable=True, comment='Days relative to start, can be null')
    date_unregistration = Column(Integer, nullable=True, comment='Days relative to start, often null')
    user = relationship("User", back_populates="registrations")
    presentation = relationship("Presentation", back_populates="registrations",
                                foreign_keys=[module_id, presentation_code],
                                primaryjoin="and_(Registration.module_id==Presentation.module_id, Registration.presentation_code==Presentation.presentation_code)")
    __table_args__ = (ForeignKeyConstraint(['module_id', 'presentation_code'],
                                           ['presentations.module_id', 'presentations.presentation_code']),
                      UniqueConstraint('student_id', 'module_id', 'presentation_code', name='uq_student_registration'),
                      Index('ix_registrations_presentation', 'module_id', 'presentation_code'))
    def __repr__(self): return f"<Registration(student_id={self.student_id}, presentation='{self.module_id}_{self.presentation_code}')>"

class Interaction(Base):
    # This table remains for DETAILED interactions if loaded later
    __tablename__ = 'interactions'
    interaction_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey('users.student_id'), index=True)
    module_id = Column(String(50))
    presentation_code = Column(String(50))
    vle_item_id = Column(Integer, ForeignKey('vle_items.vle_item_id'), index=True)
    interaction_date = Column(Integer, comment='Days relative to start (from studentVle.date)')
    total_clicks = Column(Integer, comment='Clicks on this item on this day (from studentVle.sum_click)')
    user = relationship("User", back_populates="interactions")
    vle_item = relationship("VleItem", back_populates="interactions")
    presentation = relationship("Presentation", back_populates="interactions",
                                foreign_keys=[module_id, presentation_code],
                                primaryjoin="and_(Interaction.module_id==Presentation.module_id, Interaction.presentation_code==Presentation.presentation_code)")
    __table_args__ = (ForeignKeyConstraint(['module_id', 'presentation_code'],
                                           ['presentations.module_id', 'presentations.presentation_code']),
                      Index('ix_interactions_user_presentation', 'student_id', 'module_id', 'presentation_code'),
                      Index('ix_interactions_presentation', 'module_id', 'presentation_code'))
    def __repr__(self): return f"<Interaction(student_id={self.student_id}, vle_item_id={self.vle_item_id}, date={self.interaction_date}, clicks={self.total_clicks})>"

class Assessment(Base):
    # Assumes we load the raw assessment data separately if needed
    __tablename__ = 'assessments'
    assessment_id = Column(Integer, primary_key=True)
    module_id = Column(String(50))
    presentation_code = Column(String(50))
    assessment_type = Column(String(50), comment='TMA, CMA, or Exam')
    assessment_date = Column(Integer, nullable=True, comment='Final submission date relative to start (can be null)')
    weight = Column(Float)
    presentation = relationship("Presentation", back_populates="assessments",
                                foreign_keys=[module_id, presentation_code],
                                primaryjoin="and_(Assessment.module_id==Presentation.module_id, Assessment.presentation_code==Presentation.presentation_code)")
    submissions = relationship("StudentAssessment", back_populates="assessment")
    __table_args__ = (ForeignKeyConstraint(['module_id', 'presentation_code'],
                                           ['presentations.module_id', 'presentations.presentation_code']),
                      Index('ix_assessments_presentation', 'module_id', 'presentation_code'))
    def __repr__(self): return f"<Assessment(assessment_id={self.assessment_id}, type='{self.assessment_type}')>"

class StudentAssessment(Base):
    # Assumes we load the raw student assessment data separately if needed
    __tablename__ = 'student_assessments'
    student_assessment_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey('users.student_id'), index=True)
    assessment_id = Column(Integer, ForeignKey('assessments.assessment_id'), index=True)
    submission_date = Column(Integer, comment='Submission date relative to start')
    is_banked = Column(Boolean)
    score = Column(Integer, nullable=True, comment='Assessment score (can be null)')
    user = relationship("User", back_populates="assessments_submitted")
    assessment = relationship("Assessment", back_populates="submissions")
    __table_args__ = (UniqueConstraint('student_id', 'assessment_id', name='uq_student_assessment_submission'),)
    def __repr__(self): return f"<StudentAssessment(student_id={self.student_id}, assessment_id={self.assessment_id}, score={self.score})>"

class AggregatedInteraction(Base):
    __tablename__ = 'aggregated_interactions'

    agg_interaction_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey('users.student_id'), index=True)
    module_id = Column(String(50)) # Part of FK to Presentation
    presentation_code = Column(String(50)) # Part of FK to Presentation

    # Features from aggregation (unchanged)
    total_clicks = Column(Integer)
    interaction_days = Column(Integer)
    first_interaction_date = Column(Integer)
    last_interaction_date = Column(Integer)
    implicit_feedback = Column(Float, index=True, comment='e.g., log1p(total_clicks)')

    # Relationships (unchanged)
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


# --- Engine and Metadata (Unchanged) ---
engine = None
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
    metadata = Base.metadata


# --- Utility Functions (Unchanged) ---
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
        print("Attempting to drop tables...")
        metadata.drop_all(bind=engine_to_use)
        print("Tables dropped successfully.")
    except Exception as e:
        print(f"Error dropping tables: {e}")

if __name__ == "__main__":
    if engine:
        print("\nRunning schema utility functions...")
        # drop_tables(engine) # Uncomment carefully to drop ALL tables first
        create_tables(engine)
        print("Schema utility functions finished.")
    else:
        print("\nDatabase engine not configured. Cannot run schema utilities.")