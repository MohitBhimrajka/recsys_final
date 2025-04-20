# src/database/schema.py

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Float,
    Boolean, Date, DateTime, ForeignKey, UniqueConstraint, Index, Text,
    ForeignKeyConstraint
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func # For potential server-side defaults like timestamps

import sys
from pathlib import Path

# Add project root to sys.path to import config
# This assumes the file is in RECSYS_FINAL/src/database/
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src import config # Import DATABASE_URI from config

# Define the base class for our ORM models
Base = declarative_base()

# --- Table Definitions ---

class User(Base):
    __tablename__ = 'users'

    student_id = Column(Integer, primary_key=True, comment='Unique student identifier')
    gender = Column(String(10))
    region = Column(String(255))
    highest_education = Column(String(255))
    imd_band = Column(String(50), nullable=True, comment='Index of Multiple Deprivation band, can be null')
    age_band = Column(String(50))
    num_prev_attempts = Column(Integer)
    studied_credits = Column(Integer)
    disability = Column(String(10))
    # Add processed timestamp if needed: processed_at = Column(DateTime, default=func.now())

    # Relationships (optional but useful for ORM queries)
    registrations = relationship("Registration", back_populates="user")
    assessments_submitted = relationship("StudentAssessment", back_populates="user")
    interactions = relationship("Interaction", back_populates="user")

    def __repr__(self):
        return f"<User(student_id={self.student_id}, region='{self.region}')>"

class Course(Base):
    __tablename__ = 'courses'

    module_id = Column(String(50), primary_key=True, comment='Course module code (e.g., AAA, BBB)')
    # Assuming module_title is not directly available in OULAD CSVs
    # module_title = Column(String(255), nullable=True)

    # Relationships
    presentations = relationship("Presentation", back_populates="course_module")
    vle_items = relationship("VleItem", back_populates="course_module")

    def __repr__(self):
        return f"<Course(module_id='{self.module_id}')>"

class Presentation(Base):
    __tablename__ = 'presentations'

    # Using separate columns and a composite primary key constraint
    module_id = Column(String(50), ForeignKey('courses.module_id'), primary_key=True)
    presentation_code = Column(String(50), primary_key=True, comment='Presentation identifier (e.g., 2013J, 2014B)')
    # We can derive presentation_id for easier use in application logic if needed,
    # but the composite key is cleaner for DB integrity.
    # presentation_id = Column(String(100), unique=True, index=True, comment='Composite key: module_id + presentation_code')

    module_presentation_length = Column(Integer, comment='Duration in days')

    # Relationships
    course_module = relationship("Course", back_populates="presentations")
    registrations = relationship("Registration", back_populates="presentation")
    vle_items = relationship("VleItem", back_populates="presentation")
    assessments = relationship("Assessment", back_populates="presentation")
    interactions = relationship("Interaction", back_populates="presentation") # Link interactions to context

    # Ensure combination is unique (covered by composite PK)
    # __table_args__ = (UniqueConstraint('module_id', 'presentation_code', name='uq_presentation'),)

    def __repr__(self):
        return f"<Presentation(module_id='{self.module_id}', presentation_code='{self.presentation_code}')>"

class VleItem(Base):
    __tablename__ = 'vle_items'

    vle_item_id = Column(Integer, primary_key=True, comment='Unique identifier for the VLE item (id_site)')
    module_id = Column(String(50), ForeignKey('courses.module_id'))
    presentation_code = Column(String(50)) # Part of the link to Presentation

    activity_type = Column(String(100))
    week_from = Column(Integer, nullable=True)
    week_to = Column(Integer, nullable=True)

    # Relationships
    course_module = relationship("Course", back_populates="vle_items")
    presentation = relationship("Presentation", back_populates="vle_items",
                                foreign_keys=[module_id, presentation_code], # Specify composite foreign key
                                primaryjoin="and_(VleItem.module_id==Presentation.module_id, VleItem.presentation_code==Presentation.presentation_code)")
    interactions = relationship("Interaction", back_populates="vle_item")

    # Foreign key constraint to presentations table (composite)
    __table_args__ = (ForeignKeyConstraint(['module_id', 'presentation_code'],
                                           ['presentations.module_id', 'presentations.presentation_code']),
                      Index('ix_vle_items_presentation', 'module_id', 'presentation_code')) # Index for joins


    def __repr__(self):
        return f"<VleItem(vle_item_id={self.vle_item_id}, type='{self.activity_type}')>"

class Registration(Base):
    __tablename__ = 'registrations'

    registration_id = Column(Integer, primary_key=True, autoincrement=True) # Simple auto PK
    student_id = Column(Integer, ForeignKey('users.student_id'), index=True)
    module_id = Column(String(50)) # Part of FK to Presentation
    presentation_code = Column(String(50)) # Part of FK to Presentation

    date_registration = Column(Integer, nullable=True, comment='Days relative to start, can be null')
    date_unregistration = Column(Integer, nullable=True, comment='Days relative to start, often null')

    # Relationships
    user = relationship("User", back_populates="registrations")
    presentation = relationship("Presentation", back_populates="registrations",
                                foreign_keys=[module_id, presentation_code],
                                primaryjoin="and_(Registration.module_id==Presentation.module_id, Registration.presentation_code==Presentation.presentation_code)")

    # Composite foreign key constraint
    __table_args__ = (ForeignKeyConstraint(['module_id', 'presentation_code'],
                                           ['presentations.module_id', 'presentations.presentation_code']),
                      # Add unique constraint to prevent duplicate registrations
                      UniqueConstraint('student_id', 'module_id', 'presentation_code', name='uq_student_registration'),
                      Index('ix_registrations_presentation', 'module_id', 'presentation_code'))

    def __repr__(self):
        return f"<Registration(student_id={self.student_id}, presentation='{self.module_id}_{self.presentation_code}')>"

class Interaction(Base):
    __tablename__ = 'interactions'
    # Derived from studentVle.csv

    interaction_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey('users.student_id'), index=True)
    module_id = Column(String(50)) # Part of FK to Presentation
    presentation_code = Column(String(50)) # Part of FK to Presentation
    vle_item_id = Column(Integer, ForeignKey('vle_items.vle_item_id'), index=True)

    interaction_date = Column(Integer, comment='Days relative to start (from studentVle.date)')
    total_clicks = Column(Integer, comment='Clicks on this item on this day (from studentVle.sum_click)')

    # Relationships
    user = relationship("User", back_populates="interactions")
    vle_item = relationship("VleItem", back_populates="interactions")
    presentation = relationship("Presentation", back_populates="interactions",
                                foreign_keys=[module_id, presentation_code],
                                primaryjoin="and_(Interaction.module_id==Presentation.module_id, Interaction.presentation_code==Presentation.presentation_code)")

    # Composite foreign key constraint to presentations
    __table_args__ = (ForeignKeyConstraint(['module_id', 'presentation_code'],
                                           ['presentations.module_id', 'presentations.presentation_code']),
                      Index('ix_interactions_user_presentation', 'student_id', 'module_id', 'presentation_code'), # Index for user activity queries
                      Index('ix_interactions_presentation', 'module_id', 'presentation_code'))


    def __repr__(self):
        return f"<Interaction(student_id={self.student_id}, vle_item_id={self.vle_item_id}, date={self.interaction_date}, clicks={self.total_clicks})>"


class Assessment(Base):
    __tablename__ = 'assessments'

    assessment_id = Column(Integer, primary_key=True)
    module_id = Column(String(50)) # Part of FK to Presentation
    presentation_code = Column(String(50)) # Part of FK to Presentation

    assessment_type = Column(String(50), comment='TMA, CMA, or Exam')
    assessment_date = Column(Integer, nullable=True, comment='Final submission date relative to start (can be null)')
    weight = Column(Float)

    # Relationships
    presentation = relationship("Presentation", back_populates="assessments",
                                foreign_keys=[module_id, presentation_code],
                                primaryjoin="and_(Assessment.module_id==Presentation.module_id, Assessment.presentation_code==Presentation.presentation_code)")
    submissions = relationship("StudentAssessment", back_populates="assessment")

    # Composite foreign key constraint
    __table_args__ = (ForeignKeyConstraint(['module_id', 'presentation_code'],
                                           ['presentations.module_id', 'presentations.presentation_code']),
                      Index('ix_assessments_presentation', 'module_id', 'presentation_code'))


    def __repr__(self):
        return f"<Assessment(assessment_id={self.assessment_id}, type='{self.assessment_type}')>"

class StudentAssessment(Base):
    __tablename__ = 'student_assessments'
    # Derived from studentAssessment.csv

    student_assessment_id = Column(Integer, primary_key=True, autoincrement=True) # Simple auto PK
    student_id = Column(Integer, ForeignKey('users.student_id'), index=True)
    assessment_id = Column(Integer, ForeignKey('assessments.assessment_id'), index=True)

    submission_date = Column(Integer, comment='Submission date relative to start')
    is_banked = Column(Boolean)
    score = Column(Integer, nullable=True, comment='Assessment score (can be null)') # Changed to Integer based on EDA values

    # Relationships
    user = relationship("User", back_populates="assessments_submitted")
    assessment = relationship("Assessment", back_populates="submissions")

    # Ensure a student submits a specific assessment only once
    __table_args__ = (UniqueConstraint('student_id', 'assessment_id', name='uq_student_assessment_submission'),)

    def __repr__(self):
        return f"<StudentAssessment(student_id={self.student_id}, assessment_id={self.assessment_id}, score={self.score})>"


# --- Engine and Metadata ---
# Only create engine if DATABASE_URI is set
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
    metadata = Base.metadata # Still define metadata so functions below don't break


# --- Utility Functions (can be moved to db_utils.py later, but useful here for now) ---
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
    # Example usage:
    # Be careful running drop_tables()!
    if engine:
        print("\nRunning schema utility functions...")
        # drop_tables(engine) # Uncomment to drop tables first
        # create_tables(engine) # Uncomment to create tables
        print("Schema utility functions finished.")
    else:
        print("\nDatabase engine not configured. Cannot run schema utilities.")