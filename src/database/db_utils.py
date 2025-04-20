# src/database/db_utils.py

from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import sys
from pathlib import Path
from typing import Generator

# Add project root to sys.path to import necessary modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import engine, metadata, SessionLocal, Base from schema.py
# We need these to interact with the database configuration and table definitions
from src.database.schema import engine, metadata, SessionLocal, Base

def get_db_session() -> Generator[Session, None, None]:
    """
    Provides a SQLAlchemy database session.

    Use this as a dependency in FastAPI or as a context manager elsewhere.

    Yields:
        Session: The SQLAlchemy session object.
    """
    if not SessionLocal:
        raise RuntimeError("Database session factory not configured. Check DATABASE_URI.")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    if not SessionLocal:
        raise RuntimeError("Database session factory not configured. Check DATABASE_URI.")

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def check_db_connection():
    """Tries to connect to the database and returns True if successful."""
    if not engine:
        print("Database engine not configured.")
        return False
    try:
        connection = engine.connect()
        connection.close()
        print("Database connection successful.")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def create_all_tables():
    """Creates all tables defined in schema.py metadata."""
    if not engine:
        print("Database engine not available. Cannot create tables.")
        return
    try:
        print("Attempting to create all tables...")
        Base.metadata.create_all(bind=engine)
        print("Tables created successfully (or already exist).")
    except Exception as e:
        print(f"Error creating tables: {e}")

def drop_all_tables():
    """Drops all tables defined in schema.py metadata. Use with caution!"""
    if not engine:
        print("Database engine not available. Cannot drop tables.")
        return
    try:
        confirm = input("Are you sure you want to drop all tables? This is irreversible! (yes/no): ")
        if confirm.lower() == 'yes':
            print("Attempting to drop all tables...")
            Base.metadata.drop_all(bind=engine)
            print("Tables dropped successfully.")
        else:
            print("Table drop cancelled.")
    except Exception as e:
        print(f"Error dropping tables: {e}")

# Example usage (can be called from other scripts)
if __name__ == "__main__":
    print("Running DB Utils Checks...")
    check_db_connection()
    # Example of using session_scope:
    # try:
    #     with session_scope() as session:
    #         # Perform database operations here
    #         # e.g., user = session.query(User).first()
    #         print("Session scope example executed successfully.")
    # except RuntimeError as e:
    #     print(e)
    # except Exception as e:
    #     print(f"Error during session scope example: {e}")

    # To create tables (use the function, not direct call):
    # create_all_tables()

    # To drop tables (use the function, prompts for confirmation):
    # drop_all_tables()