# src/database/db_utils.py

from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import text
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
    """
    Drops all tables defined in schema.py metadata using CASCADE.
    Use with extreme caution! Ensures dependent objects are also dropped.
    """
    if not engine:
        print("Database engine not available. Cannot drop tables.")
        return
    try:
        # Use Base.metadata.sorted_tables to get tables, potentially in a better order for dropping
        # Reverse order is generally safer for dropping with dependencies if not using CASCADE
        # However, with CASCADE, the order matters less, but reversing doesn't hurt.
        tables_to_drop = reversed(Base.metadata.sorted_tables)
        table_names = [table.name for table in tables_to_drop]

        if not table_names:
            print("No tables found in metadata to drop.")
            return

        confirm = input(f"Found tables: {', '.join(table_names)}\nAre you sure you want to drop ALL these tables using CASCADE? This is irreversible! (yes/no): ")
        if confirm.lower() == 'yes':
            print("Attempting to drop all tables with CASCADE...")
            with engine.connect() as connection:
                with connection.begin(): # Start a transaction
                    for table_name in table_names:
                        print(f" Dropping table {table_name} with CASCADE...")
                        # Use text() for raw SQL, ensure table name is quoted if needed
                        connection.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
            print("Tables dropped successfully using CASCADE.")
        else:
            print("Table drop cancelled.")
    except Exception as e:
        print(f"Error dropping tables: {e}")
        # Optionally re-raise the exception if you want the script to fail hard
        # raise e

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