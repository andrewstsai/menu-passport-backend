"""
Database configuration and session management
SQLite with SQLAlchemy ORM
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
import logging
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)
DATABASE_URL = f"sqlite:///{settings.database_path}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Enables Write-Ahead Logging for better performance"""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def get_db():
    """
    Database session dependency for FastAPI
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initializes database tables"""
    from app.models import menu  # noqa
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")

def reset_db():
    """Drops all tables and recreates"""
    from app.models import menu  # noqa
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Database reset")

def get_db_stats():
    """Gets database statistics"""
    db = SessionLocal()
    try:
        from app.models.menu import MenuItem, MenuImage
        stats = {
            "menu_images": db.query(MenuImage).count(),
            "menu_items": db.query(MenuItem).count()
        }
        return stats
    finally:
        db.close()
