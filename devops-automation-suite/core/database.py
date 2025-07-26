"""
DevOps Automation Suite - Database Management
Enterprise-grade database connection and session management
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import redis
from loguru import logger

from .config import settings


# SQLAlchemy setup
engine = create_engine(
    settings.database_url,
    poolclass=StaticPool,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

# Metadata for table creation
metadata = MetaData()


def get_db() -> Generator[Session, None, None]:
    """
    Database dependency for FastAPI
    Creates a database session and ensures it's closed after use
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_session():
    """
    Context manager for database sessions
    Usage:
        with get_db_session() as db:
            # perform database operations
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database transaction error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """Get Redis client instance"""
    return redis_client


def init_database():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


def test_database_connection():
    """Test database connectivity"""
    try:
        with get_db_session() as db:
            db.execute("SELECT 1")
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def test_redis_connection():
    """Test Redis connectivity"""
    try:
        redis_client.ping()
        logger.info("Redis connection successful")
        return True
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return False


class DatabaseManager:
    """
    Database manager class for advanced operations
    """
    
    def __init__(self):
        self.engine = engine
        self.session_factory = SessionLocal
        self.redis = redis_client
    
    def health_check(self) -> dict:
        """Comprehensive database health check"""
        db_status = test_database_connection()
        redis_status = test_redis_connection()
        
        return {
            "database": "healthy" if db_status else "unhealthy",
            "redis": "healthy" if redis_status else "unhealthy",
            "overall": "healthy" if (db_status and redis_status) else "unhealthy"
        }
    
    def get_connection_info(self) -> dict:
        """Get database connection information"""
        return {
            "database_url": settings.database_url,
            "redis_url": settings.redis_url,
            "pool_size": engine.pool.size(),
            "checked_out_connections": engine.pool.checkedout(),
        }


# Global database manager instance
db_manager = DatabaseManager()
