import os
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool

# Load environment variables
load_dotenv()

# Create declarative base for models
Base = declarative_base()

# Alternative: construct from individual components
DB_USER = os.getenv("DB_USERNAME", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_DATABASE", "")

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Global engine and session maker
_engine: Optional[AsyncEngine] = None
_async_session_maker: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine(
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        echo: bool = False,
) -> AsyncEngine:
    """
    Get or create the async database engine.

    Args:
        pool_size: Number of connections to maintain in the pool
        max_overflow: Maximum number of connections to create beyond pool_size
        pool_timeout: Timeout for getting a connection from the pool
        echo: Whether to log all SQL statements

    Returns:
        AsyncEngine instance
    """
    global _engine

    if _engine is None:
        _engine = create_async_engine(
            DATABASE_URL,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_pre_ping=True,  # Verify connections before using them
            poolclass=AsyncAdaptedQueuePool,
        )

    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the async session maker.

    Returns:
        async_sessionmaker instance
    """
    global _async_session_maker

    if _async_session_maker is None:
        engine = get_engine()
        _async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    return _async_session_maker


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    Usage:
        async with get_session() as session:
            result = await session.execute(query)
            await session.commit()

    Yields:
        AsyncSession instance
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """
    Initialize the database by creating all tables.

    Usage:
        await init_db()
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db():
    """
    Drop all tables from the database.

    Usage:
        await drop_db()
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def close_db():
    """
    Close the database connection and dispose of the engine.

    Usage:
        await close_db()
    """
    global _engine, _async_session_maker

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_maker = None


# Dependency for FastAPI (optional)
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting database sessions.

    Usage in FastAPI:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    async with get_session() as session:
        yield session


# Health check function
async def check_db_connection() -> bool:
    """
    Check if database connection is healthy.

    Returns:
        True if connection is successful, False otherwise
    """
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

