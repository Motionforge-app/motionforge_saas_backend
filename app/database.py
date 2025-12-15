from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# ====================================================
# DATABASE URL (Railway of lokale fallback)
# ====================================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

# Voor PostgreSQL op Railway moet hij altijd eindigen op ?sslmode=require
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")

if "sslmode" not in DATABASE_URL:
    DATABASE_URL += "?sslmode=require"

# ====================================================
# ENGINE
# ====================================================

engine = create_engine(
    DATABASE_URL,
    connect_args={}  # Railway heeft geen extra connect_args nodig
)

# ====================================================
# SESSION
# ====================================================

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ====================================================
# BASE CLASS voor modellen
# ====================================================

Base = declarative_base()


# ====================================================
# Dependency: database session per request
# ====================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
