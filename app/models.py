from datetime import datetime, timedelta
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from .database import Base


# ============================
# USER MODEL
# ============================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    credits = relationship("Credits", back_populates="user", uselist=False)
    transactions = relationship("CreditTransaction", back_populates="user")
    magic_tokens = relationship("MagicToken", back_populates="user")
    jobs = relationship("SplitJob", back_populates="user")


# ============================
# MAGIC LOGIN TOKENS
# ============================

class MagicToken(Base):
    __tablename__ = "magic_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    user = relationship("User", back_populates="magic_tokens")


# ============================
# CREDITS
# ============================

class Credits(Base):
    __tablename__ = "credits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    balance = Column(Integer, default=0)

    user = relationship("User", back_populates="credits")


# ============================
# CREDIT TRANSACTIONS
# ============================

class CreditTransaction(Base):
    __tablename__ = "credit_transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    change = Column(Integer, nullable=False)   # +50 for purchase, -1 for split
    reason = Column(String, nullable=False)     # "purchase", "split"
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="transactions")


# ============================
# SPLIT JOBS (OPTIONEEL MAAR BELANGRIJK)
# — Elke split-job wordt gelogd (debugging, analytics)
# ============================

class SplitJob(Base):
    __tablename__ = "split_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    input_filename = Column(String, nullable=False)
    output_count = Column(Integer, default=0)
    status = Column(String, default="processing")  # processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    user = relationship("User", back_populates="jobs")
