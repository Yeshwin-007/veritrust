# database/models.py
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class TrustRecord(Base):
    __tablename__ = 'trust_records'

    id                  = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id             = Column(String, nullable=False, index=True)
    created_at          = Column(DateTime, default=datetime.utcnow)
    trust_score         = Column(Integer, nullable=False)
    fraud_probability   = Column(Float, nullable=False)
    bias_correction     = Column(Float, default=0.0)
    explanation         = Column(Text, nullable=True)
    shap_values         = Column(JSON, nullable=True)
    feature_vector      = Column(JSON, nullable=True)
    verification_passed = Column(Boolean, default=True)
    bias_flags          = Column(JSON, default=list)
    score_breakdown     = Column(JSON, nullable=True)
    data_sources        = Column(JSON, nullable=True)
    processing_time_ms  = Column(Integer, default=0)