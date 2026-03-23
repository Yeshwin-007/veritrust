# backend/main.py
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from database.connection import get_db, init_db
from database.models import TrustRecord
from agents.orchestrator import run_pipeline
import os, uuid

app = FastAPI(title='VeriTrust AI', version='1.0.0')

app.add_middleware(CORSMiddleware,
    allow_origins=os.getenv('CORS_ORIGINS', '*').split(','),
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.on_event('startup')
async def startup():
    init_db()

class AnalyzeRequest(BaseModel):
    github_url:        Optional[str] = None
    reddit_username:   Optional[str] = None
    stackoverflow_id:  Optional[str] = None
    devto_username:    Optional[str] = None
    hashnode_username: Optional[str] = None
    resume_text:       Optional[str] = None

@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'VeriTrust AI'}

@app.post('/analyze')
async def analyze(req: AnalyzeRequest, db: Session = Depends(get_db)):
    if not any([req.github_url, req.reddit_username,
                req.stackoverflow_id, req.devto_username]):
        raise HTTPException(400, 'Provide at least one profile URL')

    result = await run_pipeline({
        'user_id':          str(uuid.uuid4()),
        'github_url':       req.github_url,
        'reddit_username':  req.reddit_username,
        'stackoverflow_id': req.stackoverflow_id,
        'devto_username':   req.devto_username,
        'hashnode_username':req.hashnode_username,
        'resume_text':      req.resume_text,
    })
    return {
        'record_id':           result.get('record_id'),
        'trust_score':         result['trust_score'],
        'breakdown':           result['score_breakdown'],
        'explanation':         result['explanation_paragraph'],
        'shap_values':         result['shap_values'],
        'bias_flags':          result['bias_flags'],
        'bias_correction':     result['bias_correction'],
        'verification_passed': result['verification_passed'],
        'processing_ms':       result.get('processing_time_ms'),
        'warnings':            result.get('warnings', []),
        'errors':              result.get('errors', []),
    }

@app.get('/record/{record_id}')
def get_record(record_id: str, db: Session = Depends(get_db)):
    rec = db.query(TrustRecord).filter(TrustRecord.id == record_id).first()
    if not rec:
        raise HTTPException(404, 'Record not found')
    return rec

@app.get('/records')
def list_records(limit: int = 20, db: Session = Depends(get_db)):
    return db.query(TrustRecord).order_by(
        TrustRecord.created_at.desc()).limit(limit).all()