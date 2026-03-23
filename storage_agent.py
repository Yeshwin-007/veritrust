# agents/storage_agent.py
from agents.state import TrustState
from database.connection import SessionLocal
from database.models import TrustRecord

def run_storage(state: TrustState) -> TrustState:
    db = None
    try:
        db = SessionLocal()
        record = TrustRecord(
            user_id=            state['user_id'],
            trust_score=        state.get('trust_score', 0),
            fraud_probability=  state.get('score_breakdown', {}).get('fraud_probability', 0),
            bias_correction=    state.get('bias_correction', 0.0),
            explanation=        state.get('explanation_paragraph', ''),
            shap_values=        state.get('shap_values', {}),
            feature_vector=     state.get('feature_vector', []),
            verification_passed=state.get('verification_passed', True),
            bias_flags=         state.get('bias_flags', []),
            score_breakdown=    state.get('score_breakdown', {}),
            data_sources={
                'github':       bool(state.get('github_data')),
                'reddit':       bool(state.get('reddit_data')),
                'stackoverflow':bool(state.get('stackoverflow_data')),
                'devto':        bool(state.get('devto_data')),
                'hashnode':     bool(state.get('hashnode_data')),
            },
            processing_time_ms=state.get('processing_time_ms', 0),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        state['record_id'] = record.id
    except Exception as e:
        state['errors'].append(f'Storage failed: {e}')
    finally:
        if db:
            db.close()
    return state