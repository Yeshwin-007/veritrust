# agents/state.py
from typing import TypedDict, Optional, List, Dict, Any

class TrustState(TypedDict):
    # ── INPUT ────────────────────────────────────────────────
    user_id:             str
    github_url:          Optional[str]
    reddit_username:     Optional[str]
    stackoverflow_id:    Optional[str]
    devto_username:      Optional[str]
    hashnode_username:   Optional[str]
    resume_text:         Optional[str]

    # ── HARVESTER OUTPUTS ────────────────────────────────────
    github_data:         Dict[str, Any]
    reddit_data:         Dict[str, Any]
    stackoverflow_data:  Dict[str, Any]
    devto_data:          Dict[str, Any]
    hashnode_data:       Dict[str, Any]

    # ── VERIFIER OUTPUTS ─────────────────────────────────────
    plagiarism_score:    float
    consistency_score:   float
    fake_cert_flags:     List[str]
    verification_passed: bool

    # ── BIAS AUDITOR OUTPUTS (SDG 10) ────────────────────────
    bias_flags:          List[str]
    bias_correction:     float
    non_english_ratio:   float

    # ── SCORING OUTPUTS ──────────────────────────────────────
    trust_score:         int
    shap_values:         Dict[str, float]
    feature_vector:      List[float]
    score_breakdown:     Dict[str, Any]

    # ── EXPLAINABILITY OUTPUT ────────────────────────────────
    explanation_paragraph: str

    # ── STORAGE OUTPUT ───────────────────────────────────────
    record_id:           Optional[str]

    # ── PIPELINE METADATA ────────────────────────────────────
    errors:              List[str]
    warnings:            List[str]
    processing_time_ms:  int