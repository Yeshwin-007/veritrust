# agents/orchestrator.py
from langgraph.graph import StateGraph, END
from agents.state import TrustState
from agents.harvester.github_agent        import run_github
from agents.harvester.reddit_agent        import run_reddit
from agents.harvester.stackoverflow_agent import run_stackoverflow
from agents.harvester.devto_agent         import run_devto
from agents.harvester.hashnode_agent      import run_hashnode
from agents.verifier_agent                import run_verifier
from agents.bias_auditor_agent            import run_bias_auditor
from agents.scoring_agent                 import run_scoring
from agents.explainability_agent          import run_explainability
from agents.storage_agent                 import run_storage
import time

def should_continue(state: TrustState) -> str:
    if len(state.get('errors', [])) >= 3:
        return 'end'
    if not state.get('verification_passed', True):
        return 'end'
    return 'continue'

def harvest_all(state: TrustState) -> TrustState:
    state = run_github(state)
    state = run_reddit(state)
    state = run_stackoverflow(state)
    state = run_devto(state)
    state = run_hashnode(state)
    return state

def build_graph():
    g = StateGraph(TrustState)
    g.add_node('harvest',    harvest_all)
    g.add_node('verify',     run_verifier)
    g.add_node('audit_bias', run_bias_auditor)
    g.add_node('score',      run_scoring)
    g.add_node('explain',    run_explainability)
    g.add_node('store',      run_storage)

    g.set_entry_point('harvest')
    g.add_edge('harvest', 'verify')
    g.add_conditional_edges('verify', should_continue,
                            {'continue': 'audit_bias', 'end': END})
    g.add_edge('audit_bias', 'score')
    g.add_edge('score',      'explain')
    g.add_edge('explain',    'store')
    g.add_edge('store',      END)
    return g.compile()

async def run_pipeline(user_input: dict) -> TrustState:
    graph = build_graph()
    initial = TrustState(
        user_id=           user_input['user_id'],
        github_url=        user_input.get('github_url'),
        reddit_username=   user_input.get('reddit_username'),
        stackoverflow_id=  user_input.get('stackoverflow_id'),
        devto_username=    user_input.get('devto_username'),
        hashnode_username= user_input.get('hashnode_username'),
        resume_text=       user_input.get('resume_text'),
        github_data={}, reddit_data={}, stackoverflow_data={},
        devto_data={}, hashnode_data={},
        plagiarism_score=0.0, consistency_score=1.0,
        fake_cert_flags=[], verification_passed=True,
        bias_flags=[], bias_correction=0.0, non_english_ratio=0.0,
        trust_score=0, shap_values={}, feature_vector=[],
        score_breakdown={}, explanation_paragraph='',
        record_id=None, errors=[], warnings=[], processing_time_ms=0,
    )
    start  = time.time()
    result = await graph.ainvoke(initial)
    result['processing_time_ms'] = int((time.time() - start) * 1000)
    return result