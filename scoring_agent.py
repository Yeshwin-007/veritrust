# agents/scoring_agent.py
# Simple direct scoring — no ML model needed
from agents.state import TrustState

def run_scoring(state: TrustState) -> TrustState:
    try:
        g  = state.get('github_data', {})
        rd = state.get('reddit_data', {})
        so = state.get('stackoverflow_data', {})
        dt = state.get('devto_data', {})
        hn = state.get('hashnode_data', {})

        score = 0.0

        # GitHub (max 40 points)
        if g:
            score += min(g.get('followers', 0) * 0.5, 8)
            score += min(g.get('total_stars', 0) * 0.1, 10)
            score += min(g.get('public_repos', 0) * 0.5, 8)
            score += min(g.get('account_age_days', 0) / 365 * 3, 6)
            score += g.get('commit_quality', 0) * 5
            score += min(g.get('lang_diversity', 0) * 0.5, 3)

        # Stack Overflow (max 25 points)
        if so:
            score += min(so.get('reputation', 0) / 1000, 10)
            score += so.get('acceptance_rate', 0) * 10
            score += min(so.get('gold_badges', 0) * 2, 5)

        # Reddit (max 10 points)
        if rd:
            score += min(rd.get('comment_karma', 0) / 5000, 5)
            score += min(rd.get('tech_comment_count', 0) / 20, 5)

        # Dev.to (max 15 points)
        if dt:
            score += min(dt.get('article_count', 0) * 0.5, 5)
            score += min(dt.get('total_reactions', 0) / 100, 5)
            score += min(dt.get('avg_reading_time', 0) * 0.3, 5)

        # Hashnode (max 10 points)
        if hn:
            score += min(hn.get('post_count', 0) * 0.5, 5)
            score += min(hn.get('followers', 0) / 100, 5)

        # Plagiarism penalty
        score -= state.get('plagiarism_score', 0) * 20

        # Bias correction
        score += state.get('bias_correction', 0)

        trust_score = int(min(max(round(score), 1), 100))
        fraud_prob  = round(1.0 - (trust_score / 100.0), 3)

        # SHAP-style breakdown
        shap_dict = {}
        if g:
            shap_dict['gh followers']      = round(min(g.get('followers', 0) * 0.5, 8), 2)
            shap_dict['gh stars']          = round(min(g.get('total_stars', 0) * 0.1, 10), 2)
            shap_dict['gh repos']          = round(min(g.get('public_repos', 0) * 0.5, 8), 2)
            shap_dict['gh account age']    = round(min(g.get('account_age_days', 0) / 365 * 3, 6), 2)
            shap_dict['gh commit quality'] = round(g.get('commit_quality', 0) * 5, 2)
            shap_dict['gh lang diversity'] = round(min(g.get('lang_diversity', 0) * 0.5, 3), 2)
        if so:
            shap_dict['so reputation']     = round(min(so.get('reputation', 0) / 1000, 10), 2)
            shap_dict['so acceptance rate']= round(so.get('acceptance_rate', 0) * 10, 2)
        if rd:
            shap_dict['rd karma']          = round(min(rd.get('comment_karma', 0) / 5000, 5), 2)
        if dt:
            shap_dict['dt articles']       = round(min(dt.get('article_count', 0) * 0.5, 5), 2)
        if hn:
            shap_dict['hn posts']          = round(min(hn.get('post_count', 0) * 0.5, 5), 2)

        # Sort by value
        shap_dict = dict(sorted(
            shap_dict.items(), key=lambda x: x[1], reverse=True))

        state['trust_score']     = trust_score
        state['shap_values']     = shap_dict
        state['feature_vector']  = list(shap_dict.values())
        state['score_breakdown'] = {
            'raw_score':            round(score, 1),
            'bias_correction':      state.get('bias_correction', 0.0),
            'final_score':          trust_score,
            'fraud_probability':    fraud_prob,
            'top_positive_factors': [k for k, v in shap_dict.items() if v > 0][:5],
            'top_negative_factors': [k for k, v in shap_dict.items() if v <= 0][:3],
            'platforms_used': {
                'github':        bool(g),
                'reddit':        bool(rd),
                'stackoverflow': bool(so),
                'devto':         bool(dt),
                'hashnode':      bool(hn),
            }
        }
    except Exception as e:
        state['errors'].append(f'Scoring failed: {e}')
        state['trust_score']     = 0
        state['score_breakdown'] = {}
    return state
