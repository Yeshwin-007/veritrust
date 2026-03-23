# agents/bias_auditor_agent.py
from agents.state import TrustState

NON_ENGLISH_TAGS = {
    'spanish', 'portuguese', 'french', 'hindi', 'arabic', 'japanese',
    'chinese', 'korean', 'russian', 'turkish', 'indonesian', 'vietnamese',
    'bengali', 'urdu', 'swahili', 'persian',
}

def run_bias_auditor(state: TrustState) -> TrustState:
    flags = []
    bias_correction = 0.0

    so = state.get('stackoverflow_data', {})
    dt = state.get('devto_data', {})
    gh = state.get('github_data', {})
    rd = state.get('reddit_data', {})

    # Signal 1: Old SO account + low English reputation
    so_rep = so.get('reputation', 0)
    so_age = so.get('account_age_days', 0)
    if so_age > 365 and so_rep < 500:
        flags.append('Possible non-English SO contributor — English data underrepresents skill')
        bias_correction += 3.0

    # Signal 2: Non-English content tags on Dev.to
    dt_tags = [t.lower() for t in dt.get('tags', [])]
    found_lang_tags = [t for t in dt_tags if t in NON_ENGLISH_TAGS]
    if found_lang_tags:
        flags.append(f'Non-English content detected on Dev.to: {found_lang_tags}')
        bias_correction += 2.0

    # Signal 3: High activity, low reputational recognition
    activity = (
        gh.get('total_commits', 0) * 0.1 +
        gh.get('total_stars', 0)   * 0.5 +
        rd.get('tech_comment_count', 0) * 0.3 +
        so.get('answer_count', 0)  * 0.4
    )
    rep_proxy = (
        gh.get('followers', 0)       * 2 +
        so.get('reputation', 0)      * 0.01 +
        rd.get('comment_karma', 0)   * 0.01
    )
    if activity > 100 and rep_proxy < 20:
        flags.append('High activity but low recognition — possible systemic underscoring')
        bias_correction += 5.0

    bias_correction = min(bias_correction, 10.0)

    state['bias_flags']       = flags
    state['bias_correction']  = round(bias_correction, 1)
    state['non_english_ratio'] = round(len(found_lang_tags) / max(len(dt_tags), 1), 2)

    if flags:
        state['warnings'].append(f'Bias auditor applied +{bias_correction} equity correction')
    return state