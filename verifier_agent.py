# agents/verifier_agent.py
# Enhanced: cross-platform consistency checks
from agents.state import TrustState

def run_verifier(state: TrustState) -> TrustState:
    flags            = []
    plagiarism_scores = []

    gh = state.get('github_data', {})
    dt = state.get('devto_data', {})
    so = state.get('stackoverflow_data', {})
    rd = state.get('reddit_data', {})

    # Check 1: All accounts brand new (bot signal)
    ages = [state.get(s, {}).get('account_age_days')
            for s in ['github_data', 'reddit_data', 'stackoverflow_data']
            if state.get(s, {}).get('account_age_days')]
    if len(ages) >= 2 and max(ages) - min(ages) == 0 and max(ages) < 30:
        flags.append('All accounts created same day — possible fake identity')
        plagiarism_scores.append(0.8)

    # Check 2: Fork farming (many repos, almost no commits)
    repos   = gh.get('public_repos', 0)
    commits = gh.get('total_commits', 0)
    if repos > 20 and commits < 10:
        flags.append('High repo count with very low commits — fork farming signal')
        plagiarism_scores.append(0.7)

    # Check 3: Commit quality too low (copy-paste commits)
    commit_quality = gh.get('commit_quality', 1.0)
    if commits > 50 and commit_quality < 0.1:
        flags.append('Very low commit message quality — possible copied/AI-generated commits')
        plagiarism_scores.append(0.5)

    # Check 4: Cross-platform skill inconsistency
    gh_langs = set(k.lower() for k in gh.get('top_languages', {}).keys())
    dt_tags  = set(t.lower() for t in dt.get('tags', []))
    so_tags  = set(t.lower() for t in so.get('top_tags', []))

    # Check GitHub vs SO consistency
    lang_map = {'python': 'python', 'javascript': 'javascript',
                'typescript': 'typescript', 'rust': 'rust',
                'go': 'go', 'java': 'java', 'c#': 'c#', 'c++': 'c++'}
    gh_norm = set(lang_map.get(l, l) for l in gh_langs)
    overlap_so = gh_norm & so_tags
    overlap_dt = gh_norm & dt_tags
    total_overlap = len(overlap_so) + len(overlap_dt)
    consistency = total_overlap / max(len(gh_norm) * 2, 1)
    state['consistency_score'] = round(min(consistency, 1.0), 2)

    if consistency < 0.05 and len(gh_norm) > 0 and (len(so_tags) > 0 or len(dt_tags) > 0):
        flags.append('No skill overlap between GitHub and other platforms — inconsistency')

    # Check 5: Answer farming on Stack Overflow
    if so.get('answer_count', 0) > 100 and so.get('reputation', 0) < 50:
        flags.append('High SO answer count with near-zero reputation — answer farming')
        plagiarism_scores.append(0.6)

    # Check 6: Stars vs followers mismatch (bought stars)
    stars     = gh.get('total_stars', 0)
    followers = gh.get('followers', 0)
    if stars > 1000 and followers < 5:
        flags.append('Very high stars with almost no followers — possible star farming')
        plagiarism_scores.append(0.4)

    plagiarism_score = (sum(plagiarism_scores) / len(plagiarism_scores)
                        if plagiarism_scores else 0.05)

    state['plagiarism_score']    = round(plagiarism_score, 3)
    state['fake_cert_flags']     = flags
    state['verification_passed'] = plagiarism_score < 0.6 and len(flags) < 3

    if not state['verification_passed']:
        state['errors'].append('Verification failed — too many suspicious signals')
    return state