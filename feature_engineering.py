# ml/feature_engineering.py
from typing import List
from agents.state import TrustState

FEATURE_NAMES = [
    # GitHub (12)
    'gh_total_stars', 'gh_total_forks', 'gh_public_repos',
    'gh_followers', 'gh_account_age_days', 'gh_total_commits',
    'gh_commit_quality', 'gh_churn_ratio', 'gh_prs_merged',
    'gh_issues_closed', 'gh_lang_diversity', 'gh_repo_descriptions',
    # Reddit (6)
    'rd_comment_karma', 'rd_tech_comment_count', 'rd_avg_tech_score',
    'rd_quality_comments', 'rd_avg_comment_length', 'rd_unique_tech_terms',
    # Stack Overflow (8)
    'so_reputation', 'so_answer_count', 'so_accepted_answers',
    'so_acceptance_rate', 'so_avg_answer_score', 'so_answer_quality',
    'so_highly_voted', 'so_gold_badges',
    # Dev.to (6)
    'dt_article_count', 'dt_total_reactions', 'dt_avg_reactions',
    'dt_high_quality', 'dt_engagement_rate', 'dt_avg_reading_time',
    # Hashnode (5)
    'hn_post_count', 'hn_followers', 'hn_avg_reactions',
    'hn_engagement_rate', 'hn_total_views',
    # Cross-platform (5)
    'plagiarism_score', 'consistency_score', 'platform_count',
    'writing_score', 'community_score',
]

def extract_features(state: TrustState) -> List[float]:
    g  = state.get('github_data', {})
    rd = state.get('reddit_data', {})
    so = state.get('stackoverflow_data', {})
    dt = state.get('devto_data', {})
    hn = state.get('hashnode_data', {})

    # Count active platforms — check for actual data keys
    platform_count = sum([
        1 if g.get('public_repos', 0) > 0 or g.get('username') else 0,
        1 if rd.get('comment_karma', 0) > 0 or rd.get('username') else 0,
        1 if so.get('reputation', 0) > 0 else 0,
        1 if dt.get('article_count', 0) > 0 or dt.get('username') else 0,
        1 if hn.get('post_count', 0) > 0 or hn.get('username') else 0,
    ])

    # Writing quality score
    writing_parts = []
    if dt.get('engagement_rate', 0) > 0:
        writing_parts.append(float(dt.get('engagement_rate', 0)))
    if hn.get('engagement_rate', 0) > 0:
        writing_parts.append(float(hn.get('engagement_rate', 0)))
    writing_score = sum(writing_parts) / max(len(writing_parts), 1)

    # Community standing
    community_score = (
        min(float(so.get('reputation', 0)) / 10000.0, 1.0) * 0.6 +
        min(float(rd.get('comment_karma', 0)) / 50000.0, 1.0) * 0.4
    )

    return [
        # GitHub
        float(g.get('total_stars', 0)),
        float(g.get('total_forks', 0)),
        float(g.get('public_repos', 0)),
        float(g.get('followers', 0)),
        float(g.get('account_age_days', 0)),
        float(g.get('total_commits', 0)),
        float(g.get('commit_quality', 0)),
        float(g.get('churn_ratio', 0.3)),
        float(g.get('prs_merged', 0)),
        float(g.get('issues_closed', 0)),
        float(g.get('lang_diversity', 0)),
        float(g.get('repo_descriptions', 0)),
        # Reddit
        float(rd.get('comment_karma', 0)),
        float(rd.get('tech_comment_count', 0)),
        float(rd.get('avg_tech_score', 0)),
        float(rd.get('quality_comments', 0)),
        float(rd.get('avg_comment_length', 0)),
        float(rd.get('unique_tech_terms', 0)),
        # Stack Overflow
        float(so.get('reputation', 0)),
        float(so.get('answer_count', 0)),
        float(so.get('accepted_answers', 0)),
        float(so.get('acceptance_rate', 0)),
        float(so.get('avg_answer_score', 0)),
        float(so.get('answer_quality', 0)),
        float(so.get('highly_voted', 0)),
        float(so.get('gold_badges', 0)),
        # Dev.to
        float(dt.get('article_count', 0)),
        float(dt.get('total_reactions', 0)),
        float(dt.get('avg_reactions', 0)),
        float(dt.get('high_quality', 0)),
        float(dt.get('engagement_rate', 0)),
        float(dt.get('avg_reading_time', 0)),
        # Hashnode
        float(hn.get('post_count', 0)),
        float(hn.get('followers', 0)),
        float(hn.get('avg_reactions', 0)),
        float(hn.get('engagement_rate', 0)),
        float(hn.get('total_views', 0)),
        # Cross-platform
        float(state.get('plagiarism_score', 0.05)),
        float(state.get('consistency_score', 0.5)),
        float(platform_count),
        float(writing_score),
        float(community_score),
    ]