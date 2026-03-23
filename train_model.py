import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import shap, joblib, os
from ml.feature_engineering import FEATURE_NAMES

np.random.seed(42)
N = 8000

def generate_dataset():
    d = {}

    # ── GitHub ───────────────────────────────────────────────
    d['gh_total_stars']       = np.random.exponential(50, N).clip(0, 10000)
    d['gh_total_forks']       = d['gh_total_stars'] * np.random.uniform(0.05, 0.3, N)
    d['gh_public_repos']      = np.random.exponential(15, N).clip(1, 300)
    d['gh_followers']         = np.random.exponential(30, N).clip(0, 5000)
    d['gh_account_age_days']  = np.random.uniform(30, 5000, N)
    d['gh_total_commits']     = np.random.exponential(300, N).clip(0, 15000)
    d['gh_commit_quality']    = np.random.beta(5, 3, N)
    d['gh_churn_ratio']       = np.random.beta(3, 5, N)
    d['gh_prs_merged']        = np.random.exponential(10, N).clip(0, 500)
    d['gh_issues_closed']     = np.random.exponential(15, N).clip(0, 500)
    d['gh_lang_diversity']    = np.random.randint(1, 10, N).astype(float)
    d['gh_repo_descriptions'] = np.random.randint(0, 10, N).astype(float)

    # ── Reddit ───────────────────────────────────────────────
    d['rd_comment_karma']      = np.random.exponential(500, N).clip(0, 100000)
    d['rd_tech_comment_count'] = np.random.exponential(40, N).clip(0, 800)
    d['rd_avg_tech_score']     = np.random.exponential(8, N).clip(0, 200)
    d['rd_quality_comments']   = np.random.exponential(5, N).clip(0, 100)
    d['rd_avg_comment_length'] = np.random.exponential(200, N).clip(20, 2000)
    d['rd_unique_tech_terms']  = np.random.exponential(50, N).clip(0, 500)

    # ── Stack Overflow ───────────────────────────────────────
    d['so_reputation']       = np.random.exponential(1000, N).clip(1, 100000)
    d['so_answer_count']     = np.random.exponential(30, N).clip(0, 2000)
    d['so_accepted_answers'] = d['so_answer_count'] * np.random.uniform(0.1, 0.6, N)
    d['so_acceptance_rate']  = d['so_accepted_answers'] / (d['so_answer_count'] + 1)
    d['so_avg_answer_score'] = np.random.exponential(4, N).clip(0, 100)
    d['so_answer_quality']   = np.random.beta(6, 3, N)
    d['so_highly_voted']     = np.random.exponential(5, N).clip(0, 200)
    d['so_gold_badges']      = np.random.exponential(1, N).clip(0, 50)

    # ── Dev.to ───────────────────────────────────────────────
    d['dt_article_count']    = np.random.exponential(5, N).clip(0, 200)
    d['dt_total_reactions']  = d['dt_article_count'] * np.random.exponential(20, N)
    d['dt_avg_reactions']    = np.random.exponential(15, N).clip(0, 500)
    d['dt_high_quality']     = np.random.exponential(3, N).clip(0, 50)
    d['dt_engagement_rate']  = np.random.beta(2, 20, N)
    d['dt_avg_reading_time'] = np.random.uniform(2, 20, N)

    # ── Hashnode ─────────────────────────────────────────────
    d['hn_post_count']       = np.random.exponential(3, N).clip(0, 100)
    d['hn_followers']        = np.random.exponential(50, N).clip(0, 5000)
    d['hn_avg_reactions']    = np.random.exponential(10, N).clip(0, 300)
    d['hn_engagement_rate']  = np.random.beta(2, 20, N)
    d['hn_total_views']      = np.random.exponential(1000, N).clip(0, 100000)

    # ── Cross-platform ───────────────────────────────────────
    d['plagiarism_score']    = np.random.beta(1, 15, N)
    d['consistency_score']   = np.random.beta(8, 2, N)
    d['platform_count']      = np.random.randint(1, 6, N).astype(float)
    d['writing_score']       = np.random.beta(3, 5, N)
    d['community_score']     = np.random.beta(3, 6, N)

    df = pd.DataFrame(d)

    # ── Trust Score Formula ──────────────────────────────────
    score = (
        # GitHub — quality over quantity
        np.log1p(df['gh_total_stars'])        * 2.5  +
        df['gh_commit_quality']               * 18.0 +
        (1 - df['gh_churn_ratio'])            * 8.0  +
        np.log1p(df['gh_prs_merged'])         * 5.0  +
        np.log1p(df['gh_issues_closed'])      * 3.0  +
        df['gh_lang_diversity']               * 1.5  +
        np.log1p(df['gh_total_commits'])      * 1.5  +
        np.log1p(df['gh_followers'])          * 2.0  +
        np.log1p(df['gh_account_age_days'])   * 1.5  +

        # Stack Overflow — quality signals
        np.log1p(df['so_reputation'])         * 4.0  +
        df['so_acceptance_rate']              * 15.0 +
        df['so_answer_quality']               * 12.0 +
        np.log1p(df['so_highly_voted'])       * 4.0  +
        df['so_gold_badges']                  * 2.0  +

        # Reddit — quality over volume
        np.log1p(df['rd_quality_comments'])   * 4.0  +
        df['rd_avg_tech_score']               * 0.3  +
        np.log1p(df['rd_tech_comment_count']) * 1.5  +

        # Writing platforms — engagement quality
        np.log1p(df['dt_avg_reactions'])      * 3.0  +
        df['dt_engagement_rate']              * 20.0 +
        np.log1p(df['dt_high_quality'])       * 4.0  +
        np.log1p(df['hn_avg_reactions'])      * 2.0  +
        df['hn_engagement_rate']              * 15.0 +

        # Cross-platform bonuses
        df['consistency_score']               * 10.0 +
        df['platform_count']                  * 2.0  +
        df['community_score']                 * 8.0  +

        # Penalties
        -df['plagiarism_score']               * 25.0
    )

    df['trust_score'] = (
        (score - score.min()) / (score.max() - score.min()) * 100
    ).clip(0, 100)
    return df


def train():
    print('Generating training data (8000 samples)...')
    df = generate_dataset()
    X = df[FEATURE_NAMES].values
    y = df['trust_score'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print('Training ensemble model...')

    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=5,
        learning_rate=0.05, subsample=0.8,
        min_samples_split=10, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_split=5, random_state=42, n_jobs=-1)

    model = VotingRegressor([('gb', gb), ('rf', rf)])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f'MAE: {mae:.2f}')
    print(f'R2:  {r2:.4f}')

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f'CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

    os.makedirs('ml', exist_ok=True)
    joblib.dump(model, 'ml/trust_model.pkl')
    print('Model saved to ml/trust_model.pkl')

    # Fit RF standalone for SHAP (needs to be fitted separately)
    print('Testing SHAP...')
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'ml/trust_model_rf.pkl')
    print('RF model saved to ml/trust_model_rf.pkl')

    explainer = shap.TreeExplainer(rf)
    vals = explainer.shap_values(X_test[:3])
    print(f'SHAP shape: {vals.shape} OK')
    print('Done!')


if __name__ == '__main__':
    train()