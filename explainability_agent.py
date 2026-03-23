# agents/explainability_agent.py
from agents.state import TrustState
import os, json

def run_explainability(state: TrustState) -> TrustState:
    shap_values = state.get('shap_values', {})
    trust_score = state.get('trust_score', 0)
    bias_flags  = state.get('bias_flags', [])
    breakdown   = state.get('score_breakdown', {})
    platforms   = breakdown.get('platforms_used', {})

    top_pos = [k for k, v in shap_values.items() if v > 0][:5]
    top_neg = [k for k, v in shap_values.items() if v <= 0][:3]
    active  = [p for p, active in platforms.items() if active]

    try:
        # Import and initialize INSIDE the function
        # so it always picks up the latest .env values
        from groq import Groq
        api_key = os.getenv('GROQ_API_KEY')

        if not api_key:
            raise ValueError('GROQ_API_KEY not set')

        client = Groq(api_key=api_key)

        prompt = f"""You are an AI trust analyst for developer credibility.

Write a 3-4 sentence explanation of a developer Trust Score of {trust_score}/100.

Active platforms: {active}
Top positive factors: {top_pos}
Top negative factors: {top_neg}
Bias correction: +{state.get('bias_correction', 0)} points

Rules:
- Write in second person (Your score...)
- Mention specific platforms by name
- Explain what the top factors mean in plain English
- End with one suggestion to improve the score
- No bullet points. Output only the paragraph."""

        resp = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        explanation = resp.choices[0].message.content.strip()

    except Exception as e:
        # Simple fallback
        explanation = (
            f'Your Trust Score of {trust_score}/100 reflects your activity on '
            f'{", ".join(active) if active else "the provided platforms"}. '
            f'Your strongest signals are: {", ".join(top_pos[:3]) if top_pos else "none yet"}. '
            f'To improve your score, consider building your Stack Overflow reputation '
            f'and contributing to more open source projects on GitHub.'
        )
        state['warnings'].append(f'Groq error: {str(e)[:100]}')

    state['explanation_paragraph'] = explanation
    return state