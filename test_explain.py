from dotenv import load_dotenv
load_dotenv()
import os
from groq import Groq

api_key = os.getenv('GROQ_API_KEY')
print('Key loaded:', api_key[:10] if api_key else 'NOT FOUND')

client = Groq(api_key=api_key)

prompt = """Write a 3 sentence explanation for a developer with Trust Score of 8/100.
They have 5 GitHub repos, 2 followers, good commit quality.
Write in second person. No bullets."""

resp = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=[{'role': 'user', 'content': prompt}],
    max_tokens=200,
    temperature=0.3,
)
print('Explanation:', resp.choices[0].message.content)
