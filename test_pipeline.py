from dotenv import load_dotenv
load_dotenv()
import asyncio
from agents.orchestrator import run_pipeline

result = asyncio.run(run_pipeline({
    'user_id': 'test-001',
    'github_url': 'https://github.com/Shyamalan-21',
    'reddit_username': None,
    'stackoverflow_id': None,
    'devto_username': None,
    'hashnode_username': None,
    'resume_text': None,
}))

print('Trust Score:', result['trust_score'])
print('Explanation:', result['explanation_paragraph'])
print('Warnings:', result['warnings'])
print('Errors:', result['errors'])
