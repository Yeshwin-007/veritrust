from dotenv import load_dotenv
load_dotenv()
from agents.harvester.github_agent import run_github

state = {
    'github_url': 'https://github.com/Shyamalan-21',
    'errors': [],
    'warnings': [],
    'github_data': {},
    'reddit_data': {},
    'stackoverflow_data': {},
    'devto_data': {},
    'hashnode_data': {}
}

result = run_github(state)
print('GitHub data:', result['github_data'])
print('Errors:', result['errors'])
