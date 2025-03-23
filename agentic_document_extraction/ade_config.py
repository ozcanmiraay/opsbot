import os
from dotenv import load_dotenv

load_dotenv()

ADE_API_URL = "https://api.va.landing.ai/v1/tools/agentic-document-analysis"
ADE_API_KEY = os.getenv("ADE_API_KEY")  # This should be added to the .env file

HEADERS = {
    "Authorization": f"Basic {ADE_API_KEY}",
}