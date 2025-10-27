"""
Configuration module for the backend.

This module loads environment variables from a .env file and provides access to them.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from google.oauth2 import service_account


# Load environment from default locations
load_dotenv()

# Correct path to project-root .env (same directory as this file)
env_path = Path(__file__).parent / ".env"

# If no project id present, try loading root .env
if not (
    os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    or os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("PROJECT_ID")
):
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded local .env file from {env_path}")
    else:
        print(f".env file not found at {env_path}, relying on existing environment variables")
else:
    print("Using environment variables already set globally")

PROJECT_ID = (
    os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    or os.getenv("GOOGLE_CLOUD_PROJECT")
    or os.getenv("PROJECT_ID")
)

LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

API_KEY = os.getenv("API_KEY")

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def get_google_credentials() -> Optional[service_account.Credentials]:
    """Create Google credentials from env vars.

    Priority:
    1) Inline service account parts: GOOGLE_PRIVATE_KEY + GOOGLE_CLIENT_EMAIL
       - PRIVATE KEY may include escaped newlines (\\n); convert to real newlines
    2) GOOGLE_APPLICATION_CREDENTIALS pointing at a JSON file
    3) Fallback: None (ADC will be used if available)
    """

    private_key_env = os.getenv("GOOGLE_PRIVATE_KEY")
    client_email = os.getenv("GOOGLE_CLIENT_EMAIL")

    if private_key_env and client_email:
        private_key = private_key_env.replace("\\n", "\n")
        info = {
            "type": "service_account",
            "private_key": private_key,
            "client_email": client_email,
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).exists():
        return service_account.Credentials.from_service_account_file(creds_path, scopes=SCOPES)

    return None
