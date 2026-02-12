"""
Configuration and path utilities for the project.

Loads environment variables from .env and provides path helpers
so that no hardcoded absolute paths appear elsewhere in the codebase.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_project_root():
    """Return the project root directory (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_path(subdir="raw"):
    """Return the path to a data subdirectory."""
    return get_project_root() / "data" / subdir


def get_results_path(subdir="figures"):
    """Return the path to a results subdirectory."""
    return get_project_root() / "results" / subdir


def get_youtube_credentials():
    """Return YouTube API credentials from environment variables."""
    return {
        "client_id": os.getenv("YOUTUBE_CLIENT_ID"),
        "client_secret": os.getenv("YOUTUBE_CLIENT_SECRET"),
        "project_id": os.getenv("YOUTUBE_PROJECT_ID"),
    }
