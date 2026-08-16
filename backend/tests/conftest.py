"""
Ensures `backend/` (this file's parent) is on sys.path so tests can use the
same bare imports the app itself uses everywhere (`from config import
settings`, `from services.rag_service import ...`), regardless of the
directory pytest is invoked from.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
