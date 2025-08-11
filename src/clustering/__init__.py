"""
Central logger configuration for the `clustering` package.

All modules under `src/clustering` import `logger` from here to ensure
consistent logging to both console and file: logs/clustering.log
"""

import logging
from pathlib import Path

# Ensure logs directory exists (repo_root/logs)
LOG_DIR = (Path(__file__).resolve().parents[2] / "logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Package-level logger
logger = logging.getLogger("clustering")
logger.setLevel(logging.DEBUG)

# Avoid duplicate handlers in interactive sessions
if not logger.handlers:
    # File handler (verbose)
    fh = logging.FileHandler(LOG_DIR / "clustering.log")
    fh.setLevel(logging.DEBUG)

    # Console handler (concise)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Unified formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)