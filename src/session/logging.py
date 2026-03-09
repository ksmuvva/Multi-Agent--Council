"""
Session logging utilities.

Provides a configured logger for the session persistence layer.
"""

import logging


def bind_session(session_id: str) -> None:
    """Bind a session ID to the current logging context (no-op stub)."""
    pass


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(f"multi_agent.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
