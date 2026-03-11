"""
CLI Module - Command Line Interface for Multi-Agent Reasoning System

This module provides the Typer-based CLI for interacting with the
multi-agent reasoning system from the command line.
"""

from .main import app

VERSION = "0.1.0"

__all__ = [
    "app",
    "VERSION",
]
