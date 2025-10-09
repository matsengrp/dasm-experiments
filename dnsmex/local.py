"""Local configuration management for dnsm-experiments-1.

This module loads user-specific configuration from local-config.py.
"""

import os

# Load user configuration
try:
    from .local_config import CONFIG
except ImportError:
    # For CI/testing environments, provide empty config since paths aren't used in tests
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        CONFIG = {}  # Empty config for CI since paths aren't used in tests
    else:
        raise ImportError(
            "local_config.py not found. Please run:\n\n"
            "    cp dnsmex/local_config.py.template dnsmex/local_config.py\n\n"
            "Then edit dnsmex/local_config.py to match your local setup."
        )


def localify(path):
    """Replace configuration placeholders with actual paths."""
    for key, value in CONFIG.items():
        path = path.replace(key, value)
    path = path.replace("~", os.path.expanduser("~"))
    return path


def get_remote_config():
    """Get remote configuration values for Makefile integration."""
    return {
        "host": CONFIG["REMOTE_HOST"],
        "dir": CONFIG["REMOTE_DIR"],
        "venv": CONFIG["REMOTE_VENV_PATH"],
    }
