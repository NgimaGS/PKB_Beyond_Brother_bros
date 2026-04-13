"""
Config Manager — Settings Persistence Layer
============================================

Architecture Rationale:
-----------------------
This module centralizes all application state that survives between restarts.
It handles serialization to `data/settings.json`, ensuring that user preferences
(Model choices, paths, etc.) are always restored.

Design Note: Default-Merging
We use a 'Merge with Defaults' strategy. This ensures that when new features
are added to the code, existing user config files are automatically patched
with the missing default keys.
"""

import json
import os

class ConfigManager:
    """Manages persistent application settings via a JSON file."""

    
    DEFAULT_CONFIG = {
        "log_path": "data/",
        "log_pattern": "ingestion_errors_%Y-%m-%d_%H-%M-%S.log",
        "model_nickname": "Engine",
        "ollama_host": "http://localhost:11434",
        "clearing_threshold": 40,
        "engine_mode": "Deep Learning",
        "ingestion_size_limit_mb": 50,
        "ingestion_size_limit_active": False,
        "chat_model": "gemma4:e4b",
        "embedding_model": "mxbai-embed-large"
    }
    
    def __init__(self, config_path="data/settings.json"):
        self.config_path = config_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        self.config = self.load()

    def load(self):
        """Loads configuration from disk or returns defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    # Merge defaults with loaded data to handle schema updates
                    return {**self.DEFAULT_CONFIG, **data}
            except Exception:
                return self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()

    def save(self, updates):
        """Saves updated settings to disk."""
        self.config.update(updates)
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception:
            return False

    def get(self, key):
        """Retrieves a specific setting."""
        return self.config.get(key, self.DEFAULT_CONFIG.get(key))
