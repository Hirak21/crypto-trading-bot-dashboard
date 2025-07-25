"""
Configuration management with encrypted credential storage.

This module handles loading, saving, and validating bot configuration,
with special handling for sensitive API credentials.
"""

import json
import os
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from cryptography.fernet import Fernet
import logging

from ..interfaces import IConfigManager


class ConfigManager(IConfigManager):
    """Configuration manager with encrypted credential storage."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.config_file = self.config_dir / "bot_config.json"
        self.credentials_file = self.config_dir / "credentials.enc"
        self.key_file = self.config_dir / ".key"
        
        self.logger = logging.getLogger(__name__)
        self._encryption_key = self._get_or_create_key()
        
    def _get_or_create_key(self) -> bytes:
        """Get existing encryption key or create new one."""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions on key file
            os.chmod(self.key_file, 0o600)
            return key
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from storage."""
        try:
            if not self.config_file.exists():
                return self._get_default_config()
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            self.logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to storage."""
        try:
            # Validate config before saving
            if not self.validate_config(config):
                self.logger.error("Invalid configuration, not saving")
                return False
            
            # Remove sensitive data before saving to main config
            safe_config = {k: v for k, v in config.items() 
                          if k not in ['api_key', 'api_secret']}
            
            with open(self.config_file, 'w') as f:
                json.dump(safe_config, f, indent=2, default=str)
            
            # Save credentials separately if provided
            if 'api_key' in config and 'api_secret' in config:
                self._save_credentials(config['api_key'], config['api_secret'])
            
            self.logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_api_credentials(self) -> Tuple[str, str]:
        """Get encrypted API credentials."""
        try:
            if not self.credentials_file.exists():
                raise FileNotFoundError("Credentials file not found")
            
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            
            fernet = Fernet(self._encryption_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())
            
            return credentials['api_key'], credentials['api_secret']
            
        except Exception as e:
            self.logger.error(f"Failed to load API credentials: {e}")
            raise
    
    def _save_credentials(self, api_key: str, api_secret: str) -> None:
        """Save encrypted API credentials."""
        try:
            credentials = {
                'api_key': api_key,
                'api_secret': api_secret
            }
            
            fernet = Fernet(self._encryption_key)
            encrypted_data = fernet.encrypt(json.dumps(credentials).encode())
            
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions on credentials file
            os.chmod(self.credentials_file, 0o600)
            
        except Exception as e:
            self.logger.error(f"Failed to save credentials: {e}")
            raise
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        try:
            from ..models.config import validate_bot_config
            return validate_bot_config(config)
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    

    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        from ..models.config import create_default_config
        return create_default_config().to_dict()