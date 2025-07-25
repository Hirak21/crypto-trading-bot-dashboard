"""
Security utilities for the crypto trading bot.

This module provides encryption, credential management, and security
validation functions.
"""

import os
import secrets
import hashlib
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json
import logging


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class CredentialManager:
    """Secure credential management with encryption."""
    
    def __init__(self, credentials_dir: str = "config"):
        self.credentials_dir = Path(credentials_dir)
        self.credentials_dir.mkdir(exist_ok=True, mode=0o700)  # Restrictive permissions
        
        self.credentials_file = self.credentials_dir / "credentials.enc"
        self.key_file = self.credentials_dir / ".key"
        self.salt_file = self.credentials_dir / ".salt"
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption
        self._encryption_key = self._get_or_create_key()
    
    def _get_or_create_key(self) -> bytes:
        """Get existing encryption key or create new one."""
        try:
            if self.key_file.exists() and self.salt_file.exists():
                # Load existing key and salt
                with open(self.key_file, 'rb') as f:
                    key = f.read()
                return key
            else:
                # Generate new key and salt
                salt = os.urandom(16)
                key = Fernet.generate_key()
                
                # Save key and salt with restrictive permissions
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                os.chmod(self.key_file, 0o600)
                
                with open(self.salt_file, 'wb') as f:
                    f.write(salt)
                os.chmod(self.salt_file, 0o600)
                
                return key
        
        except Exception as e:
            raise SecurityError(f"Failed to initialize encryption key: {e}")
    
    def store_credentials(self, api_key: str, api_secret: str, 
                         additional_data: Optional[Dict[str, str]] = None) -> bool:
        """Store encrypted API credentials."""
        try:
            # Validate credentials
            if not self._validate_api_credentials(api_key, api_secret):
                raise SecurityError("Invalid API credentials format")
            
            # Prepare credential data
            credentials = {
                'api_key': api_key,
                'api_secret': api_secret,
                'created_at': str(int(os.time.time())),
                'checksum': self._calculate_checksum(api_key + api_secret)
            }
            
            if additional_data:
                credentials.update(additional_data)
            
            # Encrypt and save
            fernet = Fernet(self._encryption_key)
            encrypted_data = fernet.encrypt(json.dumps(credentials).encode())
            
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(self.credentials_file, 0o600)
            
            self.logger.info("Credentials stored successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store credentials: {e}")
            return False
    
    def load_credentials(self) -> Tuple[str, str]:
        """Load and decrypt API credentials."""
        try:
            if not self.credentials_file.exists():
                raise SecurityError("Credentials file not found")
            
            # Read and decrypt
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            
            fernet = Fernet(self._encryption_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())
            
            # Validate integrity
            api_key = credentials['api_key']
            api_secret = credentials['api_secret']
            stored_checksum = credentials.get('checksum', '')
            
            if stored_checksum != self._calculate_checksum(api_key + api_secret):
                raise SecurityError("Credential integrity check failed")
            
            return api_key, api_secret
            
        except Exception as e:
            self.logger.error(f"Failed to load credentials: {e}")
            raise SecurityError(f"Credential loading failed: {e}")
    
    def credentials_exist(self) -> bool:
        """Check if credentials file exists."""
        return self.credentials_file.exists()
    
    def delete_credentials(self) -> bool:
        """Securely delete stored credentials."""
        try:
            if self.credentials_file.exists():
                # Overwrite file with random data before deletion
                file_size = self.credentials_file.stat().st_size
                with open(self.credentials_file, 'wb') as f:
                    f.write(os.urandom(file_size))
                
                self.credentials_file.unlink()
            
            self.logger.info("Credentials deleted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete credentials: {e}")
            return False
    
    def rotate_encryption_key(self) -> bool:
        """Rotate encryption key (re-encrypt credentials with new key)."""
        try:
            # Load existing credentials
            if self.credentials_exist():
                api_key, api_secret = self.load_credentials()
            else:
                self.logger.info("No existing credentials to rotate")
                return True
            
            # Delete old key files
            if self.key_file.exists():
                self.key_file.unlink()
            if self.salt_file.exists():
                self.salt_file.unlink()
            
            # Generate new key
            self._encryption_key = self._get_or_create_key()
            
            # Re-encrypt credentials with new key
            return self.store_credentials(api_key, api_secret)
            
        except Exception as e:
            self.logger.error(f"Failed to rotate encryption key: {e}")
            return False
    
    def _validate_api_credentials(self, api_key: str, api_secret: str) -> bool:
        """Validate API credential format."""
        if not api_key or not api_secret:
            return False
        
        # Basic format validation for Binance API keys
        if len(api_key) < 32 or len(api_secret) < 32:
            return False
        
        # Check for valid characters (alphanumeric)
        if not api_key.replace('-', '').replace('_', '').isalnum():
            return False
        
        if not api_secret.replace('-', '').replace('_', '').isalnum():
            return False
        
        return True
    
    def _calculate_checksum(self, data: str) -> str:
        """Calculate SHA-256 checksum for data integrity."""
        return hashlib.sha256(data.encode()).hexdigest()


class ParameterValidator:
    """Validate configuration parameters for security."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_symbols(self, symbols: list) -> bool:
        """Validate trading symbols for security."""
        if not symbols or not isinstance(symbols, list):
            return False
        
        # Check symbol format and prevent injection
        import re
        symbol_pattern = re.compile(r'^[A-Z]{2,10}USDT?$')
        
        for symbol in symbols:
            if not isinstance(symbol, str):
                return False
            
            if not symbol_pattern.match(symbol):
                return False
            
            # Prevent potential injection attacks
            if any(char in symbol for char in ['<', '>', '&', '"', "'"]):
                return False
        
        return True
    
    def validate_numeric_range(self, value: float, min_val: float, max_val: float, 
                              field_name: str) -> bool:
        """Validate numeric parameter within safe range."""
        if not isinstance(value, (int, float)):
            self.logger.error(f"{field_name} must be numeric")
            return False
        
        if not min_val <= value <= max_val:
            self.logger.error(f"{field_name} must be between {min_val} and {max_val}")
            return False
        
        return True
    
    def validate_string_parameter(self, value: str, max_length: int = 255, 
                                 allowed_chars: Optional[str] = None) -> bool:
        """Validate string parameter for security."""
        if not isinstance(value, str):
            return False
        
        if len(value) > max_length:
            return False
        
        # Check for potential injection patterns
        dangerous_patterns = ['<script', 'javascript:', 'data:', 'vbscript:', '<?php']
        for pattern in dangerous_patterns:
            if pattern.lower() in value.lower():
                return False
        
        # Check allowed characters if specified
        if allowed_chars:
            if not all(char in allowed_chars for char in value):
                return False
        
        return True
    
    def sanitize_log_message(self, message: str) -> str:
        """Sanitize log message to prevent log injection."""
        if not isinstance(message, str):
            return str(message)
        
        # Remove or escape dangerous characters
        sanitized = message.replace('\n', '\\n').replace('\r', '\\r')
        sanitized = sanitized.replace('\t', '\\t')
        
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:997] + "..."
        
        return sanitized


class SecurityAuditor:
    """Security auditing and monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security-related events."""
        sanitized_details = {
            k: ParameterValidator().sanitize_log_message(str(v)) 
            for k, v in details.items()
        }
        
        self.logger.warning(
            f"Security event: {event_type}",
            extra={
                'event_type': event_type,
                'details': sanitized_details,
                'timestamp': str(int(os.time.time()))
            }
        )
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if identifier is rate limited."""
        import time
        current_time = time.time()
        
        if identifier in self.failed_attempts:
            attempts, last_attempt = self.failed_attempts[identifier]
            
            # Reset if lockout period has passed
            if current_time - last_attempt > self.lockout_duration:
                del self.failed_attempts[identifier]
                return True
            
            # Check if still locked out
            if attempts >= self.max_failed_attempts:
                return False
        
        return True
    
    def record_failed_attempt(self, identifier: str) -> None:
        """Record a failed authentication attempt."""
        import time
        current_time = time.time()
        
        if identifier in self.failed_attempts:
            attempts, _ = self.failed_attempts[identifier]
            self.failed_attempts[identifier] = (attempts + 1, current_time)
        else:
            self.failed_attempts[identifier] = (1, current_time)
        
        self.log_security_event("failed_authentication", {
            'identifier': identifier,
            'attempt_count': self.failed_attempts[identifier][0]
        })
    
    def clear_failed_attempts(self, identifier: str) -> None:
        """Clear failed attempts for identifier."""
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]


# Global instances
credential_manager = CredentialManager()
parameter_validator = ParameterValidator()
security_auditor = SecurityAuditor()


# Convenience functions
def store_api_credentials(api_key: str, api_secret: str) -> bool:
    """Store API credentials securely."""
    return credential_manager.store_credentials(api_key, api_secret)


def load_api_credentials() -> Tuple[str, str]:
    """Load API credentials securely."""
    return credential_manager.load_credentials()


def validate_config_security(config: Dict[str, Any]) -> bool:
    """Validate configuration for security issues."""
    try:
        # Validate symbols
        if 'symbols' in config:
            if not parameter_validator.validate_symbols(config['symbols']):
                return False
        
        # Validate numeric parameters
        if 'risk_config' in config:
            risk_config = config['risk_config']
            
            numeric_params = [
                ('max_position_size', 0.001, 0.5),
                ('daily_loss_limit', 0.01, 1.0),
                ('max_drawdown', 0.05, 1.0),
                ('stop_loss_pct', 0.005, 0.2),
                ('take_profit_pct', 0.01, 0.5)
            ]
            
            for param, min_val, max_val in numeric_params:
                if param in risk_config:
                    if not parameter_validator.validate_numeric_range(
                        risk_config[param], min_val, max_val, param
                    ):
                        return False
        
        return True
        
    except Exception as e:
        security_auditor.log_security_event("config_validation_error", {'error': str(e)})
        return False