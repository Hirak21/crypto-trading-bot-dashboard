"""
Dynamic configuration management system.

This module provides dynamic configuration updates, validation, backup,
and rollback functionality for the crypto trading bot.
"""

import json
import shutil
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from threading import Lock
import logging

from ..models.config import BotConfig, RiskConfig, StrategyConfig, NotificationConfig, LoggingConfig
from ..utils.config import ConfigManager


@dataclass
class ConfigChange:
    """Represents a configuration change."""
    timestamp: datetime
    section: str
    key: str
    old_value: Any
    new_value: Any
    user: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class ConfigBackup:
    """Represents a configuration backup."""
    timestamp: datetime
    config: Dict[str, Any]
    version: str
    description: Optional[str] = None


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigUpdateError(Exception):
    """Raised when configuration update fails."""
    pass


class DynamicConfigManager:
    """
    Dynamic configuration manager with hot-reload capabilities.
    
    Provides:
    - Real-time configuration updates without restart
    - Configuration validation and rollback
    - Change tracking and audit logging
    - Backup and restore functionality
    """
    
    def __init__(self, config_dir: str = "config", backup_dir: str = "config/backups"):
        self.config_dir = Path(config_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_manager = ConfigManager(config_dir)
        self.logger = logging.getLogger(__name__)
        
        # Current configuration
        self._current_config: Optional[BotConfig] = None
        self._config_lock = Lock()
        
        # Change tracking
        self._change_history: List[ConfigChange] = []
        self._change_callbacks: Dict[str, List[Callable]] = {}
        
        # Backup management
        self._backups: List[ConfigBackup] = []
        self._max_backups = 50
        self._auto_backup_interval = timedelta(hours=1)
        self._last_backup_time: Optional[datetime] = None
        
        # Validation rules
        self._validation_rules: Dict[str, Callable] = {}
        self._setup_default_validation_rules()
        
        # Load initial configuration
        self._load_initial_config()
    
    def _setup_default_validation_rules(self) -> None:
        """Setup default validation rules for configuration sections."""
        self._validation_rules = {
            'risk_config': self._validate_risk_config,
            'strategies': self._validate_strategies_config,
            'notification_config': self._validate_notification_config,
            'logging_config': self._validate_logging_config,
            'symbols': self._validate_symbols,
            'trading_enabled': self._validate_boolean,
            'dry_run': self._validate_boolean,
            'testnet': self._validate_boolean
        }
    
    def _load_initial_config(self) -> None:
        """Load initial configuration from storage."""
        try:
            config_dict = self.config_manager.load_config()
            self._current_config = BotConfig.from_dict(config_dict)
            self.logger.info("Initial configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load initial configuration: {e}")
            self._current_config = BotConfig()  # Use default config
    
    @property
    def current_config(self) -> BotConfig:
        """Get current configuration (thread-safe)."""
        with self._config_lock:
            return self._current_config
    
    def get_config_section(self, section: str) -> Any:
        """Get specific configuration section."""
        with self._config_lock:
            return getattr(self._current_config, section, None)
    
    def update_config(self, updates: Dict[str, Any], user: Optional[str] = None, 
                     reason: Optional[str] = None, validate: bool = True) -> bool:
        """
        Update configuration with validation and change tracking.
        
        Args:
            updates: Dictionary of configuration updates
            user: User making the change (for audit)
            reason: Reason for the change (for audit)
            validate: Whether to validate changes before applying
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            with self._config_lock:
                # Create backup before making changes
                self._create_backup("Before update", auto=True)
                
                # Store original config for rollback
                original_config = self._current_config.to_dict()
                changes = []
                
                # Apply updates and track changes
                new_config_dict = original_config.copy()
                
                for key, new_value in updates.items():
                    if key in new_config_dict:
                        old_value = new_config_dict[key]
                        
                        # Validate individual change if requested
                        if validate and not self._validate_config_change(key, new_value):
                            raise ConfigValidationError(f"Validation failed for {key}")
                        
                        new_config_dict[key] = new_value
                        
                        # Track change
                        change = ConfigChange(
                            timestamp=datetime.now(),
                            section=key,
                            key=key,
                            old_value=old_value,
                            new_value=new_value,
                            user=user,
                            reason=reason
                        )
                        changes.append(change)
                    else:
                        self.logger.warning(f"Unknown configuration key: {key}")
                
                # Validate complete configuration
                if validate:
                    try:
                        new_config = BotConfig.from_dict(new_config_dict)
                    except Exception as e:
                        raise ConfigValidationError(f"Configuration validation failed: {e}")
                else:
                    new_config = BotConfig.from_dict(new_config_dict)
                
                # Apply changes
                self._current_config = new_config
                self._change_history.extend(changes)
                
                # Save to storage
                if not self.config_manager.save_config(new_config_dict):
                    raise ConfigUpdateError("Failed to save configuration to storage")
                
                # Notify callbacks
                self._notify_config_changes(changes)
                
                self.logger.info(f"Configuration updated successfully. Changes: {len(changes)}")
                return True
                
        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
            return False
    
    def update_strategy_config(self, strategy_name: str, config_updates: Dict[str, Any],
                             user: Optional[str] = None, reason: Optional[str] = None) -> bool:
        """Update configuration for a specific strategy."""
        try:
            with self._config_lock:
                if strategy_name not in self._current_config.strategies:
                    raise ValueError(f"Unknown strategy: {strategy_name}")
                
                # Get current strategy config
                current_strategy_config = self._current_config.strategies[strategy_name]
                updated_strategy_dict = current_strategy_config.to_dict()
                updated_strategy_dict.update(config_updates)
                
                # Validate strategy config
                new_strategy_config = StrategyConfig.from_dict(updated_strategy_dict)
                
                # Update in main config
                strategies_update = {
                    'strategies': {
                        **{name: config.to_dict() for name, config in self._current_config.strategies.items()},
                        strategy_name: new_strategy_config.to_dict()
                    }
                }
                
                return self.update_config(strategies_update, user, reason)
                
        except Exception as e:
            self.logger.error(f"Strategy config update failed: {e}")
            return False
    
    def update_risk_config(self, risk_updates: Dict[str, Any], 
                          user: Optional[str] = None, reason: Optional[str] = None) -> bool:
        """Update risk management configuration."""
        try:
            current_risk_dict = self._current_config.risk_config.to_dict()
            current_risk_dict.update(risk_updates)
            
            # Validate risk config
            new_risk_config = RiskConfig.from_dict(current_risk_dict)
            
            return self.update_config({'risk_config': new_risk_config.to_dict()}, user, reason)
            
        except Exception as e:
            self.logger.error(f"Risk config update failed: {e}")
            return False
    
    def _validate_config_change(self, key: str, value: Any) -> bool:
        """Validate a single configuration change."""
        try:
            if key in self._validation_rules:
                return self._validation_rules[key](value)
            return True  # No specific validation rule
        except Exception as e:
            self.logger.error(f"Validation error for {key}: {e}")
            return False
    
    def _validate_risk_config(self, config: Dict[str, Any]) -> bool:
        """Validate risk configuration."""
        try:
            RiskConfig.from_dict(config)
            return True
        except Exception:
            return False
    
    def _validate_strategies_config(self, config: Dict[str, Any]) -> bool:
        """Validate strategies configuration."""
        try:
            for strategy_name, strategy_config in config.items():
                StrategyConfig.from_dict(strategy_config)
            return True
        except Exception:
            return False
    
    def _validate_notification_config(self, config: Dict[str, Any]) -> bool:
        """Validate notification configuration."""
        try:
            NotificationConfig.from_dict(config)
            return True
        except Exception:
            return False
    
    def _validate_logging_config(self, config: Dict[str, Any]) -> bool:
        """Validate logging configuration."""
        try:
            LoggingConfig.from_dict(config)
            return True
        except Exception:
            return False
    
    def _validate_symbols(self, symbols: List[str]) -> bool:
        """Validate trading symbols."""
        if not isinstance(symbols, list) or not symbols:
            return False
        
        import re
        symbol_pattern = r'^[A-Z]{2,10}USDT?$'
        return all(re.match(symbol_pattern, symbol) for symbol in symbols)
    
    def _validate_boolean(self, value: Any) -> bool:
        """Validate boolean value."""
        return isinstance(value, bool)
    
    def register_change_callback(self, section: str, callback: Callable[[ConfigChange], None]) -> None:
        """Register callback for configuration changes in specific section."""
        if section not in self._change_callbacks:
            self._change_callbacks[section] = []
        self._change_callbacks[section].append(callback)
    
    def _notify_config_changes(self, changes: List[ConfigChange]) -> None:
        """Notify registered callbacks about configuration changes."""
        for change in changes:
            if change.section in self._change_callbacks:
                for callback in self._change_callbacks[change.section]:
                    try:
                        callback(change)
                    except Exception as e:
                        self.logger.error(f"Error in change callback: {e}")
    
    def _create_backup(self, description: Optional[str] = None, auto: bool = False) -> bool:
        """Create configuration backup."""
        try:
            now = datetime.now()
            
            # Check if auto backup is needed
            if auto and self._last_backup_time:
                if now - self._last_backup_time < self._auto_backup_interval:
                    return True  # Skip backup, too recent
            
            backup = ConfigBackup(
                timestamp=now,
                config=self._current_config.to_dict(),
                version=f"{now.strftime('%Y%m%d_%H%M%S')}",
                description=description
            )
            
            # Save backup to file
            backup_file = self.backup_dir / f"config_backup_{backup.version}.json"
            backup_data = {
                'timestamp': backup.timestamp.isoformat(),
                'version': backup.version,
                'description': backup.description,
                'config': backup.config
            }
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            self._backups.append(backup)
            self._last_backup_time = now
            
            # Clean old backups
            self._cleanup_old_backups()
            
            self.logger.info(f"Configuration backup created: {backup.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
    
    def create_manual_backup(self, description: str) -> bool:
        """Create manual configuration backup."""
        return self._create_backup(description, auto=False)
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available configuration backups."""
        backups = []
        for backup in sorted(self._backups, key=lambda x: x.timestamp, reverse=True):
            backups.append({
                'version': backup.version,
                'timestamp': backup.timestamp.isoformat(),
                'description': backup.description
            })
        return backups
    
    def restore_from_backup(self, version: str, user: Optional[str] = None) -> bool:
        """Restore configuration from backup."""
        try:
            # Find backup
            backup = None
            for b in self._backups:
                if b.version == version:
                    backup = b
                    break
            
            if not backup:
                # Try to load from file
                backup_file = self.backup_dir / f"config_backup_{version}.json"
                if backup_file.exists():
                    with open(backup_file, 'r') as f:
                        backup_data = json.load(f)
                    backup = ConfigBackup(
                        timestamp=datetime.fromisoformat(backup_data['timestamp']),
                        config=backup_data['config'],
                        version=backup_data['version'],
                        description=backup_data.get('description')
                    )
                else:
                    raise ValueError(f"Backup version not found: {version}")
            
            # Create backup of current config before restore
            self._create_backup("Before restore", auto=True)
            
            # Restore configuration
            with self._config_lock:
                self._current_config = BotConfig.from_dict(backup.config)
                
                # Save to storage
                if not self.config_manager.save_config(backup.config):
                    raise ConfigUpdateError("Failed to save restored configuration")
                
                # Track restore as change
                change = ConfigChange(
                    timestamp=datetime.now(),
                    section="system",
                    key="restore",
                    old_value="current_config",
                    new_value=f"backup_{version}",
                    user=user,
                    reason=f"Restored from backup {version}"
                )
                self._change_history.append(change)
                
                self.logger.info(f"Configuration restored from backup: {version}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {e}")
            return False
    
    def _cleanup_old_backups(self) -> None:
        """Clean up old backup files."""
        if len(self._backups) <= self._max_backups:
            return
        
        # Sort by timestamp and remove oldest
        self._backups.sort(key=lambda x: x.timestamp)
        old_backups = self._backups[:-self._max_backups]
        
        for backup in old_backups:
            try:
                backup_file = self.backup_dir / f"config_backup_{backup.version}.json"
                if backup_file.exists():
                    backup_file.unlink()
                self._backups.remove(backup)
            except Exception as e:
                self.logger.error(f"Failed to cleanup backup {backup.version}: {e}")
    
    def get_change_history(self, limit: Optional[int] = None, 
                          section: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        changes = self._change_history
        
        if section:
            changes = [c for c in changes if c.section == section]
        
        if limit:
            changes = changes[-limit:]
        
        return [
            {
                'timestamp': change.timestamp.isoformat(),
                'section': change.section,
                'key': change.key,
                'old_value': change.old_value,
                'new_value': change.new_value,
                'user': change.user,
                'reason': change.reason
            }
            for change in sorted(changes, key=lambda x: x.timestamp, reverse=True)
        ]
    
    def validate_current_config(self) -> Dict[str, Any]:
        """Validate current configuration and return validation results."""
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            with self._config_lock:
                config_dict = self._current_config.to_dict()
                
                # Validate each section
                for section, validator in self._validation_rules.items():
                    if section in config_dict:
                        try:
                            if not validator(config_dict[section]):
                                results['errors'].append(f"Validation failed for section: {section}")
                                results['valid'] = False
                        except Exception as e:
                            results['errors'].append(f"Validation error in {section}: {str(e)}")
                            results['valid'] = False
                
                # Additional cross-section validations
                self._validate_cross_section_rules(config_dict, results)
                
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Configuration validation error: {str(e)}")
        
        return results
    
    def _validate_cross_section_rules(self, config: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Validate rules that span multiple configuration sections."""
        # Example: Check if risk limits are compatible with strategy settings
        try:
            risk_config = config.get('risk_config', {})
            strategies = config.get('strategies', {})
            
            # Check if any strategy has parameters that conflict with risk settings
            max_position = risk_config.get('max_position_size', 0.02)
            
            for strategy_name, strategy_config in strategies.items():
                if strategy_config.get('enabled', False):
                    strategy_params = strategy_config.get('parameters', {})
                    
                    # Example validation: position size consistency
                    if 'position_size' in strategy_params:
                        if strategy_params['position_size'] > max_position:
                            results['warnings'].append(
                                f"Strategy {strategy_name} position size exceeds risk limit"
                            )
                            
        except Exception as e:
            results['warnings'].append(f"Cross-section validation warning: {str(e)}")
    
    def export_config(self, file_path: str, include_sensitive: bool = False) -> bool:
        """Export current configuration to file."""
        try:
            with self._config_lock:
                config_dict = self._current_config.to_dict()
                
                if not include_sensitive:
                    # Remove sensitive information
                    config_dict.pop('api_key', None)
                    config_dict.pop('api_secret', None)
                
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'config_version': '1.0',
                    'config': config_dict
                }
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Configuration exported to: {file_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, file_path: str, user: Optional[str] = None, 
                     validate: bool = True) -> bool:
        """Import configuration from file."""
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            if 'config' not in import_data:
                raise ValueError("Invalid import file format")
            
            imported_config = import_data['config']
            
            return self.update_config(
                imported_config, 
                user=user, 
                reason=f"Imported from {file_path}",
                validate=validate
            )
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return False