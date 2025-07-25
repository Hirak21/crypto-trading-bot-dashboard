"""
Tests for dynamic configuration management system.

This module contains comprehensive tests for configuration validation,
updates, backup, and rollback functionality.
"""

import pytest
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from crypto_trading_bot.managers.config_manager import (
    DynamicConfigManager, ConfigChange, ConfigBackup,
    ConfigValidationError, ConfigUpdateError
)
from crypto_trading_bot.models.config import BotConfig, RiskConfig, StrategyConfig


class TestDynamicConfigManager:
    """Test suite for DynamicConfigManager."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create DynamicConfigManager instance for testing."""
        return DynamicConfigManager(
            config_dir=temp_config_dir,
            backup_dir=f"{temp_config_dir}/backups"
        )
    
    @pytest.fixture
    def sample_config_updates(self):
        """Sample configuration updates for testing."""
        return {
            'trading_enabled': False,
            'dry_run': True,
            'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        }
    
    def test_initialization(self, config_manager):
        """Test DynamicConfigManager initialization."""
        assert config_manager is not None
        assert config_manager.current_config is not None
        assert isinstance(config_manager.current_config, BotConfig)
        assert config_manager.backup_dir.exists()
    
    def test_get_config_section(self, config_manager):
        """Test getting specific configuration sections."""
        risk_config = config_manager.get_config_section('risk_config')
        assert isinstance(risk_config, RiskConfig)
        
        strategies = config_manager.get_config_section('strategies')
        assert isinstance(strategies, dict)
        assert 'liquidity' in strategies
        
        # Test non-existent section
        invalid_section = config_manager.get_config_section('invalid_section')
        assert invalid_section is None
    
    def test_update_config_success(self, config_manager, sample_config_updates):
        """Test successful configuration update."""
        original_trading_enabled = config_manager.current_config.trading_enabled
        
        success = config_manager.update_config(
            sample_config_updates,
            user="test_user",
            reason="Testing config update"
        )
        
        assert success is True
        assert config_manager.current_config.trading_enabled != original_trading_enabled
        assert config_manager.current_config.dry_run is True
        assert len(config_manager.current_config.symbols) == 3
    
    def test_update_config_validation_failure(self, config_manager):
        """Test configuration update with validation failure."""
        invalid_updates = {
            'symbols': [],  # Empty symbols list should fail validation
            'trading_enabled': 'invalid_boolean'  # Invalid boolean value
        }
        
        success = config_manager.update_config(invalid_updates, validate=True)
        assert success is False
    
    def test_update_config_without_validation(self, config_manager):
        """Test configuration update without validation."""
        # This should work even with potentially invalid data
        updates = {'trading_enabled': False}
        
        success = config_manager.update_config(updates, validate=False)
        assert success is True
    
    def test_update_strategy_config(self, config_manager):
        """Test updating specific strategy configuration."""
        strategy_updates = {
            'enabled': False,
            'weight': 2.0,
            'min_confidence': 0.8
        }
        
        success = config_manager.update_strategy_config(
            'liquidity',
            strategy_updates,
            user="test_user",
            reason="Disable liquidity strategy"
        )
        
        assert success is True
        liquidity_config = config_manager.current_config.strategies['liquidity']
        assert liquidity_config.enabled is False
        assert liquidity_config.weight == 2.0
        assert liquidity_config.min_confidence == 0.8
    
    def test_update_strategy_config_invalid_strategy(self, config_manager):
        """Test updating configuration for non-existent strategy."""
        success = config_manager.update_strategy_config(
            'invalid_strategy',
            {'enabled': False}
        )
        
        assert success is False
    
    def test_update_risk_config(self, config_manager):
        """Test updating risk management configuration."""
        risk_updates = {
            'max_position_size': 0.03,
            'daily_loss_limit': 0.08,
            'stop_loss_pct': 0.025
        }
        
        success = config_manager.update_risk_config(
            risk_updates,
            user="test_user",
            reason="Adjust risk parameters"
        )
        
        assert success is True
        risk_config = config_manager.current_config.risk_config
        assert risk_config.max_position_size == 0.03
        assert risk_config.daily_loss_limit == 0.08
        assert risk_config.stop_loss_pct == 0.025
    
    def test_update_risk_config_invalid_values(self, config_manager):
        """Test updating risk configuration with invalid values."""
        invalid_risk_updates = {
            'max_position_size': 1.5,  # Too high (>50%)
            'stop_loss_pct': 0.3       # Too high (>20%)
        }
        
        success = config_manager.update_risk_config(invalid_risk_updates)
        assert success is False
    
    def test_validation_rules(self, config_manager):
        """Test individual validation rules."""
        # Test risk config validation
        valid_risk = {'max_position_size': 0.02, 'daily_loss_limit': 0.05}
        assert config_manager._validate_risk_config(valid_risk) is True
        
        invalid_risk = {'max_position_size': 2.0}  # Too high
        assert config_manager._validate_risk_config(invalid_risk) is False
        
        # Test symbols validation
        valid_symbols = ['BTCUSDT', 'ETHUSDT']
        assert config_manager._validate_symbols(valid_symbols) is True
        
        invalid_symbols = ['INVALID', 'btcusdt']  # Wrong format
        assert config_manager._validate_symbols(invalid_symbols) is False
        
        # Test boolean validation
        assert config_manager._validate_boolean(True) is True
        assert config_manager._validate_boolean(False) is True
        assert config_manager._validate_boolean('true') is False
    
    def test_change_callbacks(self, config_manager):
        """Test configuration change callbacks."""
        callback_called = False
        callback_change = None
        
        def test_callback(change: ConfigChange):
            nonlocal callback_called, callback_change
            callback_called = True
            callback_change = change
        
        # Register callback
        config_manager.register_change_callback('trading_enabled', test_callback)
        
        # Make a change
        config_manager.update_config({'trading_enabled': False})
        
        assert callback_called is True
        assert callback_change is not None
        assert callback_change.section == 'trading_enabled'
        assert callback_change.new_value is False
    
    def test_backup_creation(self, config_manager):
        """Test configuration backup creation."""
        # Create manual backup
        success = config_manager.create_manual_backup("Test backup")
        assert success is True
        
        # Check backup was created
        backups = config_manager.list_backups()
        assert len(backups) > 0
        assert backups[0]['description'] == "Test backup"
    
    def test_backup_restore(self, config_manager):
        """Test configuration backup and restore."""
        # Make initial change
        original_trading_enabled = config_manager.current_config.trading_enabled
        config_manager.update_config({'trading_enabled': not original_trading_enabled})
        
        # Create backup
        config_manager.create_manual_backup("Before restore test")
        backups = config_manager.list_backups()
        backup_version = backups[0]['version']
        
        # Make another change
        config_manager.update_config({'dry_run': True})
        assert config_manager.current_config.dry_run is True
        
        # Restore from backup
        success = config_manager.restore_from_backup(backup_version, user="test_user")
        assert success is True
        
        # Verify restoration
        # Note: The backup contains the state after the first change
        assert config_manager.current_config.trading_enabled == (not original_trading_enabled)
    
    def test_backup_cleanup(self, config_manager):
        """Test automatic backup cleanup."""
        # Set low max_backups for testing
        config_manager._max_backups = 3
        
        # Create multiple backups
        for i in range(5):
            config_manager.create_manual_backup(f"Backup {i}")
        
        # Check that only max_backups are kept
        backups = config_manager.list_backups()
        assert len(backups) <= 3
    
    def test_change_history(self, config_manager):
        """Test configuration change history tracking."""
        # Make some changes
        config_manager.update_config(
            {'trading_enabled': False},
            user="user1",
            reason="Disable trading"
        )
        
        config_manager.update_config(
            {'dry_run': True},
            user="user2",
            reason="Enable dry run"
        )
        
        # Get change history
        history = config_manager.get_change_history()
        assert len(history) >= 2
        
        # Check latest change
        latest_change = history[0]  # Should be sorted by timestamp desc
        assert latest_change['section'] == 'dry_run'
        assert latest_change['user'] == 'user2'
        assert latest_change['reason'] == 'Enable dry run'
        
        # Test filtered history
        trading_history = config_manager.get_change_history(section='trading_enabled')
        assert len(trading_history) >= 1
        assert all(change['section'] == 'trading_enabled' for change in trading_history)
        
        # Test limited history
        limited_history = config_manager.get_change_history(limit=1)
        assert len(limited_history) == 1
    
    def test_config_validation(self, config_manager):
        """Test current configuration validation."""
        # Test with valid configuration
        validation_result = config_manager.validate_current_config()
        assert validation_result['valid'] is True
        assert len(validation_result['errors']) == 0
        
        # Test with invalid configuration (manually corrupt it)
        with config_manager._config_lock:
            # Temporarily corrupt the config for testing
            original_max_position = config_manager._current_config.risk_config.max_position_size
            config_manager._current_config.risk_config.max_position_size = 2.0  # Invalid value
            
            validation_result = config_manager.validate_current_config()
            assert validation_result['valid'] is False
            assert len(validation_result['errors']) > 0
            
            # Restore valid value
            config_manager._current_config.risk_config.max_position_size = original_max_position
    
    def test_config_export_import(self, config_manager, temp_config_dir):
        """Test configuration export and import."""
        export_file = f"{temp_config_dir}/exported_config.json"
        
        # Make some changes to have something to export
        config_manager.update_config({'trading_enabled': False, 'dry_run': True})
        
        # Export configuration
        success = config_manager.export_config(export_file, include_sensitive=False)
        assert success is True
        assert Path(export_file).exists()
        
        # Verify export content
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        assert 'config' in export_data
        assert 'export_timestamp' in export_data
        assert export_data['config']['trading_enabled'] is False
        assert export_data['config']['dry_run'] is True
        
        # Make different changes
        config_manager.update_config({'trading_enabled': True, 'dry_run': False})
        
        # Import configuration
        success = config_manager.import_config(export_file, user="test_user")
        assert success is True
        
        # Verify import worked
        assert config_manager.current_config.trading_enabled is False
        assert config_manager.current_config.dry_run is True
    
    def test_concurrent_access(self, config_manager):
        """Test thread-safe configuration access."""
        import threading
        import time
        
        results = []
        errors = []
        
        def update_config(thread_id):
            try:
                for i in range(5):
                    success = config_manager.update_config(
                        {'trading_enabled': i % 2 == 0},
                        user=f"thread_{thread_id}",
                        reason=f"Update {i}"
                    )
                    results.append(success)
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=update_config, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0  # No errors should occur
        assert len(results) == 15  # 3 threads * 5 updates each
        assert all(results)  # All updates should succeed
    
    def test_auto_backup_interval(self, config_manager):
        """Test automatic backup interval functionality."""
        # Set short interval for testing
        config_manager._auto_backup_interval = timedelta(seconds=1)
        
        # First update should create backup
        config_manager.update_config({'trading_enabled': False})
        initial_backup_count = len(config_manager.list_backups())
        
        # Immediate second update should not create backup (too soon)
        config_manager.update_config({'dry_run': True})
        assert len(config_manager.list_backups()) == initial_backup_count
        
        # Wait for interval to pass
        import time
        time.sleep(1.1)
        
        # Third update should create backup
        config_manager.update_config({'trading_enabled': True})
        assert len(config_manager.list_backups()) > initial_backup_count
    
    def test_cross_section_validation(self, config_manager):
        """Test validation rules that span multiple configuration sections."""
        # This test verifies the _validate_cross_section_rules method
        validation_result = config_manager.validate_current_config()
        
        # Should not have errors for default configuration
        assert validation_result['valid'] is True
        
        # Test scenario where strategy position size exceeds risk limit
        # First update risk config to have low position size
        config_manager.update_risk_config({'max_position_size': 0.01})
        
        # Then try to set strategy with higher position size
        strategy_updates = {
            'parameters': {'position_size': 0.02}  # Higher than risk limit
        }
        config_manager.update_strategy_config('liquidity', strategy_updates)
        
        # Validation should show warnings
        validation_result = config_manager.validate_current_config()
        # Note: This might show warnings but still be valid depending on implementation
        assert isinstance(validation_result['warnings'], list)


class TestConfigChange:
    """Test suite for ConfigChange dataclass."""
    
    def test_config_change_creation(self):
        """Test ConfigChange creation."""
        change = ConfigChange(
            timestamp=datetime.now(),
            section="risk_config",
            key="max_position_size",
            old_value=0.02,
            new_value=0.03,
            user="test_user",
            reason="Increase position size"
        )
        
        assert change.section == "risk_config"
        assert change.key == "max_position_size"
        assert change.old_value == 0.02
        assert change.new_value == 0.03
        assert change.user == "test_user"
        assert change.reason == "Increase position size"


class TestConfigBackup:
    """Test suite for ConfigBackup dataclass."""
    
    def test_config_backup_creation(self):
        """Test ConfigBackup creation."""
        config_dict = {'trading_enabled': True, 'dry_run': False}
        
        backup = ConfigBackup(
            timestamp=datetime.now(),
            config=config_dict,
            version="20231201_120000",
            description="Test backup"
        )
        
        assert backup.config == config_dict
        assert backup.version == "20231201_120000"
        assert backup.description == "Test backup"


# Integration tests
class TestConfigManagerIntegration:
    """Integration tests for configuration management."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_workflow(self, temp_config_dir):
        """Test complete configuration management workflow."""
        # Initialize manager
        config_manager = DynamicConfigManager(
            config_dir=temp_config_dir,
            backup_dir=f"{temp_config_dir}/backups"
        )
        
        # 1. Initial state
        assert config_manager.current_config.trading_enabled is True
        
        # 2. Create backup
        config_manager.create_manual_backup("Initial state")
        
        # 3. Update configuration
        updates = {
            'trading_enabled': False,
            'dry_run': True,
            'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        }
        
        success = config_manager.update_config(
            updates,
            user="admin",
            reason="Maintenance mode"
        )
        assert success is True
        
        # 4. Verify changes
        assert config_manager.current_config.trading_enabled is False
        assert config_manager.current_config.dry_run is True
        assert len(config_manager.current_config.symbols) == 3
        
        # 5. Update strategy
        config_manager.update_strategy_config(
            'momentum',
            {'enabled': False, 'weight': 0.5},
            user="admin",
            reason="Disable momentum strategy"
        )
        
        # 6. Update risk config
        config_manager.update_risk_config(
            {'max_position_size': 0.015, 'daily_loss_limit': 0.03},
            user="admin",
            reason="Reduce risk"
        )
        
        # 7. Validate configuration
        validation_result = config_manager.validate_current_config()
        assert validation_result['valid'] is True
        
        # 8. Check change history
        history = config_manager.get_change_history()
        assert len(history) >= 3  # At least 3 changes made
        
        # 9. Export configuration
        export_file = f"{temp_config_dir}/final_config.json"
        config_manager.export_config(export_file)
        assert Path(export_file).exists()
        
        # 10. Restore from backup
        backups = config_manager.list_backups()
        initial_backup = backups[-1]  # Oldest backup (initial state)
        
        config_manager.restore_from_backup(
            initial_backup['version'],
            user="admin"
        )
        
        # 11. Verify restoration
        assert config_manager.current_config.trading_enabled is True
        assert config_manager.current_config.dry_run is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])