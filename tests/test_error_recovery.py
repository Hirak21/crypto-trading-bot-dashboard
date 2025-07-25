"""
Unit tests for error recovery system.

Tests error recovery manager, state persistence, component recovery,
and error handling mechanisms.
"""

import unittest
import asyncio
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, Callable

from crypto_trading_bot.utils.error_recovery import (
    ErrorRecoveryManager, ApplicationState, ComponentState,
    RecoveryStrategy, ErrorSeverity, with_error_recovery
)
from tests.test_mock_data import MockDataGenerator


class TestApplicationState(unittest.TestCase):
    """Test cases for ApplicationState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.component_states = {
            'market_manager': {'status': 'running', 'last_update': datetime.now()},
            'trade_manager': {'status': 'running', 'active_trades': 5}
        }
        self.active_positions = {
            'BTCUSDT': {'quantity': 0.1, 'entry_price': 50000}
        }
        self.portfolio_metrics = {'total_pnl': 100.0, 'win_rate': 0.6}
        self.configuration = {'api_key': 'test_key', 'symbols': ['BTCUSDT']}
        self.performance_metrics = {'total_trades': 10, 'uptime': 3600}
        
        self.app_state = ApplicationState(
            timestamp=datetime.now(),
            component_states=self.component_states,
            active_positions=self.active_positions,
            portfolio_metrics=self.portfolio_metrics,
            configuration=self.configuration,
            performance_metrics=self.performance_metrics
        )
    
    def test_initialization(self):
        """Test application state initialization."""
        self.assertIsInstance(self.app_state.timestamp, datetime)
        self.assertEqual(self.app_state.component_states, self.component_states)
        self.assertEqual(self.app_state.active_positions, self.active_positions)
        self.assertEqual(self.app_state.portfolio_metrics, self.portfolio_metrics)
        self.assertEqual(self.app_state.configuration, self.configuration)
        self.assertEqual(self.app_state.performance_metrics, self.performance_metrics)
    
    def test_to_dict(self):
        """Test converting application state to dictionary."""
        state_dict = self.app_state.to_dict()
        
        required_keys = [
            'timestamp', 'component_states', 'active_positions',
            'portfolio_metrics', 'configuration', 'performance_metrics'
        ]
        
        for key in required_keys:
            self.assertIn(key, state_dict)
        
        # Check timestamp serialization
        self.assertIsInstance(state_dict['timestamp'], str)
    
    def test_from_dict(self):
        """Test creating application state from dictionary."""
        state_dict = self.app_state.to_dict()
        restored_state = ApplicationState.from_dict(state_dict)
        
        self.assertEqual(restored_state.component_states, self.component_states)
        self.assertEqual(restored_state.active_positions, self.active_positions)
        self.assertIsInstance(restored_state.timestamp, datetime)
    
    def test_is_recent(self):
        """Test checking if state is recent."""
        # Recent state
        recent_state = ApplicationState(
            timestamp=datetime.now() - timedelta(minutes=5),
            component_states={},
            active_positions={},
            portfolio_metrics={},
            configuration={},
            performance_metrics={}
        )
        self.assertTrue(recent_state.is_recent(timedelta(hours=1)))
        
        # Old state
        old_state = ApplicationState(
            timestamp=datetime.now() - timedelta(hours=2),
            component_states={},
            active_positions={},
            portfolio_metrics={},
            configuration={},
            performance_metrics={}
        )
        self.assertFalse(old_state.is_recent(timedelta(hours=1)))


class TestComponentState(unittest.TestCase):
    """Test cases for ComponentState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.component_state = ComponentState(
            name="test_component",
            status="running",
            last_error=None,
            error_count=0,
            last_recovery=None,
            recovery_attempts=0,
            health_score=1.0,
            metadata={'key': 'value'}
        )
    
    def test_initialization(self):
        """Test component state initialization."""
        self.assertEqual(self.component_state.name, "test_component")
        self.assertEqual(self.component_state.status, "running")
        self.assertEqual(self.component_state.error_count, 0)
        self.assertEqual(self.component_state.health_score, 1.0)
    
    def test_record_error(self):
        """Test recording an error."""
        error = Exception("Test error")
        self.component_state.record_error(error)
        
        self.assertEqual(self.component_state.error_count, 1)
        self.assertEqual(self.component_state.last_error, str(error))
        self.assertIsInstance(self.component_state.last_error_time, datetime)
        self.assertLess(self.component_state.health_score, 1.0)
    
    def test_record_recovery(self):
        """Test recording a recovery."""
        self.component_state.record_recovery()
        
        self.assertEqual(self.component_state.recovery_attempts, 1)
        self.assertIsInstance(self.component_state.last_recovery, datetime)
    
    def test_reset_errors(self):
        """Test resetting error count."""
        # Record some errors first
        self.component_state.record_error(Exception("Error 1"))
        self.component_state.record_error(Exception("Error 2"))
        
        self.assertEqual(self.component_state.error_count, 2)
        
        # Reset errors
        self.component_state.reset_errors()
        
        self.assertEqual(self.component_state.error_count, 0)
        self.assertIsNone(self.component_state.last_error)
        self.assertEqual(self.component_state.health_score, 1.0)
    
    def test_is_healthy(self):
        """Test health check."""
        # Healthy component
        self.assertTrue(self.component_state.is_healthy())
        
        # Unhealthy component (many errors)
        for i in range(10):
            self.component_state.record_error(Exception(f"Error {i}"))
        
        self.assertFalse(self.component_state.is_healthy())
    
    def test_to_dict(self):
        """Test converting component state to dictionary."""
        state_dict = self.component_state.to_dict()
        
        required_keys = [
            'name', 'status', 'error_count', 'recovery_attempts',
            'health_score', 'metadata'
        ]
        
        for key in required_keys:
            self.assertIn(key, state_dict)


class TestErrorRecoveryManager(unittest.TestCase):
    """Test cases for ErrorRecoveryManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'state_dir': self.temp_dir,
            'max_recovery_attempts': 3,
            'recovery_cooldown_minutes': 5,
            'critical_error_threshold': 5,
            'state_save_interval_minutes': 10
        }
        self.recovery_manager = ErrorRecoveryManager(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test error recovery manager initialization."""
        self.assertEqual(self.recovery_manager.config, self.config)
        self.assertEqual(self.recovery_manager.max_recovery_attempts, 3)
        self.assertEqual(self.recovery_manager.critical_error_threshold, 5)
        self.assertIsInstance(self.recovery_manager.component_states, dict)
    
    def test_register_component_recovery(self):
        """Test registering component recovery callback."""
        async def mock_recovery():
            pass
        
        self.recovery_manager.register_component_recovery('test_component', mock_recovery)
        
        self.assertIn('test_component', self.recovery_manager.recovery_callbacks)
        self.assertEqual(self.recovery_manager.recovery_callbacks['test_component'], mock_recovery)
    
    async def test_handle_error_first_time(self):
        """Test handling error for the first time."""
        error = Exception("Test error")
        component = "test_component"
        context = {'operation': 'test_operation'}
        
        # Register mock recovery callback
        mock_recovery = AsyncMock()
        self.recovery_manager.register_component_recovery(component, mock_recovery)
        
        result = await self.recovery_manager.handle_error(error, component, context)
        
        self.assertTrue(result)
        self.assertIn(component, self.recovery_manager.component_states)
        self.assertEqual(self.recovery_manager.component_states[component].error_count, 1)
        mock_recovery.assert_called_once()
    
    async def test_handle_error_max_attempts_exceeded(self):
        """Test handling error when max attempts exceeded."""
        error = Exception("Test error")
        component = "test_component"
        context = {}
        
        # Register mock recovery callback
        mock_recovery = AsyncMock()
        self.recovery_manager.register_component_recovery(component, mock_recovery)
        
        # Simulate multiple errors to exceed max attempts
        component_state = ComponentState(component, "error", None, 0, None, 0, 1.0, {})
        component_state.recovery_attempts = self.config['max_recovery_attempts']
        self.recovery_manager.component_states[component] = component_state
        
        result = await self.recovery_manager.handle_error(error, component, context)
        
        self.assertFalse(result)
        mock_recovery.assert_not_called()
    
    async def test_handle_error_cooldown_period(self):
        """Test handling error during cooldown period."""
        error = Exception("Test error")
        component = "test_component"
        context = {}
        
        # Register mock recovery callback
        mock_recovery = AsyncMock()
        self.recovery_manager.register_component_recovery(component, mock_recovery)
        
        # Set recent recovery time
        component_state = ComponentState(component, "recovering", None, 1, datetime.now(), 1, 0.8, {})
        self.recovery_manager.component_states[component] = component_state
        
        result = await self.recovery_manager.handle_error(error, component, context)
        
        self.assertFalse(result)
        mock_recovery.assert_not_called()
    
    def test_save_application_state(self):
        """Test saving application state."""
        component_states = {'test_component': {'status': 'running'}}
        active_positions = {'BTCUSDT': {'quantity': 0.1}}
        portfolio_metrics = {'pnl': 100.0}
        configuration = {'api_key': 'test'}
        performance_metrics = {'trades': 10}
        
        success = self.recovery_manager.save_application_state(
            component_states=component_states,
            active_positions=active_positions,
            portfolio_metrics=portfolio_metrics,
            configuration=configuration,
            performance_metrics=performance_metrics
        )
        
        self.assertTrue(success)
        
        # Check that state file was created
        state_file = os.path.join(self.temp_dir, 'application_state.json')
        self.assertTrue(os.path.exists(state_file))
    
    def test_load_application_state(self):
        """Test loading application state."""
        # First save a state
        component_states = {'test_component': {'status': 'running'}}
        active_positions = {'BTCUSDT': {'quantity': 0.1}}
        
        self.recovery_manager.save_application_state(
            component_states=component_states,
            active_positions=active_positions,
            portfolio_metrics={},
            configuration={},
            performance_metrics={}
        )
        
        # Then load it
        loaded_state = self.recovery_manager.load_application_state()
        
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state.component_states, component_states)
        self.assertEqual(loaded_state.active_positions, active_positions)
    
    def test_load_application_state_no_file(self):
        """Test loading application state when no file exists."""
        loaded_state = self.recovery_manager.load_application_state()
        self.assertIsNone(loaded_state)
    
    async def test_perform_health_check(self):
        """Test performing health check."""
        health_checks = {
            'component1': lambda: True,
            'component2': lambda: False,
            'component3': lambda: True
        }
        
        results = await self.recovery_manager.perform_health_check(health_checks)
        
        self.assertEqual(results['component1'], True)
        self.assertEqual(results['component2'], False)
        self.assertEqual(results['component3'], True)
    
    async def test_perform_health_check_with_exception(self):
        """Test health check with exception in check function."""
        def failing_check():
            raise Exception("Health check failed")
        
        health_checks = {
            'component1': lambda: True,
            'failing_component': failing_check
        }
        
        results = await self.recovery_manager.perform_health_check(health_checks)
        
        self.assertEqual(results['component1'], True)
        self.assertEqual(results['failing_component'], False)
    
    def test_get_error_statistics(self):
        """Test getting error statistics."""
        # Add some component states with errors
        component1 = ComponentState("component1", "running", None, 2, None, 1, 0.8, {})
        component2 = ComponentState("component2", "error", "Test error", 5, None, 2, 0.5, {})
        
        self.recovery_manager.component_states['component1'] = component1
        self.recovery_manager.component_states['component2'] = component2
        
        stats = self.recovery_manager.get_error_statistics()
        
        self.assertIn('total_errors', stats)
        self.assertIn('total_recoveries', stats)
        self.assertIn('component_stats', stats)
        self.assertEqual(stats['total_errors'], 7)  # 2 + 5
        self.assertEqual(stats['total_recoveries'], 3)  # 1 + 2
    
    def test_get_component_health(self):
        """Test getting component health."""
        # Add component state
        component_state = ComponentState("test_component", "running", None, 1, None, 0, 0.9, {})
        self.recovery_manager.component_states['test_component'] = component_state
        
        health = self.recovery_manager.get_component_health('test_component')
        
        self.assertIn('name', health)
        self.assertIn('status', health)
        self.assertIn('health_score', health)
        self.assertIn('error_count', health)
        self.assertEqual(health['name'], 'test_component')
        self.assertEqual(health['health_score'], 0.9)
    
    def test_get_component_health_nonexistent(self):
        """Test getting health for non-existent component."""
        health = self.recovery_manager.get_component_health('nonexistent')
        
        self.assertIn('name', health)
        self.assertIn('status', health)
        self.assertEqual(health['name'], 'nonexistent')
        self.assertEqual(health['status'], 'unknown')
    
    def test_reset_component_errors(self):
        """Test resetting component errors."""
        # Add component with errors
        component_state = ComponentState("test_component", "error", "Test error", 5, None, 2, 0.5, {})
        self.recovery_manager.component_states['test_component'] = component_state
        
        self.recovery_manager.reset_component_errors('test_component')
        
        self.assertEqual(component_state.error_count, 0)
        self.assertIsNone(component_state.last_error)
        self.assertEqual(component_state.health_score, 1.0)
    
    def test_is_component_in_cooldown(self):
        """Test checking if component is in cooldown."""
        component = "test_component"
        
        # Component not in cooldown (no recent recovery)
        component_state = ComponentState(component, "running", None, 0, None, 0, 1.0, {})
        self.recovery_manager.component_states[component] = component_state
        
        self.assertFalse(self.recovery_manager._is_component_in_cooldown(component))
        
        # Component in cooldown (recent recovery)
        component_state.last_recovery = datetime.now()
        
        self.assertTrue(self.recovery_manager._is_component_in_cooldown(component))
    
    def test_determine_recovery_strategy(self):
        """Test determining recovery strategy."""
        # Low error count - restart strategy
        strategy = self.recovery_manager._determine_recovery_strategy(2, 1)
        self.assertEqual(strategy, RecoveryStrategy.RESTART)
        
        # Medium error count - reinitialize strategy
        strategy = self.recovery_manager._determine_recovery_strategy(4, 2)
        self.assertEqual(strategy, RecoveryStrategy.REINITIALIZE)
        
        # High error count - failsafe strategy
        strategy = self.recovery_manager._determine_recovery_strategy(6, 3)
        self.assertEqual(strategy, RecoveryStrategy.FAILSAFE)
    
    def test_classify_error_severity(self):
        """Test classifying error severity."""
        # Connection error - high severity
        connection_error = ConnectionError("Connection failed")
        severity = self.recovery_manager._classify_error_severity(connection_error)
        self.assertEqual(severity, ErrorSeverity.HIGH)
        
        # Timeout error - medium severity
        timeout_error = TimeoutError("Request timed out")
        severity = self.recovery_manager._classify_error_severity(timeout_error)
        self.assertEqual(severity, ErrorSeverity.MEDIUM)
        
        # Generic exception - low severity
        generic_error = Exception("Generic error")
        severity = self.recovery_manager._classify_error_severity(generic_error)
        self.assertEqual(severity, ErrorSeverity.LOW)


class TestErrorRecoveryDecorator(unittest.TestCase):
    """Test cases for error recovery decorator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'state_dir': self.temp_dir,
            'max_recovery_attempts': 3,
            'recovery_cooldown_minutes': 5,
            'critical_error_threshold': 5
        }
        self.recovery_manager = ErrorRecoveryManager(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_decorator_success(self):
        """Test decorator with successful function execution."""
        @with_error_recovery(self.recovery_manager, 'test_component')
        async def test_function():
            return "success"
        
        result = await test_function()
        self.assertEqual(result, "success")
    
    async def test_decorator_with_error(self):
        """Test decorator with function that raises error."""
        # Register mock recovery callback
        mock_recovery = AsyncMock()
        self.recovery_manager.register_component_recovery('test_component', mock_recovery)
        
        call_count = 0
        
        @with_error_recovery(self.recovery_manager, 'test_component')
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test error")
            return "success after recovery"
        
        result = await test_function()
        
        self.assertEqual(result, "success after recovery")
        self.assertEqual(call_count, 2)  # Original call + retry after recovery
        mock_recovery.assert_called_once()
    
    async def test_decorator_max_retries_exceeded(self):
        """Test decorator when max retries are exceeded."""
        # Register mock recovery callback that always fails
        mock_recovery = AsyncMock(side_effect=Exception("Recovery failed"))
        self.recovery_manager.register_component_recovery('test_component', mock_recovery)
        
        @with_error_recovery(self.recovery_manager, 'test_component', max_retries=2)
        async def test_function():
            raise Exception("Persistent error")
        
        with self.assertRaises(Exception) as context:
            await test_function()
        
        self.assertIn("Persistent error", str(context.exception))
    
    async def test_decorator_with_context(self):
        """Test decorator with context information."""
        mock_recovery = AsyncMock()
        self.recovery_manager.register_component_recovery('test_component', mock_recovery)
        
        @with_error_recovery(self.recovery_manager, 'test_component', context={'operation': 'test_op'})
        async def test_function():
            raise Exception("Test error")
        
        with self.assertRaises(Exception):
            await test_function()
        
        # Verify that error was handled with context
        self.assertIn('test_component', self.recovery_manager.component_states)
        component_state = self.recovery_manager.component_states['test_component']
        self.assertEqual(component_state.error_count, 1)


class TestRecoveryStrategies(unittest.TestCase):
    """Test cases for recovery strategies."""
    
    def test_recovery_strategy_enum(self):
        """Test RecoveryStrategy enum values."""
        self.assertEqual(RecoveryStrategy.RESTART.value, "restart")
        self.assertEqual(RecoveryStrategy.REINITIALIZE.value, "reinitialize")
        self.assertEqual(RecoveryStrategy.FAILSAFE.value, "failsafe")
        self.assertEqual(RecoveryStrategy.IGNORE.value, "ignore")
    
    def test_error_severity_enum(self):
        """Test ErrorSeverity enum values."""
        self.assertEqual(ErrorSeverity.LOW.value, "low")
        self.assertEqual(ErrorSeverity.MEDIUM.value, "medium")
        self.assertEqual(ErrorSeverity.HIGH.value, "high")
        self.assertEqual(ErrorSeverity.CRITICAL.value, "critical")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios for error recovery system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'state_dir': self.temp_dir,
            'max_recovery_attempts': 3,
            'recovery_cooldown_minutes': 1,  # Short cooldown for testing
            'critical_error_threshold': 3
        }
        self.recovery_manager = ErrorRecoveryManager(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_component_failure_and_recovery_cycle(self):
        """Test complete component failure and recovery cycle."""
        component = "market_manager"
        recovery_call_count = 0
        
        async def mock_recovery():
            nonlocal recovery_call_count
            recovery_call_count += 1
            if recovery_call_count <= 2:
                # Simulate recovery failure for first two attempts
                raise Exception("Recovery failed")
            # Third attempt succeeds
            return True
        
        self.recovery_manager.register_component_recovery(component, mock_recovery)
        
        # Simulate multiple errors
        error1 = Exception("Connection lost")
        error2 = Exception("API timeout")
        error3 = Exception("Data corruption")
        
        # First error - should trigger recovery
        result1 = await self.recovery_manager.handle_error(error1, component, {})
        self.assertFalse(result1)  # Recovery failed
        
        # Second error - should trigger recovery again
        result2 = await self.recovery_manager.handle_error(error2, component, {})
        self.assertFalse(result2)  # Recovery failed again
        
        # Third error - should trigger successful recovery
        result3 = await self.recovery_manager.handle_error(error3, component, {})
        self.assertTrue(result3)  # Recovery succeeded
        
        # Verify component state
        component_state = self.recovery_manager.component_states[component]
        self.assertEqual(component_state.error_count, 3)
        self.assertEqual(component_state.recovery_attempts, 3)
        self.assertEqual(recovery_call_count, 3)
    
    async def test_state_persistence_across_restarts(self):
        """Test state persistence across application restarts."""
        # Create initial state
        component_states = {
            'market_manager': {'status': 'running', 'errors': 2},
            'trade_manager': {'status': 'error', 'last_error': 'Connection failed'}
        }
        active_positions = {
            'BTCUSDT': {'quantity': 0.1, 'entry_price': 50000, 'pnl': 100}
        }
        
        # Save state
        success = self.recovery_manager.save_application_state(
            component_states=component_states,
            active_positions=active_positions,
            portfolio_metrics={'total_pnl': 100},
            configuration={'api_key': 'test'},
            performance_metrics={'uptime': 3600}
        )
        self.assertTrue(success)
        
        # Simulate application restart by creating new recovery manager
        new_recovery_manager = ErrorRecoveryManager(self.config)
        
        # Load state
        loaded_state = new_recovery_manager.load_application_state()
        
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state.component_states, component_states)
        self.assertEqual(loaded_state.active_positions, active_positions)
        self.assertEqual(loaded_state.portfolio_metrics['total_pnl'], 100)
    
    async def test_cascading_failure_scenario(self):
        """Test handling cascading failures across multiple components."""
        components = ['market_manager', 'trade_manager', 'portfolio_manager']
        recovery_calls = {comp: 0 for comp in components}
        
        async def create_recovery_callback(component_name):
            async def recovery_callback():
                recovery_calls[component_name] += 1
                # Simulate that recovery of one component might fail others
                if component_name == 'market_manager' and recovery_calls[component_name] == 1:
                    # First market manager recovery triggers trade manager failure
                    await self.recovery_manager.handle_error(
                        Exception("Cascade failure"), 'trade_manager', {}
                    )
                return True
            return recovery_callback
        
        # Register recovery callbacks
        for component in components:
            callback = await create_recovery_callback(component)
            self.recovery_manager.register_component_recovery(component, callback)
        
        # Trigger initial failure
        initial_error = Exception("Market data connection lost")
        result = await self.recovery_manager.handle_error(initial_error, 'market_manager', {})
        
        # Verify that cascading recovery was handled
        self.assertTrue(result)
        self.assertEqual(recovery_calls['market_manager'], 1)
        self.assertEqual(recovery_calls['trade_manager'], 1)  # Triggered by cascade
        
        # Verify component states
        for component in ['market_manager', 'trade_manager']:
            self.assertIn(component, self.recovery_manager.component_states)
            self.assertGreaterEqual(self.recovery_manager.component_states[component].error_count, 1)


if __name__ == '__main__':
    # Custom test runner for async tests
    class AsyncTestRunner:
        def run_async_tests(self):
            # Get all test classes
            test_classes = [
                TestApplicationState,
                TestComponentState,
                TestErrorRecoveryManager,
                TestErrorRecoveryDecorator,
                TestRecoveryStrategies,
                TestIntegrationScenarios
            ]
            
            # Run synchronous tests normally
            sync_test_classes = [TestApplicationState, TestComponentState, TestRecoveryStrategies]
            for test_class in sync_test_classes:
                suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
                unittest.TextTestRunner(verbosity=2).run(suite)
            
            # Run async tests with event loop
            async_test_classes = [
                TestErrorRecoveryManager,
                TestErrorRecoveryDecorator,
                TestIntegrationScenarios
            ]
            
            for test_class in async_test_classes:
                suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
                for test in suite:
                    if hasattr(test, '_testMethodName'):
                        method = getattr(test, test._testMethodName)
                        if asyncio.iscoroutinefunction(method):
                            # Run async test
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                loop.run_until_complete(method())
                                print(f"✓ {test_class.__name__}.{test._testMethodName}")
                            except Exception as e:
                                print(f"✗ {test_class.__name__}.{test._testMethodName}: {e}")
                            finally:
                                loop.close()
                        else:
                            # Run sync test normally
                            unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([test]))
    
    if __name__ == '__main__':
        runner = AsyncTestRunner()
        runner.run_async_tests()
    else:
        unittest.main()