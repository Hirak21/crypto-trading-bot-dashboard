"""
Error recovery and resilience utilities for the crypto trading bot.

This module provides comprehensive error handling, automatic recovery,
state persistence, and system resilience features.
"""

import asyncio
import logging
import json
import pickle
from typing import Dict, Any, Optional, Callable, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import traceback
from dataclasses import dataclass, asdict
import threading
from collections import deque, defaultdict


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Recovery action types."""
    RETRY = "retry"
    RESTART_COMPONENT = "restart_component"
    RESTART_APPLICATION = "restart_application"
    MANUAL_INTERVENTION = "manual_intervention"
    IGNORE = "ignore"


@dataclass
class ErrorRecord:
    """Error record for tracking and analysis."""
    timestamp: datetime
    error_type: str
    error_message: str
    component: str
    severity: ErrorSeverity
    stack_trace: str
    context: Dict[str, Any]
    recovery_action: Optional[RecoveryAction] = None
    recovery_attempts: int = 0
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class ApplicationState:
    """Persistent application state."""
    timestamp: datetime
    component_states: Dict[str, Dict[str, Any]]
    active_positions: Dict[str, Any]
    portfolio_metrics: Dict[str, Any]
    configuration: Dict[str, Any]
    error_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]


class CircuitBreaker:
    """Circuit breaker pattern implementation for component resilience."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func):
        """Decorator to wrap functions with circuit breaker."""
        async def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                    self.logger.info(f"Circuit breaker HALF_OPEN for {func.__name__}")
                else:
                    raise Exception(f"Circuit breaker OPEN for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout))
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.failure_count = 0
            self.logger.info("Circuit breaker CLOSED - service recovered")
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self.logger.warning(f"Circuit breaker OPEN - failure threshold reached")


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)
    
    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Retry async function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = min(self.base_delay * (self.backoff_factor ** (attempt - 1)), self.max_delay)
                    self.logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                
                return await func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                
                if attempt == self.max_retries:
                    self.logger.error(f"All retry attempts failed for {func.__name__}")
                    break
        
        raise last_exception
    
    def retry_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Retry sync function with exponential backoff."""
        import time
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = min(self.base_delay * (self.backoff_factor ** (attempt - 1)), self.max_delay)
                    self.logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                
                if attempt == self.max_retries:
                    self.logger.error(f"All retry attempts failed for {func.__name__}")
                    break
        
        raise last_exception


class StateManager:
    """Manages persistent application state for recovery."""
    
    def __init__(self, state_dir: str = "state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        
        self.state_file = self.state_dir / "application_state.json"
        self.backup_dir = self.state_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def save_state(self, state: ApplicationState) -> bool:
        """Save application state to persistent storage."""
        try:
            with self._lock:
                # Create backup of current state
                if self.state_file.exists():
                    backup_file = self.backup_dir / f"state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.state_file.rename(backup_file)
                    
                    # Keep only last 10 backups
                    backups = sorted(self.backup_dir.glob("state_backup_*.json"))
                    for old_backup in backups[:-10]:
                        old_backup.unlink()
                
                # Save current state
                state_dict = asdict(state)
                # Convert datetime objects to ISO strings
                state_dict['timestamp'] = state.timestamp.isoformat()
                if state_dict.get('error_history'):
                    for error in state_dict['error_history']:
                        if 'timestamp' in error:
                            error['timestamp'] = error['timestamp'].isoformat() if isinstance(error['timestamp'], datetime) else error['timestamp']
                        if 'resolution_time' in error and error['resolution_time']:
                            error['resolution_time'] = error['resolution_time'].isoformat() if isinstance(error['resolution_time'], datetime) else error['resolution_time']
                
                with open(self.state_file, 'w') as f:
                    json.dump(state_dict, f, indent=2, default=str)
                
                self.logger.debug("Application state saved successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save application state: {e}")
            return False
    
    def load_state(self) -> Optional[ApplicationState]:
        """Load application state from persistent storage."""
        try:
            if not self.state_file.exists():
                self.logger.info("No saved state found")
                return None
            
            with open(self.state_file, 'r') as f:
                state_dict = json.load(f)
            
            # Convert ISO strings back to datetime objects
            state_dict['timestamp'] = datetime.fromisoformat(state_dict['timestamp'])
            if state_dict.get('error_history'):
                for error in state_dict['error_history']:
                    if 'timestamp' in error:
                        error['timestamp'] = datetime.fromisoformat(error['timestamp'])
                    if 'resolution_time' in error and error['resolution_time']:
                        error['resolution_time'] = datetime.fromisoformat(error['resolution_time'])
            
            state = ApplicationState(**state_dict)
            self.logger.info(f"Application state loaded from {state.timestamp}")
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to load application state: {e}")
            return None
    
    def clear_state(self) -> bool:
        """Clear saved application state."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
                self.logger.info("Application state cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear application state: {e}")
            return False


class ErrorRecoveryManager:
    """Comprehensive error recovery and resilience manager."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.component_errors = defaultdict(list)
        
        # Recovery configuration
        self.max_recovery_attempts = self.config.get('max_recovery_attempts', 3)
        self.recovery_cooldown = self.config.get('recovery_cooldown_minutes', 5)
        self.critical_error_threshold = self.config.get('critical_error_threshold', 5)
        
        # State management
        self.state_manager = StateManager(self.config.get('state_dir', 'state'))
        self.retry_manager = RetryManager()
        
        # Circuit breakers for components
        self.circuit_breakers = {}
        
        # Recovery callbacks
        self.recovery_callbacks = {}
        
        # Statistics
        self.recovery_stats = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'automatic_restarts': 0,
            'manual_interventions': 0
        }
    
    def register_component_recovery(self, component_name: str, recovery_callback: Callable):
        """Register recovery callback for a component."""
        self.recovery_callbacks[component_name] = recovery_callback
        self.circuit_breakers[component_name] = CircuitBreaker()
        self.logger.info(f"Registered recovery callback for {component_name}")
    
    async def handle_error(self, error: Exception, component: str, context: Dict[str, Any] = None) -> RecoveryAction:
        """Handle error and determine recovery action."""
        try:
            # Create error record
            error_record = ErrorRecord(
                timestamp=datetime.now(),
                error_type=type(error).__name__,
                error_message=str(error),
                component=component,
                severity=self._determine_severity(error, component),
                stack_trace=traceback.format_exc(),
                context=context or {}
            )
            
            # Add to history
            self.error_history.append(error_record)
            self.error_counts[error_record.error_type] += 1
            self.component_errors[component].append(error_record)
            self.recovery_stats['total_errors'] += 1
            
            # Log error
            self.logger.error(f"Error in {component}: {error_record.error_message}")
            self.logger.debug(f"Error context: {context}")
            
            # Determine recovery action
            recovery_action = self._determine_recovery_action(error_record)
            error_record.recovery_action = recovery_action
            
            # Execute recovery
            success = await self._execute_recovery(error_record)
            
            if success:
                error_record.resolved = True
                error_record.resolution_time = datetime.now()
                self.recovery_stats['successful_recoveries'] += 1
                self.logger.info(f"Successfully recovered from error in {component}")
            else:
                self.recovery_stats['failed_recoveries'] += 1
                self.logger.error(f"Failed to recover from error in {component}")
            
            return recovery_action
            
        except Exception as e:
            self.logger.error(f"Error in error recovery manager: {e}")
            return RecoveryAction.MANUAL_INTERVENTION
    
    def _determine_severity(self, error: Exception, component: str) -> ErrorSeverity:
        """Determine error severity based on error type and component."""
        # Critical errors
        critical_errors = [
            'ConnectionError',
            'AuthenticationError',
            'InsufficientFundsError',
            'SystemExit',
            'KeyboardInterrupt'
        ]
        
        if type(error).__name__ in critical_errors:
            return ErrorSeverity.CRITICAL
        
        # High severity for core components
        core_components = ['market_manager', 'trade_manager', 'risk_manager']
        if component in core_components:
            return ErrorSeverity.HIGH
        
        # Medium severity for network/API errors
        network_errors = ['TimeoutError', 'HTTPError', 'APIError']
        if type(error).__name__ in network_errors:
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def _determine_recovery_action(self, error_record: ErrorRecord) -> RecoveryAction:
        """Determine appropriate recovery action for error."""
        # Check error frequency
        recent_errors = [
            err for err in self.component_errors[error_record.component]
            if datetime.now() - err.timestamp < timedelta(minutes=10)
        ]
        
        # Critical errors or too many recent errors
        if (error_record.severity == ErrorSeverity.CRITICAL or 
            len(recent_errors) > self.critical_error_threshold):
            return RecoveryAction.RESTART_APPLICATION
        
        # High severity errors
        if error_record.severity == ErrorSeverity.HIGH:
            return RecoveryAction.RESTART_COMPONENT
        
        # Medium severity - retry
        if error_record.severity == ErrorSeverity.MEDIUM:
            return RecoveryAction.RETRY
        
        # Low severity - ignore
        return RecoveryAction.IGNORE
    
    async def _execute_recovery(self, error_record: ErrorRecord) -> bool:
        """Execute recovery action."""
        try:
            action = error_record.recovery_action
            component = error_record.component
            
            if action == RecoveryAction.IGNORE:
                return True
            
            elif action == RecoveryAction.RETRY:
                # Retry is handled by the retry manager in calling code
                return True
            
            elif action == RecoveryAction.RESTART_COMPONENT:
                if component in self.recovery_callbacks:
                    self.logger.info(f"Restarting component: {component}")
                    await self.recovery_callbacks[component]()
                    return True
                else:
                    self.logger.warning(f"No recovery callback for component: {component}")
                    return False
            
            elif action == RecoveryAction.RESTART_APPLICATION:
                self.logger.critical("Application restart required")
                self.recovery_stats['automatic_restarts'] += 1
                # This should trigger application restart
                return False
            
            elif action == RecoveryAction.MANUAL_INTERVENTION:
                self.logger.critical(f"Manual intervention required for {component}")
                self.recovery_stats['manual_interventions'] += 1
                return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing recovery action: {e}")
            return False
    
    def save_application_state(self, component_states: Dict[str, Dict[str, Any]], 
                             active_positions: Dict[str, Any],
                             portfolio_metrics: Dict[str, Any],
                             configuration: Dict[str, Any],
                             performance_metrics: Dict[str, Any]) -> bool:
        """Save current application state for recovery."""
        try:
            state = ApplicationState(
                timestamp=datetime.now(),
                component_states=component_states,
                active_positions=active_positions,
                portfolio_metrics=portfolio_metrics,
                configuration=configuration,
                error_history=[asdict(err) for err in list(self.error_history)[-100:]],  # Last 100 errors
                performance_metrics=performance_metrics
            )
            
            return self.state_manager.save_state(state)
            
        except Exception as e:
            self.logger.error(f"Failed to save application state: {e}")
            return False
    
    def load_application_state(self) -> Optional[ApplicationState]:
        """Load saved application state for recovery."""
        return self.state_manager.load_state()
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        recent_errors = [
            err for err in self.error_history
            if datetime.now() - err.timestamp < timedelta(hours=24)
        ]
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors_24h': len(recent_errors),
            'error_types': dict(self.error_counts),
            'component_error_counts': {
                comp: len(errors) for comp, errors in self.component_errors.items()
            },
            'recovery_stats': self.recovery_stats.copy(),
            'most_common_errors': [
                (error_type, count) for error_type, count in 
                sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
        }
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status for a specific component."""
        component_errors = self.component_errors.get(component, [])
        recent_errors = [
            err for err in component_errors
            if datetime.now() - err.timestamp < timedelta(hours=1)
        ]
        
        circuit_breaker = self.circuit_breakers.get(component)
        
        return {
            'component': component,
            'total_errors': len(component_errors),
            'recent_errors_1h': len(recent_errors),
            'circuit_breaker_state': circuit_breaker.state if circuit_breaker else 'N/A',
            'last_error': component_errors[-1].timestamp if component_errors else None,
            'error_rate_1h': len(recent_errors),  # errors per hour
            'health_status': 'healthy' if len(recent_errors) < 3 else 'degraded' if len(recent_errors) < 10 else 'unhealthy'
        }
    
    async def perform_health_check(self, components: Dict[str, Callable]) -> Dict[str, bool]:
        """Perform health check on all components."""
        health_results = {}
        
        for component_name, health_check_func in components.items():
            try:
                if asyncio.iscoroutinefunction(health_check_func):
                    is_healthy = await health_check_func()
                else:
                    is_healthy = health_check_func()
                
                health_results[component_name] = is_healthy
                
                if not is_healthy:
                    self.logger.warning(f"Health check failed for {component_name}")
                
            except Exception as e:
                self.logger.error(f"Health check error for {component_name}: {e}")
                health_results[component_name] = False
        
        return health_results
    
    def clear_error_history(self, component: str = None) -> None:
        """Clear error history for component or all components."""
        if component:
            self.component_errors[component].clear()
            self.logger.info(f"Cleared error history for {component}")
        else:
            self.error_history.clear()
            self.component_errors.clear()
            self.error_counts.clear()
            self.logger.info("Cleared all error history")


# Decorator for automatic error handling
def with_error_recovery(recovery_manager: ErrorRecoveryManager, component: str):
    """Decorator to add automatic error recovery to functions."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                await recovery_manager.handle_error(e, component, {
                    'function': func.__name__,
                    'args': str(args)[:200],  # Truncate for logging
                    'kwargs': str(kwargs)[:200]
                })
                raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we can't await, so we schedule the error handling
                asyncio.create_task(recovery_manager.handle_error(e, component, {
                    'function': func.__name__,
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200]
                }))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator