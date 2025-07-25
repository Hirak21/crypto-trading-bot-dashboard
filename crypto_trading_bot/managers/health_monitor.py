"""
System Health Monitoring for Crypto Trading Bot

This module provides comprehensive system health monitoring including:
- Connection health monitoring
- Performance metrics tracking
- Error rate monitoring and alerting
- Resource usage monitoring
- System availability tracking
"""

import asyncio
import logging
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import deque, defaultdict
from enum import Enum

from .notification_manager import NotificationManager, Alert, AlertType, AlertLevel


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Represents a health metric"""
    name: str
    value: float
    status: HealthStatus
    timestamp: datetime = field(default_factory=datetime.now)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: str = ""
    description: str = ""


@dataclass
class ConnectionHealth:
    """Connection health information"""
    service: str
    is_connected: bool
    last_successful: Optional[datetime] = None
    last_failed: Optional[datetime] = None
    consecutive_failures: int = 0
    total_failures: int = 0
    uptime_percentage: float = 0.0
    response_time_ms: Optional[float] = None


class PerformanceTracker:
    """Tracks system performance metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
    def record_metric(self, name: str, value: float, timestamp: Optional[datetime] = None):
        """Record a performance metric"""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.metrics[name].append(value)
        self.timestamps[name].append(timestamp)
    
    def get_average(self, name: str, duration: Optional[timedelta] = None) -> Optional[float]:
        """Get average value for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None
            
        if duration is None:
            return sum(self.metrics[name]) / len(self.metrics[name])
        
        # Filter by time window
        cutoff = datetime.now() - duration
        values = []
        for value, timestamp in zip(self.metrics[name], self.timestamps[name]):
            if timestamp >= cutoff:
                values.append(value)
        
        return sum(values) / len(values) if values else None
    
    def get_percentile(self, name: str, percentile: float, duration: Optional[timedelta] = None) -> Optional[float]:
        """Get percentile value for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None
            
        values = list(self.metrics[name])
        if duration is not None:
            cutoff = datetime.now() - duration
            values = [
                value for value, timestamp in zip(self.metrics[name], self.timestamps[name])
                if timestamp >= cutoff
            ]
        
        if not values:
            return None
            
        values.sort()
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None
        return self.metrics[name][-1]


class ErrorRateMonitor:
    """Monitors error rates and patterns"""
    
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.errors: Dict[str, deque] = defaultdict(lambda: deque())
        self.total_requests: Dict[str, deque] = defaultdict(lambda: deque())
        
    def record_request(self, component: str, is_error: bool = False):
        """Record a request and whether it was an error"""
        timestamp = datetime.now()
        
        # Clean old entries
        self._clean_old_entries(component, timestamp)
        
        # Record request
        self.total_requests[component].append(timestamp)
        if is_error:
            self.errors[component].append(timestamp)
    
    def get_error_rate(self, component: str) -> float:
        """Get current error rate for a component"""
        timestamp = datetime.now()
        self._clean_old_entries(component, timestamp)
        
        total = len(self.total_requests[component])
        errors = len(self.errors[component])
        
        return errors / total if total > 0 else 0.0
    
    def get_error_count(self, component: str) -> int:
        """Get current error count for a component"""
        timestamp = datetime.now()
        self._clean_old_entries(component, timestamp)
        return len(self.errors[component])
    
    def _clean_old_entries(self, component: str, current_time: datetime):
        """Remove entries older than the window"""
        cutoff = current_time - timedelta(minutes=self.window_minutes)
        
        # Clean errors
        while (self.errors[component] and 
               self.errors[component][0] < cutoff):
            self.errors[component].popleft()
        
        # Clean total requests
        while (self.total_requests[component] and 
               self.total_requests[component][0] < cutoff):
            self.total_requests[component].popleft()


class ResourceMonitor:
    """Monitors system resource usage"""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return self.process.cpu_percent()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': memory_percent
        }
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage information"""
        disk_usage = psutil.disk_usage('/')
        return {
            'total_gb': disk_usage.total / 1024 / 1024 / 1024,
            'used_gb': disk_usage.used / 1024 / 1024 / 1024,
            'free_gb': disk_usage.free / 1024 / 1024 / 1024,
            'percent': (disk_usage.used / disk_usage.total) * 100
        }
    
    def get_network_io(self) -> Dict[str, int]:
        """Get network I/O statistics"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }


class ConnectionMonitor:
    """Monitors connection health for various services"""
    
    def __init__(self):
        self.connections: Dict[str, ConnectionHealth] = {}
        self.check_intervals: Dict[str, int] = {}  # seconds
        self.check_functions: Dict[str, Callable] = {}
        
    def register_connection(self, service: str, check_function: Callable, 
                          check_interval: int = 60):
        """Register a connection to monitor"""
        self.connections[service] = ConnectionHealth(
            service=service,
            is_connected=False
        )
        self.check_functions[service] = check_function
        self.check_intervals[service] = check_interval
    
    async def check_connection(self, service: str) -> bool:
        """Check connection status for a service"""
        if service not in self.connections:
            return False
            
        connection = self.connections[service]
        check_function = self.check_functions[service]
        
        try:
            start_time = time.time()
            is_connected = await check_function()
            response_time = (time.time() - start_time) * 1000  # ms
            
            connection.response_time_ms = response_time
            
            if is_connected:
                connection.is_connected = True
                connection.last_successful = datetime.now()
                connection.consecutive_failures = 0
            else:
                connection.is_connected = False
                connection.last_failed = datetime.now()
                connection.consecutive_failures += 1
                connection.total_failures += 1
                
            return is_connected
            
        except Exception as e:
            connection.is_connected = False
            connection.last_failed = datetime.now()
            connection.consecutive_failures += 1
            connection.total_failures += 1
            logging.error(f"Connection check failed for {service}: {e}")
            return False
    
    def get_connection_status(self, service: str) -> Optional[ConnectionHealth]:
        """Get connection status for a service"""
        return self.connections.get(service)
    
    def calculate_uptime(self, service: str, duration: timedelta) -> float:
        """Calculate uptime percentage for a service over a duration"""
        # This is a simplified calculation
        # In a real implementation, you'd track detailed uptime history
        connection = self.connections.get(service)
        if not connection:
            return 0.0
            
        if connection.is_connected:
            return 100.0
        else:
            # Estimate based on failure rate
            if connection.total_failures == 0:
                return 100.0
            
            # Simple estimation - in practice you'd want more detailed tracking
            failure_rate = connection.consecutive_failures / max(connection.total_failures, 1)
            return max(0.0, 100.0 - (failure_rate * 100))


class HealthMonitor:
    """Main system health monitoring class"""
    
    def __init__(self, notification_manager: NotificationManager, config: Dict[str, Any]):
        self.notification_manager = notification_manager
        self.config = config
        
        # Initialize monitoring components
        self.performance_tracker = PerformanceTracker()
        self.error_monitor = ErrorRateMonitor()
        self.resource_monitor = ResourceMonitor()
        self.connection_monitor = ConnectionMonitor()
        
        # Configuration
        self.check_interval = config.get('check_interval', 60)  # seconds
        self.cpu_warning_threshold = config.get('cpu_warning_threshold', 80.0)
        self.cpu_critical_threshold = config.get('cpu_critical_threshold', 95.0)
        self.memory_warning_threshold = config.get('memory_warning_threshold', 80.0)
        self.memory_critical_threshold = config.get('memory_critical_threshold', 95.0)
        self.error_rate_warning = config.get('error_rate_warning', 0.05)  # 5%
        self.error_rate_critical = config.get('error_rate_critical', 0.15)  # 15%
        
        # Monitoring state
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_health_check = datetime.now()
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start health monitoring"""
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Health monitoring started")
    
    async def stop(self):
        """Stop health monitoring"""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        self.last_health_check = datetime.now()
        
        # Check system resources
        await self._check_system_resources()
        
        # Check error rates
        await self._check_error_rates()
        
        # Check connections
        await self._check_connections()
        
        # Check performance metrics
        await self._check_performance_metrics()
    
    async def _check_system_resources(self):
        """Check system resource usage"""
        # CPU usage
        cpu_usage = self.resource_monitor.get_cpu_usage()
        self.performance_tracker.record_metric('cpu_usage', cpu_usage)
        
        if cpu_usage >= self.cpu_critical_threshold:
            self.notification_manager.send_alert(Alert(
                type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.CRITICAL,
                title="Critical CPU Usage",
                message=f"CPU usage at {cpu_usage:.1f}% (critical threshold: {self.cpu_critical_threshold:.1f}%)",
                data={'cpu_usage': cpu_usage, 'threshold': self.cpu_critical_threshold}
            ))
        elif cpu_usage >= self.cpu_warning_threshold:
            self.notification_manager.send_alert(Alert(
                type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.WARNING,
                title="High CPU Usage",
                message=f"CPU usage at {cpu_usage:.1f}% (warning threshold: {self.cpu_warning_threshold:.1f}%)",
                data={'cpu_usage': cpu_usage, 'threshold': self.cpu_warning_threshold}
            ))
        
        # Memory usage
        memory_info = self.resource_monitor.get_memory_usage()
        memory_percent = memory_info['percent']
        self.performance_tracker.record_metric('memory_usage', memory_percent)
        
        if memory_percent >= self.memory_critical_threshold:
            self.notification_manager.send_alert(Alert(
                type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.CRITICAL,
                title="Critical Memory Usage",
                message=f"Memory usage at {memory_percent:.1f}% ({memory_info['rss_mb']:.1f} MB)",
                data={'memory_percent': memory_percent, 'memory_mb': memory_info['rss_mb']}
            ))
        elif memory_percent >= self.memory_warning_threshold:
            self.notification_manager.send_alert(Alert(
                type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.WARNING,
                title="High Memory Usage",
                message=f"Memory usage at {memory_percent:.1f}% ({memory_info['rss_mb']:.1f} MB)",
                data={'memory_percent': memory_percent, 'memory_mb': memory_info['rss_mb']}
            ))
    
    async def _check_error_rates(self):
        """Check error rates for all monitored components"""
        for component in self.error_monitor.total_requests.keys():
            error_rate = self.error_monitor.get_error_rate(component)
            error_count = self.error_monitor.get_error_count(component)
            
            self.performance_tracker.record_metric(f'{component}_error_rate', error_rate)
            
            if error_rate >= self.error_rate_critical:
                self.notification_manager.send_alert(Alert(
                    type=AlertType.SYSTEM_ERROR,
                    level=AlertLevel.CRITICAL,
                    title=f"Critical Error Rate - {component}",
                    message=f"Error rate: {error_rate:.1%} ({error_count} errors in last hour)",
                    data={
                        'component': component,
                        'error_rate': error_rate,
                        'error_count': error_count
                    }
                ))
            elif error_rate >= self.error_rate_warning:
                self.notification_manager.send_alert(Alert(
                    type=AlertType.SYSTEM_ERROR,
                    level=AlertLevel.WARNING,
                    title=f"High Error Rate - {component}",
                    message=f"Error rate: {error_rate:.1%} ({error_count} errors in last hour)",
                    data={
                        'component': component,
                        'error_rate': error_rate,
                        'error_count': error_count
                    }
                ))
    
    async def _check_connections(self):
        """Check all registered connections"""
        for service in self.connection_monitor.connections.keys():
            is_connected = await self.connection_monitor.check_connection(service)
            connection = self.connection_monitor.get_connection_status(service)
            
            if connection:
                # Alert on connection loss
                if not is_connected and connection.consecutive_failures == 1:
                    self.notification_manager.connection_lost(service)
                
                # Alert on connection restoration
                elif is_connected and connection.consecutive_failures == 0 and connection.last_failed:
                    # Only if we had a recent failure
                    if (datetime.now() - connection.last_failed).total_seconds() < 300:  # 5 minutes
                        self.notification_manager.connection_restored(service)
                
                # Track response time
                if connection.response_time_ms:
                    self.performance_tracker.record_metric(
                        f'{service}_response_time', 
                        connection.response_time_ms
                    )
    
    async def _check_performance_metrics(self):
        """Check performance metrics for anomalies"""
        # Check for performance degradation
        for metric_name in ['api_response_time', 'signal_generation_time', 'trade_execution_time']:
            current_avg = self.performance_tracker.get_average(metric_name, timedelta(minutes=15))
            historical_avg = self.performance_tracker.get_average(metric_name, timedelta(hours=24))
            
            if current_avg and historical_avg and current_avg > historical_avg * 2:
                self.notification_manager.send_alert(Alert(
                    type=AlertType.PERFORMANCE_ALERT,
                    level=AlertLevel.WARNING,
                    title=f"Performance Degradation - {metric_name}",
                    message=f"Current avg: {current_avg:.2f}ms vs historical: {historical_avg:.2f}ms",
                    data={
                        'metric': metric_name,
                        'current_avg': current_avg,
                        'historical_avg': historical_avg
                    }
                ))
    
    # Public methods for recording metrics
    def record_api_request(self, component: str, is_error: bool = False, response_time_ms: Optional[float] = None):
        """Record an API request"""
        self.error_monitor.record_request(component, is_error)
        if response_time_ms is not None:
            self.performance_tracker.record_metric(f'{component}_response_time', response_time_ms)
    
    def record_signal_generation_time(self, strategy: str, time_ms: float):
        """Record signal generation time"""
        self.performance_tracker.record_metric(f'{strategy}_signal_time', time_ms)
        self.performance_tracker.record_metric('signal_generation_time', time_ms)
    
    def record_trade_execution_time(self, time_ms: float):
        """Record trade execution time"""
        self.performance_tracker.record_metric('trade_execution_time', time_ms)
    
    def register_connection_check(self, service: str, check_function: Callable, check_interval: int = 60):
        """Register a connection to monitor"""
        self.connection_monitor.register_connection(service, check_function, check_interval)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': HealthStatus.HEALTHY.value,
            'system_resources': {
                'cpu_usage': self.performance_tracker.get_latest('cpu_usage'),
                'memory_usage': self.performance_tracker.get_latest('memory_usage'),
            },
            'connections': {},
            'error_rates': {},
            'performance_metrics': {}
        }
        
        # Connection status
        for service, connection in self.connection_monitor.connections.items():
            summary['connections'][service] = {
                'is_connected': connection.is_connected,
                'consecutive_failures': connection.consecutive_failures,
                'response_time_ms': connection.response_time_ms
            }
            
            if not connection.is_connected:
                summary['overall_status'] = HealthStatus.WARNING.value
        
        # Error rates
        for component in self.error_monitor.total_requests.keys():
            error_rate = self.error_monitor.get_error_rate(component)
            summary['error_rates'][component] = error_rate
            
            if error_rate >= self.error_rate_critical:
                summary['overall_status'] = HealthStatus.CRITICAL.value
            elif error_rate >= self.error_rate_warning and summary['overall_status'] == HealthStatus.HEALTHY.value:
                summary['overall_status'] = HealthStatus.WARNING.value
        
        # Performance metrics
        for metric in ['signal_generation_time', 'trade_execution_time']:
            avg_time = self.performance_tracker.get_average(metric, timedelta(minutes=15))
            if avg_time:
                summary['performance_metrics'][metric] = avg_time
        
        return summary
    
    def get_health_metrics(self) -> List[HealthMetric]:
        """Get detailed health metrics"""
        metrics = []
        
        # CPU metric
        cpu_usage = self.performance_tracker.get_latest('cpu_usage')
        if cpu_usage is not None:
            status = HealthStatus.HEALTHY
            if cpu_usage >= self.cpu_critical_threshold:
                status = HealthStatus.CRITICAL
            elif cpu_usage >= self.cpu_warning_threshold:
                status = HealthStatus.WARNING
                
            metrics.append(HealthMetric(
                name="CPU Usage",
                value=cpu_usage,
                status=status,
                threshold_warning=self.cpu_warning_threshold,
                threshold_critical=self.cpu_critical_threshold,
                unit="%",
                description="Current CPU usage percentage"
            ))
        
        # Memory metric
        memory_usage = self.performance_tracker.get_latest('memory_usage')
        if memory_usage is not None:
            status = HealthStatus.HEALTHY
            if memory_usage >= self.memory_critical_threshold:
                status = HealthStatus.CRITICAL
            elif memory_usage >= self.memory_warning_threshold:
                status = HealthStatus.WARNING
                
            metrics.append(HealthMetric(
                name="Memory Usage",
                value=memory_usage,
                status=status,
                threshold_warning=self.memory_warning_threshold,
                threshold_critical=self.memory_critical_threshold,
                unit="%",
                description="Current memory usage percentage"
            ))
        
        # Error rate metrics
        for component in self.error_monitor.total_requests.keys():
            error_rate = self.error_monitor.get_error_rate(component)
            status = HealthStatus.HEALTHY
            if error_rate >= self.error_rate_critical:
                status = HealthStatus.CRITICAL
            elif error_rate >= self.error_rate_warning:
                status = HealthStatus.WARNING
                
            metrics.append(HealthMetric(
                name=f"{component} Error Rate",
                value=error_rate * 100,  # Convert to percentage
                status=status,
                threshold_warning=self.error_rate_warning * 100,
                threshold_critical=self.error_rate_critical * 100,
                unit="%",
                description=f"Error rate for {component} in the last hour"
            ))
        
        return metrics