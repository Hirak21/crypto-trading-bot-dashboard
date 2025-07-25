"""
Comprehensive Monitoring System Integration

This module integrates all monitoring components and provides a unified
interface for system monitoring, health checks, and alerting.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from .notification_manager import NotificationManager, Alert, AlertType, AlertLevel
from .health_monitor import HealthMonitor, HealthStatus, HealthMetric
from ..models.config import NotificationConfig
from ..models.trading import Trade, Position


@dataclass
class SystemStatus:
    """Overall system status"""
    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    active_connections: int
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    last_trade_time: Optional[datetime] = None
    total_trades: int = 0
    active_positions: int = 0


class MonitoringSystem:
    """
    Comprehensive monitoring system that coordinates all monitoring components
    and provides unified system health and performance tracking.
    """
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.start_time = datetime.now()
        
        # Initialize monitoring components
        self.notification_manager = NotificationManager(config)
        self.health_monitor = HealthMonitor(
            self.notification_manager, 
            config.to_dict()
        )
        
        # System state tracking
        self.is_running = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.trade_count = 0
        self.position_count = 0
        self.last_trade_time: Optional[datetime] = None
        self.error_counts: Dict[str, int] = {}
        
        # Connection tracking
        self.registered_connections: Dict[str, Callable] = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the monitoring system"""
        if self.is_running:
            self.logger.warning("Monitoring system already running")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start notification manager
        await self.notification_manager.start()
        
        # Start health monitor
        await self.health_monitor.start()
        
        # Start periodic tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._periodic_health_report()),
            asyncio.create_task(self._periodic_performance_summary()),
            asyncio.create_task(self._connection_health_check())
        ]
        
        self.logger.info("Monitoring system started successfully")
        
        # Send startup notification
        self.notification_manager.send_alert(Alert(
            type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.INFO,
            title="Trading Bot Started",
            message="Monitoring system initialized and all components started",
            data={'start_time': self.start_time.isoformat()}
        ))
    
    async def stop(self):
        """Stop the monitoring system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        # Stop components
        await self.health_monitor.stop()
        await self.notification_manager.stop()
        
        self.logger.info("Monitoring system stopped")
        
        # Send shutdown notification
        uptime = datetime.now() - self.start_time
        self.notification_manager.send_alert(Alert(
            type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.INFO,
            title="Trading Bot Stopped",
            message=f"System shutdown after {uptime.total_seconds():.0f} seconds uptime",
            data={
                'uptime_seconds': uptime.total_seconds(),
                'total_trades': self.trade_count,
                'shutdown_time': datetime.now().isoformat()
            }
        ))
    
    async def _periodic_health_report(self):
        """Send periodic health reports"""
        while self.is_running:
            try:
                # Wait for configured interval (default 1 hour)
                interval = self.config.performance_report_interval_hours * 3600
                await asyncio.sleep(interval)
                
                if not self.is_running:
                    break
                
                # Generate health report
                health_summary = self.health_monitor.get_health_summary()
                system_status = self.get_system_status()
                
                # Send health report
                self.notification_manager.send_alert(Alert(
                    type=AlertType.PERFORMANCE_ALERT,
                    level=AlertLevel.INFO,
                    title="Periodic Health Report",
                    message=self._format_health_report(system_status, health_summary),
                    data={
                        'system_status': system_status.__dict__,
                        'health_summary': health_summary
                    }
                ))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic health report: {e}")
    
    async def _periodic_performance_summary(self):
        """Send periodic performance summaries"""
        while self.is_running:
            try:
                # Wait 6 hours between performance summaries
                await asyncio.sleep(6 * 3600)
                
                if not self.is_running:
                    break
                
                # Generate performance summary
                uptime = datetime.now() - self.start_time
                
                self.notification_manager.send_alert(Alert(
                    type=AlertType.PERFORMANCE_ALERT,
                    level=AlertLevel.INFO,
                    title="Performance Summary",
                    message=f"Uptime: {uptime.total_seconds():.0f}s, Trades: {self.trade_count}, Active Positions: {self.position_count}",
                    data={
                        'uptime_seconds': uptime.total_seconds(),
                        'total_trades': self.trade_count,
                        'active_positions': self.position_count,
                        'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None
                    }
                ))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance summary: {e}")
    
    async def _connection_health_check(self):
        """Periodic connection health checks"""
        while self.is_running:
            try:
                # Check every 2 minutes
                await asyncio.sleep(120)
                
                if not self.is_running:
                    break
                
                # Check all registered connections
                for service_name, check_func in self.registered_connections.items():
                    try:
                        is_healthy = await check_func()
                        if not is_healthy:
                            self.notification_manager.connection_lost(service_name)
                    except Exception as e:
                        self.logger.error(f"Connection check failed for {service_name}: {e}")
                        self.notification_manager.connection_lost(service_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in connection health check: {e}")
    
    def _format_health_report(self, status: SystemStatus, health_summary: Dict[str, Any]) -> str:
        """Format health report message"""
        uptime_hours = status.uptime_seconds / 3600
        
        report = f"""System Health Report:
        
Status: {status.status.value.upper()}
Uptime: {uptime_hours:.1f} hours
CPU Usage: {status.cpu_usage_percent:.1f}%
Memory Usage: {status.memory_usage_mb:.1f} MB
Error Rate: {status.error_rate:.2%}
Active Connections: {status.active_connections}
Total Trades: {status.total_trades}
Active Positions: {status.active_positions}
Last Trade: {status.last_trade_time.strftime('%Y-%m-%d %H:%M:%S') if status.last_trade_time else 'None'}
"""
        
        # Add connection status
        if 'connections' in health_summary:
            report += "\nConnection Status:\n"
            for service, conn_info in health_summary['connections'].items():
                status_text = "✓ Connected" if conn_info['is_connected'] else "✗ Disconnected"
                response_time = f" ({conn_info['response_time_ms']:.0f}ms)" if conn_info['response_time_ms'] else ""
                report += f"  {service}: {status_text}{response_time}\n"
        
        return report
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        uptime = datetime.now() - self.start_time
        health_summary = self.health_monitor.get_health_summary()
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        if health_summary['overall_status'] == 'critical':
            overall_status = HealthStatus.CRITICAL
        elif health_summary['overall_status'] == 'warning':
            overall_status = HealthStatus.WARNING
        
        return SystemStatus(
            status=overall_status,
            timestamp=datetime.now(),
            uptime_seconds=uptime.total_seconds(),
            active_connections=len([
                conn for conn in health_summary.get('connections', {}).values()
                if conn.get('is_connected', False)
            ]),
            error_rate=sum(health_summary.get('error_rates', {}).values()) / max(len(health_summary.get('error_rates', {})), 1),
            memory_usage_mb=health_summary.get('system_resources', {}).get('memory_usage', 0),
            cpu_usage_percent=health_summary.get('system_resources', {}).get('cpu_usage', 0),
            last_trade_time=self.last_trade_time,
            total_trades=self.trade_count,
            active_positions=self.position_count
        )
    
    def get_health_metrics(self) -> List[HealthMetric]:
        """Get detailed health metrics"""
        return self.health_monitor.get_health_metrics()
    
    # Event recording methods
    def record_trade(self, trade: Trade):
        """Record a trade execution"""
        self.trade_count += 1
        self.last_trade_time = datetime.now()
        
        # Notify trade execution
        self.notification_manager.trade_executed(trade)
        
        # Record performance metrics
        self.health_monitor.record_trade_execution_time(100)  # Placeholder timing
    
    def record_position_opened(self, position: Position):
        """Record position opening"""
        self.position_count += 1
        self.notification_manager.position_opened(position)
    
    def record_position_closed(self, position: Position, pnl: float):
        """Record position closing"""
        self.position_count = max(0, self.position_count - 1)
        self.notification_manager.position_closed(position, pnl)
    
    def record_api_request(self, component: str, is_error: bool = False, response_time_ms: Optional[float] = None):
        """Record API request"""
        self.health_monitor.record_api_request(component, is_error, response_time_ms)
        
        if is_error:
            self.error_counts[component] = self.error_counts.get(component, 0) + 1
    
    def record_signal_generation(self, strategy: str, time_ms: float):
        """Record signal generation timing"""
        self.health_monitor.record_signal_generation_time(strategy, time_ms)
    
    def record_market_data(self, symbol: str, price: float, volume: float):
        """Record market data and check for events"""
        self.notification_manager.check_market_events(symbol, price, volume)
    
    def record_technical_indicators(self, symbol: str, indicators: Dict[str, float]):
        """Record technical indicators and check for extremes"""
        self.notification_manager.check_technical_indicators(symbol, indicators)
    
    def register_connection_check(self, service: str, check_function: Callable):
        """Register a connection health check"""
        self.registered_connections[service] = check_function
        self.health_monitor.register_connection_check(service, check_function)
    
    # Alert methods
    def alert_risk_limit_reached(self, limit_type: str, current_value: float, limit_value: float):
        """Alert for risk limit violations"""
        self.notification_manager.risk_limit_reached(limit_type, current_value, limit_value)
    
    def alert_api_error(self, error_message: str, error_code: Optional[str] = None):
        """Alert for API errors"""
        self.notification_manager.api_error(error_message, error_code)
    
    def alert_system_error(self, error_message: str, component: str):
        """Alert for system errors"""
        self.notification_manager.system_error(error_message, component)
    
    def alert_connection_lost(self, service: str):
        """Alert for connection loss"""
        self.notification_manager.connection_lost(service)
    
    def alert_connection_restored(self, service: str):
        """Alert for connection restoration"""
        self.notification_manager.connection_restored(service)
    
    # Configuration methods
    def update_notification_config(self, config: NotificationConfig):
        """Update notification configuration"""
        self.config = config
        # Note: In a full implementation, you'd need to restart the notification manager
        # with the new configuration
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics"""
        uptime = datetime.now() - self.start_time
        health_summary = self.health_monitor.get_health_summary()
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'start_time': self.start_time.isoformat(),
            'total_trades': self.trade_count,
            'active_positions': self.position_count,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'error_counts': self.error_counts,
            'health_summary': health_summary,
            'system_status': self.get_system_status().__dict__,
            'registered_connections': list(self.registered_connections.keys())
        }
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run immediate health check and return results"""
        # Force a health check
        await self.health_monitor._perform_health_check()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': self.get_system_status().__dict__,
            'health_metrics': [metric.__dict__ for metric in self.get_health_metrics()],
            'health_summary': self.health_monitor.get_health_summary()
        }
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        status = self.get_system_status()
        return status.status in [HealthStatus.HEALTHY, HealthStatus.WARNING]
    
    def get_uptime(self) -> timedelta:
        """Get system uptime"""
        return datetime.now() - self.start_time
    
    def get_error_rate(self) -> float:
        """Get overall error rate"""
        total_errors = sum(self.error_counts.values())
        total_requests = max(total_errors * 10, 1)  # Estimate total requests
        return total_errors / total_requests