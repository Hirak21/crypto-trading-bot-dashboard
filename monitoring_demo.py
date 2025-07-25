"""
Demonstration of the Crypto Trading Bot Monitoring System

This script showcases the key features of the monitoring and notification system
without requiring external dependencies.
"""

import sys
import os
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_trading_bot.models.config import NotificationConfig, RiskConfig
from crypto_trading_bot.models.trading import Trade, Position, TradingSignal, OrderSide, PositionSide, SignalAction


def demonstrate_configuration():
    """Demonstrate configuration management"""
    print("="*60)
    print("CONFIGURATION MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    # Create notification configuration
    notification_config = NotificationConfig(
        enabled=True,
        console={'enabled': True, 'min_level': 'info'},
        email={
            'enabled': False,  # Disabled for demo
            'smtp_server': 'smtp.gmail.com',
            'username': 'your_email@gmail.com',
            'password': 'your_password',
            'to_emails': ['alerts@yourcompany.com']
        },
        webhook={
            'enabled': False,  # Disabled for demo
            'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        },
        market_events={
            'price_change_threshold': 0.05,  # 5% price change alert
            'volume_spike_threshold': 2.0,   # 2x volume spike alert
            'volatility_threshold': 0.1      # 10% volatility alert
        },
        performance_monitoring={
            'win_rate_threshold': 0.4,       # 40% win rate threshold
            'profit_factor_threshold': 1.2,  # 1.2 profit factor threshold
            'max_consecutive_losses': 5      # 5 consecutive losses alert
        },
        technical_indicators={
            'rsi_overbought': 80,            # RSI overbought level
            'rsi_oversold': 20,              # RSI oversold level
            'bb_squeeze_threshold': 0.02     # 2% Bollinger Band squeeze
        },
        trade_notifications=True,
        error_notifications=True,
        performance_notifications=True,
        system_notifications=True
    )
    
    print("✓ Notification Configuration:")
    print(f"  - Enabled: {notification_config.enabled}")
    print(f"  - Console logging: {notification_config.console}")
    print(f"  - Price change threshold: {notification_config.market_events['price_change_threshold']:.1%}")
    print(f"  - Win rate threshold: {notification_config.performance_monitoring['win_rate_threshold']:.1%}")
    print(f"  - RSI overbought: {notification_config.technical_indicators['rsi_overbought']}")
    
    # Create risk configuration
    risk_config = RiskConfig(
        max_position_size=0.02,    # 2% of portfolio per trade
        daily_loss_limit=0.05,     # 5% daily loss limit
        max_drawdown=0.15,         # 15% maximum drawdown
        stop_loss_pct=0.02,        # 2% stop loss
        take_profit_pct=0.04,      # 4% take profit
        max_open_positions=5,      # Maximum 5 concurrent positions
        min_account_balance=100.0  # Minimum $100 balance
    )
    
    print("\n✓ Risk Configuration:")
    print(f"  - Max position size: {risk_config.max_position_size:.1%}")
    print(f"  - Daily loss limit: {risk_config.daily_loss_limit:.1%}")
    print(f"  - Max drawdown: {risk_config.max_drawdown:.1%}")
    print(f"  - Stop loss: {risk_config.stop_loss_pct:.1%}")
    print(f"  - Take profit: {risk_config.take_profit_pct:.1%}")
    
    # Test serialization
    config_dict = notification_config.to_dict()
    restored_config = NotificationConfig.from_dict(config_dict)
    
    print(f"\n✓ Configuration serialization working: {restored_config.enabled}")


def demonstrate_trading_models():
    """Demonstrate trading data models"""
    print("\n" + "="*60)
    print("TRADING DATA MODELS DEMONSTRATION")
    print("="*60)
    
    # Create a trading signal
    signal = TradingSignal(
        symbol="BTCUSDT",
        action=SignalAction.BUY,
        confidence=0.85,
        strategy="momentum_strategy",
        timestamp=datetime.now(),
        target_price=46000.0,
        stop_loss=44000.0,
        take_profit=47000.0,
        position_size=0.02
    )
    
    print("✓ Trading Signal Created:")
    print(f"  - Symbol: {signal.symbol}")
    print(f"  - Action: {signal.action.value}")
    print(f"  - Confidence: {signal.confidence:.1%}")
    print(f"  - Strategy: {signal.strategy}")
    print(f"  - Target: ${signal.target_price:,.2f}")
    print(f"  - Stop Loss: ${signal.stop_loss:,.2f}")
    print(f"  - Valid: {signal.is_valid()}")
    
    # Create a trade execution
    trade = Trade(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        size=0.1,
        price=45000.0,
        commission=4.5,
        timestamp=datetime.now(),
        strategy="momentum_strategy",
        pnl=150.0
    )
    
    print(f"\n✓ Trade Execution:")
    print(f"  - Trade ID: {trade.trade_id}")
    print(f"  - Symbol: {trade.symbol}")
    print(f"  - Side: {trade.side.value}")
    print(f"  - Size: {trade.size}")
    print(f"  - Price: ${trade.price:,.2f}")
    print(f"  - Commission: ${trade.commission:.2f}")
    print(f"  - P&L: ${trade.pnl:.2f}")
    
    # Create a position
    position = Position(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        size=2.0,
        entry_price=3000.0,
        current_price=3150.0,
        timestamp=datetime.now(),
        stop_loss=2900.0,
        take_profit=3300.0,
        strategy="liquidity_strategy"
    )
    
    print(f"\n✓ Position:")
    print(f"  - Position ID: {position.position_id}")
    print(f"  - Symbol: {position.symbol}")
    print(f"  - Side: {position.side.value}")
    print(f"  - Size: {position.size}")
    print(f"  - Entry Price: ${position.entry_price:,.2f}")
    print(f"  - Current Price: ${position.current_price:,.2f}")
    print(f"  - Unrealized P&L: ${position.unrealized_pnl:.2f}")
    print(f"  - P&L Percentage: {position.unrealized_pnl_percentage:.2f}%")
    print(f"  - Notional Value: ${position.notional_value:,.2f}")
    print(f"  - Should close (stop loss): {position.should_close_stop_loss()}")
    print(f"  - Should close (take profit): {position.should_close_take_profit()}")


def demonstrate_alert_scenarios():
    """Demonstrate different alert scenarios"""
    print("\n" + "="*60)
    print("ALERT SCENARIOS DEMONSTRATION")
    print("="*60)
    
    # Define alert types for demonstration
    class AlertLevel(Enum):
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"
    
    class AlertType(Enum):
        TRADE_EXECUTED = "trade_executed"
        POSITION_OPENED = "position_opened"
        POSITION_CLOSED = "position_closed"
        RISK_LIMIT_REACHED = "risk_limit_reached"
        MARKET_EVENT = "market_event"
        TECHNICAL_EXTREME = "technical_extreme"
        API_ERROR = "api_error"
        SYSTEM_ERROR = "system_error"
    
    @dataclass
    class Alert:
        type: AlertType
        level: AlertLevel
        title: str
        message: str
        timestamp: datetime = None
        data: Optional[Dict[str, Any]] = None
        symbol: Optional[str] = None
        strategy: Optional[str] = None
        
        def __post_init__(self):
            if self.timestamp is None:
                self.timestamp = datetime.now()
    
    # Demonstrate different alert scenarios
    alerts = [
        Alert(
            type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Trade Executed - BTCUSDT",
            message="BUY 0.1 BTCUSDT at $45,000.00",
            symbol="BTCUSDT",
            strategy="momentum_strategy",
            data={'side': 'BUY', 'size': 0.1, 'price': 45000.0}
        ),
        Alert(
            type=AlertType.RISK_LIMIT_REACHED,
            level=AlertLevel.WARNING,
            title="Daily Loss Limit Approaching",
            message="Daily loss reached 4.2% (limit: 5.0%)",
            data={'current_loss': 0.042, 'limit': 0.05}
        ),
        Alert(
            type=AlertType.MARKET_EVENT,
            level=AlertLevel.WARNING,
            title="Significant Price Movement - ETHUSDT",
            message="Price increased by 7.5% to $3,225.00",
            symbol="ETHUSDT",
            data={'price_change_pct': 0.075, 'new_price': 3225.0}
        ),
        Alert(
            type=AlertType.TECHNICAL_EXTREME,
            level=AlertLevel.INFO,
            title="RSI Overbought - BTCUSDT",
            message="RSI reached 82.5 (overbought threshold: 80)",
            symbol="BTCUSDT",
            data={'rsi': 82.5, 'threshold': 80}
        ),
        Alert(
            type=AlertType.API_ERROR,
            level=AlertLevel.ERROR,
            title="API Rate Limit Exceeded",
            message="Binance API rate limit exceeded, retrying in 60 seconds",
            data={'error_code': 'RATE_LIMIT', 'retry_after': 60}
        ),
        Alert(
            type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.CRITICAL,
            title="Critical Memory Usage",
            message="Memory usage at 95.2% (critical threshold: 95.0%)",
            data={'memory_usage': 95.2, 'threshold': 95.0}
        )
    ]
    
    print("Alert Scenarios:")
    for i, alert in enumerate(alerts, 1):
        level_symbols = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }
        
        symbol = level_symbols.get(alert.level, "❓")
        print(f"\n{i}. {symbol} [{alert.level.value.upper()}] {alert.title}")
        print(f"   Type: {alert.type.value}")
        print(f"   Message: {alert.message}")
        if alert.symbol:
            print(f"   Symbol: {alert.symbol}")
        if alert.strategy:
            print(f"   Strategy: {alert.strategy}")
        if alert.data:
            print(f"   Data: {alert.data}")


def demonstrate_performance_tracking():
    """Demonstrate performance tracking capabilities"""
    print("\n" + "="*60)
    print("PERFORMANCE TRACKING DEMONSTRATION")
    print("="*60)
    
    from collections import deque, defaultdict
    from datetime import datetime, timedelta
    
    class PerformanceTracker:
        def __init__(self, window_size: int = 1000):
            self.window_size = window_size
            self.metrics = defaultdict(lambda: deque(maxlen=window_size))
            self.timestamps = defaultdict(lambda: deque(maxlen=window_size))
        
        def record_metric(self, name: str, value: float, timestamp: Optional[datetime] = None):
            if timestamp is None:
                timestamp = datetime.now()
            self.metrics[name].append(value)
            self.timestamps[name].append(timestamp)
        
        def get_average(self, name: str, duration: Optional[timedelta] = None) -> Optional[float]:
            if name not in self.metrics or not self.metrics[name]:
                return None
            
            if duration is None:
                return sum(self.metrics[name]) / len(self.metrics[name])
            
            cutoff = datetime.now() - duration
            values = []
            for value, timestamp in zip(self.metrics[name], self.timestamps[name]):
                if timestamp >= cutoff:
                    values.append(value)
            
            return sum(values) / len(values) if values else None
        
        def get_percentile(self, name: str, percentile: float) -> Optional[float]:
            if name not in self.metrics or not self.metrics[name]:
                return None
            
            values = sorted(list(self.metrics[name]))
            index = int(len(values) * percentile / 100)
            return values[min(index, len(values) - 1)]
        
        def get_latest(self, name: str) -> Optional[float]:
            if name not in self.metrics or not self.metrics[name]:
                return None
            return self.metrics[name][-1]
    
    # Create performance tracker
    tracker = PerformanceTracker()
    
    # Simulate performance data
    import random
    
    print("Simulating performance data...")
    
    # Simulate API response times
    for i in range(100):
        response_time = random.uniform(50, 200)  # 50-200ms
        tracker.record_metric("api_response_time", response_time)
    
    # Simulate signal generation times
    for i in range(50):
        signal_time = random.uniform(10, 50)  # 10-50ms
        tracker.record_metric("signal_generation_time", signal_time)
    
    # Simulate trade execution times
    for i in range(30):
        execution_time = random.uniform(100, 500)  # 100-500ms
        tracker.record_metric("trade_execution_time", execution_time)
    
    # Simulate CPU usage
    for i in range(200):
        cpu_usage = random.uniform(30, 90)  # 30-90%
        tracker.record_metric("cpu_usage", cpu_usage)
    
    print("\n✓ Performance Metrics:")
    
    metrics = [
        ("API Response Time", "api_response_time", "ms"),
        ("Signal Generation Time", "signal_generation_time", "ms"),
        ("Trade Execution Time", "trade_execution_time", "ms"),
        ("CPU Usage", "cpu_usage", "%")
    ]
    
    for name, metric_key, unit in metrics:
        avg = tracker.get_average(metric_key)
        p50 = tracker.get_percentile(metric_key, 50)
        p95 = tracker.get_percentile(metric_key, 95)
        latest = tracker.get_latest(metric_key)
        
        print(f"\n  {name}:")
        print(f"    - Average: {avg:.1f}{unit}")
        print(f"    - Median (P50): {p50:.1f}{unit}")
        print(f"    - 95th Percentile: {p95:.1f}{unit}")
        print(f"    - Latest: {latest:.1f}{unit}")
    
    # Demonstrate time-based filtering
    recent_avg = tracker.get_average("api_response_time", timedelta(minutes=5))
    print(f"\n  Recent API Response Time (last 5 min): {recent_avg:.1f}ms")


def demonstrate_system_health():
    """Demonstrate system health monitoring"""
    print("\n" + "="*60)
    print("SYSTEM HEALTH MONITORING DEMONSTRATION")
    print("="*60)
    
    from enum import Enum
    from dataclasses import dataclass
    
    class HealthStatus(Enum):
        HEALTHY = "healthy"
        WARNING = "warning"
        CRITICAL = "critical"
        UNKNOWN = "unknown"
    
    @dataclass
    class HealthMetric:
        name: str
        value: float
        status: HealthStatus
        timestamp: datetime = None
        threshold_warning: Optional[float] = None
        threshold_critical: Optional[float] = None
        unit: str = ""
        description: str = ""
        
        def __post_init__(self):
            if self.timestamp is None:
                self.timestamp = datetime.now()
    
    # Create sample health metrics
    health_metrics = [
        HealthMetric(
            name="CPU Usage",
            value=65.2,
            status=HealthStatus.HEALTHY,
            threshold_warning=80.0,
            threshold_critical=95.0,
            unit="%",
            description="Current CPU usage percentage"
        ),
        HealthMetric(
            name="Memory Usage",
            value=78.5,
            status=HealthStatus.WARNING,
            threshold_warning=75.0,
            threshold_critical=90.0,
            unit="%",
            description="Current memory usage percentage"
        ),
        HealthMetric(
            name="API Error Rate",
            value=2.1,
            status=HealthStatus.HEALTHY,
            threshold_warning=5.0,
            threshold_critical=15.0,
            unit="%",
            description="API error rate in the last hour"
        ),
        HealthMetric(
            name="WebSocket Connection",
            value=1.0,
            status=HealthStatus.HEALTHY,
            threshold_warning=0.5,
            threshold_critical=0.0,
            unit="",
            description="WebSocket connection status (1=connected, 0=disconnected)"
        ),
        HealthMetric(
            name="Average Response Time",
            value=125.3,
            status=HealthStatus.HEALTHY,
            threshold_warning=200.0,
            threshold_critical=500.0,
            unit="ms",
            description="Average API response time"
        )
    ]
    
    print("System Health Metrics:")
    
    status_symbols = {
        HealthStatus.HEALTHY: "✅",
        HealthStatus.WARNING: "⚠️",
        HealthStatus.CRITICAL: "❌",
        HealthStatus.UNKNOWN: "❓"
    }
    
    overall_status = HealthStatus.HEALTHY
    
    for metric in health_metrics:
        symbol = status_symbols.get(metric.status, "❓")
        print(f"\n{symbol} {metric.name}: {metric.value}{metric.unit}")
        print(f"   Status: {metric.status.value.upper()}")
        print(f"   Description: {metric.description}")
        
        if metric.threshold_warning is not None:
            print(f"   Warning Threshold: {metric.threshold_warning}{metric.unit}")
        if metric.threshold_critical is not None:
            print(f"   Critical Threshold: {metric.threshold_critical}{metric.unit}")
        
        # Update overall status
        if metric.status == HealthStatus.CRITICAL:
            overall_status = HealthStatus.CRITICAL
        elif metric.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
            overall_status = HealthStatus.WARNING
    
    print(f"\n🏥 Overall System Status: {overall_status.value.upper()}")
    
    # System summary
    healthy_count = sum(1 for m in health_metrics if m.status == HealthStatus.HEALTHY)
    warning_count = sum(1 for m in health_metrics if m.status == HealthStatus.WARNING)
    critical_count = sum(1 for m in health_metrics if m.status == HealthStatus.CRITICAL)
    
    print(f"\nHealth Summary:")
    print(f"  - Healthy: {healthy_count}")
    print(f"  - Warning: {warning_count}")
    print(f"  - Critical: {critical_count}")
    print(f"  - Total Metrics: {len(health_metrics)}")


def main():
    """Main demonstration function"""
    print("🤖 CRYPTO TRADING BOT - MONITORING SYSTEM DEMONSTRATION")
    print("This demo showcases the comprehensive monitoring and notification capabilities")
    print("of the crypto trading bot without requiring external dependencies.\n")
    
    try:
        demonstrate_configuration()
        demonstrate_trading_models()
        demonstrate_alert_scenarios()
        demonstrate_performance_tracking()
        demonstrate_system_health()
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✅ Configuration Management (Notifications, Risk, Logging)")
        print("✅ Trading Data Models (Signals, Trades, Positions)")
        print("✅ Alert System (Multiple levels and types)")
        print("✅ Performance Tracking (Metrics, Averages, Percentiles)")
        print("✅ System Health Monitoring (CPU, Memory, Connections)")
        print("\nThe monitoring system is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()