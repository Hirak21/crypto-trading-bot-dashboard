"""
Notification and Monitoring System for Crypto Trading Bot

This module provides comprehensive notification and monitoring capabilities including:
- Multi-channel notifications (email, webhook, console)
- Market event detection and alerting
- Bot performance monitoring
- Technical indicator extreme level alerts
- System health monitoring
"""

import asyncio
import json
import logging
import smtplib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from urllib.parse import urljoin
import aiohttp
import requests

from ..models.trading import TradingSignal, Position, Trade
from ..models.config import NotificationConfig


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts that can be generated"""
    TRADE_EXECUTED = "trade_executed"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    RISK_LIMIT_REACHED = "risk_limit_reached"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_WARNING = "drawdown_warning"
    API_ERROR = "api_error"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    STRATEGY_PERFORMANCE = "strategy_performance"
    TECHNICAL_EXTREME = "technical_extreme"
    MARKET_EVENT = "market_event"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_ALERT = "performance_alert"


@dataclass
class Alert:
    """Represents a system alert"""
    type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = None
    symbol: Optional[str] = None
    strategy: Optional[str] = None


class NotificationChannel(ABC):
    """Abstract base class for notification channels"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.min_level = AlertLevel(config.get('min_level', 'info'))
        
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert through this channel"""
        pass
    
    def should_send(self, alert: Alert) -> bool:
        """Check if alert should be sent through this channel"""
        if not self.enabled:
            return False
            
        level_priority = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.ERROR: 2,
            AlertLevel.CRITICAL: 3
        }
        
        return level_priority[alert.level] >= level_priority[self.min_level]


class ConsoleNotificationChannel(NotificationChannel):
    """Console/logging notification channel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger('notifications')
        
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to console/logs"""
        try:
            log_message = f"[{alert.type.value.upper()}] {alert.title}: {alert.message}"
            
            if alert.level == AlertLevel.INFO:
                self.logger.info(log_message)
            elif alert.level == AlertLevel.WARNING:
                self.logger.warning(log_message)
            elif alert.level == AlertLevel.ERROR:
                self.logger.error(log_message)
            elif alert.level == AlertLevel.CRITICAL:
                self.logger.critical(log_message)
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to send console alert: {e}")
            return False


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email', self.username)
        self.to_emails = config.get('to_emails', [])
        
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        if not self.username or not self.password or not self.to_emails:
            return False
            
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.level.value.upper()}] Trading Bot Alert: {alert.title}"
            
            # Create email body
            body = self._format_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
            return False
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format alert as HTML email body"""
        color_map = {
            AlertLevel.INFO: '#17a2b8',
            AlertLevel.WARNING: '#ffc107',
            AlertLevel.ERROR: '#dc3545',
            AlertLevel.CRITICAL: '#721c24'
        }
        
        color = color_map.get(alert.level, '#6c757d')
        
        html = f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                    <h2 style="margin: 0;">{alert.title}</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">
                        {alert.level.value.upper()} - {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </div>
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 0 0 5px 5px;">
                    <p><strong>Message:</strong> {alert.message}</p>
                    {f'<p><strong>Symbol:</strong> {alert.symbol}</p>' if alert.symbol else ''}
                    {f'<p><strong>Strategy:</strong> {alert.strategy}</p>' if alert.strategy else ''}
                    {self._format_alert_data(alert.data) if alert.data else ''}
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_alert_data(self, data: Dict[str, Any]) -> str:
        """Format alert data as HTML"""
        if not data:
            return ""
            
        html = "<p><strong>Additional Data:</strong></p><ul>"
        for key, value in data.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        return html


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.timeout = config.get('timeout', 10)
        
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        if not self.webhook_url:
            return False
            
        try:
            payload = {
                'type': alert.type.value,
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'symbol': alert.symbol,
                'strategy': alert.strategy,
                'data': alert.data
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    return response.status < 400
                    
        except Exception as e:
            logging.error(f"Failed to send webhook alert: {e}")
            return False


class MarketEventDetector:
    """Detects significant market events for alerting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.price_change_threshold = config.get('price_change_threshold', 0.05)  # 5%
        self.volume_spike_threshold = config.get('volume_spike_threshold', 2.0)  # 2x average
        self.volatility_threshold = config.get('volatility_threshold', 0.1)  # 10%
        
        # Store recent data for comparison
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.last_prices: Dict[str, float] = {}
        
    def detect_events(self, symbol: str, price: float, volume: float) -> List[Alert]:
        """Detect market events and return alerts"""
        alerts = []
        
        # Initialize history if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
            self.last_prices[symbol] = price
            return alerts
        
        # Check for significant price changes
        if self.last_prices[symbol] > 0:
            price_change = abs(price - self.last_prices[symbol]) / self.last_prices[symbol]
            if price_change > self.price_change_threshold:
                direction = "increased" if price > self.last_prices[symbol] else "decreased"
                alerts.append(Alert(
                    type=AlertType.MARKET_EVENT,
                    level=AlertLevel.WARNING,
                    title=f"Significant Price Movement - {symbol}",
                    message=f"Price {direction} by {price_change:.2%} to ${price:.2f}",
                    symbol=symbol,
                    data={
                        'price_change_pct': price_change,
                        'old_price': self.last_prices[symbol],
                        'new_price': price
                    }
                ))
        
        # Check for volume spikes
        if len(self.volume_history[symbol]) >= 20:  # Need history for average
            avg_volume = sum(self.volume_history[symbol][-20:]) / 20
            if volume > avg_volume * self.volume_spike_threshold:
                alerts.append(Alert(
                    type=AlertType.MARKET_EVENT,
                    level=AlertLevel.INFO,
                    title=f"Volume Spike - {symbol}",
                    message=f"Volume spike detected: {volume:.0f} vs avg {avg_volume:.0f}",
                    symbol=symbol,
                    data={
                        'current_volume': volume,
                        'average_volume': avg_volume,
                        'spike_ratio': volume / avg_volume
                    }
                ))
        
        # Update history
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        self.last_prices[symbol] = price
        
        # Keep only recent history (last 100 data points)
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
            self.volume_history[symbol] = self.volume_history[symbol][-100:]
        
        return alerts


class PerformanceMonitor:
    """Monitors bot performance and generates alerts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.win_rate_threshold = config.get('win_rate_threshold', 0.4)  # 40%
        self.profit_factor_threshold = config.get('profit_factor_threshold', 1.2)
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)
        
        # Performance tracking
        self.trades: List[Trade] = []
        self.consecutive_losses = 0
        self.last_performance_check = datetime.now()
        self.check_interval = timedelta(hours=1)
        
    def add_trade(self, trade: Trade) -> List[Alert]:
        """Add trade and check for performance alerts"""
        alerts = []
        self.trades.append(trade)
        
        # Track consecutive losses
        if trade.pnl and trade.pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check for consecutive loss alert
        if self.consecutive_losses >= self.max_consecutive_losses:
            alerts.append(Alert(
                type=AlertType.PERFORMANCE_ALERT,
                level=AlertLevel.WARNING,
                title="Consecutive Losses Alert",
                message=f"Bot has {self.consecutive_losses} consecutive losses",
                data={'consecutive_losses': self.consecutive_losses}
            ))
        
        # Periodic performance check
        if datetime.now() - self.last_performance_check > self.check_interval:
            alerts.extend(self._check_performance())
            self.last_performance_check = datetime.now()
        
        return alerts
    
    def _check_performance(self) -> List[Alert]:
        """Check overall performance metrics"""
        alerts = []
        
        if len(self.trades) < 10:  # Need minimum trades for meaningful analysis
            return alerts
        
        # Calculate recent performance (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_trades = [t for t in self.trades if t.timestamp > recent_cutoff]
        
        if not recent_trades:
            return alerts
        
        # Calculate win rate
        winning_trades = [t for t in recent_trades if t.pnl and t.pnl > 0]
        win_rate = len(winning_trades) / len(recent_trades)
        
        if win_rate < self.win_rate_threshold:
            alerts.append(Alert(
                type=AlertType.PERFORMANCE_ALERT,
                level=AlertLevel.WARNING,
                title="Low Win Rate Alert",
                message=f"Win rate dropped to {win_rate:.1%} (threshold: {self.win_rate_threshold:.1%})",
                data={
                    'win_rate': win_rate,
                    'total_trades': len(recent_trades),
                    'winning_trades': len(winning_trades)
                }
            ))
        
        # Calculate profit factor
        total_profit = sum(t.pnl for t in recent_trades if t.pnl and t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in recent_trades if t.pnl and t.pnl < 0))
        
        if total_loss > 0:
            profit_factor = total_profit / total_loss
            if profit_factor < self.profit_factor_threshold:
                alerts.append(Alert(
                    type=AlertType.PERFORMANCE_ALERT,
                    level=AlertLevel.WARNING,
                    title="Low Profit Factor Alert",
                    message=f"Profit factor dropped to {profit_factor:.2f} (threshold: {self.profit_factor_threshold:.2f})",
                    data={
                        'profit_factor': profit_factor,
                        'total_profit': total_profit,
                        'total_loss': total_loss
                    }
                ))
        
        return alerts


class TechnicalIndicatorMonitor:
    """Monitors technical indicators for extreme levels"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rsi_overbought = config.get('rsi_overbought', 80)
        self.rsi_oversold = config.get('rsi_oversold', 20)
        self.bb_squeeze_threshold = config.get('bb_squeeze_threshold', 0.02)  # 2%
        
        # Track last alert times to avoid spam
        self.last_alerts: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(minutes=30)
        
    def check_indicators(self, symbol: str, indicators: Dict[str, float]) -> List[Alert]:
        """Check indicators for extreme levels"""
        alerts = []
        current_time = datetime.now()
        
        # RSI checks
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            alert_key = f"{symbol}_rsi"
            
            if (alert_key not in self.last_alerts or 
                current_time - self.last_alerts[alert_key] > self.alert_cooldown):
                
                if rsi >= self.rsi_overbought:
                    alerts.append(Alert(
                        type=AlertType.TECHNICAL_EXTREME,
                        level=AlertLevel.INFO,
                        title=f"RSI Overbought - {symbol}",
                        message=f"RSI reached {rsi:.1f} (overbought threshold: {self.rsi_overbought})",
                        symbol=symbol,
                        data={'rsi': rsi, 'threshold': self.rsi_overbought}
                    ))
                    self.last_alerts[alert_key] = current_time
                    
                elif rsi <= self.rsi_oversold:
                    alerts.append(Alert(
                        type=AlertType.TECHNICAL_EXTREME,
                        level=AlertLevel.INFO,
                        title=f"RSI Oversold - {symbol}",
                        message=f"RSI reached {rsi:.1f} (oversold threshold: {self.rsi_oversold})",
                        symbol=symbol,
                        data={'rsi': rsi, 'threshold': self.rsi_oversold}
                    ))
                    self.last_alerts[alert_key] = current_time
        
        # Bollinger Bands squeeze check
        if all(k in indicators for k in ['bb_upper', 'bb_lower', 'price']):
            bb_width = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['price']
            alert_key = f"{symbol}_bb_squeeze"
            
            if (bb_width < self.bb_squeeze_threshold and
                (alert_key not in self.last_alerts or 
                 current_time - self.last_alerts[alert_key] > self.alert_cooldown)):
                
                alerts.append(Alert(
                    type=AlertType.TECHNICAL_EXTREME,
                    level=AlertLevel.INFO,
                    title=f"Bollinger Bands Squeeze - {symbol}",
                    message=f"BB width: {bb_width:.3f} indicates potential breakout",
                    symbol=symbol,
                    data={'bb_width': bb_width, 'threshold': self.bb_squeeze_threshold}
                ))
                self.last_alerts[alert_key] = current_time
        
        return alerts


class NotificationManager:
    """Main notification and monitoring system"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.channels: List[NotificationChannel] = []
        self.market_detector = MarketEventDetector(config.market_events)
        self.performance_monitor = PerformanceMonitor(config.performance_monitoring)
        self.indicator_monitor = TechnicalIndicatorMonitor(config.technical_indicators)
        
        # Initialize notification channels
        self._setup_channels()
        
        # Alert queue for batch processing
        self.alert_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
        
    def _setup_channels(self):
        """Initialize notification channels based on configuration"""
        # Console channel (always enabled)
        self.channels.append(ConsoleNotificationChannel(
            self.config.console or {'enabled': True}
        ))
        
        # Email channel
        if self.config.email and self.config.email.get('enabled'):
            self.channels.append(EmailNotificationChannel(self.config.email))
        
        # Webhook channel
        if self.config.webhook and self.config.webhook.get('enabled'):
            self.channels.append(WebhookNotificationChannel(self.config.webhook))
    
    async def start(self):
        """Start the notification system"""
        self.processing_task = asyncio.create_task(self._process_alerts())
        self.logger.info("Notification system started")
    
    async def stop(self):
        """Stop the notification system"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Notification system stopped")
    
    async def _process_alerts(self):
        """Process alerts from the queue"""
        while True:
            try:
                alert = await self.alert_queue.get()
                await self._send_alert(alert)
                self.alert_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing alert: {e}")
    
    async def _send_alert(self, alert: Alert):
        """Send alert through all appropriate channels"""
        tasks = []
        for channel in self.channels:
            if channel.should_send(alert):
                tasks.append(channel.send_alert(alert))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            self.logger.debug(f"Alert sent through {success_count}/{len(tasks)} channels")
    
    def send_alert(self, alert: Alert):
        """Queue an alert for sending"""
        try:
            self.alert_queue.put_nowait(alert)
        except asyncio.QueueFull:
            self.logger.warning("Alert queue full, dropping alert")
    
    # Convenience methods for common alerts
    def trade_executed(self, trade: Trade):
        """Alert for trade execution"""
        alert = Alert(
            type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title=f"Trade Executed - {trade.symbol}",
            message=f"{trade.side} {trade.size} {trade.symbol} at ${trade.price:.2f}",
            symbol=trade.symbol,
            strategy=trade.strategy,
            data={
                'trade_id': trade.id,
                'side': trade.side,
                'size': trade.size,
                'price': trade.price,
                'commission': trade.commission
            }
        )
        self.send_alert(alert)
        
        # Check for performance alerts
        perf_alerts = self.performance_monitor.add_trade(trade)
        for alert in perf_alerts:
            self.send_alert(alert)
    
    def position_opened(self, position: Position):
        """Alert for position opening"""
        alert = Alert(
            type=AlertType.POSITION_OPENED,
            level=AlertLevel.INFO,
            title=f"Position Opened - {position.symbol}",
            message=f"Opened {position.side} position: {position.size} {position.symbol} at ${position.entry_price:.2f}",
            symbol=position.symbol,
            data={
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            }
        )
        self.send_alert(alert)
    
    def position_closed(self, position: Position, pnl: float):
        """Alert for position closing"""
        alert = Alert(
            type=AlertType.POSITION_CLOSED,
            level=AlertLevel.INFO,
            title=f"Position Closed - {position.symbol}",
            message=f"Closed {position.side} position: P&L ${pnl:.2f}",
            symbol=position.symbol,
            data={
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'exit_price': position.current_price,
                'pnl': pnl,
                'unrealized_pnl': position.unrealized_pnl
            }
        )
        self.send_alert(alert)
    
    def risk_limit_reached(self, limit_type: str, current_value: float, limit_value: float):
        """Alert for risk limit violations"""
        alert = Alert(
            type=AlertType.RISK_LIMIT_REACHED,
            level=AlertLevel.WARNING,
            title=f"Risk Limit Reached: {limit_type}",
            message=f"{limit_type} reached {current_value:.2%} (limit: {limit_value:.2%})",
            data={
                'limit_type': limit_type,
                'current_value': current_value,
                'limit_value': limit_value
            }
        )
        self.send_alert(alert)
    
    def api_error(self, error_message: str, error_code: Optional[str] = None):
        """Alert for API errors"""
        alert = Alert(
            type=AlertType.API_ERROR,
            level=AlertLevel.ERROR,
            title="API Error",
            message=error_message,
            data={'error_code': error_code} if error_code else None
        )
        self.send_alert(alert)
    
    def connection_lost(self, service: str):
        """Alert for connection loss"""
        alert = Alert(
            type=AlertType.CONNECTION_LOST,
            level=AlertLevel.WARNING,
            title=f"Connection Lost - {service}",
            message=f"Lost connection to {service}, attempting to reconnect...",
            data={'service': service}
        )
        self.send_alert(alert)
    
    def connection_restored(self, service: str):
        """Alert for connection restoration"""
        alert = Alert(
            type=AlertType.CONNECTION_RESTORED,
            level=AlertLevel.INFO,
            title=f"Connection Restored - {service}",
            message=f"Successfully reconnected to {service}",
            data={'service': service}
        )
        self.send_alert(alert)
    
    def check_market_events(self, symbol: str, price: float, volume: float):
        """Check for market events and send alerts"""
        alerts = self.market_detector.detect_events(symbol, price, volume)
        for alert in alerts:
            self.send_alert(alert)
    
    def check_technical_indicators(self, symbol: str, indicators: Dict[str, float]):
        """Check technical indicators and send alerts"""
        alerts = self.indicator_monitor.check_indicators(symbol, indicators)
        for alert in alerts:
            self.send_alert(alert)
    
    def system_error(self, error_message: str, component: str):
        """Alert for system errors"""
        alert = Alert(
            type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.ERROR,
            title=f"System Error - {component}",
            message=error_message,
            data={'component': component}
        )
        self.send_alert(alert)