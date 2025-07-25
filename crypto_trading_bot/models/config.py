"""
Configuration models for the crypto trading bot.

This module defines configuration data structures with validation
and parameter management capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json
from pathlib import Path


class LogLevel(Enum):
    """Logging level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class NotificationChannel(Enum):
    """Notification channel enumeration."""
    CONSOLE = "console"
    EMAIL = "email"
    WEBHOOK = "webhook"
    TELEGRAM = "telegram"
    SLACK = "slack"


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size: float = 0.02  # 2% of portfolio per trade
    daily_loss_limit: float = 0.05   # 5% daily loss limit
    max_drawdown: float = 0.15       # 15% maximum drawdown
    stop_loss_pct: float = 0.02      # 2% stop loss
    take_profit_pct: float = 0.04    # 4% take profit
    max_open_positions: int = 5      # Maximum concurrent positions
    min_account_balance: float = 100.0  # Minimum account balance to trade
    risk_free_rate: float = 0.02     # Risk-free rate for Sharpe ratio calculation
    
    def __post_init__(self):
        """Validate risk configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate risk configuration parameters."""
        if not 0.001 <= self.max_position_size <= 0.5:
            raise ValueError("max_position_size must be between 0.1% and 50%")
        
        if not 0.01 <= self.daily_loss_limit <= 1.0:
            raise ValueError("daily_loss_limit must be between 1% and 100%")
        
        if not 0.05 <= self.max_drawdown <= 1.0:
            raise ValueError("max_drawdown must be between 5% and 100%")
        
        if not 0.005 <= self.stop_loss_pct <= 0.2:
            raise ValueError("stop_loss_pct must be between 0.5% and 20%")
        
        if not 0.01 <= self.take_profit_pct <= 0.5:
            raise ValueError("take_profit_pct must be between 1% and 50%")
        
        if self.stop_loss_pct >= self.take_profit_pct:
            raise ValueError("stop_loss_pct must be less than take_profit_pct")
        
        if not 1 <= self.max_open_positions <= 20:
            raise ValueError("max_open_positions must be between 1 and 20")
        
        if self.min_account_balance < 0:
            raise ValueError("min_account_balance cannot be negative")
        
        if not 0.0 <= self.risk_free_rate <= 0.1:
            raise ValueError("risk_free_rate must be between 0% and 10%")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_position_size': self.max_position_size,
            'daily_loss_limit': self.daily_loss_limit,
            'max_drawdown': self.max_drawdown,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_open_positions': self.max_open_positions,
            'min_account_balance': self.min_account_balance,
            'risk_free_rate': self.risk_free_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class NotificationConfig:
    """Notification system configuration."""
    enabled: bool = True
    
    # Channel configurations
    console: Optional[Dict[str, Any]] = field(default_factory=lambda: {'enabled': True, 'min_level': 'info'})
    email: Optional[Dict[str, Any]] = None
    webhook: Optional[Dict[str, Any]] = None
    
    # Market event detection settings
    market_events: Dict[str, Any] = field(default_factory=lambda: {
        'price_change_threshold': 0.05,  # 5%
        'volume_spike_threshold': 2.0,   # 2x average
        'volatility_threshold': 0.1      # 10%
    })
    
    # Performance monitoring settings
    performance_monitoring: Dict[str, Any] = field(default_factory=lambda: {
        'win_rate_threshold': 0.4,       # 40%
        'profit_factor_threshold': 1.2,
        'max_consecutive_losses': 5
    })
    
    # Technical indicator monitoring settings
    technical_indicators: Dict[str, Any] = field(default_factory=lambda: {
        'rsi_overbought': 80,
        'rsi_oversold': 20,
        'bb_squeeze_threshold': 0.02     # 2%
    })
    
    # Notification settings
    trade_notifications: bool = True
    error_notifications: bool = True
    performance_notifications: bool = True
    system_notifications: bool = True
    
    # Notification thresholds
    min_trade_value: float = 0.0  # Minimum trade value to notify
    error_cooldown_minutes: int = 5  # Cooldown between error notifications
    performance_report_interval_hours: int = 24  # Performance report frequency
    
    def __post_init__(self):
        """Validate notification configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate notification configuration parameters."""
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        
        if self.min_trade_value < 0:
            raise ValueError("min_trade_value cannot be negative")
        
        if not 1 <= self.error_cooldown_minutes <= 60:
            raise ValueError("error_cooldown_minutes must be between 1 and 60")
        
        if not 1 <= self.performance_report_interval_hours <= 168:
            raise ValueError("performance_report_interval_hours must be between 1 and 168 (1 week)")
        
        # Validate channel-specific configurations
        if self.email and self.email.get('enabled'):
            required_email_fields = ['smtp_server', 'username', 'password', 'to_emails']
            for field in required_email_fields:
                if field not in self.email:
                    raise ValueError(f"email config missing required field: {field}")
        
        if self.webhook and self.webhook.get('enabled'):
            if 'webhook_url' not in self.webhook:
                raise ValueError("webhook config missing required field: webhook_url")
        
        # Validate threshold values
        if 'price_change_threshold' in self.market_events:
            if not 0.01 <= self.market_events['price_change_threshold'] <= 1.0:
                raise ValueError("price_change_threshold must be between 1% and 100%")
        
        if 'win_rate_threshold' in self.performance_monitoring:
            if not 0.1 <= self.performance_monitoring['win_rate_threshold'] <= 1.0:
                raise ValueError("win_rate_threshold must be between 10% and 100%")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enabled': self.enabled,
            'console': self.console,
            'email': self.email,
            'webhook': self.webhook,
            'market_events': self.market_events,
            'performance_monitoring': self.performance_monitoring,
            'technical_indicators': self.technical_indicators,
            'trade_notifications': self.trade_notifications,
            'error_notifications': self.error_notifications,
            'performance_notifications': self.performance_notifications,
            'system_notifications': self.system_notifications,
            'min_trade_value': self.min_trade_value,
            'error_cooldown_minutes': self.error_cooldown_minutes,
            'performance_report_interval_hours': self.performance_report_interval_hours
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NotificationConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class StrategyConfig:
    """Individual strategy configuration."""
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Strategy weight in signal aggregation
    min_confidence: float = 0.5  # Minimum confidence to generate signals
    cooldown_minutes: int = 1  # Minimum time between signals
    
    def __post_init__(self):
        """Validate strategy configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate strategy configuration parameters."""
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        
        if not 0.1 <= self.weight <= 10.0:
            raise ValueError("weight must be between 0.1 and 10.0")
        
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        
        if not 0 <= self.cooldown_minutes <= 60:
            raise ValueError("cooldown_minutes must be between 0 and 60")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enabled': self.enabled,
            'parameters': self.parameters,
            'weight': self.weight,
            'min_confidence': self.min_confidence,
            'cooldown_minutes': self.cooldown_minutes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    log_dir: str = "logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    structured_logging: bool = True
    console_logging: bool = True
    
    def __post_init__(self):
        """Validate logging configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate logging configuration parameters."""
        if not isinstance(self.level, LogLevel):
            if isinstance(self.level, str):
                self.level = LogLevel(self.level.upper())
            else:
                raise ValueError("Invalid log level")
        
        if not self.log_dir:
            raise ValueError("log_dir cannot be empty")
        
        if not 1024 * 1024 <= self.max_file_size <= 100 * 1024 * 1024:  # 1MB to 100MB
            raise ValueError("max_file_size must be between 1MB and 100MB")
        
        if not 1 <= self.backup_count <= 20:
            raise ValueError("backup_count must be between 1 and 20")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'level': self.level.value,
            'log_dir': self.log_dir,
            'max_file_size': self.max_file_size,
            'backup_count': self.backup_count,
            'structured_logging': self.structured_logging,
            'console_logging': self.console_logging
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoggingConfig':
        """Create from dictionary."""
        data = data.copy()
        if 'level' in data:
            data['level'] = LogLevel(data['level'].upper())
        return cls(**data)


@dataclass
class BotConfig:
    """Main bot configuration."""
    # API Configuration
    testnet: bool = True
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    
    # Strategy Configuration
    strategies: Dict[str, StrategyConfig] = field(default_factory=lambda: {
        "liquidity": StrategyConfig(),
        "momentum": StrategyConfig(),
        "chart_patterns": StrategyConfig(),
        "candlestick_patterns": StrategyConfig()
    })
    
    # Component Configuration
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    notification_config: NotificationConfig = field(default_factory=NotificationConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Trading Configuration
    trading_enabled: bool = True
    dry_run: bool = False  # Paper trading mode
    auto_start: bool = False  # Auto-start trading on bot startup
    
    # Data Configuration
    data_retention_days: int = 30  # How long to keep historical data
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    def __post_init__(self):
        """Validate bot configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate bot configuration parameters."""
        if not isinstance(self.testnet, bool):
            raise ValueError("testnet must be a boolean")
        
        if not self.symbols:
            raise ValueError("At least one symbol must be specified")
        
        # Validate symbols format
        symbol_pattern = r'^[A-Z]{2,10}USDT?$'
        import re
        for symbol in self.symbols:
            if not re.match(symbol_pattern, symbol):
                raise ValueError(f"Invalid symbol format: {symbol}")
        
        # Validate strategies
        if not self.strategies:
            raise ValueError("At least one strategy must be configured")
        
        required_strategies = ["liquidity", "momentum", "chart_patterns", "candlestick_patterns"]
        for strategy_name in required_strategies:
            if strategy_name not in self.strategies:
                raise ValueError(f"Required strategy missing: {strategy_name}")
        
        # Check if at least one strategy is enabled
        if not any(config.enabled for config in self.strategies.values()):
            raise ValueError("At least one strategy must be enabled")
        
        if not 1 <= self.data_retention_days <= 365:
            raise ValueError("data_retention_days must be between 1 and 365")
        
        if not 1 <= self.backup_interval_hours <= 168:
            raise ValueError("backup_interval_hours must be between 1 and 168 (1 week)")
    
    def get_enabled_strategies(self) -> List[str]:
        """Get list of enabled strategy names."""
        return [name for name, config in self.strategies.items() if config.enabled]
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get configuration for specific strategy."""
        return self.strategies.get(strategy_name)
    
    def update_strategy_config(self, strategy_name: str, config: StrategyConfig) -> None:
        """Update configuration for specific strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        self.strategies[strategy_name] = config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'testnet': self.testnet,
            'symbols': self.symbols,
            'strategies': {name: config.to_dict() for name, config in self.strategies.items()},
            'risk_config': self.risk_config.to_dict(),
            'notification_config': self.notification_config.to_dict(),
            'logging_config': self.logging_config.to_dict(),
            'trading_enabled': self.trading_enabled,
            'dry_run': self.dry_run,
            'auto_start': self.auto_start,
            'data_retention_days': self.data_retention_days,
            'backup_enabled': self.backup_enabled,
            'backup_interval_hours': self.backup_interval_hours
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BotConfig':
        """Create from dictionary."""
        data = data.copy()
        
        # Convert nested configurations
        if 'strategies' in data:
            data['strategies'] = {
                name: StrategyConfig.from_dict(config) 
                for name, config in data['strategies'].items()
            }
        
        if 'risk_config' in data:
            data['risk_config'] = RiskConfig.from_dict(data['risk_config'])
        
        if 'notification_config' in data:
            data['notification_config'] = NotificationConfig.from_dict(data['notification_config'])
        
        if 'logging_config' in data:
            data['logging_config'] = LoggingConfig.from_dict(data['logging_config'])
        
        return cls(**data)
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'BotConfig':
        """Load configuration from JSON file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


# Default configuration factory
def create_default_config() -> BotConfig:
    """Create default bot configuration."""
    return BotConfig()


# Configuration validation functions
def validate_risk_config(config: Dict[str, Any]) -> bool:
    """Validate risk configuration dictionary."""
    try:
        RiskConfig.from_dict(config)
        return True
    except (ValueError, TypeError):
        return False


def validate_notification_config(config: Dict[str, Any]) -> bool:
    """Validate notification configuration dictionary."""
    try:
        NotificationConfig.from_dict(config)
        return True
    except (ValueError, TypeError):
        return False


def validate_strategy_config(config: Dict[str, Any]) -> bool:
    """Validate strategy configuration dictionary."""
    try:
        StrategyConfig.from_dict(config)
        return True
    except (ValueError, TypeError):
        return False


def validate_bot_config(config: Dict[str, Any]) -> bool:
    """Validate bot configuration dictionary."""
    try:
        BotConfig.from_dict(config)
        return True
    except (ValueError, TypeError):
        return False