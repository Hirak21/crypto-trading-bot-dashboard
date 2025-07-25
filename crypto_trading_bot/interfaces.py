"""
Core interfaces and abstract base classes for the crypto trading bot.

This module defines the contracts that all components must implement,
ensuring consistency and enabling dependency injection.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models.trading import TradingSignal, MarketData, Position, Trade


class IStrategy(ABC):
    """Interface for trading strategies."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the strategy name."""
        pass
    
    @abstractmethod
    def analyze(self, market_data: MarketData) -> Optional[TradingSignal]:
        """Analyze market data and generate trading signal if conditions are met."""
        pass
    
    @abstractmethod
    def get_confidence(self) -> float:
        """Return current strategy confidence level (0.0 - 1.0)."""
        pass
    
    @abstractmethod
    def update_performance(self, trade: Trade) -> None:
        """Update strategy performance metrics based on completed trade."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters."""
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Update strategy parameters. Return True if successful."""
        pass


class IMarketDataProvider(ABC):
    """Interface for market data providers."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to market data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to market data source."""
        pass
    
    @abstractmethod
    async def subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to market data for a symbol."""
        pass
    
    @abstractmethod
    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe from market data for a symbol."""
        pass
    
    @abstractmethod
    async def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get the latest market data for a symbol."""
        pass


class ITradeExecutor(ABC):
    """Interface for trade execution."""
    
    @abstractmethod
    async def execute_trade(self, signal: TradingSignal) -> Optional[Trade]:
        """Execute a trade based on trading signal."""
        pass
    
    @abstractmethod
    async def close_position(self, position: Position) -> Optional[Trade]:
        """Close an existing position."""
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> float:
        """Get current account balance."""
        pass
    
    @abstractmethod
    async def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        pass


class IRiskManager(ABC):
    """Interface for risk management."""
    
    @abstractmethod
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate if signal meets risk criteria."""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, account_balance: float) -> float:
        """Calculate appropriate position size based on risk parameters."""
        pass
    
    @abstractmethod
    def should_stop_trading(self) -> bool:
        """Check if trading should be halted due to risk conditions."""
        pass
    
    @abstractmethod
    def update_portfolio_metrics(self, positions: List[Position]) -> None:
        """Update portfolio risk metrics."""
        pass


class INotificationService(ABC):
    """Interface for notification services."""
    
    @abstractmethod
    async def send_alert(self, message: str, level: str = "INFO") -> bool:
        """Send alert notification."""
        pass
    
    @abstractmethod
    async def send_trade_notification(self, trade: Trade) -> bool:
        """Send trade execution notification."""
        pass
    
    @abstractmethod
    async def send_performance_report(self, report: Dict[str, Any]) -> bool:
        """Send performance report."""
        pass


class IDataStorage(ABC):
    """Interface for data storage and persistence."""
    
    @abstractmethod
    async def save_trade(self, trade: Trade) -> bool:
        """Save trade record."""
        pass
    
    @abstractmethod
    async def save_position(self, position: Position) -> bool:
        """Save position record."""
        pass
    
    @abstractmethod
    async def get_trades(self, symbol: str = None, start_date: datetime = None, 
                        end_date: datetime = None) -> List[Trade]:
        """Retrieve trade records with optional filters."""
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: str = None) -> List[Position]:
        """Retrieve position records with optional filters."""
        pass
    
    @abstractmethod
    async def save_market_data(self, market_data: MarketData) -> bool:
        """Save market data for backtesting."""
        pass


class IConfigManager(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from storage."""
        pass
    
    @abstractmethod
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to storage."""
        pass
    
    @abstractmethod
    def get_api_credentials(self) -> tuple[str, str]:
        """Get encrypted API credentials."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        pass