"""
Mock data generators for testing trading bot components.

This module provides utilities to generate realistic test data for
market data, trading signals, and other trading-related objects.
"""

import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from crypto_trading_bot.models.trading import (
    MarketData, TradingSignal, SignalAction, Trade, Position, OrderBook
)
from crypto_trading_bot.models.config import BotConfig, RiskConfig, NotificationConfig, StrategyConfig


class MockDataGenerator:
    """Generates mock data for testing purposes."""
    
    def __init__(self, seed: int = 42):
        """Initialize with optional seed for reproducible tests."""
        random.seed(seed)
        self.base_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'ADAUSDT': 1.5,
            'DOTUSDT': 25.0,
            'LINKUSDT': 20.0
        }
    
    def generate_market_data(self, symbol: str = 'BTCUSDT', 
                           count: int = 100,
                           start_time: Optional[datetime] = None,
                           price_volatility: float = 0.02) -> List[MarketData]:
        """Generate realistic market data with price movements."""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=count)
        
        base_price = self.base_prices.get(symbol, 50000.0)
        current_price = base_price
        market_data = []
        
        for i in range(count):
            # Generate price movement (random walk with drift)
            price_change = random.gauss(0, price_volatility)
            current_price *= (1 + price_change)
            
            # Generate volume (correlated with price volatility)
            base_volume = 1000000
            volume_multiplier = 1 + abs(price_change) * 10
            volume = base_volume * volume_multiplier * random.uniform(0.5, 2.0)
            
            # Generate high/low based on current price
            high_24h = current_price * random.uniform(1.0, 1.05)
            low_24h = current_price * random.uniform(0.95, 1.0)
            
            # Generate bid/ask spread
            spread_pct = random.uniform(0.001, 0.005)  # 0.1% to 0.5%
            bid = current_price * (1 - spread_pct / 2)
            ask = current_price * (1 + spread_pct / 2)
            
            timestamp = start_time + timedelta(minutes=i)
            
            market_data.append(MarketData(
                symbol=symbol,
                timestamp=timestamp,
                price=current_price,
                volume=volume,
                high_24h=high_24h,
                low_24h=low_24h,
                bid=bid,
                ask=ask,
                change_24h=random.uniform(-0.1, 0.1)
            ))
        
        return market_data
    
    def generate_trading_signal(self, symbol: str = 'BTCUSDT',
                              action: Optional[SignalAction] = None,
                              confidence: Optional[float] = None,
                              strategy: str = 'test_strategy') -> TradingSignal:
        """Generate a trading signal."""
        if action is None:
            action = random.choice([SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD])
        
        if confidence is None:
            confidence = random.uniform(0.1, 1.0)
        
        base_price = self.base_prices.get(symbol, 50000.0)
        current_price = base_price * random.uniform(0.95, 1.05)
        
        # Generate stop loss and target price
        if action == SignalAction.BUY:
            stop_loss = current_price * random.uniform(0.95, 0.98)
            target_price = current_price * random.uniform(1.02, 1.08)
        elif action == SignalAction.SELL:
            stop_loss = current_price * random.uniform(1.02, 1.05)
            target_price = current_price * random.uniform(0.92, 0.98)
        else:
            stop_loss = None
            target_price = None
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            strategy=strategy,
            timestamp=datetime.now(),
            metadata={
                'stop_loss': stop_loss,
                'target_price': target_price,
                'current_price': current_price,
                'indicators': {
                    'rsi': random.uniform(20, 80),
                    'macd': random.uniform(-1, 1),
                    'volume_ratio': random.uniform(0.5, 2.0)
                }
            }
        )
    
    def generate_trade(self, symbol: str = 'BTCUSDT',
                      strategy: str = 'test_strategy') -> Trade:
        """Generate a completed trade."""
        base_price = self.base_prices.get(symbol, 50000.0)
        price = base_price * random.uniform(0.95, 1.05)
        quantity = random.uniform(0.001, 1.0)
        side = random.choice(['BUY', 'SELL'])
        
        # Generate realistic P&L
        pnl = random.gauss(0, price * quantity * 0.02)  # 2% volatility
        commission = price * quantity * 0.001  # 0.1% commission
        
        return Trade(
            trade_id=f"trade_{random.randint(1000, 9999)}",
            symbol=symbol,
            side=side,
            size=quantity,
            price=price,
            commission=commission,
            timestamp=datetime.now() - timedelta(minutes=random.randint(1, 1440)),
            strategy=strategy,
            pnl=pnl
        )
    
    def generate_position(self, symbol: str = 'BTCUSDT') -> Position:
        """Generate an open position."""
        base_price = self.base_prices.get(symbol, 50000.0)
        entry_price = base_price * random.uniform(0.95, 1.05)
        current_price = entry_price * random.uniform(0.95, 1.05)
        quantity = random.uniform(0.001, 1.0)
        side = random.choice(['LONG', 'SHORT'])
        
        # Calculate unrealized P&L
        if side == 'LONG':
            unrealized_pnl = (current_price - entry_price) * quantity
        else:
            unrealized_pnl = (entry_price - current_price) * quantity
        
        return Position(
            symbol=symbol,
            side=side,
            size=quantity,
            entry_price=entry_price,
            current_price=current_price,
            timestamp=datetime.now() - timedelta(minutes=random.randint(1, 1440))
        )
    
    def generate_order_book(self, symbol: str = 'BTCUSDT',
                           depth: int = 20) -> OrderBook:
        """Generate order book data."""
        base_price = self.base_prices.get(symbol, 50000.0)
        current_price = base_price * random.uniform(0.95, 1.05)
        
        bids = []
        asks = []
        
        # Generate bids (below current price)
        for i in range(depth):
            price = current_price * (1 - (i + 1) * 0.001)  # 0.1% increments
            quantity = random.uniform(0.1, 10.0)
            bids.append([price, quantity])
        
        # Generate asks (above current price)
        for i in range(depth):
            price = current_price * (1 + (i + 1) * 0.001)  # 0.1% increments
            quantity = random.uniform(0.1, 10.0)
            asks.append([price, quantity])
        
        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.now()
        )
    
    def generate_bot_config(self) -> BotConfig:
        """Generate a test bot configuration."""
        return BotConfig(
            testnet=True,
            symbols=['BTCUSDT', 'ETHUSDT'],
            strategies={
                'liquidity': StrategyConfig(enabled=True),
                'momentum': StrategyConfig(enabled=True),
                'chart_patterns': StrategyConfig(enabled=False),
                'candlestick_patterns': StrategyConfig(enabled=True)
            },
            risk_config=self.generate_risk_config(),
            notification_config=self.generate_notification_config(),
            trading_enabled=True,
            dry_run=True
        )
    
    def generate_risk_config(self) -> RiskConfig:
        """Generate a test risk configuration."""
        return RiskConfig(
            max_position_size=0.02,
            daily_loss_limit=0.05,
            max_drawdown=0.15,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            max_open_positions=5,
            min_account_balance=100.0,
            risk_free_rate=0.02
        )
    
    def generate_notification_config(self) -> NotificationConfig:
        """Generate a test notification configuration."""
        return NotificationConfig(
            enabled=True,
            console={'enabled': True, 'min_level': 'info'},
            email=None,
            webhook=None,
            trade_notifications=True,
            error_notifications=True,
            performance_notifications=True,
            system_notifications=True
        )
    
    def generate_price_series(self, length: int = 100,
                            start_price: float = 50000.0,
                            volatility: float = 0.02,
                            trend: float = 0.0) -> List[float]:
        """Generate a price series with optional trend."""
        prices = [start_price]
        
        for i in range(1, length):
            # Random walk with trend
            change = random.gauss(trend, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        return prices
    
    def generate_indicator_data(self, length: int = 100) -> Dict[str, List[float]]:
        """Generate mock technical indicator data."""
        return {
            'rsi': [random.uniform(20, 80) for _ in range(length)],
            'macd': [random.gauss(0, 0.5) for _ in range(length)],
            'sma_20': [random.uniform(45000, 55000) for _ in range(length)],
            'ema_12': [random.uniform(45000, 55000) for _ in range(length)],
            'bollinger_upper': [random.uniform(52000, 58000) for _ in range(length)],
            'bollinger_lower': [random.uniform(42000, 48000) for _ in range(length)],
            'volume': [random.uniform(500000, 2000000) for _ in range(length)]
        }


# Global instance for easy access
mock_generator = MockDataGenerator()


# Convenience functions
def create_mock_market_data(count: int = 100, symbol: str = 'BTCUSDT') -> List[MarketData]:
    """Create mock market data."""
    return mock_generator.generate_market_data(symbol=symbol, count=count)


def create_mock_trading_signal(action: SignalAction = SignalAction.BUY,
                             confidence: float = 0.8) -> TradingSignal:
    """Create mock trading signal."""
    return mock_generator.generate_trading_signal(action=action, confidence=confidence)


def create_mock_trade() -> Trade:
    """Create mock trade."""
    return mock_generator.generate_trade()


def create_mock_position() -> Position:
    """Create mock position."""
    return mock_generator.generate_position()


def create_mock_config() -> BotConfig:
    """Create mock bot configuration."""
    return mock_generator.generate_bot_config()


def create_price_series(length: int = 100, trend: float = 0.0) -> List[float]:
    """Create mock price series."""
    return mock_generator.generate_price_series(length=length, trend=trend)