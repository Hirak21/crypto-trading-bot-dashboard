"""
Core trading data models for the crypto trading bot.

This module defines the fundamental data structures used throughout
the trading system with validation and serialization capabilities.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import json
import uuid


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"


class SignalAction(Enum):
    """Trading signal action enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class OrderBook:
    """Order book data structure."""
    symbol: str
    timestamp: datetime
    bids: List[tuple[float, float]]  # [(price, quantity), ...]
    asks: List[tuple[float, float]]  # [(price, quantity), ...]
    
    def __post_init__(self):
        """Validate order book data after initialization."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        
        if not self.bids or not self.asks:
            raise ValueError("Order book must have both bids and asks")
        
        # Sort bids (highest price first) and asks (lowest price first)
        self.bids.sort(key=lambda x: x[0], reverse=True)
        self.asks.sort(key=lambda x: x[0])
    
    @property
    def best_bid(self) -> tuple[float, float]:
        """Get best bid (highest price)."""
        return self.bids[0] if self.bids else (0.0, 0.0)
    
    @property
    def best_ask(self) -> tuple[float, float]:
        """Get best ask (lowest price)."""
        return self.asks[0] if self.asks else (0.0, 0.0)
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if not self.bids or not self.asks:
            return 0.0
        return self.best_ask[0] - self.best_bid[0]
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        if not self.bids or not self.asks:
            return 0.0
        return (self.best_bid[0] + self.best_ask[0]) / 2
    
    def get_depth(self, levels: int = 10) -> Dict[str, Any]:
        """Get order book depth for specified levels."""
        return {
            'bids': self.bids[:levels],
            'asks': self.asks[:levels],
            'bid_volume': sum(qty for _, qty in self.bids[:levels]),
            'ask_volume': sum(qty for _, qty in self.asks[:levels])
        }


@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: float
    ask: float
    orderbook: Optional[OrderBook] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    
    def __post_init__(self):
        """Validate market data after initialization."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        
        if self.price <= 0:
            raise ValueError("Price must be positive")
        
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
        
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("Bid and ask prices must be positive")
        
        if self.bid >= self.ask:
            raise ValueError("Bid price must be less than ask price")
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_percentage(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid_price = (self.bid + self.ask) / 2
        return (self.spread / mid_price) * 100 if mid_price > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'change_24h': self.change_24h,
            'volume_24h': self.volume_24h
        }


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    symbol: str
    action: SignalAction
    confidence: float
    strategy: str
    timestamp: datetime = field(default_factory=datetime.now)
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Validate trading signal after initialization."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        
        if not isinstance(self.action, SignalAction):
            if isinstance(self.action, str):
                self.action = SignalAction(self.action)
            else:
                raise ValueError("Invalid signal action")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if not self.strategy:
            raise ValueError("Strategy name cannot be empty")
        
        # Validate price levels if provided
        if self.target_price is not None and self.target_price <= 0:
            raise ValueError("Target price must be positive")
        
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("Stop loss price must be positive")
        
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("Take profit price must be positive")
        
        if self.position_size is not None and self.position_size <= 0:
            raise ValueError("Position size must be positive")
    
    def is_valid(self) -> bool:
        """Check if signal is valid for execution."""
        return (
            self.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.CLOSE] and
            self.confidence > 0.0 and
            bool(self.symbol) and
            bool(self.strategy)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'action': self.action.value,
            'confidence': self.confidence,
            'strategy': self.strategy,
            'timestamp': self.timestamp.isoformat(),
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Create TradingSignal from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['action'] = SignalAction(data['action'])
        return cls(**data)


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    timestamp: datetime = field(default_factory=datetime.now)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: Optional[str] = None
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate position after initialization."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        
        if not isinstance(self.side, PositionSide):
            if isinstance(self.side, str):
                self.side = PositionSide(self.side)
            else:
                raise ValueError("Invalid position side")
        
        if self.size <= 0:
            raise ValueError("Position size must be positive")
        
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        
        if self.current_price <= 0:
            raise ValueError("Current price must be positive")
        
        # Validate stop loss and take profit levels
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("Stop loss price must be positive")
        
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("Take profit price must be positive")
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.size
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.size
    
    @property
    def unrealized_pnl_percentage(self) -> float:
        """Calculate unrealized P&L as percentage."""
        if self.entry_price == 0:
            return 0.0
        
        if self.side == PositionSide.LONG:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            return ((self.entry_price - self.current_price) / self.entry_price) * 100
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of position."""
        return self.current_price * self.size
    
    def update_price(self, new_price: float) -> None:
        """Update current price."""
        if new_price <= 0:
            raise ValueError("Price must be positive")
        self.current_price = new_price
    
    def should_close_stop_loss(self) -> bool:
        """Check if position should be closed due to stop loss."""
        if self.stop_loss is None:
            return False
        
        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss
        else:  # SHORT
            return self.current_price >= self.stop_loss
    
    def should_close_take_profit(self) -> bool:
        """Check if position should be closed due to take profit."""
        if self.take_profit is None:
            return False
        
        if self.side == PositionSide.LONG:
            return self.current_price >= self.take_profit
        else:  # SHORT
            return self.current_price <= self.take_profit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'timestamp': self.timestamp.isoformat(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'strategy': self.strategy,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_percentage': self.unrealized_pnl_percentage,
            'notional_value': self.notional_value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create Position from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['side'] = PositionSide(data['side'])
        # Remove calculated fields
        for field in ['unrealized_pnl', 'unrealized_pnl_percentage', 'notional_value']:
            data.pop(field, None)
        return cls(**data)


@dataclass
class Trade:
    """Trade execution data structure."""
    symbol: str
    side: OrderSide
    size: float
    price: float
    commission: float
    timestamp: datetime = field(default_factory=datetime.now)
    strategy: Optional[str] = None
    order_type: OrderType = OrderType.MARKET
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    pnl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate trade after initialization."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        
        if not isinstance(self.side, OrderSide):
            if isinstance(self.side, str):
                self.side = OrderSide(self.side)
            else:
                raise ValueError("Invalid order side")
        
        if not isinstance(self.order_type, OrderType):
            if isinstance(self.order_type, str):
                self.order_type = OrderType(self.order_type)
            else:
                raise ValueError("Invalid order type")
        
        if self.size <= 0:
            raise ValueError("Trade size must be positive")
        
        if self.price <= 0:
            raise ValueError("Trade price must be positive")
        
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of trade."""
        return self.price * self.size
    
    @property
    def net_pnl(self) -> float:
        """Calculate net P&L including commission."""
        if self.pnl is None:
            return -self.commission
        return self.pnl - self.commission
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'price': self.price,
            'commission': self.commission,
            'timestamp': self.timestamp.isoformat(),
            'strategy': self.strategy,
            'order_type': self.order_type.value,
            'order_id': self.order_id,
            'position_id': self.position_id,
            'pnl': self.pnl,
            'net_pnl': self.net_pnl,
            'notional_value': self.notional_value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create Trade from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['side'] = OrderSide(data['side'])
        data['order_type'] = OrderType(data['order_type'])
        # Remove calculated fields
        for field in ['net_pnl', 'notional_value']:
            data.pop(field, None)
        return cls(**data)


# Utility functions for serialization
def serialize_to_json(obj) -> str:
    """Serialize trading objects to JSON string."""
    if hasattr(obj, 'to_dict'):
        return json.dumps(obj.to_dict(), indent=2)
    else:
        raise ValueError(f"Object {type(obj)} does not support serialization")


def deserialize_from_json(json_str: str, obj_type: type):
    """Deserialize JSON string to trading object."""
    data = json.loads(json_str)
    
    if hasattr(obj_type, 'from_dict'):
        return obj_type.from_dict(data)
    else:
        raise ValueError(f"Object type {obj_type} does not support deserialization")