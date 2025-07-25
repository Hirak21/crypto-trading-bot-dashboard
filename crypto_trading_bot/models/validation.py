"""
Data validation utilities for trading models.

This module provides comprehensive validation functions for ensuring
data integrity throughout the trading system.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import re
import logging

from .trading import (
    TradingSignal, Position, Trade, MarketData, OrderBook,
    SignalAction, PositionSide, OrderSide, OrderType
)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """Comprehensive data validator for trading models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Symbol validation pattern (e.g., BTCUSDT, ETHUSDT)
        self.symbol_pattern = re.compile(r'^[A-Z]{2,10}USDT?$')
        
        # Price validation limits
        self.min_price = 0.000001
        self.max_price = 1000000.0
        
        # Volume validation limits
        self.min_volume = 0.0
        self.max_volume = 1000000000.0
        
        # Time validation limits
        self.max_age_minutes = 60  # Maximum age for market data
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate trading symbol format."""
        if not symbol or not isinstance(symbol, str):
            return False
        
        return bool(self.symbol_pattern.match(symbol.upper()))
    
    def validate_price(self, price: float, field_name: str = "price") -> bool:
        """Validate price value."""
        if not isinstance(price, (int, float)):
            raise ValidationError(f"{field_name} must be a number")
        
        if price <= 0:
            raise ValidationError(f"{field_name} must be positive")
        
        if price < self.min_price or price > self.max_price:
            raise ValidationError(
                f"{field_name} {price} is outside valid range "
                f"({self.min_price} - {self.max_price})"
            )
        
        return True
    
    def validate_volume(self, volume: float, field_name: str = "volume") -> bool:
        """Validate volume value."""
        if not isinstance(volume, (int, float)):
            raise ValidationError(f"{field_name} must be a number")
        
        if volume < 0:
            raise ValidationError(f"{field_name} cannot be negative")
        
        if volume > self.max_volume:
            raise ValidationError(
                f"{field_name} {volume} exceeds maximum allowed ({self.max_volume})"
            )
        
        return True
    
    def validate_timestamp(self, timestamp: datetime, field_name: str = "timestamp") -> bool:
        """Validate timestamp value."""
        if not isinstance(timestamp, datetime):
            raise ValidationError(f"{field_name} must be a datetime object")
        
        now = datetime.now()
        max_age = timedelta(minutes=self.max_age_minutes)
        
        # Check if timestamp is too old
        if now - timestamp > max_age:
            raise ValidationError(
                f"{field_name} is too old (max age: {self.max_age_minutes} minutes)"
            )
        
        # Check if timestamp is in the future (with small tolerance)
        future_tolerance = timedelta(seconds=30)
        if timestamp > now + future_tolerance:
            raise ValidationError(f"{field_name} cannot be in the future")
        
        return True
    
    def validate_confidence(self, confidence: float) -> bool:
        """Validate confidence score."""
        if not isinstance(confidence, (int, float)):
            raise ValidationError("Confidence must be a number")
        
        if not 0.0 <= confidence <= 1.0:
            raise ValidationError("Confidence must be between 0.0 and 1.0")
        
        return True
    
    def validate_market_data(self, market_data: MarketData) -> bool:
        """Validate MarketData object."""
        try:
            # Validate symbol
            if not self.validate_symbol(market_data.symbol):
                raise ValidationError(f"Invalid symbol: {market_data.symbol}")
            
            # Validate timestamp
            self.validate_timestamp(market_data.timestamp)
            
            # Validate prices
            self.validate_price(market_data.price, "price")
            self.validate_price(market_data.bid, "bid")
            self.validate_price(market_data.ask, "ask")
            
            # Validate volume
            self.validate_volume(market_data.volume)
            
            # Validate bid-ask relationship
            if market_data.bid >= market_data.ask:
                raise ValidationError("Bid price must be less than ask price")
            
            # Validate spread reasonableness (not more than 10%)
            spread_pct = market_data.spread_percentage
            if spread_pct > 10.0:
                raise ValidationError(f"Bid-ask spread too wide: {spread_pct:.2f}%")
            
            # Validate optional 24h data
            if market_data.high_24h is not None:
                self.validate_price(market_data.high_24h, "high_24h")
            
            if market_data.low_24h is not None:
                self.validate_price(market_data.low_24h, "low_24h")
            
            if market_data.volume_24h is not None:
                self.validate_volume(market_data.volume_24h, "volume_24h")
            
            # Validate order book if present
            if market_data.orderbook:
                self.validate_order_book(market_data.orderbook)
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Market data validation error: {e}")
    
    def validate_order_book(self, order_book: OrderBook) -> bool:
        """Validate OrderBook object."""
        try:
            # Validate symbol
            if not self.validate_symbol(order_book.symbol):
                raise ValidationError(f"Invalid order book symbol: {order_book.symbol}")
            
            # Validate timestamp
            self.validate_timestamp(order_book.timestamp)
            
            # Validate bids and asks
            if not order_book.bids:
                raise ValidationError("Order book must have bids")
            
            if not order_book.asks:
                raise ValidationError("Order book must have asks")
            
            # Validate bid levels
            for i, (price, qty) in enumerate(order_book.bids):
                self.validate_price(price, f"bid[{i}].price")
                self.validate_volume(qty, f"bid[{i}].quantity")
            
            # Validate ask levels
            for i, (price, qty) in enumerate(order_book.asks):
                self.validate_price(price, f"ask[{i}].price")
                self.validate_volume(qty, f"ask[{i}].quantity")
            
            # Validate bid-ask relationship
            best_bid = order_book.best_bid[0]
            best_ask = order_book.best_ask[0]
            
            if best_bid >= best_ask:
                raise ValidationError(
                    f"Best bid ({best_bid}) must be less than best ask ({best_ask})"
                )
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Order book validation error: {e}")
    
    def validate_trading_signal(self, signal: TradingSignal) -> bool:
        """Validate TradingSignal object."""
        try:
            # Validate symbol
            if not self.validate_symbol(signal.symbol):
                raise ValidationError(f"Invalid signal symbol: {signal.symbol}")
            
            # Validate action
            if signal.action not in SignalAction:
                raise ValidationError(f"Invalid signal action: {signal.action}")
            
            # Validate confidence
            self.validate_confidence(signal.confidence)
            
            # Validate strategy name
            if not signal.strategy or not isinstance(signal.strategy, str):
                raise ValidationError("Strategy name must be a non-empty string")
            
            # Validate timestamp
            self.validate_timestamp(signal.timestamp)
            
            # Validate optional price levels
            if signal.target_price is not None:
                self.validate_price(signal.target_price, "target_price")
            
            if signal.stop_loss is not None:
                self.validate_price(signal.stop_loss, "stop_loss")
            
            if signal.take_profit is not None:
                self.validate_price(signal.take_profit, "take_profit")
            
            if signal.position_size is not None:
                if signal.position_size <= 0:
                    raise ValidationError("Position size must be positive")
            
            # Validate signal logic consistency
            if signal.action == SignalAction.BUY:
                if signal.stop_loss and signal.target_price:
                    if signal.stop_loss >= signal.target_price:
                        raise ValidationError(
                            "For BUY signal, stop loss must be less than target price"
                        )
            
            elif signal.action == SignalAction.SELL:
                if signal.stop_loss and signal.target_price:
                    if signal.stop_loss <= signal.target_price:
                        raise ValidationError(
                            "For SELL signal, stop loss must be greater than target price"
                        )
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Trading signal validation error: {e}")
    
    def validate_position(self, position: Position) -> bool:
        """Validate Position object."""
        try:
            # Validate symbol
            if not self.validate_symbol(position.symbol):
                raise ValidationError(f"Invalid position symbol: {position.symbol}")
            
            # Validate side
            if position.side not in PositionSide:
                raise ValidationError(f"Invalid position side: {position.side}")
            
            # Validate size
            if position.size <= 0:
                raise ValidationError("Position size must be positive")
            
            # Validate prices
            self.validate_price(position.entry_price, "entry_price")
            self.validate_price(position.current_price, "current_price")
            
            # Validate timestamp
            self.validate_timestamp(position.timestamp)
            
            # Validate optional stop loss and take profit
            if position.stop_loss is not None:
                self.validate_price(position.stop_loss, "stop_loss")
            
            if position.take_profit is not None:
                self.validate_price(position.take_profit, "take_profit")
            
            # Validate position logic consistency
            if position.side == PositionSide.LONG:
                if position.stop_loss and position.stop_loss >= position.entry_price:
                    raise ValidationError(
                        "For LONG position, stop loss must be below entry price"
                    )
                if position.take_profit and position.take_profit <= position.entry_price:
                    raise ValidationError(
                        "For LONG position, take profit must be above entry price"
                    )
            
            else:  # SHORT position
                if position.stop_loss and position.stop_loss <= position.entry_price:
                    raise ValidationError(
                        "For SHORT position, stop loss must be above entry price"
                    )
                if position.take_profit and position.take_profit >= position.entry_price:
                    raise ValidationError(
                        "For SHORT position, take profit must be below entry price"
                    )
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Position validation error: {e}")
    
    def validate_trade(self, trade: Trade) -> bool:
        """Validate Trade object."""
        try:
            # Validate symbol
            if not self.validate_symbol(trade.symbol):
                raise ValidationError(f"Invalid trade symbol: {trade.symbol}")
            
            # Validate side
            if trade.side not in OrderSide:
                raise ValidationError(f"Invalid trade side: {trade.side}")
            
            # Validate order type
            if trade.order_type not in OrderType:
                raise ValidationError(f"Invalid order type: {trade.order_type}")
            
            # Validate size
            if trade.size <= 0:
                raise ValidationError("Trade size must be positive")
            
            # Validate price
            self.validate_price(trade.price, "price")
            
            # Validate commission
            if trade.commission < 0:
                raise ValidationError("Commission cannot be negative")
            
            # Validate timestamp
            self.validate_timestamp(trade.timestamp)
            
            # Validate strategy name if provided
            if trade.strategy and not isinstance(trade.strategy, str):
                raise ValidationError("Strategy name must be a string")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Trade validation error: {e}")
    
    def validate_batch(self, objects: List[Any]) -> Dict[str, Any]:
        """Validate a batch of objects and return results."""
        results = {
            'valid': [],
            'invalid': [],
            'errors': []
        }
        
        for i, obj in enumerate(objects):
            try:
                if isinstance(obj, MarketData):
                    self.validate_market_data(obj)
                elif isinstance(obj, TradingSignal):
                    self.validate_trading_signal(obj)
                elif isinstance(obj, Position):
                    self.validate_position(obj)
                elif isinstance(obj, Trade):
                    self.validate_trade(obj)
                elif isinstance(obj, OrderBook):
                    self.validate_order_book(obj)
                else:
                    raise ValidationError(f"Unsupported object type: {type(obj)}")
                
                results['valid'].append(i)
                
            except ValidationError as e:
                results['invalid'].append(i)
                results['errors'].append(f"Object {i}: {str(e)}")
                self.logger.warning(f"Validation failed for object {i}: {e}")
            
            except Exception as e:
                results['invalid'].append(i)
                results['errors'].append(f"Object {i}: Unexpected error - {str(e)}")
                self.logger.error(f"Unexpected validation error for object {i}: {e}")
        
        return results


# Global validator instance
validator = DataValidator()


# Convenience functions
def validate_market_data(market_data: MarketData) -> bool:
    """Validate market data."""
    return validator.validate_market_data(market_data)


def validate_trading_signal(signal: TradingSignal) -> bool:
    """Validate trading signal."""
    return validator.validate_trading_signal(signal)


def validate_position(position: Position) -> bool:
    """Validate position."""
    return validator.validate_position(position)


def validate_trade(trade: Trade) -> bool:
    """Validate trade."""
    return validator.validate_trade(trade)