"""
Trade Manager for order execution and position management.

This manager handles order execution, automatic stop-loss placement,
position tracking, and comprehensive trade logging.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

from ..models.trading import TradingSignal, MarketData, SignalAction
from ..models.config import TradingConfig
from ..api.binance_client import BinanceClient
from ..utils.logging_config import setup_logging


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT_LIMIT"
    OCO = "OCO"  # One-Cancels-Other


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class Order(NamedTuple):
    """Order structure."""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float
    avg_fill_price: float
    timestamp: datetime
    strategy_name: str
    metadata: Dict[str, Any]


class Trade(NamedTuple):
    """Trade structure for completed transactions."""
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    realized_pnl: float
    commission: float
    strategy_name: str
    orders: List[str]  # List of order IDs
    metadata: Dict[str, Any]


class Position:
    """Active position tracking."""
    
    def __init__(self, symbol: str, side: OrderSide, quantity: float, 
                 entry_price: float, strategy_name: str, entry_order_id: str):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.strategy_name = strategy_name
        self.entry_order_id = entry_order_id
        self.entry_time = datetime.now()
        
        # Position tracking
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.position_value = abs(quantity * entry_price)
        
        # Risk management orders
        self.stop_loss_order_id: Optional[str] = None
        self.take_profit_order_id: Optional[str] = None
        self.stop_loss_price: Optional[float] = None
        self.take_profit_price: Optional[float] = None
        
        # Tracking
        self.price_updates = deque(maxlen=1000)
        self.last_updated = datetime.now()
        
        # Status
        self.is_active = True
        self.exit_reason: Optional[str] = None
    
    def update_price(self, current_price: float):
        """Update position with current market price."""
        try:
            self.current_price = current_price
            self.price_updates.append((datetime.now(), current_price))
            
            # Calculate unrealized P&L
            if self.side == OrderSide.BUY:
                self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            else:  # SELL (short position)
                self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            
            self.unrealized_pnl_pct = self.unrealized_pnl / self.position_value if self.position_value > 0 else 0
            self.last_updated = datetime.now()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error updating position price for {self.symbol}: {e}")
    
    def set_stop_loss(self, stop_loss_price: float, order_id: str):
        """Set stop loss for the position."""
        self.stop_loss_price = stop_loss_price
        self.stop_loss_order_id = order_id
    
    def set_take_profit(self, take_profit_price: float, order_id: str):
        """Set take profit for the position."""
        self.take_profit_price = take_profit_price
        self.take_profit_order_id = order_id
    
    def close_position(self, exit_reason: str):
        """Mark position as closed."""
        self.is_active = False
        self.exit_reason = exit_reason
    
    def get_position_info(self) -> Dict[str, Any]:
        """Get comprehensive position information."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'position_value': self.position_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'strategy_name': self.strategy_name,
            'entry_time': self.entry_time,
            'entry_order_id': self.entry_order_id,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'stop_loss_order_id': self.stop_loss_order_id,
            'take_profit_order_id': self.take_profit_order_id,
            'is_active': self.is_active,
            'exit_reason': self.exit_reason,
            'last_updated': self.last_updated
        }


class TradeManager:
    """Comprehensive trade execution and position management system."""
    
    def __init__(self, config: TradingConfig, binance_client: BinanceClient):
        self.logger = setup_logging(__name__)
        self.config = config
        self.binance_client = binance_client
        
        # Order and position tracking
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.completed_trades: List[Trade] = []
        
        # Order management
        self.pending_orders: Dict[str, Order] = {}
        self.order_status_cache: Dict[str, OrderStatus] = {}
        self.order_updates_queue = deque(maxlen=10000)
        
        # Risk management integration
        self.auto_stop_loss_enabled = config.auto_stop_loss_enabled
        self.auto_take_profit_enabled = config.auto_take_profit_enabled
        self.default_stop_loss_pct = config.default_stop_loss_pct
        self.default_take_profit_pct = config.default_take_profit_pct
        
        # Execution settings
        self.max_slippage_pct = config.max_slippage_pct
        self.order_timeout_seconds = config.order_timeout_seconds
        self.retry_attempts = config.retry_attempts
        self.retry_delay_seconds = config.retry_delay_seconds
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.order_monitor_thread = None
        self.position_monitor_thread = None
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.execution_metrics = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'avg_execution_time': 0.0,
            'total_slippage': 0.0,
            'avg_slippage': 0.0,
            'total_commission': 0.0
        }
        
        # Trade logging
        self.trade_log = deque(maxlen=10000)
        self.error_log = deque(maxlen=1000)
        
        # Start monitoring threads
        self._start_monitoring_threads()
        
        self.logger.info("Trade Manager initialized successfully")
    
    def execute_trade(self, signal: TradingSignal, position_size: float, 
                     market_data: MarketData) -> Tuple[bool, str, Optional[str]]:
        """Execute a trading signal with comprehensive order management."""
        try:
            self.logger.info(f"Executing trade: {signal.action.value} {signal.symbol} "
                           f"size: {position_size:.6f} @ {market_data.price:.6f}")
            
            # Generate unique client order ID
            client_order_id = f"{signal.strategy_name}_{signal.symbol}_{int(datetime.now().timestamp())}"
            
            # Determine order side
            order_side = OrderSide.BUY if signal.action == SignalAction.BUY else OrderSide.SELL
            
            # Execute market order
            success, message, order_id = self._execute_market_order(
                symbol=signal.symbol,
                side=order_side,
                quantity=abs(position_size),
                client_order_id=client_order_id,
                strategy_name=signal.strategy_name,
                signal_metadata=signal.metadata
            )
            
            if not success:
                self.logger.error(f"Failed to execute trade: {message}")
                return False, message, None
            
            # Create position tracking
            if order_id:
                self._create_position_from_order(order_id, signal, market_data)
                
                # Place automatic stop loss and take profit orders
                if self.auto_stop_loss_enabled or self.auto_take_profit_enabled:
                    self._place_risk_management_orders(order_id, signal, market_data)
            
            self.logger.info(f"Trade executed successfully: {order_id}")
            return True, "Trade executed successfully", order_id
            
        except Exception as e:
            error_msg = f"Error executing trade: {str(e)}"
            self.logger.error(error_msg)
            self.error_log.append({
                'timestamp': datetime.now(),
                'error': error_msg,
                'signal': signal,
                'position_size': position_size
            })
            return False, error_msg, None
    
    async def _execute_market_order(self, symbol: str, side: OrderSide, quantity: float,
                             client_order_id: str, strategy_name: str,
                             signal_metadata: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        """Execute a market order with retry logic."""
        try:
            start_time = datetime.now()
            
            for attempt in range(self.retry_attempts):
                try:
                    # Execute order through Binance client
                    order_result = self.binance_client.place_market_order(
                        symbol=symbol,
                        side=side.value,
                        quantity=quantity,
                        client_order_id=client_order_id
                    )
                    
                    if order_result and 'orderId' in order_result:
                        order_id = str(order_result['orderId'])
                        
                        # Create order record
                        order = Order(
                            order_id=order_id,
                            client_order_id=client_order_id,
                            symbol=symbol,
                            side=side,
                            order_type=OrderType.MARKET,
                            quantity=quantity,
                            price=None,  # Market order
                            stop_price=None,
                            status=OrderStatus.SUBMITTED,
                            filled_quantity=0.0,
                            avg_fill_price=0.0,
                            timestamp=datetime.now(),
                            strategy_name=strategy_name,
                            metadata=signal_metadata
                        )
                        
                        # Store order
                        self.orders[order_id] = order
                        self.pending_orders[order_id] = order
                        
                        # Update metrics
                        execution_time = (datetime.now() - start_time).total_seconds()
                        self._update_execution_metrics(execution_time, True)
                        
                        return True, "Order placed successfully", order_id
                    
                    else:
                        error_msg = f"Invalid order result: {order_result}"
                        self.logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                        
                except Exception as e:
                    error_msg = f"Order execution attempt {attempt + 1} failed: {str(e)}"
                    self.logger.warning(error_msg)
                    
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(self.retry_delay_seconds)
            
            # All attempts failed
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_execution_metrics(execution_time, False)
            
            return False, f"Order failed after {self.retry_attempts} attempts", None
            
        except Exception as e:
            error_msg = f"Critical error in order execution: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg, None
    
    def _create_position_from_order(self, order_id: str, signal: TradingSignal, market_data: MarketData):
        """Create position tracking from executed order."""
        try:
            if order_id not in self.orders:
                self.logger.error(f"Order {order_id} not found for position creation")
                return
            
            order = self.orders[order_id]
            
            # Wait for order fill information
            fill_info = self._wait_for_order_fill(order_id, timeout_seconds=30)
            if not fill_info:
                self.logger.error(f"Failed to get fill information for order {order_id}")
                return
            
            # Create position
            order_side = OrderSide.BUY if signal.action == SignalAction.BUY else OrderSide.SELL
            
            position = Position(
                symbol=signal.symbol,
                side=order_side,
                quantity=fill_info['filled_quantity'],
                entry_price=fill_info['avg_fill_price'],
                strategy_name=signal.strategy_name,
                entry_order_id=order_id
            )
            
            # Store position
            self.positions[signal.symbol] = position
            
            self.logger.info(f"Created position: {signal.symbol} {order_side.value} "
                           f"{fill_info['filled_quantity']:.6f} @ {fill_info['avg_fill_price']:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error creating position from order {order_id}: {e}")
    
    def _wait_for_order_fill(self, order_id: str, timeout_seconds: int = 30) -> Optional[Dict[str, Any]]:
        """Wait for order to be filled and return fill information."""
        try:
            start_time = datetime.now()
            timeout_delta = timedelta(seconds=timeout_seconds)
            
            while datetime.now() - start_time < timeout_delta:
                # Check order status
                order_status = self.binance_client.get_order_status(order_id)
                
                if order_status and order_status.get('status') == 'FILLED':
                    return {
                        'filled_quantity': float(order_status.get('executedQty', 0)),
                        'avg_fill_price': float(order_status.get('price', 0)) if order_status.get('price') else 
                                         float(order_status.get('avgPrice', 0)),
                        'commission': float(order_status.get('commission', 0))
                    }
                
                elif order_status and order_status.get('status') in ['CANCELLED', 'REJECTED', 'EXPIRED']:
                    self.logger.error(f"Order {order_id} failed with status: {order_status.get('status')}")
                    return None
                
                # Wait before next check
                await asyncio.sleep(1)
            
            self.logger.warning(f"Timeout waiting for order {order_id} to fill")
            return None
            
        except Exception as e:
            self.logger.error(f"Error waiting for order fill {order_id}: {e}")
            return None 
   
    def _place_risk_management_orders(self, entry_order_id: str, signal: TradingSignal, market_data: MarketData):
        """Place automatic stop loss and take profit orders."""
        try:
            if entry_order_id not in self.orders:
                return
            
            entry_order = self.orders[entry_order_id]
            position = self.positions.get(signal.symbol)
            
            if not position:
                return
            
            # Calculate stop loss and take profit prices
            stop_loss_price = self._calculate_stop_loss_price(signal, market_data, position)
            take_profit_price = self._calculate_take_profit_price(signal, market_data, position)
            
            # Place stop loss order
            if self.auto_stop_loss_enabled and stop_loss_price:
                stop_order_id = self._place_stop_loss_order(position, stop_loss_price)
                if stop_order_id:
                    position.set_stop_loss(stop_loss_price, stop_order_id)
                    self.logger.info(f"Stop loss placed: {signal.symbol} @ {stop_loss_price:.6f}")
            
            # Place take profit order
            if self.auto_take_profit_enabled and take_profit_price:
                tp_order_id = self._place_take_profit_order(position, take_profit_price)
                if tp_order_id:
                    position.set_take_profit(take_profit_price, tp_order_id)
                    self.logger.info(f"Take profit placed: {signal.symbol} @ {take_profit_price:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error placing risk management orders: {e}")
    
    def _calculate_stop_loss_price(self, signal: TradingSignal, market_data: MarketData, 
                                  position: Position) -> Optional[float]:
        """Calculate stop loss price."""
        try:
            # Check if stop loss is provided in signal metadata
            if 'stop_loss' in signal.metadata and signal.metadata['stop_loss']:
                return float(signal.metadata['stop_loss'])
            
            # Use default percentage-based stop loss
            if position.side == OrderSide.BUY:
                return position.entry_price * (1 - self.default_stop_loss_pct)
            else:  # SELL (short position)
                return position.entry_price * (1 + self.default_stop_loss_pct)
                
        except Exception as e:
            self.logger.error(f"Error calculating stop loss price: {e}")
            return None
    
    def _calculate_take_profit_price(self, signal: TradingSignal, market_data: MarketData, 
                                   position: Position) -> Optional[float]:
        """Calculate take profit price."""
        try:
            # Check if take profit is provided in signal metadata
            if 'target_price' in signal.metadata and signal.metadata['target_price']:
                return float(signal.metadata['target_price'])
            
            # Use default percentage-based take profit
            if position.side == OrderSide.BUY:
                return position.entry_price * (1 + self.default_take_profit_pct)
            else:  # SELL (short position)
                return position.entry_price * (1 - self.default_take_profit_pct)
                
        except Exception as e:
            self.logger.error(f"Error calculating take profit price: {e}")
            return None
    
    def _place_stop_loss_order(self, position: Position, stop_price: float) -> Optional[str]:
        """Place stop loss order."""
        try:
            # Determine order side (opposite of position)
            order_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
            
            # Generate client order ID
            client_order_id = f"SL_{position.symbol}_{int(datetime.now().timestamp())}"
            
            # Place stop loss order
            order_result = self.binance_client.place_stop_loss_order(
                symbol=position.symbol,
                side=order_side.value,
                quantity=abs(position.quantity),
                stop_price=stop_price,
                client_order_id=client_order_id
            )
            
            if order_result and 'orderId' in order_result:
                order_id = str(order_result['orderId'])
                
                # Create order record
                order = Order(
                    order_id=order_id,
                    client_order_id=client_order_id,
                    symbol=position.symbol,
                    side=order_side,
                    order_type=OrderType.STOP_LOSS,
                    quantity=abs(position.quantity),
                    price=None,
                    stop_price=stop_price,
                    status=OrderStatus.SUBMITTED,
                    filled_quantity=0.0,
                    avg_fill_price=0.0,
                    timestamp=datetime.now(),
                    strategy_name=position.strategy_name,
                    metadata={'position_entry_order': position.entry_order_id}
                )
                
                self.orders[order_id] = order
                self.pending_orders[order_id] = order
                
                return order_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error placing stop loss order: {e}")
            return None
    
    def _place_take_profit_order(self, position: Position, target_price: float) -> Optional[str]:
        """Place take profit order."""
        try:
            # Determine order side (opposite of position)
            order_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
            
            # Generate client order ID
            client_order_id = f"TP_{position.symbol}_{int(datetime.now().timestamp())}"
            
            # Place take profit order
            order_result = self.binance_client.place_limit_order(
                symbol=position.symbol,
                side=order_side.value,
                quantity=abs(position.quantity),
                price=target_price,
                client_order_id=client_order_id
            )
            
            if order_result and 'orderId' in order_result:
                order_id = str(order_result['orderId'])
                
                # Create order record
                order = Order(
                    order_id=order_id,
                    client_order_id=client_order_id,
                    symbol=position.symbol,
                    side=order_side,
                    order_type=OrderType.TAKE_PROFIT,
                    quantity=abs(position.quantity),
                    price=target_price,
                    stop_price=None,
                    status=OrderStatus.SUBMITTED,
                    filled_quantity=0.0,
                    avg_fill_price=0.0,
                    timestamp=datetime.now(),
                    strategy_name=position.strategy_name,
                    metadata={'position_entry_order': position.entry_order_id}
                )
                
                self.orders[order_id] = order
                self.pending_orders[order_id] = order
                
                return order_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error placing take profit order: {e}")
            return None
    
    def close_position(self, symbol: str, reason: str = "Manual close") -> Tuple[bool, str]:
        """Manually close a position."""
        try:
            if symbol not in self.positions:
                return False, f"No active position found for {symbol}"
            
            position = self.positions[symbol]
            if not position.is_active:
                return False, f"Position {symbol} is already closed"
            
            # Determine order side (opposite of position)
            order_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
            
            # Generate client order ID
            client_order_id = f"CLOSE_{symbol}_{int(datetime.now().timestamp())}"
            
            # Execute market order to close position
            success, message, order_id = self._execute_market_order(
                symbol=symbol,
                side=order_side,
                quantity=abs(position.quantity),
                client_order_id=client_order_id,
                strategy_name=position.strategy_name,
                signal_metadata={'close_reason': reason}
            )
            
            if success and order_id:
                # Cancel any pending stop loss or take profit orders
                self._cancel_position_orders(position)
                
                # Mark position as closed
                position.close_position(reason)
                
                # Wait for fill and create trade record
                fill_info = self._wait_for_order_fill(order_id)
                if fill_info:
                    self._create_trade_record(position, fill_info, reason)
                
                self.logger.info(f"Position closed: {symbol} - {reason}")
                return True, f"Position {symbol} closed successfully"
            
            return False, f"Failed to close position {symbol}: {message}"
            
        except Exception as e:
            error_msg = f"Error closing position {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def _cancel_position_orders(self, position: Position):
        """Cancel stop loss and take profit orders for a position."""
        try:
            orders_to_cancel = []
            
            if position.stop_loss_order_id:
                orders_to_cancel.append(position.stop_loss_order_id)
            
            if position.take_profit_order_id:
                orders_to_cancel.append(position.take_profit_order_id)
            
            for order_id in orders_to_cancel:
                try:
                    self.binance_client.cancel_order(position.symbol, order_id)
                    self.logger.info(f"Cancelled order: {order_id}")
                    
                    # Update order status
                    if order_id in self.orders:
                        updated_order = self.orders[order_id]._replace(status=OrderStatus.CANCELLED)
                        self.orders[order_id] = updated_order
                        
                        if order_id in self.pending_orders:
                            del self.pending_orders[order_id]
                            
                except Exception as e:
                    self.logger.warning(f"Failed to cancel order {order_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error cancelling position orders: {e}")
    
    def _create_trade_record(self, position: Position, exit_fill_info: Dict[str, Any], exit_reason: str):
        """Create a completed trade record."""
        try:
            # Calculate realized P&L
            if position.side == OrderSide.BUY:
                realized_pnl = (exit_fill_info['avg_fill_price'] - position.entry_price) * position.quantity
            else:  # SELL (short position)
                realized_pnl = (position.entry_price - exit_fill_info['avg_fill_price']) * position.quantity
            
            # Create trade record
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                symbol=position.symbol,
                side=position.side,
                quantity=position.quantity,
                entry_price=position.entry_price,
                exit_price=exit_fill_info['avg_fill_price'],
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                realized_pnl=realized_pnl,
                commission=exit_fill_info.get('commission', 0),
                strategy_name=position.strategy_name,
                orders=[position.entry_order_id],
                metadata={
                    'exit_reason': exit_reason,
                    'position_duration': (datetime.now() - position.entry_time).total_seconds(),
                    'max_unrealized_pnl': getattr(position, 'max_unrealized_pnl', 0),
                    'min_unrealized_pnl': getattr(position, 'min_unrealized_pnl', 0)
                }
            )
            
            self.completed_trades.append(trade)
            
            # Log trade
            self.trade_log.append({
                'timestamp': datetime.now(),
                'action': 'trade_completed',
                'trade': trade,
                'realized_pnl': realized_pnl
            })
            
            self.logger.info(f"Trade completed: {position.symbol} P&L: {realized_pnl:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error creating trade record: {e}")
    
    def update_positions(self, market_data: Dict[str, MarketData]):
        """Update all positions with current market data."""
        try:
            for symbol, position in self.positions.items():
                if symbol in market_data and position.is_active:
                    position.update_price(market_data[symbol].price)
                    
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _start_monitoring_threads(self):
        """Start background monitoring threads."""
        try:
            # Order monitoring thread
            self.order_monitor_thread = threading.Thread(
                target=self._monitor_orders,
                daemon=True,
                name="OrderMonitor"
            )
            self.order_monitor_thread.start()
            
            # Position monitoring thread
            self.position_monitor_thread = threading.Thread(
                target=self._monitor_positions,
                daemon=True,
                name="PositionMonitor"
            )
            self.position_monitor_thread.start()
            
            self.logger.info("Monitoring threads started")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring threads: {e}")
    
    def _monitor_orders(self):
        """Monitor pending orders for status updates."""
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Check pending orders
                    orders_to_remove = []
                    
                    for order_id, order in self.pending_orders.items():
                        try:
                            # Get order status from exchange
                            status_result = self.binance_client.get_order_status(order_id)
                            
                            if status_result:
                                new_status = self._map_order_status(status_result.get('status'))
                                
                                if new_status != order.status:
                                    # Update order
                                    updated_order = order._replace(
                                        status=new_status,
                                        filled_quantity=float(status_result.get('executedQty', 0)),
                                        avg_fill_price=float(status_result.get('avgPrice', 0)) if status_result.get('avgPrice') else 0
                                    )
                                    
                                    self.orders[order_id] = updated_order
                                    
                                    # Log status change
                                    self.order_updates_queue.append({
                                        'timestamp': datetime.now(),
                                        'order_id': order_id,
                                        'old_status': order.status,
                                        'new_status': new_status,
                                        'filled_quantity': updated_order.filled_quantity
                                    })
                                    
                                    # Remove from pending if final status
                                    if new_status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                                                    OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                                        orders_to_remove.append(order_id)
                                        
                                        # Handle position closure if this was a stop loss or take profit
                                        if new_status == OrderStatus.FILLED:
                                            self._handle_position_exit_order(updated_order)
                        
                        except Exception as e:
                            self.logger.warning(f"Error checking order {order_id}: {e}")
                    
                    # Remove completed orders from pending
                    for order_id in orders_to_remove:
                        if order_id in self.pending_orders:
                            del self.pending_orders[order_id]
                    
                    # Sleep before next check
                    self.shutdown_event.wait(5)  # Check every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in order monitoring loop: {e}")
                    self.shutdown_event.wait(10)  # Wait longer on error
                    
        except Exception as e:
            self.logger.error(f"Critical error in order monitoring: {e}")
    
    def _monitor_positions(self):
        """Monitor positions for risk management."""
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Check positions for risk management triggers
                    for symbol, position in self.positions.items():
                        if not position.is_active:
                            continue
                        
                        # Check for position timeout (optional)
                        position_age = datetime.now() - position.entry_time
                        if position_age > timedelta(hours=24):  # 24 hour max position
                            self.logger.warning(f"Position {symbol} has been open for {position_age}")
                    
                    # Sleep before next check
                    self.shutdown_event.wait(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in position monitoring loop: {e}")
                    self.shutdown_event.wait(60)  # Wait longer on error
                    
        except Exception as e:
            self.logger.error(f"Critical error in position monitoring: {e}")
    
    def _handle_position_exit_order(self, order: Order):
        """Handle position exit when stop loss or take profit is filled."""
        try:
            # Find the position this order belongs to
            position = None
            for pos in self.positions.values():
                if (order.order_id == pos.stop_loss_order_id or 
                    order.order_id == pos.take_profit_order_id):
                    position = pos
                    break
            
            if not position:
                return
            
            # Determine exit reason
            exit_reason = "stop_loss" if order.order_id == position.stop_loss_order_id else "take_profit"
            
            # Close position
            position.close_position(exit_reason)
            
            # Cancel remaining orders
            self._cancel_position_orders(position)
            
            # Create trade record
            fill_info = {
                'avg_fill_price': order.avg_fill_price,
                'commission': 0  # Would need to get from order details
            }
            self._create_trade_record(position, fill_info, exit_reason)
            
            self.logger.info(f"Position {position.symbol} closed via {exit_reason}")
            
        except Exception as e:
            self.logger.error(f"Error handling position exit order: {e}")
    
    def _map_order_status(self, exchange_status: str) -> OrderStatus:
        """Map exchange order status to internal status."""
        status_mapping = {
            'NEW': OrderStatus.SUBMITTED,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED
        }
        
        return status_mapping.get(exchange_status, OrderStatus.PENDING)
    
    def _update_execution_metrics(self, execution_time: float, success: bool):
        """Update execution performance metrics."""
        try:
            self.execution_metrics['total_orders'] += 1
            
            if success:
                self.execution_metrics['successful_orders'] += 1
            else:
                self.execution_metrics['failed_orders'] += 1
            
            # Update average execution time
            total_orders = self.execution_metrics['total_orders']
            current_avg = self.execution_metrics['avg_execution_time']
            new_avg = ((current_avg * (total_orders - 1)) + execution_time) / total_orders
            self.execution_metrics['avg_execution_time'] = new_avg
            
        except Exception as e:
            self.logger.error(f"Error updating execution metrics: {e}")
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active positions."""
        try:
            return {
                symbol: position.get_position_info()
                for symbol, position in self.positions.items()
                if position.is_active
            }
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_orders(self, status_filter: Optional[OrderStatus] = None) -> List[Dict[str, Any]]:
        """Get orders with optional status filter."""
        try:
            orders = []
            for order in self.orders.values():
                if status_filter is None or order.status == status_filter:
                    orders.append({
                        'order_id': order.order_id,
                        'client_order_id': order.client_order_id,
                        'symbol': order.symbol,
                        'side': order.side.value,
                        'order_type': order.order_type.value,
                        'quantity': order.quantity,
                        'price': order.price,
                        'stop_price': order.stop_price,
                        'status': order.status.value,
                        'filled_quantity': order.filled_quantity,
                        'avg_fill_price': order.avg_fill_price,
                        'timestamp': order.timestamp,
                        'strategy_name': order.strategy_name
                    })
            
            return sorted(orders, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return []
    
    def get_trade_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get completed trade history."""
        try:
            trades = []
            trade_list = self.completed_trades[-limit:] if limit else self.completed_trades
            
            for trade in trade_list:
                trades.append({
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'side': trade.side.value,
                    'quantity': trade.quantity,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'realized_pnl': trade.realized_pnl,
                    'commission': trade.commission,
                    'strategy_name': trade.strategy_name,
                    'duration': (trade.exit_time - trade.entry_time).total_seconds() if trade.exit_time else 0,
                    'return_pct': (trade.realized_pnl / (trade.entry_price * trade.quantity)) * 100 if trade.entry_price > 0 else 0
                })
            
            return sorted(trades, key=lambda x: x['exit_time'] or datetime.min, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            # Calculate trade statistics
            total_trades = len(self.completed_trades)
            winning_trades = len([t for t in self.completed_trades if t.realized_pnl > 0])
            losing_trades = len([t for t in self.completed_trades if t.realized_pnl < 0])
            
            total_pnl = sum(t.realized_pnl for t in self.completed_trades)
            total_commission = sum(t.commission for t in self.completed_trades)
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate average win/loss
            winning_pnl = [t.realized_pnl for t in self.completed_trades if t.realized_pnl > 0]
            losing_pnl = [t.realized_pnl for t in self.completed_trades if t.realized_pnl < 0]
            
            avg_win = sum(winning_pnl) / len(winning_pnl) if winning_pnl else 0
            avg_loss = sum(losing_pnl) / len(losing_pnl) if losing_pnl else 0
            
            return {
                'execution_metrics': self.execution_metrics,
                'trade_statistics': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'total_commission': total_commission,
                    'net_pnl': total_pnl - total_commission,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
                },
                'current_positions': len([p for p in self.positions.values() if p.is_active]),
                'pending_orders': len(self.pending_orders),
                'total_orders': len(self.orders),
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown trade manager and cleanup resources."""
        try:
            self.logger.info("Shutting down Trade Manager...")
            
            # Signal shutdown to monitoring threads
            self.shutdown_event.set()
            
            # Wait for threads to complete
            if self.order_monitor_thread and self.order_monitor_thread.is_alive():
                self.order_monitor_thread.join(timeout=10)
            
            if self.position_monitor_thread and self.position_monitor_thread.is_alive():
                self.position_monitor_thread.join(timeout=10)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Log final summary
            summary = self.get_performance_summary()
            self.logger.info(f"Final trade summary: {summary}")
            
            self.logger.info("Trade Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during trade manager shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()