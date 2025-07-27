#!/usr/bin/env python3
"""
WebSocket Market Data Client
Real-time market data streaming with auto-reconnection and circuit breaker
"""

import asyncio
import json
import websockets
import logging
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import time
import traceback

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"

@dataclass
class MarketData:
    """Market data structure for WebSocket streams"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    close: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0

@dataclass
class OrderBookData:
    """Order book data structure"""
    symbol: str
    timestamp: datetime
    bids: List[List[float]]  # [price, quantity]
    asks: List[List[float]]  # [price, quantity]
    last_update_id: int = 0

@dataclass
class TradeData:
    """Individual trade data"""
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    is_buyer_maker: bool
    trade_id: int

class CircuitBreaker:
    """Circuit breaker for connection failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logging.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                logging.info("Circuit breaker moved to half-open state")
                return True
            return False
        
        # half_open state
        return True

class WebSocketClient:
    """WebSocket client for Binance Futures market data"""
    
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        # Use the correct WebSocket URLs for Binance Futures
        if testnet:
            self.base_url = "wss://stream.binancefuture.com"
        else:
            self.base_url = "wss://fstream.binance.com"
        
        # Use combined stream approach
        self.use_combined_stream = True
        
        # Connection management
        self.websocket = None
        self.connection_state = ConnectionState.DISCONNECTED
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # seconds
        self.max_reconnect_delay = 300  # 5 minutes
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Subscriptions and callbacks
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.stream_callbacks: Dict[str, Callable] = {}
        
        # Health monitoring
        self.last_ping_time = None
        self.ping_interval = 30  # seconds
        self.ping_timeout = 10  # seconds
        self.last_message_time = None
        self.message_timeout = 60  # seconds
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'connection_attempts': 0,
            'successful_connections': 0,
            'reconnections': 0,
            'errors': 0,
            'last_error': None,
            'uptime_start': None
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        if not self.circuit_breaker.can_execute():
            self.logger.warning("Circuit breaker is open, cannot connect")
            return False
        
        try:
            self.connection_state = ConnectionState.CONNECTING
            self.stats['connection_attempts'] += 1
            
            # Build WebSocket URL
            if self.use_combined_stream and self.subscriptions:
                # Use combined stream for multiple subscriptions
                streams = list(self.subscriptions.keys())
                stream_list = '/'.join(streams)
                url = f"{self.base_url}/stream?streams={stream_list}"
            else:
                # Use individual stream endpoint
                url = f"{self.base_url}/ws"
            
            self.logger.info(f"Connecting to {url}")
            
            # Connect with timeout
            self.websocket = await asyncio.wait_for(
                websockets.connect(url, ping_interval=None),
                timeout=10
            )
            
            self.connection_state = ConnectionState.CONNECTED
            self.stats['successful_connections'] += 1
            self.stats['uptime_start'] = datetime.now()
            self.last_message_time = time.time()
            self.reconnect_attempts = 0
            
            self.circuit_breaker.record_success()
            self.logger.info("WebSocket connected successfully")
            
            # Start health monitoring
            asyncio.create_task(self._health_monitor())
            
            return True
            
        except Exception as e:
            self.connection_state = ConnectionState.FAILED
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            self.circuit_breaker.record_failure()
            
            self.logger.error(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Gracefully disconnect WebSocket"""
        if self.websocket:
            self.connection_state = ConnectionState.DISCONNECTED
            await self.websocket.close()
            self.websocket = None
            self.logger.info("WebSocket disconnected")
    
    async def _reconnect(self) -> bool:
        """Handle reconnection with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            self.connection_state = ConnectionState.FAILED
            return False
        
        self.connection_state = ConnectionState.RECONNECTING
        self.reconnect_attempts += 1
        
        # Exponential backoff
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 
                   self.max_reconnect_delay)
        
        self.logger.info(f"Reconnecting in {delay} seconds (attempt {self.reconnect_attempts})")
        await asyncio.sleep(delay)
        
        success = await self.connect()
        if success:
            self.stats['reconnections'] += 1
            # Resubscribe to all streams
            await self._resubscribe_all()
        
        return success
    
    async def _resubscribe_all(self):
        """Resubscribe to all active streams after reconnection"""
        for stream_name in list(self.subscriptions.keys()):
            try:
                await self._send_subscription(stream_name, "SUBSCRIBE")
                self.logger.info(f"Resubscribed to {stream_name}")
            except Exception as e:
                self.logger.error(f"Failed to resubscribe to {stream_name}: {e}")
    
    async def _health_monitor(self):
        """Monitor connection health and handle reconnections"""
        while self.connection_state in [ConnectionState.CONNECTED, ConnectionState.RECONNECTING]:
            try:
                current_time = time.time()
                
                # Check message timeout
                if (self.last_message_time and 
                    current_time - self.last_message_time > self.message_timeout):
                    self.logger.warning("Message timeout detected, reconnecting...")
                    await self._reconnect()
                    continue
                
                # Send ping if needed
                if (not self.last_ping_time or 
                    current_time - self.last_ping_time > self.ping_interval):
                    await self._send_ping()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _send_ping(self):
        """Send ping to keep connection alive"""
        if self.websocket and self.connection_state == ConnectionState.CONNECTED:
            try:
                await self.websocket.ping()
                self.last_ping_time = time.time()
            except Exception as e:
                self.logger.error(f"Ping failed: {e}")
                await self._reconnect()
    
    async def _send_subscription(self, stream: str, method: str):
        """Send subscription/unsubscription message"""
        if not self.websocket or self.connection_state != ConnectionState.CONNECTED:
            raise Exception("WebSocket not connected")
        
        message = {
            "method": method,
            "params": [stream],
            "id": int(time.time())
        }
        
        await self.websocket.send(json.dumps(message))
        self.logger.debug(f"Sent {method} for {stream}")
    
    async def subscribe_ticker(self, symbol: str, callback: Callable[[MarketData], None]):
        """Subscribe to 24hr ticker stream"""
        stream = f"{symbol.lower()}@ticker"
        
        if stream not in self.subscriptions:
            self.subscriptions[stream] = []
        
        self.subscriptions[stream].append(callback)
        self.stream_callbacks[stream] = self._handle_ticker_data
        
        if self.connection_state == ConnectionState.CONNECTED:
            await self._send_subscription(stream, "SUBSCRIBE")
        
        self.logger.info(f"Subscribed to ticker for {symbol}")
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable[[OrderBookData], None]):
        """Subscribe to order book stream"""
        stream = f"{symbol.lower()}@depth20@100ms"
        
        if stream not in self.subscriptions:
            self.subscriptions[stream] = []
        
        self.subscriptions[stream].append(callback)
        self.stream_callbacks[stream] = self._handle_orderbook_data
        
        if self.connection_state == ConnectionState.CONNECTED:
            await self._send_subscription(stream, "SUBSCRIBE")
        
        self.logger.info(f"Subscribed to orderbook for {symbol}")
    
    async def subscribe_trades(self, symbol: str, callback: Callable[[TradeData], None]):
        """Subscribe to trade stream"""
        stream = f"{symbol.lower()}@aggTrade"
        
        if stream not in self.subscriptions:
            self.subscriptions[stream] = []
        
        self.subscriptions[stream].append(callback)
        self.stream_callbacks[stream] = self._handle_trade_data
        
        if self.connection_state == ConnectionState.CONNECTED:
            await self._send_subscription(stream, "SUBSCRIBE")
        
        self.logger.info(f"Subscribed to trades for {symbol}")
    
    async def unsubscribe(self, symbol: str, stream_type: str):
        """Unsubscribe from a stream"""
        stream_map = {
            'ticker': f"{symbol.lower()}@ticker",
            'orderbook': f"{symbol.lower()}@depth20@100ms",
            'trades': f"{symbol.lower()}@aggTrade"
        }
        
        stream = stream_map.get(stream_type)
        if not stream:
            raise ValueError(f"Unknown stream type: {stream_type}")
        
        if stream in self.subscriptions:
            del self.subscriptions[stream]
            del self.stream_callbacks[stream]
            
            if self.connection_state == ConnectionState.CONNECTED:
                await self._send_subscription(stream, "UNSUBSCRIBE")
            
            self.logger.info(f"Unsubscribed from {stream}") 
   
    def _handle_ticker_data(self, data: Dict[str, Any]) -> MarketData:
        """Parse ticker data from WebSocket"""
        return MarketData(
            symbol=data['s'],
            timestamp=datetime.fromtimestamp(data['E'] / 1000),
            price=float(data['c']),
            volume=float(data['v']),
            bid=float(data['b']),
            ask=float(data['a']),
            high=float(data['h']),
            low=float(data['l']),
            open=float(data['o']),
            close=float(data['c']),
            change=float(data['P']),
            change_percent=float(data['p'])
        )
    
    def _handle_orderbook_data(self, data: Dict[str, Any]) -> OrderBookData:
        """Parse order book data from WebSocket"""
        return OrderBookData(
            symbol=data['s'],
            timestamp=datetime.fromtimestamp(data['E'] / 1000),
            bids=[[float(bid[0]), float(bid[1])] for bid in data['b']],
            asks=[[float(ask[0]), float(ask[1])] for ask in data['a']],
            last_update_id=data['u']
        )
    
    def _handle_trade_data(self, data: Dict[str, Any]) -> TradeData:
        """Parse trade data from WebSocket"""
        return TradeData(
            symbol=data['s'],
            timestamp=datetime.fromtimestamp(data['T'] / 1000),
            price=float(data['p']),
            quantity=float(data['q']),
            is_buyer_maker=data['m'],
            trade_id=data['a']
        )
    
    async def start_listening(self):
        """Start listening for WebSocket messages"""
        if not self.websocket or self.connection_state != ConnectionState.CONNECTED:
            if not await self.connect():
                return
        
        self.logger.info("Started listening for WebSocket messages")
        
        try:
            async for message in self.websocket:
                try:
                    self.last_message_time = time.time()
                    self.stats['messages_received'] += 1
                    
                    data = json.loads(message)
                    
                    # Handle subscription responses
                    if 'result' in data:
                        if data.get('result') is None:
                            self.logger.info(f"Subscription confirmed: {data.get('id')}")
                        continue
                    
                    # Handle stream data - Binance sends data directly or in stream format
                    if 'stream' in data and 'data' in data:
                        # Multi-stream format
                        stream_name = data['stream']
                        stream_data = data['data']
                    elif 'e' in data:
                        # Single stream format - construct stream name from event type and symbol
                        event_type = data['e']
                        symbol = data.get('s', '').lower()
                        
                        if event_type == '24hrTicker':
                            stream_name = f"{symbol}@ticker"
                        elif event_type == 'depthUpdate':
                            stream_name = f"{symbol}@depth20@100ms"
                        elif event_type == 'aggTrade':
                            stream_name = f"{symbol}@aggTrade"
                        else:
                            continue
                        
                        stream_data = data
                    else:
                        continue
                    
                    if stream_name in self.stream_callbacks:
                        try:
                            # Parse data using appropriate handler
                            parsed_data = self.stream_callbacks[stream_name](stream_data)
                            
                            # Call all registered callbacks for this stream
                            if stream_name in self.subscriptions:
                                for callback in self.subscriptions[stream_name]:
                                    try:
                                        callback(parsed_data)
                                    except Exception as e:
                                        self.logger.error(f"Callback error for {stream_name}: {e}")
                        except Exception as e:
                            self.logger.error(f"Data parsing error for {stream_name}: {e}")
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    self.logger.error(f"Message processing error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            await self._reconnect()
        except Exception as e:
            self.logger.error(f"WebSocket listening error: {e}")
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
            await self._reconnect()
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        uptime = None
        if self.stats['uptime_start']:
            uptime = (datetime.now() - self.stats['uptime_start']).total_seconds()
        
        return {
            'connection_state': self.connection_state.value,
            'messages_received': self.stats['messages_received'],
            'connection_attempts': self.stats['connection_attempts'],
            'successful_connections': self.stats['successful_connections'],
            'reconnections': self.stats['reconnections'],
            'errors': self.stats['errors'],
            'last_error': self.stats['last_error'],
            'uptime_seconds': uptime,
            'active_subscriptions': len(self.subscriptions),
            'circuit_breaker_state': self.circuit_breaker.state,
            'circuit_breaker_failures': self.circuit_breaker.failure_count
        }
    
    async def run_forever(self):
        """Run WebSocket client with automatic reconnection"""
        while True:
            try:
                if self.connection_state == ConnectionState.DISCONNECTED:
                    await self.connect()
                
                if self.connection_state == ConnectionState.CONNECTED:
                    await self.start_listening()
                else:
                    await asyncio.sleep(5)
                    
            except KeyboardInterrupt:
                self.logger.info("Shutting down WebSocket client")
                await self.disconnect()
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in run_forever: {e}")
                await asyncio.sleep(5)

class MarketDataManager:
    """High-level market data manager using WebSocket client"""
    
    def __init__(self, testnet: bool = True):
        self.ws_client = WebSocketClient(testnet)
        self.market_data_cache: Dict[str, MarketData] = {}
        self.orderbook_cache: Dict[str, OrderBookData] = {}
        self.trade_callbacks: List[Callable] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the market data manager"""
        await self.ws_client.connect()
        # Start listening in background
        asyncio.create_task(self.ws_client.run_forever())
        self.logger.info("Market data manager started")
    
    async def stop(self):
        """Stop the market data manager"""
        await self.ws_client.disconnect()
        self.logger.info("Market data manager stopped")
    
    async def subscribe_symbol(self, symbol: str, include_orderbook: bool = True, include_trades: bool = False):
        """Subscribe to all data streams for a symbol"""
        # Subscribe to ticker
        await self.ws_client.subscribe_ticker(symbol, self._on_ticker_update)
        
        # Subscribe to orderbook if requested
        if include_orderbook:
            await self.ws_client.subscribe_orderbook(symbol, self._on_orderbook_update)
        
        # Subscribe to trades if requested
        if include_trades:
            await self.ws_client.subscribe_trades(symbol, self._on_trade_update)
        
        self.logger.info(f"Subscribed to data streams for {symbol}")
    
    def _on_ticker_update(self, data: MarketData):
        """Handle ticker updates"""
        self.market_data_cache[data.symbol] = data
        self.logger.debug(f"Updated ticker for {data.symbol}: ${data.price}")
    
    def _on_orderbook_update(self, data: OrderBookData):
        """Handle orderbook updates"""
        self.orderbook_cache[data.symbol] = data
        self.logger.debug(f"Updated orderbook for {data.symbol}")
    
    def _on_trade_update(self, data: TradeData):
        """Handle trade updates"""
        # Call registered trade callbacks
        for callback in self.trade_callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Trade callback error: {e}")
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for symbol"""
        return self.market_data_cache.get(symbol)
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        """Get latest orderbook for symbol"""
        return self.orderbook_cache.get(symbol)
    
    def add_trade_callback(self, callback: Callable[[TradeData], None]):
        """Add callback for trade updates"""
        self.trade_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection and data statistics"""
        ws_stats = self.ws_client.get_connection_stats()
        return {
            **ws_stats,
            'cached_symbols': len(self.market_data_cache),
            'cached_orderbooks': len(self.orderbook_cache),
            'trade_callbacks': len(self.trade_callbacks)
        }

# Example usage and testing
async def example_usage():
    """Example of how to use the WebSocket client"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create market data manager
    manager = MarketDataManager(testnet=True)
    
    # Define callbacks
    def on_trade(trade_data: TradeData):
        print(f"Trade: {trade_data.symbol} - ${trade_data.price} x {trade_data.quantity}")
    
    # Add trade callback
    manager.add_trade_callback(on_trade)
    
    try:
        # Start manager
        await manager.start()
        
        # Subscribe to some symbols
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        for symbol in symbols:
            await manager.subscribe_symbol(symbol, include_trades=True)
        
        # Run for a while and print stats
        for i in range(60):  # Run for 1 minute
            await asyncio.sleep(1)
            
            if i % 10 == 0:  # Print stats every 10 seconds
                stats = manager.get_stats()
                print(f"Stats: {stats['messages_received']} messages, "
                      f"{stats['cached_symbols']} symbols, "
                      f"State: {stats['connection_state']}")
                
                # Print latest prices
                for symbol in symbols:
                    data = manager.get_market_data(symbol)
                    if data:
                        print(f"{symbol}: ${data.price} ({data.change_percent:+.2f}%)")
        
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(example_usage())