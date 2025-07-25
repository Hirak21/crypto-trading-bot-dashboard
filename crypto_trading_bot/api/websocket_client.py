"""
Binance WebSocket client for real-time market data.

This module provides WebSocket connectivity for live market data streaming
with automatic reconnection and error handling.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional, List, Set
import websockets
from datetime import datetime
import time

from ..models.trading import MarketData, OrderBook
from ..interfaces import IMarketDataProvider


class WebSocketError(Exception):
    """Custom exception for WebSocket errors."""
    pass


class BinanceWebSocketClient(IMarketDataProvider):
    """Binance WebSocket client for real-time market data."""
    
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        
        # WebSocket URLs
        if testnet:
            self.ws_url = "wss://stream.binancefuture.com/ws"
            self.stream_url = "wss://stream.binancefuture.com/stream"
        else:
            self.ws_url = "wss://fstream.binance.com/ws"
            self.stream_url = "wss://fstream.binance.com/stream"
        
        # Connection management
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5.0
        
        # Subscriptions
        self.subscribed_symbols: Set[str] = set()
        self.stream_handlers: Dict[str, Callable] = {}
        
        # Data storage
        self.latest_data: Dict[str, MarketData] = {}
        self.order_books: Dict[str, OrderBook] = {}
        
        # Event handlers
        self.on_message: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_connect: Optional[Callable] = None
        self.on_disconnect: Optional[Callable] = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Keep-alive
        self.ping_interval = 30  # seconds
        self.ping_task: Optional[asyncio.Task] = None
        
        # Circuit breaker
        self.error_count = 0
        self.max_errors = 5
        self.error_reset_time = 300  # 5 minutes
        self.last_error_time = 0
    
    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            if self.is_connected:
                return True
            
            self.logger.info(f"Connecting to Binance {'Testnet' if self.testnet else 'Live'} WebSocket...")
            
            # Reset circuit breaker if enough time has passed
            current_time = time.time()
            if current_time - self.last_error_time > self.error_reset_time:
                self.error_count = 0
            
            # Check circuit breaker
            if self.error_count >= self.max_errors:
                self.logger.error("Circuit breaker activated - too many errors")
                return False
            
            # Establish connection
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.is_connected = True
            self.reconnect_attempts = 0
            
            # Start ping task
            self.ping_task = asyncio.create_task(self._ping_loop())
            
            # Start message handler
            asyncio.create_task(self._message_handler())
            
            self.logger.info("WebSocket connected successfully")
            
            if self.on_connect:
                await self.on_connect()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect WebSocket: {e}")
            self.error_count += 1
            self.last_error_time = time.time()
            
            if self.on_error:
                await self.on_error(e)
            
            return False
    
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        try:
            self.is_connected = False
            
            # Cancel ping task
            if self.ping_task:
                self.ping_task.cancel()
                try:
                    await self.ping_task
                except asyncio.CancelledError:
                    pass
            
            # Close WebSocket
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            
            self.logger.info("WebSocket disconnected")
            
            if self.on_disconnect:
                await self.on_disconnect()
                
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    async def _ping_loop(self) -> None:
        """Keep-alive ping loop."""
        try:
            while self.is_connected and self.websocket:
                await asyncio.sleep(self.ping_interval)
                
                if self.websocket and not self.websocket.closed:
                    await self.websocket.ping()
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Ping loop error: {e}")
    
    async def _message_handler(self) -> None:
        """Handle incoming WebSocket messages."""
        try:
            while self.is_connected and self.websocket:
                try:
                    message = await self.websocket.recv()
                    await self._process_message(message)
                    
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("WebSocket connection closed")
                    self.is_connected = False
                    await self._reconnect()
                    break
                    
                except Exception as e:
                    self.logger.error(f"Message handler error: {e}")
                    self.error_count += 1
                    self.last_error_time = time.time()
                    
                    if self.on_error:
                        await self.on_error(e)
                    
                    # Break if too many errors
                    if self.error_count >= self.max_errors:
                        break
                        
        except Exception as e:
            self.logger.error(f"Message handler crashed: {e}")
            self.is_connected = False
    
    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if 'stream' in data:
                stream_name = data['stream']
                stream_data = data['data']
                
                # Route to appropriate handler
                if '@ticker' in stream_name:
                    await self._handle_ticker_data(stream_name, stream_data)
                elif '@depth' in stream_name:
                    await self._handle_depth_data(stream_name, stream_data)
                elif '@kline' in stream_name:
                    await self._handle_kline_data(stream_name, stream_data)
                
            elif 'e' in data:  # Event type
                event_type = data['e']
                
                if event_type == '24hrTicker':
                    await self._handle_ticker_event(data)
                elif event_type == 'depthUpdate':
                    await self._handle_depth_event(data)
                elif event_type == 'kline':
                    await self._handle_kline_event(data)
            
            # Call custom message handler if set
            if self.on_message:
                await self.on_message(data)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    async def _handle_ticker_data(self, stream_name: str, data: Dict[str, Any]) -> None:
        """Handle ticker data."""
        try:
            symbol = data['s']
            
            # Create MarketData object
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(data['E'] / 1000),
                price=float(data['c']),
                volume=float(data['v']),
                bid=float(data['b']),
                ask=float(data['a']),
                high_24h=float(data['h']),
                low_24h=float(data['l']),
                change_24h=float(data['P']),
                volume_24h=float(data['q'])
            )
            
            # Store latest data
            self.latest_data[symbol] = market_data
            
            self.logger.debug(f"Ticker update: {symbol} @ {market_data.price}")
            
        except Exception as e:
            self.logger.error(f"Error handling ticker data: {e}")
    
    async def _handle_depth_data(self, stream_name: str, data: Dict[str, Any]) -> None:
        """Handle order book depth data."""
        try:
            symbol = data['s']
            
            # Convert bid/ask data
            bids = [(float(price), float(qty)) for price, qty in data['b']]
            asks = [(float(price), float(qty)) for price, qty in data['a']]
            
            # Create OrderBook object
            order_book = OrderBook(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(data['E'] / 1000),
                bids=bids,
                asks=asks
            )
            
            # Store order book
            self.order_books[symbol] = order_book
            
            # Update market data with order book
            if symbol in self.latest_data:
                self.latest_data[symbol].orderbook = order_book
            
            self.logger.debug(f"Depth update: {symbol} - {len(bids)} bids, {len(asks)} asks")
            
        except Exception as e:
            self.logger.error(f"Error handling depth data: {e}")
    
    async def _handle_kline_data(self, stream_name: str, data: Dict[str, Any]) -> None:
        """Handle candlestick data."""
        try:
            kline = data['k']
            symbol = kline['s']
            
            # Only process closed candles
            if kline['x']:  # Kline is closed
                self.logger.debug(f"Kline closed: {symbol} {kline['i']} - O:{kline['o']} H:{kline['h']} L:{kline['l']} C:{kline['c']}")
            
        except Exception as e:
            self.logger.error(f"Error handling kline data: {e}")
    
    async def _handle_ticker_event(self, data: Dict[str, Any]) -> None:
        """Handle ticker event."""
        await self._handle_ticker_data('', data)
    
    async def _handle_depth_event(self, data: Dict[str, Any]) -> None:
        """Handle depth update event."""
        await self._handle_depth_data('', data)
    
    async def _handle_kline_event(self, data: Dict[str, Any]) -> None:
        """Handle kline event."""
        await self._handle_kline_data('', data)
    
    async def _reconnect(self) -> None:
        """Attempt to reconnect WebSocket."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        delay = self.reconnect_delay * self.reconnect_attempts
        
        self.logger.info(f"Reconnecting in {delay} seconds (attempt {self.reconnect_attempts})")
        await asyncio.sleep(delay)
        
        success = await self.connect()
        if success:
            # Resubscribe to all symbols
            for symbol in self.subscribed_symbols.copy():
                await self.subscribe_symbol(symbol)
    
    # IMarketDataProvider Implementation
    
    async def subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to market data for a symbol."""
        try:
            if not self.is_connected:
                await self.connect()
            
            if not self.is_connected:
                return False
            
            # Subscribe to ticker and depth streams
            streams = [
                f"{symbol.lower()}@ticker",
                f"{symbol.lower()}@depth20@100ms"
            ]
            
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": int(time.time())
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            self.subscribed_symbols.add(symbol)
            
            self.logger.info(f"Subscribed to {symbol} market data")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False
    
    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe from market data for a symbol."""
        try:
            if not self.is_connected:
                return False
            
            streams = [
                f"{symbol.lower()}@ticker",
                f"{symbol.lower()}@depth20@100ms"
            ]
            
            unsubscribe_message = {
                "method": "UNSUBSCRIBE",
                "params": streams,
                "id": int(time.time())
            }
            
            await self.websocket.send(json.dumps(unsubscribe_message))
            self.subscribed_symbols.discard(symbol)
            
            # Clean up stored data
            self.latest_data.pop(symbol, None)
            self.order_books.pop(symbol, None)
            
            self.logger.info(f"Unsubscribed from {symbol} market data")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from {symbol}: {e}")
            return False
    
    async def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get the latest market data for a symbol."""
        return self.latest_data.get(symbol)
    
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get the latest order book for a symbol."""
        return self.order_books.get(symbol)
    
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of subscribed symbols."""
        return list(self.subscribed_symbols)
    
    def is_symbol_subscribed(self, symbol: str) -> bool:
        """Check if symbol is subscribed."""
        return symbol in self.subscribed_symbols