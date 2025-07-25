"""
Market Manager for coordinating market data collection and distribution.

This module manages real-time market data from multiple sources,
validates data quality, and distributes it to trading strategies.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from pathlib import Path

from ..models.trading import MarketData, OrderBook
from ..models.validation import validate_market_data, ValidationError
from ..api.websocket_client import BinanceWebSocketClient
from ..api.binance_client import BinanceRestClient


class MarketDataCache:
    """Cache for market data with automatic cleanup."""
    
    def __init__(self, max_age_minutes: int = 5, max_items_per_symbol: int = 1000):
        self.max_age = timedelta(minutes=max_age_minutes)
        self.max_items_per_symbol = max_items_per_symbol
        
        # Data storage
        self.market_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_items_per_symbol))
        self.order_books: Dict[str, OrderBook] = {}
        self.latest_data: Dict[str, MarketData] = {}
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'validation_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def add_market_data(self, data: MarketData) -> bool:
        """Add market data to cache."""
        try:
            # Validate data
            if not validate_market_data(data):
                self.stats['validation_errors'] += 1
                return False
            
            # Add to cache
            self.market_data[data.symbol].append(data)
            self.latest_data[data.symbol] = data
            
            # Update order book if present
            if data.orderbook:
                self.order_books[data.symbol] = data.orderbook
            
            self.stats['total_updates'] += 1
            return True
            
        except ValidationError as e:
            self.logger.warning(f"Market data validation failed: {e}")
            self.stats['validation_errors'] += 1
            return False
        except Exception as e:
            self.logger.error(f"Error adding market data: {e}")
            return False
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for symbol."""
        if symbol in self.latest_data:
            self.stats['cache_hits'] += 1
            return self.latest_data[symbol]
        else:
            self.stats['cache_misses'] += 1
            return None
    
    def get_historical_data(self, symbol: str, count: int = 100) -> List[MarketData]:
        """Get historical market data for symbol."""
        if symbol in self.market_data:
            self.stats['cache_hits'] += 1
            return list(self.market_data[symbol])[-count:]
        else:
            self.stats['cache_misses'] += 1
            return []
    
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get latest order book for symbol."""
        return self.order_books.get(symbol)
    
    def cleanup_old_data(self) -> None:
        """Remove old data from cache."""
        current_time = datetime.now()
        cleaned_count = 0
        
        for symbol in list(self.market_data.keys()):
            # Clean market data
            data_queue = self.market_data[symbol]
            original_length = len(data_queue)
            
            # Remove old items
            while data_queue and current_time - data_queue[0].timestamp > self.max_age:
                data_queue.popleft()
                cleaned_count += 1
            
            # Remove empty queues
            if not data_queue:
                del self.market_data[symbol]
                self.latest_data.pop(symbol, None)
                self.order_books.pop(symbol, None)
        
        if cleaned_count > 0:
            self.logger.debug(f"Cleaned {cleaned_count} old market data items")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self.stats,
            'symbols_count': len(self.latest_data),
            'total_cached_items': sum(len(queue) for queue in self.market_data.values()),
            'order_books_count': len(self.order_books)
        }


class MarketDataValidator:
    """Validates market data quality and detects anomalies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds
        self.max_price_change_pct = 10.0  # 10% max price change
        self.min_volume_threshold = 0.0
        self.max_spread_pct = 5.0  # 5% max spread
        
        # Anomaly detection
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.anomaly_counts: Dict[str, int] = defaultdict(int)
    
    def validate_data_quality(self, data: MarketData, previous_data: Optional[MarketData] = None) -> bool:
        """Validate market data quality."""
        try:
            # Basic validation
            if not validate_market_data(data):
                return False
            
            # Price change validation
            if previous_data and self._is_price_change_anomalous(data, previous_data):
                self.logger.warning(f"Anomalous price change detected for {data.symbol}")
                self.anomaly_counts[data.symbol] += 1
                return False
            
            # Spread validation
            if data.spread_percentage > self.max_spread_pct:
                self.logger.warning(f"Wide spread detected for {data.symbol}: {data.spread_percentage:.2f}%")
                return False
            
            # Volume validation
            if data.volume < self.min_volume_threshold:
                self.logger.warning(f"Low volume detected for {data.symbol}: {data.volume}")
                return False
            
            # Update price history
            self.price_history[data.symbol].append(data.price)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return False
    
    def _is_price_change_anomalous(self, current: MarketData, previous: MarketData) -> bool:
        """Check if price change is anomalous."""
        if previous.price == 0:
            return False
        
        price_change_pct = abs((current.price - previous.price) / previous.price) * 100
        return price_change_pct > self.max_price_change_pct
    
    def get_anomaly_stats(self) -> Dict[str, int]:
        """Get anomaly statistics."""
        return dict(self.anomaly_counts)


class MarketManager:
    """Market Manager for coordinating market data collection and distribution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbols = config.get('symbols', ['BTCUSDT'])
        self.testnet = config.get('testnet', True)
        
        # Components
        self.ws_client: Optional[BinanceWebSocketClient] = None
        self.rest_client: Optional[BinanceRestClient] = None
        self.cache = MarketDataCache()
        self.validator = MarketDataValidator()
        
        # Event handlers
        self.data_handlers: List[Callable[[MarketData], None]] = []
        self.error_handlers: List[Callable[[Exception], None]] = []
        
        # State management
        self.is_running = False
        self.subscribed_symbols: Set[str] = set()
        
        # Tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'errors_count': 0,
            'last_update_time': None
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> bool:
        """Start the market manager."""
        try:
            self.logger.info("Starting Market Manager...")
            
            # Initialize WebSocket client
            self.ws_client = BinanceWebSocketClient(testnet=self.testnet)
            
            # Set up event handlers
            self.ws_client.on_message = self._handle_ws_message
            self.ws_client.on_error = self._handle_ws_error
            self.ws_client.on_connect = self._handle_ws_connect
            self.ws_client.on_disconnect = self._handle_ws_disconnect
            
            # Connect WebSocket
            connected = await self.ws_client.connect()
            if not connected:
                self.logger.error("Failed to connect WebSocket")
                return False
            
            # Initialize REST client for fallback data
            try:
                from ..utils.config import ConfigManager
                config_manager = ConfigManager()
                api_key, api_secret = config_manager.get_api_credentials()
                
                self.rest_client = BinanceRestClient(api_key, api_secret, self.testnet)
                await self.rest_client.connect()
                
            except Exception as e:
                self.logger.warning(f"REST client initialization failed: {e}")
                self.rest_client = None
            
            # Subscribe to symbols
            for symbol in self.symbols:
                success = await self.subscribe_symbol(symbol)
                if not success:
                    self.logger.warning(f"Failed to subscribe to {symbol}")
            
            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            self.is_running = True
            self.logger.info(f"Market Manager started with {len(self.subscribed_symbols)} symbols")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Market Manager: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> None:
        """Stop the market manager."""
        try:
            self.logger.info("Stopping Market Manager...")
            self.is_running = False
            
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Disconnect clients
            if self.ws_client:
                await self.ws_client.disconnect()
            
            if self.rest_client:
                await self.rest_client.disconnect()
            
            self.logger.info("Market Manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping Market Manager: {e}")
    
    async def subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to market data for a symbol."""
        try:
            if not self.ws_client:
                return False
            
            success = await self.ws_client.subscribe_symbol(symbol)
            if success:
                self.subscribed_symbols.add(symbol)
                self.logger.info(f"Subscribed to {symbol}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False
    
    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe from market data for a symbol."""
        try:
            if not self.ws_client:
                return False
            
            success = await self.ws_client.unsubscribe_symbol(symbol)
            if success:
                self.subscribed_symbols.discard(symbol)
                self.logger.info(f"Unsubscribed from {symbol}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from {symbol}: {e}")
            return False
    
    def add_data_handler(self, handler: Callable[[MarketData], None]) -> None:
        """Add a market data handler."""
        self.data_handlers.append(handler)
    
    def remove_data_handler(self, handler: Callable[[MarketData], None]) -> None:
        """Remove a market data handler."""
        if handler in self.data_handlers:
            self.data_handlers.remove(handler)
    
    def add_error_handler(self, handler: Callable[[Exception], None]) -> None:
        """Add an error handler."""
        self.error_handlers.append(handler)
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for symbol."""
        return self.cache.get_latest_data(symbol)
    
    def get_historical_data(self, symbol: str, count: int = 100) -> List[MarketData]:
        """Get historical market data for symbol."""
        return self.cache.get_historical_data(symbol, count)
    
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get latest order book for symbol."""
        return self.cache.get_order_book(symbol)
    
    async def get_candlestick_data(self, symbol: str, interval: str = '1m', limit: int = 100) -> List[List]:
        """Get candlestick data from REST API."""
        if not self.rest_client:
            return []
        
        try:
            return await self.rest_client.get_klines(symbol, interval, limit)
        except Exception as e:
            self.logger.error(f"Failed to get candlestick data: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get market manager statistics."""
        cache_stats = self.cache.get_stats()
        anomaly_stats = self.validator.get_anomaly_stats()
        
        return {
            'manager_stats': self.stats,
            'cache_stats': cache_stats,
            'anomaly_stats': anomaly_stats,
            'subscribed_symbols': list(self.subscribed_symbols),
            'is_running': self.is_running
        }
    
    async def _handle_ws_message(self, message: Dict[str, Any]) -> None:
        """Handle WebSocket message."""
        try:
            self.stats['messages_received'] += 1
            
            # The WebSocket client already processes the message
            # We just need to get the processed data
            if self.ws_client:
                for symbol in self.subscribed_symbols:
                    latest_data = await self.ws_client.get_latest_data(symbol)
                    if latest_data:
                        await self._process_market_data(latest_data)
            
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
            self.stats['errors_count'] += 1
    
    async def _process_market_data(self, data: MarketData) -> None:
        """Process incoming market data."""
        try:
            # Get previous data for validation
            previous_data = self.cache.get_latest_data(data.symbol)
            
            # Validate data quality
            if not self.validator.validate_data_quality(data, previous_data):
                return
            
            # Add to cache
            if self.cache.add_market_data(data):
                self.stats['messages_processed'] += 1
                self.stats['last_update_time'] = datetime.now()
                
                # Notify handlers
                for handler in self.data_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(data)
                        else:
                            handler(data)
                    except Exception as e:
                        self.logger.error(f"Error in data handler: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            self.stats['errors_count'] += 1
    
    async def _handle_ws_error(self, error: Exception) -> None:
        """Handle WebSocket error."""
        self.logger.error(f"WebSocket error: {error}")
        self.stats['errors_count'] += 1
        
        # Notify error handlers
        for handler in self.error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error)
                else:
                    handler(error)
            except Exception as e:
                self.logger.error(f"Error in error handler: {e}")
    
    async def _handle_ws_connect(self) -> None:
        """Handle WebSocket connection."""
        self.logger.info("WebSocket connected")
    
    async def _handle_ws_disconnect(self) -> None:
        """Handle WebSocket disconnection."""
        self.logger.warning("WebSocket disconnected")
    
    async def _cleanup_loop(self) -> None:
        """Background task for cache cleanup."""
        try:
            while self.is_running:
                await asyncio.sleep(60)  # Run every minute
                self.cache.cleanup_old_data()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Cleanup loop error: {e}")
    
    async def _health_check_loop(self) -> None:
        """Background task for health monitoring."""
        try:
            while self.is_running:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check if we're receiving data
                if self.stats['last_update_time']:
                    time_since_update = datetime.now() - self.stats['last_update_time']
                    if time_since_update > timedelta(minutes=2):
                        self.logger.warning("No market data received for 2 minutes")
                
                # Check WebSocket connection
                if self.ws_client and not self.ws_client.is_connected:
                    self.logger.warning("WebSocket not connected, attempting reconnection")
                    await self.ws_client.connect()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Health check loop error: {e}")
    
    async def save_data_snapshot(self, file_path: str) -> bool:
        """Save current market data snapshot to file."""
        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'symbols': list(self.subscribed_symbols),
                'latest_data': {},
                'stats': self.get_stats()
            }
            
            # Add latest data for each symbol
            for symbol in self.subscribed_symbols:
                data = self.get_latest_data(symbol)
                if data:
                    snapshot['latest_data'][symbol] = data.to_dict()
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(snapshot, f, indent=2)
            
            self.logger.info(f"Data snapshot saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save data snapshot: {e}")
            return False