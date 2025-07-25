"""
Binance REST API client with authentication and rate limiting.

This module provides a comprehensive Binance API client for futures trading
with proper error handling, rate limiting, and security features.
"""

import asyncio
import hashlib
import hmac
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlencode
import aiohttp
import logging
from datetime import datetime, timedelta

from ..models.trading import Trade, Position, OrderSide, OrderType, PositionSide
from ..interfaces import ITradeExecutor


class BinanceAPIError(Exception):
    """Custom exception for Binance API errors."""
    
    def __init__(self, message: str, error_code: int = None, response: Dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.response = response or {}


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int = 1200, requests_per_second: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        
        # Track requests
        self.minute_requests = []
        self.second_requests = []
        
        # Locks for thread safety
        self.minute_lock = asyncio.Lock()
        self.second_lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire rate limit permission."""
        current_time = time.time()
        
        # Check per-second limit
        async with self.second_lock:
            # Remove old requests (older than 1 second)
            self.second_requests = [
                req_time for req_time in self.second_requests 
                if current_time - req_time < 1.0
            ]
            
            if len(self.second_requests) >= self.requests_per_second:
                sleep_time = 1.0 - (current_time - self.second_requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.second_requests.append(current_time)
        
        # Check per-minute limit
        async with self.minute_lock:
            # Remove old requests (older than 1 minute)
            self.minute_requests = [
                req_time for req_time in self.minute_requests 
                if current_time - req_time < 60.0
            ]
            
            if len(self.minute_requests) >= self.requests_per_minute:
                sleep_time = 60.0 - (current_time - self.minute_requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.minute_requests.append(current_time)


class BinanceRestClient(ITradeExecutor):
    """Binance REST API client for futures trading."""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # API endpoints
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
        
        # Rate limiter
        self.rate_limiter = RateLimiter()
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Request timeout
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Server time offset
        self.time_offset = 0
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Initialize HTTP session and sync server time."""
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
                headers={
                    'X-MBX-APIKEY': self.api_key,
                    'Content-Type': 'application/json'
                }
            )
            
            # Sync server time
            await self._sync_server_time()
            
            self.logger.info(f"Connected to Binance {'Testnet' if self.testnet else 'Live'} API")
    
    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("Disconnected from Binance API")
    
    async def _sync_server_time(self) -> None:
        """Synchronize with Binance server time."""
        try:
            response = await self._make_request('GET', '/fapi/v1/time')
            server_time = response['serverTime']
            local_time = int(time.time() * 1000)
            self.time_offset = server_time - local_time
            
            self.logger.info(f"Server time synchronized, offset: {self.time_offset}ms")
            
        except Exception as e:
            self.logger.warning(f"Failed to sync server time: {e}")
            self.time_offset = 0
    
    def _get_timestamp(self) -> int:
        """Get current timestamp adjusted for server time."""
        return int(time.time() * 1000) + self.time_offset
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for authenticated requests."""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                           signed: bool = False, retries: int = 0) -> Dict[str, Any]:
        """Make HTTP request to Binance API with error handling and retries."""
        if not self.session:
            await self.connect()
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Prepare parameters
        params = params or {}
        
        if signed:
            params['timestamp'] = self._get_timestamp()
            query_string = urlencode(params)
            params['signature'] = self._generate_signature(query_string)
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = await self.session.get(url, params=params)
            elif method == 'POST':
                response = await self.session.post(url, data=params)
            elif method == 'PUT':
                response = await self.session.put(url, data=params)
            elif method == 'DELETE':
                response = await self.session.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Handle response
            response_text = await response.text()
            
            if response.status == 200:
                return json.loads(response_text)
            else:
                # Parse error response
                try:
                    error_data = json.loads(response_text)
                    error_code = error_data.get('code', response.status)
                    error_msg = error_data.get('msg', 'Unknown error')
                except:
                    error_code = response.status
                    error_msg = response_text
                
                # Handle specific error codes
                if error_code == -1021:  # Timestamp outside recv window
                    if retries < self.max_retries:
                        await self._sync_server_time()
                        await asyncio.sleep(self.retry_delay)
                        return await self._make_request(method, endpoint, params, signed, retries + 1)
                
                elif error_code == -1003:  # Rate limit exceeded
                    if retries < self.max_retries:
                        await asyncio.sleep(self.retry_delay * (retries + 1))
                        return await self._make_request(method, endpoint, params, signed, retries + 1)
                
                raise BinanceAPIError(
                    f"API request failed: {error_msg}",
                    error_code=error_code,
                    response=error_data if 'error_data' in locals() else {}
                )
        
        except aiohttp.ClientError as e:
            if retries < self.max_retries:
                await asyncio.sleep(self.retry_delay * (retries + 1))
                return await self._make_request(method, endpoint, params, signed, retries + 1)
            else:
                raise BinanceAPIError(f"Network error: {str(e)}")
        
        except asyncio.TimeoutError:
            if retries < self.max_retries:
                await asyncio.sleep(self.retry_delay * (retries + 1))
                return await self._make_request(method, endpoint, params, signed, retries + 1)
            else:
                raise BinanceAPIError("Request timeout")
    
    # Account Information Methods
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get futures account information."""
        return await self._make_request('GET', '/fapi/v2/account', signed=True)
    
    async def get_account_balance(self) -> float:
        """Get current account balance in USDT."""
        try:
            account_info = await self.get_account_info()
            
            for asset in account_info.get('assets', []):
                if asset['asset'] == 'USDT':
                    return float(asset['walletBalance'])
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to get account balance: {e}")
            raise BinanceAPIError(f"Failed to get account balance: {e}")
    
    async def get_position_info(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get position information."""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return await self._make_request('GET', '/fapi/v2/positionRisk', params, signed=True)
    
    async def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        try:
            positions_data = await self.get_position_info()
            positions = []
            
            for pos_data in positions_data:
                position_amt = float(pos_data['positionAmt'])
                
                # Skip positions with zero size
                if position_amt == 0:
                    continue
                
                # Determine position side
                side = PositionSide.LONG if position_amt > 0 else PositionSide.SHORT
                
                position = Position(
                    symbol=pos_data['symbol'],
                    side=side,
                    size=abs(position_amt),
                    entry_price=float(pos_data['entryPrice']),
                    current_price=float(pos_data['markPrice']),
                    timestamp=datetime.now()
                )
                
                positions.append(position)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to get open positions: {e}")
            raise BinanceAPIError(f"Failed to get open positions: {e}")
    
    # Order Management Methods
    
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         quantity: float, price: float = None, 
                         stop_price: float = None, time_in_force: str = 'GTC') -> Dict[str, Any]:
        """Place a new order."""
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': order_type.value,
            'quantity': str(quantity),
            'timeInForce': time_in_force
        }
        
        if price is not None:
            params['price'] = str(price)
        
        if stop_price is not None:
            params['stopPrice'] = str(stop_price)
        
        return await self._make_request('POST', '/fapi/v1/order', params, signed=True)
    
    async def execute_market_order(self, symbol: str, side: OrderSide, 
                                  quantity: float) -> Optional[Trade]:
        """Execute a market order."""
        try:
            order_response = await self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity
            )
            
            # Convert to Trade object
            trade = Trade(
                symbol=symbol,
                side=side,
                size=float(order_response['executedQty']),
                price=float(order_response['avgPrice']) if order_response['avgPrice'] != '0' else float(order_response['price']),
                commission=0.0,  # Will be updated when we get trade details
                order_type=OrderType.MARKET,
                order_id=str(order_response['orderId']),
                timestamp=datetime.fromtimestamp(order_response['updateTime'] / 1000)
            )
            
            self.logger.info(f"Market order executed: {side.value} {quantity} {symbol}")
            return trade
            
        except Exception as e:
            self.logger.error(f"Failed to execute market order: {e}")
            raise BinanceAPIError(f"Failed to execute market order: {e}")
    
    async def execute_limit_order(self, symbol: str, side: OrderSide, 
                                 quantity: float, price: float) -> Optional[Trade]:
        """Execute a limit order."""
        try:
            order_response = await self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price
            )
            
            # For limit orders, we need to check if it was filled
            if order_response['status'] == 'FILLED':
                trade = Trade(
                    symbol=symbol,
                    side=side,
                    size=float(order_response['executedQty']),
                    price=float(order_response['avgPrice']),
                    commission=0.0,
                    order_type=OrderType.LIMIT,
                    order_id=str(order_response['orderId']),
                    timestamp=datetime.fromtimestamp(order_response['updateTime'] / 1000)
                )
                
                self.logger.info(f"Limit order filled: {side.value} {quantity} {symbol} @ {price}")
                return trade
            else:
                self.logger.info(f"Limit order placed: {side.value} {quantity} {symbol} @ {price}")
                return None  # Order placed but not filled
                
        except Exception as e:
            self.logger.error(f"Failed to execute limit order: {e}")
            raise BinanceAPIError(f"Failed to execute limit order: {e}")
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order."""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        return await self._make_request('DELETE', '/fapi/v1/order', params, signed=True)
    
    async def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        return await self._make_request('GET', '/fapi/v1/order', params, signed=True)
    
    # ITradeExecutor Implementation
    
    async def execute_trade(self, signal) -> Optional[Trade]:
        """Execute a trade based on trading signal."""
        try:
            from ..models.trading import SignalAction
            
            if signal.action == SignalAction.BUY:
                side = OrderSide.BUY
            elif signal.action == SignalAction.SELL:
                side = OrderSide.SELL
            else:
                self.logger.warning(f"Unsupported signal action: {signal.action}")
                return None
            
            # Use market order for immediate execution
            quantity = signal.position_size or 0.001  # Default small size for testing
            
            trade = await self.execute_market_order(signal.symbol, side, quantity)
            
            if trade:
                trade.strategy = signal.strategy
                
                # Set stop loss if specified
                if signal.stop_loss:
                    await self._set_stop_loss_order(trade, signal.stop_loss)
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Failed to execute trade from signal: {e}")
            return None
    
    async def close_position(self, position: Position) -> Optional[Trade]:
        """Close an existing position."""
        try:
            # Determine opposite side
            side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
            
            # Execute market order to close position
            trade = await self.execute_market_order(position.symbol, side, position.size)
            
            if trade:
                trade.position_id = position.position_id
                self.logger.info(f"Position closed: {position.side.value} {position.size} {position.symbol}")
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Failed to close position: {e}")
            return None
    
    async def _set_stop_loss_order(self, trade: Trade, stop_price: float) -> None:
        """Set stop loss order for a trade."""
        try:
            # Determine stop loss side (opposite of original trade)
            stop_side = OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY
            
            await self.place_order(
                symbol=trade.symbol,
                side=stop_side,
                order_type=OrderType.STOP_LOSS,
                quantity=trade.size,
                stop_price=stop_price
            )
            
            self.logger.info(f"Stop loss set for {trade.symbol} at {stop_price}")
            
        except Exception as e:
            self.logger.error(f"Failed to set stop loss: {e}")
    
    # Market Data Methods
    
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol information."""
        response = await self._make_request('GET', '/fapi/v1/exchangeInfo')
        
        for symbol_info in response['symbols']:
            if symbol_info['symbol'] == symbol:
                return symbol_info
        
        raise BinanceAPIError(f"Symbol {symbol} not found")
    
    async def get_ticker_price(self, symbol: str) -> float:
        """Get current ticker price."""
        params = {'symbol': symbol}
        response = await self._make_request('GET', '/fapi/v1/ticker/price', params)
        return float(response['price'])
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book data."""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        return await self._make_request('GET', '/fapi/v1/depth', params)
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List]:
        """Get candlestick data."""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        return await self._make_request('GET', '/fapi/v1/klines', params)
    
    # Health Check Methods
    
    async def ping(self) -> bool:
        """Ping the API to check connectivity."""
        try:
            await self._make_request('GET', '/fapi/v1/ping')
            return True
        except:
            return False
    
    async def get_server_time(self) -> int:
        """Get server time."""
        response = await self._make_request('GET', '/fapi/v1/time')
        return response['serverTime']