#!/usr/bin/env python3
"""
Simplified Market Data Client for Binance Futures
Uses individual WebSocket connections for better reliability
"""

import asyncio
import json
import websockets
import logging
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import time

@dataclass
class MarketTicker:
    """Market ticker data"""
    symbol: str
    price: float
    change_percent: float
    volume: float
    high: float
    low: float
    timestamp: datetime

@dataclass
class OrderBook:
    """Order book data"""
    symbol: str
    bids: List[List[float]]
    asks: List[List[float]]
    timestamp: datetime

class StreamConnection:
    """Individual WebSocket stream connection"""
    
    def __init__(self, url: str, callback: Callable):
        self.url = url
        self.callback = callback
        self.websocket = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        self.logger = logging.getLogger(f"Stream-{url.split('/')[-1]}")
    
    async def start(self):
        """Start the stream connection"""
        self.running = True
        while self.running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                self.logger.error(f"Stream error: {e}")
                if self.running:
                    await self._handle_reconnect()
    
    async def _connect_and_listen(self):
        """Connect and listen for messages"""
        self.logger.info(f"Connecting to {self.url}")
        
        async with websockets.connect(self.url) as websocket:
            self.websocket = websocket
            self.reconnect_attempts = 0
            self.logger.info("Connected successfully")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.callback(data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
    
    async def _handle_reconnect(self):
        """Handle reconnection with backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            self.running = False
            return
        
        self.reconnect_attempts += 1
        delay = min(5 * (2 ** (self.reconnect_attempts - 1)), 60)
        
        self.logger.info(f"Reconnecting in {delay}s (attempt {self.reconnect_attempts})")
        await asyncio.sleep(delay)
    
    async def stop(self):
        """Stop the stream connection"""
        self.running = False
        if self.websocket:
            await self.websocket.close()

class MarketDataClient:
    """Market data client using individual stream connections"""
    
    def __init__(self, testnet: bool = False):
        self.testnet = testnet
        self.base_url = "wss://stream.binancefuture.com" if testnet else "wss://fstream.binance.com"
        
        # Data storage
        self.tickers: Dict[str, MarketTicker] = {}
        self.orderbooks: Dict[str, OrderBook] = {}
        
        # Stream connections
        self.streams: Dict[str, StreamConnection] = {}
        
        # Callbacks
        self.ticker_callbacks: List[Callable[[MarketTicker], None]] = []
        self.orderbook_callbacks: List[Callable[[OrderBook], None]] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def subscribe_ticker(self, symbol: str):
        """Subscribe to ticker stream for a symbol"""
        stream_name = f"{symbol.lower()}@ticker"
        url = f"{self.base_url}/ws/{stream_name}"
        
        async def ticker_callback(data):
            ticker = MarketTicker(
                symbol=data['s'],
                price=float(data['c']),
                change_percent=float(data['P']),
                volume=float(data['v']),
                high=float(data['h']),
                low=float(data['l']),
                timestamp=datetime.fromtimestamp(data['E'] / 1000)
            )
            
            self.tickers[symbol] = ticker
            
            # Call registered callbacks
            for callback in self.ticker_callbacks:
                try:
                    callback(ticker)
                except Exception as e:
                    self.logger.error(f"Ticker callback error: {e}")
        
        stream = StreamConnection(url, ticker_callback)
        self.streams[stream_name] = stream
        
        # Start the stream
        asyncio.create_task(stream.start())
        self.logger.info(f"Subscribed to ticker for {symbol}")
    
    async def subscribe_orderbook(self, symbol: str):
        """Subscribe to orderbook stream for a symbol"""
        stream_name = f"{symbol.lower()}@depth20@100ms"
        url = f"{self.base_url}/ws/{stream_name}"
        
        async def orderbook_callback(data):
            orderbook = OrderBook(
                symbol=data['s'],
                bids=[[float(bid[0]), float(bid[1])] for bid in data['b']],
                asks=[[float(ask[0]), float(ask[1])] for ask in data['a']],
                timestamp=datetime.fromtimestamp(data['E'] / 1000)
            )
            
            self.orderbooks[symbol] = orderbook
            
            # Call registered callbacks
            for callback in self.orderbook_callbacks:
                try:
                    callback(orderbook)
                except Exception as e:
                    self.logger.error(f"Orderbook callback error: {e}")
        
        stream = StreamConnection(url, orderbook_callback)
        self.streams[stream_name] = stream
        
        # Start the stream
        asyncio.create_task(stream.start())
        self.logger.info(f"Subscribed to orderbook for {symbol}")
    
    def add_ticker_callback(self, callback: Callable[[MarketTicker], None]):
        """Add callback for ticker updates"""
        self.ticker_callbacks.append(callback)
    
    def add_orderbook_callback(self, callback: Callable[[OrderBook], None]):
        """Add callback for orderbook updates"""
        self.orderbook_callbacks.append(callback)
    
    def get_ticker(self, symbol: str) -> Optional[MarketTicker]:
        """Get latest ticker for symbol"""
        return self.tickers.get(symbol)
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBook]:
        """Get latest orderbook for symbol"""
        return self.orderbooks.get(symbol)
    
    def get_all_tickers(self) -> Dict[str, MarketTicker]:
        """Get all cached tickers"""
        return self.tickers.copy()
    
    async def stop_all_streams(self):
        """Stop all stream connections"""
        for stream in self.streams.values():
            await stream.stop()
        self.streams.clear()
        self.logger.info("All streams stopped")

# Test the client
async def test_market_data_client():
    """Test the market data client"""
    
    print("🔌 Testing Market Data Client")
    print("=" * 40)
    
    client = MarketDataClient(testnet=False)
    
    # Add callbacks
    def on_ticker_update(ticker: MarketTicker):
        print(f"📊 {ticker.symbol}: ${ticker.price:.4f} ({ticker.change_percent:+.2f}%)")
    
    def on_orderbook_update(orderbook: OrderBook):
        if orderbook.bids and orderbook.asks:
            spread = orderbook.asks[0][0] - orderbook.bids[0][0]
            print(f"📖 {orderbook.symbol} spread: ${spread:.4f}")
    
    client.add_ticker_callback(on_ticker_update)
    client.add_orderbook_callback(on_orderbook_update)
    
    try:
        # Subscribe to some symbols
        symbols = ['BTCUSDT', 'ETHUSDT']
        
        for symbol in symbols:
            await client.subscribe_ticker(symbol)
            await client.subscribe_orderbook(symbol)
        
        print("⏳ Collecting data for 30 seconds...")
        
        # Run for 30 seconds
        for i in range(30):
            await asyncio.sleep(1)
            
            if i % 5 == 0:  # Print status every 5 seconds
                print(f"\n⏱️  {i}s - Active streams: {len(client.streams)}")
                print(f"   Cached tickers: {len(client.tickers)}")
                print(f"   Cached orderbooks: {len(client.orderbooks)}")
        
        print("\n📈 Final Results:")
        print("=" * 20)
        
        for symbol in symbols:
            ticker = client.get_ticker(symbol)
            orderbook = client.get_orderbook(symbol)
            
            if ticker:
                print(f"✅ {symbol}: ${ticker.price:.4f} ({ticker.change_percent:+.2f}%)")
            else:
                print(f"❌ {symbol}: No ticker data")
            
            if orderbook and orderbook.bids and orderbook.asks:
                spread = orderbook.asks[0][0] - orderbook.bids[0][0]
                print(f"   Spread: ${spread:.4f}")
            else:
                print(f"   No orderbook data")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        await client.stop_all_streams()
        print("✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_market_data_client())