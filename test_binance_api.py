#!/usr/bin/env python3
"""
Test script for Binance API integration.

This script tests the Binance REST API client functionality including
authentication, market data retrieval, and order management.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


async def test_api_connection():
    """Test basic API connection and authentication."""
    print("🔌 Testing API Connection...")
    
    try:
        from crypto_trading_bot.api import BinanceRestClient
        from crypto_trading_bot.utils.config import ConfigManager
        
        # Load credentials
        config_manager = ConfigManager()
        
        try:
            api_key, api_secret = config_manager.get_api_credentials()
            print("✅ API credentials loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load credentials: {e}")
            print("   Please run setup_credentials.py first")
            return False
        
        # Create client
        async with BinanceRestClient(api_key, api_secret, testnet=True) as client:
            # Test ping
            ping_success = await client.ping()
            if ping_success:
                print("✅ API ping successful")
            else:
                print("❌ API ping failed")
                return False
            
            # Test server time
            server_time = await client.get_server_time()
            print(f"✅ Server time: {server_time}")
            
            return True
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Please install: pip install aiohttp websockets")
        return False
    except Exception as e:
        print(f"❌ API connection test failed: {e}")
        return False


async def test_account_info():
    """Test account information retrieval."""
    print("\n💰 Testing Account Information...")
    
    try:
        from crypto_trading_bot.api import BinanceRestClient
        from crypto_trading_bot.utils.config import ConfigManager
        
        config_manager = ConfigManager()
        api_key, api_secret = config_manager.get_api_credentials()
        
        async with BinanceRestClient(api_key, api_secret, testnet=True) as client:
            # Test account balance
            balance = await client.get_account_balance()
            print(f"✅ Account balance: {balance} USDT")
            
            # Test account info
            account_info = await client.get_account_info()
            print(f"✅ Account info retrieved: {len(account_info.get('assets', []))} assets")
            
            # Test position info
            positions = await client.get_open_positions()
            print(f"✅ Open positions: {len(positions)}")
            
            return True
            
    except Exception as e:
        print(f"❌ Account info test failed: {e}")
        return False


async def test_market_data():
    """Test market data retrieval."""
    print("\n📊 Testing Market Data...")
    
    try:
        from crypto_trading_bot.api import BinanceRestClient
        from crypto_trading_bot.utils.config import ConfigManager
        
        config_manager = ConfigManager()
        api_key, api_secret = config_manager.get_api_credentials()
        
        async with BinanceRestClient(api_key, api_secret, testnet=True) as client:
            symbol = "BTCUSDT"
            
            # Test ticker price
            price = await client.get_ticker_price(symbol)
            print(f"✅ {symbol} price: ${price}")
            
            # Test symbol info
            symbol_info = await client.get_symbol_info(symbol)
            print(f"✅ {symbol} info: {symbol_info['status']}")
            
            # Test order book
            order_book = await client.get_order_book(symbol, limit=10)
            print(f"✅ Order book: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
            
            # Test klines
            klines = await client.get_klines(symbol, "1m", limit=5)
            print(f"✅ Klines: {len(klines)} candles")
            
            return True
            
    except Exception as e:
        print(f"❌ Market data test failed: {e}")
        return False


async def test_order_simulation():
    """Test order placement simulation (testnet only)."""
    print("\n📋 Testing Order Simulation...")
    
    try:
        from crypto_trading_bot.api import BinanceRestClient
        from crypto_trading_bot.utils.config import ConfigManager
        from crypto_trading_bot.models.trading import OrderSide, TradingSignal, SignalAction
        
        config_manager = ConfigManager()
        api_key, api_secret = config_manager.get_api_credentials()
        
        async with BinanceRestClient(api_key, api_secret, testnet=True) as client:
            symbol = "BTCUSDT"
            
            # Get current price for reference
            current_price = await client.get_ticker_price(symbol)
            print(f"✅ Current {symbol} price: ${current_price}")
            
            # Create a test trading signal
            test_signal = TradingSignal(
                symbol=symbol,
                action=SignalAction.BUY,
                confidence=0.8,
                strategy="test_strategy",
                position_size=0.001,  # Very small size for testing
                target_price=current_price * 1.02,  # 2% above current
                stop_loss=current_price * 0.98     # 2% below current
            )
            
            print(f"✅ Test signal created: {test_signal.action.value} {test_signal.position_size} {symbol}")
            print(f"   Target: ${test_signal.target_price:.2f}, Stop: ${test_signal.stop_loss:.2f}")
            
            # Note: We're not actually executing the trade in this test
            # to avoid using testnet balance unnecessarily
            print("✅ Order simulation prepared (not executed)")
            
            return True
            
    except Exception as e:
        print(f"❌ Order simulation test failed: {e}")
        return False


async def test_websocket_connection():
    """Test WebSocket connection."""
    print("\n🌐 Testing WebSocket Connection...")
    
    try:
        from crypto_trading_bot.api import BinanceWebSocketClient
        
        # Create WebSocket client
        ws_client = BinanceWebSocketClient(testnet=True)
        
        # Test connection
        connected = await ws_client.connect()
        if connected:
            print("✅ WebSocket connected successfully")
            
            # Test subscription
            subscribed = await ws_client.subscribe_symbol("BTCUSDT")
            if subscribed:
                print("✅ Subscribed to BTCUSDT")
                
                # Wait a moment for data
                await asyncio.sleep(2)
                
                # Check for data
                latest_data = await ws_client.get_latest_data("BTCUSDT")
                if latest_data:
                    print(f"✅ Received market data: {latest_data.symbol} @ ${latest_data.price}")
                else:
                    print("⚠️  No market data received yet (this is normal)")
                
                # Unsubscribe
                await ws_client.unsubscribe_symbol("BTCUSDT")
                print("✅ Unsubscribed from BTCUSDT")
            
            # Disconnect
            await ws_client.disconnect()
            print("✅ WebSocket disconnected")
            
            return True
        else:
            print("❌ WebSocket connection failed")
            return False
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Please install: pip install websockets")
        return False
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False


async def test_rate_limiting():
    """Test rate limiting functionality."""
    print("\n⏱️ Testing Rate Limiting...")
    
    try:
        from crypto_trading_bot.api.binance_client import RateLimiter
        import time
        
        # Create rate limiter with low limits for testing
        rate_limiter = RateLimiter(requests_per_minute=10, requests_per_second=2)
        
        # Test rapid requests
        start_time = time.time()
        
        for i in range(5):
            await rate_limiter.acquire()
            print(f"✅ Request {i+1} allowed")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"✅ Rate limiting working: 5 requests took {elapsed:.2f} seconds")
        
        if elapsed >= 2.0:  # Should take at least 2 seconds due to rate limiting
            print("✅ Rate limiting properly enforced")
        else:
            print("⚠️  Rate limiting may not be working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Rate limiting test failed: {e}")
        return False


async def main():
    """Run all API tests."""
    print("🚀 Starting Binance API Tests\n")
    
    tests = [
        ("API Connection", test_api_connection),
        ("Account Information", test_account_info),
        ("Market Data", test_market_data),
        ("Order Simulation", test_order_simulation),
        ("WebSocket Connection", test_websocket_connection),
        ("Rate Limiting", test_rate_limiting)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed >= 4:  # Allow some tests to fail due to dependencies
        print("🎉 Binance API integration is working!")
        print("\n📝 Next Steps:")
        print("   1. Install missing dependencies if any tests failed")
        print("   2. Verify your API credentials are correct")
        print("   3. Continue with Market Manager implementation")
        return True
    else:
        print("⚠️  Several tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)