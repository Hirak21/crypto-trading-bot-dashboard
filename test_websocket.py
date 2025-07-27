#!/usr/bin/env python3
"""
Test script for WebSocket market data client
"""

import asyncio
import logging
from websocket_client import MarketDataManager, MarketData, OrderBookData, TradeData

async def test_websocket_client():
    """Test the WebSocket client functionality"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🔌 Testing WebSocket Market Data Client")
    print("=" * 50)
    
    # Create manager
    manager = MarketDataManager(testnet=True)
    
    # Track received data
    ticker_updates = 0
    orderbook_updates = 0
    trade_updates = 0
    
    def on_trade(trade_data: TradeData):
        nonlocal trade_updates
        trade_updates += 1
        if trade_updates <= 5:  # Show first 5 trades
            print(f"📊 Trade: {trade_data.symbol} - ${trade_data.price:.4f} x {trade_data.quantity:.4f}")
    
    # Add trade callback
    manager.add_trade_callback(on_trade)
    
    try:
        print("🚀 Starting WebSocket connection...")
        await manager.start()
        
        # Wait a moment for connection
        await asyncio.sleep(2)
        
        # Subscribe to test symbols
        test_symbols = ['BTCUSDT', 'ETHUSDT']
        print(f"📡 Subscribing to {test_symbols}...")
        
        for symbol in test_symbols:
            await manager.subscribe_symbol(symbol, include_trades=True)
        
        print("⏳ Collecting data for 30 seconds...")
        
        # Collect data for 30 seconds
        for i in range(30):
            await asyncio.sleep(1)
            
            # Count updates
            ticker_updates = len(manager.market_data_cache)
            orderbook_updates = len(manager.orderbook_cache)
            
            if i % 5 == 0:  # Print status every 5 seconds
                stats = manager.get_stats()
                print(f"⏱️  {i}s - Messages: {stats['messages_received']}, "
                      f"Tickers: {ticker_updates}, Orderbooks: {orderbook_updates}, "
                      f"Trades: {trade_updates}")
                
                # Show current prices
                for symbol in test_symbols:
                    data = manager.get_market_data(symbol)
                    if data:
                        print(f"   💰 {symbol}: ${data.price:.4f} ({data.change_percent:+.2f}%)")
        
        # Final statistics
        print("\n📈 Final Test Results:")
        print("=" * 30)
        
        final_stats = manager.get_stats()
        print(f"Connection State: {final_stats['connection_state']}")
        print(f"Messages Received: {final_stats['messages_received']}")
        print(f"Successful Connections: {final_stats['successful_connections']}")
        print(f"Reconnections: {final_stats['reconnections']}")
        print(f"Errors: {final_stats['errors']}")
        print(f"Active Subscriptions: {final_stats['active_subscriptions']}")
        print(f"Cached Symbols: {final_stats['cached_symbols']}")
        print(f"Trade Updates: {trade_updates}")
        
        # Test data quality
        print("\n🔍 Data Quality Check:")
        for symbol in test_symbols:
            market_data = manager.get_market_data(symbol)
            orderbook_data = manager.get_orderbook(symbol)
            
            if market_data:
                print(f"✅ {symbol} ticker data: Price=${market_data.price:.4f}, Volume={market_data.volume:.0f}")
            else:
                print(f"❌ {symbol} ticker data: Missing")
            
            if orderbook_data:
                best_bid = orderbook_data.bids[0][0] if orderbook_data.bids else 0
                best_ask = orderbook_data.asks[0][0] if orderbook_data.asks else 0
                print(f"✅ {symbol} orderbook: Bid=${best_bid:.4f}, Ask=${best_ask:.4f}")
            else:
                print(f"❌ {symbol} orderbook: Missing")
        
        # Test connection resilience
        print("\n🔧 Testing Connection Resilience...")
        print("Simulating brief disconnection...")
        
        # Disconnect and reconnect
        await manager.ws_client.disconnect()
        await asyncio.sleep(2)
        
        # Should auto-reconnect
        await asyncio.sleep(5)
        
        reconnect_stats = manager.get_stats()
        if reconnect_stats['connection_state'] == 'connected':
            print("✅ Auto-reconnection successful!")
        else:
            print(f"⚠️ Reconnection status: {reconnect_stats['connection_state']}")
        
        print(f"Total reconnections: {reconnect_stats['reconnections']}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🛑 Stopping WebSocket client...")
        await manager.stop()
        print("✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_websocket_client())