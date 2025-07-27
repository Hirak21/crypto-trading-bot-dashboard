#!/usr/bin/env python3
"""
Simple WebSocket test to verify Binance connection
"""

import asyncio
import json
import websockets
import logging

async def test_simple_websocket():
    """Test basic WebSocket connection to Binance"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Use the correct Binance Futures WebSocket URL
    url = "wss://fstream.binance.com/ws/btcusdt@ticker"
    
    print(f"🔌 Connecting to {url}")
    
    try:
        async with websockets.connect(url) as websocket:
            print("✅ Connected successfully!")
            
            # Listen for messages
            message_count = 0
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_count += 1
                    
                    if message_count <= 5:  # Show first 5 messages
                        print(f"📊 Message {message_count}:")
                        print(f"   Symbol: {data.get('s')}")
                        print(f"   Price: ${float(data.get('c', 0)):.4f}")
                        print(f"   Change: {float(data.get('P', 0)):.2f}%")
                        print(f"   Volume: {float(data.get('v', 0)):,.0f}")
                        print()
                    
                    if message_count >= 10:
                        print(f"✅ Received {message_count} messages successfully!")
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                except Exception as e:
                    print(f"❌ Message processing error: {e}")
                    
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_websocket())