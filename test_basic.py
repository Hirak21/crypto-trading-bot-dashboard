#!/usr/bin/env python3
"""
Basic test script to verify core functionality.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that core modules can be imported."""
    print("Testing imports...")
    
    try:
        from crypto_trading_bot.models.trading import MarketData, TradingSignal, SignalAction
        print("✅ Trading models imported successfully")
    except Exception as e:
        print(f"❌ Failed to import trading models: {e}")
        return False
    
    try:
        from crypto_trading_bot.utils.config import ConfigManager
        print("✅ Config manager imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config manager: {e}")
        return False
    
    try:
        from tests.test_mock_data import MockDataGenerator
        print("✅ Mock data generator imported successfully")
    except Exception as e:
        print(f"❌ Failed to import mock data generator: {e}")
        return False
    
    return True

def test_mock_data_generation():
    """Test mock data generation."""
    print("\nTesting mock data generation...")
    
    try:
        from tests.test_mock_data import MockDataGenerator
        from crypto_trading_bot.models.trading import SignalAction
        
        generator = MockDataGenerator()
        
        # Test market data generation
        market_data = generator.generate_market_data(count=5)
        print(f"✅ Generated {len(market_data)} market data points")
        
        # Test trading signal generation
        signal = generator.generate_trading_signal(action=SignalAction.BUY)
        print(f"✅ Generated trading signal: {signal.action.value} with confidence {signal.confidence:.2f}")
        
        # Test trade generation
        trade = generator.generate_trade()
        print(f"✅ Generated trade: {trade.side} {trade.size} @ {trade.price}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock data generation failed: {e}")
        return False

def test_technical_analysis():
    """Test technical analysis functions."""
    print("\nTesting technical analysis...")
    
    try:
        from crypto_trading_bot.utils.technical_analysis import MovingAverages
        
        # Test simple moving average
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        sma_values = MovingAverages.sma(prices, 3)
        
        print(f"✅ SMA calculation successful: {len(sma_values)} values")
        
        # Test EMA
        ema_values = MovingAverages.ema(prices, 3)
        print(f"✅ EMA calculation successful: {len(ema_values)} values")
        
        return True
        
    except Exception as e:
        print(f"❌ Technical analysis test failed: {e}")
        return False

def test_config_management():
    """Test configuration management."""
    print("\nTesting configuration management...")
    
    try:
        from crypto_trading_bot.utils.config import ConfigManager
        from tests.test_mock_data import create_mock_config
        
        # Test mock config creation
        config = create_mock_config()
        print(f"✅ Mock config created with {len(config.symbols)} symbols")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run basic tests."""
    print("🚀 Running Basic Crypto Trading Bot Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_mock_data_generation,
        test_technical_analysis,
        test_config_management
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("❌ Test failed")
        except Exception as e:
            print(f"❌ Test error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return True
    else:
        print("💔 Some tests failed")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)