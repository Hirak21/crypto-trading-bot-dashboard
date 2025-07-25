#!/usr/bin/env python3
"""
Test script for the crypto trading bot.

This script tests the basic functionality including configuration loading,
credential management, and data model validation.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_trading_bot.main import TradingBotApplication
from crypto_trading_bot.utils.config import ConfigManager
from crypto_trading_bot.utils.security import store_api_credentials, load_api_credentials
from crypto_trading_bot.models import (
    TradingSignal, MarketData, SignalAction, 
    create_default_config, validate_trading_signal
)
from datetime import datetime


def test_configuration():
    """Test configuration management."""
    print("🔧 Testing Configuration Management...")
    
    try:
        # Test default config creation
        config = create_default_config()
        print(f"✅ Default config created with {len(config.symbols)} symbols")
        print(f"   Enabled strategies: {config.get_enabled_strategies()}")
        
        # Test config manager
        config_manager = ConfigManager()
        loaded_config = config_manager.load_config()
        print(f"✅ Configuration loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_credentials():
    """Test credential management."""
    print("\n🔐 Testing Credential Management...")
    
    try:
        # Test storing credentials (using dummy values for testing)
        api_key = "test_api_key_12345"
        api_secret = "test_api_secret_67890"
        
        success = store_api_credentials(api_key, api_secret)
        if success:
            print("✅ Credentials stored successfully")
        else:
            print("❌ Failed to store credentials")
            return False
        
        # Test loading credentials
        loaded_key, loaded_secret = load_api_credentials()
        if loaded_key == api_key and loaded_secret == api_secret:
            print("✅ Credentials loaded and verified successfully")
        else:
            print("❌ Credential verification failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Credential test failed: {e}")
        return False


def test_data_models():
    """Test data model creation and validation."""
    print("\n📊 Testing Data Models...")
    
    try:
        # Test MarketData creation
        market_data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            price=45000.0,
            volume=1000.0,
            bid=44999.0,
            ask=45001.0
        )
        print(f"✅ MarketData created: {market_data.symbol} @ ${market_data.price}")
        
        # Test TradingSignal creation
        signal = TradingSignal(
            symbol="BTCUSDT",
            action=SignalAction.BUY,
            confidence=0.85,
            strategy="test_strategy",
            target_price=46000.0,
            stop_loss=44000.0
        )
        print(f"✅ TradingSignal created: {signal.action.value} {signal.symbol} (confidence: {signal.confidence})")
        
        # Test validation
        is_valid = validate_trading_signal(signal)
        if is_valid:
            print("✅ Signal validation passed")
        else:
            print("❌ Signal validation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data model test failed: {e}")
        return False


def test_logging():
    """Test logging system."""
    print("\n📝 Testing Logging System...")
    
    try:
        from crypto_trading_bot.utils.logging_config import setup_logging, get_logger
        
        # Setup logging
        setup_logging()
        logger = get_logger("test_logger")
        
        # Test different log levels
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        print("✅ Logging system working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False


async def test_bot_initialization():
    """Test bot initialization."""
    print("\n🤖 Testing Bot Initialization...")
    
    try:
        # Create bot application
        app = TradingBotApplication()
        print("✅ Bot application created successfully")
        
        # Test configuration validation
        config_manager = ConfigManager()
        
        # Add test credentials to config for validation
        test_config = config_manager.load_config()
        test_config['api_key'] = "test_key"
        test_config['api_secret'] = "test_secret"
        
        # Save config with credentials
        success = config_manager.save_config(test_config)
        if success:
            print("✅ Configuration saved with credentials")
        else:
            print("❌ Failed to save configuration")
            return False
        
        print("✅ Bot initialization test completed")
        return True
        
    except Exception as e:
        print(f"❌ Bot initialization test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Crypto Trading Bot Tests\n")
    
    tests = [
        ("Configuration", test_configuration),
        ("Credentials", test_credentials),
        ("Data Models", test_data_models),
        ("Logging", test_logging),
        ("Bot Initialization", test_bot_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The bot is ready for development.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
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
        sys.exit(1)