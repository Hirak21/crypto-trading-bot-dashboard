#!/usr/bin/env python3
"""
Simple test script for the crypto trading bot core functionality.

This script tests basic functionality without external dependencies.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_basic_imports():
    """Test basic imports without external dependencies."""
    print("🔧 Testing Basic Imports...")
    
    try:
        # Test enum imports
        from crypto_trading_bot.models.trading import SignalAction, PositionSide, OrderSide
        print("✅ Trading enums imported successfully")
        
        # Test basic data structures (without validation that requires external libs)
        from crypto_trading_bot.models.trading import TradingSignal, Position, Trade
        print("✅ Trading models imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic import test failed: {e}")
        return False


def test_data_model_creation():
    """Test creating data models with basic validation."""
    print("\n📊 Testing Data Model Creation...")
    
    try:
        from crypto_trading_bot.models.trading import (
            TradingSignal, SignalAction, MarketData, Position, PositionSide
        )
        
        # Test TradingSignal creation
        signal = TradingSignal(
            symbol="BTCUSDT",
            action=SignalAction.BUY,
            confidence=0.85,
            strategy="test_strategy"
        )
        print(f"✅ TradingSignal created: {signal.action.value} {signal.symbol}")
        print(f"   Signal ID: {signal.signal_id}")
        print(f"   Confidence: {signal.confidence}")
        
        # Test Position creation
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=45000.0,
            current_price=45500.0
        )
        print(f"✅ Position created: {position.side.value} {position.size} {position.symbol}")
        print(f"   Unrealized P&L: ${position.unrealized_pnl:.2f}")
        print(f"   P&L %: {position.unrealized_pnl_percentage:.2f}%")
        
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
        print(f"   Spread: ${market_data.spread:.2f} ({market_data.spread_percentage:.3f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_serialization():
    """Test basic serialization functionality."""
    print("\n💾 Testing Serialization...")
    
    try:
        from crypto_trading_bot.models.trading import TradingSignal, SignalAction
        
        # Create a signal
        signal = TradingSignal(
            symbol="ETHUSDT",
            action=SignalAction.SELL,
            confidence=0.75,
            strategy="momentum",
            target_price=3000.0,
            stop_loss=3200.0
        )
        
        # Test to_dict serialization
        signal_dict = signal.to_dict()
        print(f"✅ Signal serialized to dict with {len(signal_dict)} fields")
        
        # Test from_dict deserialization
        restored_signal = TradingSignal.from_dict(signal_dict)
        print(f"✅ Signal restored from dict: {restored_signal.action.value} {restored_signal.symbol}")
        
        # Verify data integrity
        if (restored_signal.symbol == signal.symbol and 
            restored_signal.action == signal.action and
            restored_signal.confidence == signal.confidence):
            print("✅ Serialization data integrity verified")
        else:
            print("❌ Serialization data integrity check failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_models():
    """Test configuration models without external dependencies."""
    print("\n⚙️ Testing Configuration Models...")
    
    try:
        from crypto_trading_bot.models.config import (
            RiskConfig, StrategyConfig, LogLevel, NotificationChannel
        )
        
        # Test RiskConfig
        risk_config = RiskConfig(
            max_position_size=0.03,
            daily_loss_limit=0.08,
            stop_loss_pct=0.025
        )
        print(f"✅ RiskConfig created with {risk_config.max_position_size*100}% max position size")
        
        # Test StrategyConfig
        strategy_config = StrategyConfig(
            enabled=True,
            weight=1.5,
            min_confidence=0.6
        )
        print(f"✅ StrategyConfig created with weight {strategy_config.weight}")
        
        # Test enums
        log_level = LogLevel.INFO
        notification_channel = NotificationChannel.CONSOLE
        print(f"✅ Enums working: {log_level.value}, {notification_channel.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_interfaces():
    """Test interface definitions."""
    print("\n🔌 Testing Interfaces...")
    
    try:
        from crypto_trading_bot.interfaces import (
            IStrategy, IMarketDataProvider, ITradeExecutor, IRiskManager
        )
        print("✅ All interfaces imported successfully")
        
        # Test that interfaces are abstract
        try:
            # This should fail because IStrategy is abstract
            strategy = IStrategy()
            print("❌ Interface abstraction not working")
            return False
        except TypeError:
            print("✅ Interface abstraction working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Interface test failed: {e}")
        return False


def test_project_structure():
    """Test project structure and file organization."""
    print("\n📁 Testing Project Structure...")
    
    try:
        # Check main directories exist
        required_dirs = [
            "crypto_trading_bot",
            "crypto_trading_bot/models",
            "crypto_trading_bot/strategies", 
            "crypto_trading_bot/managers",
            "crypto_trading_bot/utils"
        ]
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                print(f"❌ Missing directory: {dir_path}")
                return False
        
        print("✅ All required directories exist")
        
        # Check key files exist
        required_files = [
            "crypto_trading_bot/__init__.py",
            "crypto_trading_bot/main.py",
            "crypto_trading_bot/interfaces.py",
            "crypto_trading_bot/models/trading.py",
            "crypto_trading_bot/models/config.py",
            "crypto_trading_bot/strategies/base_strategy.py",
            "requirements.txt",
            "setup.py"
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                print(f"❌ Missing file: {file_path}")
                return False
        
        print("✅ All required files exist")
        return True
        
    except Exception as e:
        print(f"❌ Project structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Simple Crypto Trading Bot Tests\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Model Creation", test_data_model_creation),
        ("Serialization", test_serialization),
        ("Configuration Models", test_configuration_models),
        ("Interfaces", test_interfaces),
        ("Project Structure", test_project_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core tests passed! The bot foundation is solid.")
        print("\n📝 Next Steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Add your Binance API credentials")
        print("   3. Continue with Binance API integration (Task 3)")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)