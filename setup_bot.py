#!/usr/bin/env python3
"""
Automated Bot Setup Script
Sets up the entire trading bot system automatically
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_header():
    print("🚀 CRYPTO TRADING BOT - AUTOMATED SETUP")
    print("=" * 60)
    print("This script will set up your trading bot automatically")
    print("=" * 60)

def check_python():
    """Check Python version"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Please upgrade Python.")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ["config", "logs", "results"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_credentials():
    """Setup API credentials"""
    print("\n🔑 Setting up API credentials...")
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    credentials_file = config_dir / "credentials.json"
    
    if credentials_file.exists():
        print("✅ Credentials file already exists")
        return True
    
    print("Please enter your Binance API credentials:")
    api_key = input("API Key: ").strip()
    api_secret = input("API Secret: ").strip()
    
    if not api_key or not api_secret:
        print("❌ API credentials cannot be empty")
        return False
    
    credentials = {
        "binance": {
            "api_key": api_key,
            "api_secret": api_secret
        }
    }
    
    with open(credentials_file, 'w') as f:
        json.dump(credentials, f, indent=2)
    
    print("✅ Credentials saved successfully")
    return True

def create_bot_config():
    """Create bot configuration file"""
    print("\n⚙️ Creating bot configuration...")
    
    config_file = Path("config/bot_config.json")
    
    if config_file.exists():
        print("✅ Bot config already exists")
        return
    
    bot_config = {
        "testnet": True,
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "strategies": {
            "liquidity": {"enabled": True, "weight": 1.0},
            "momentum": {"enabled": True, "weight": 1.0},
            "chart_patterns": {"enabled": True, "weight": 1.0},
            "candlestick_patterns": {"enabled": True, "weight": 1.0}
        },
        "risk_config": {
            "max_position_size": 0.02,
            "daily_loss_limit": 0.05,
            "max_drawdown": 0.15,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04
        },
        "notification_config": {
            "enabled": True,
            "channels": ["console"]
        },
        "logging_config": {
            "level": "INFO",
            "log_dir": "logs"
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(bot_config, f, indent=2)
    
    print("✅ Bot configuration created")

def test_setup():
    """Test the setup"""
    print("\n🧪 Testing setup...")
    
    try:
        # Test credentials loading
        with open('config/credentials.json', 'r') as f:
            creds = json.load(f)
        
        if 'binance' in creds and 'api_key' in creds['binance']:
            print("✅ Credentials format is valid")
        else:
            print("❌ Invalid credentials format")
            return False
        
        # Test imports
        try:
            from binance.client import Client
            print("✅ Binance library imported successfully")
        except ImportError:
            print("❌ Failed to import Binance library")
            return False
        
        print("✅ Setup test passed")
        return True
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False

def show_next_steps():
    """Show next steps to user"""
    print("\n🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("Your trading bot is ready to use!")
    print("\n📋 Next Steps:")
    print("\n1. 🔍 Run Adaptive Scalping Scanner:")
    print("   python scalping_scanner.py")
    print("\n2. 🎯 Run Strategy Scanner:")
    print("   python strategy_scanner.py")
    print("\n3. 🤖 Execute Trades:")
    print("   python trading_executor.py")
    print("\n📚 Documentation:")
    print("   - Read QUICK_START.md for detailed instructions")
    print("   - Check README.md for full documentation")
    print("\n🛡️ Safety Note:")
    print("   - Bot is configured for TESTNET by default")
    print("   - Change 'testnet': false in config/bot_config.json for live trading")
    print("\n🚀 Happy Trading!")

def main():
    """Main setup function"""
    print_header()
    
    # Check prerequisites
    if not check_python():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create directories
    create_directories()
    
    # Setup credentials
    if not setup_credentials():
        return False
    
    # Create bot config
    create_bot_config()
    
    # Test setup
    if not test_setup():
        return False
    
    # Show next steps
    show_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)