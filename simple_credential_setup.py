#!/usr/bin/env python3
"""
Simple credential setup without external dependencies.
This creates a basic configuration file for testing.
"""

import json
import os
from pathlib import Path


def create_simple_config():
    """Create a simple configuration file."""
    print("🔧 Creating Simple Configuration...")
    
    # Get API credentials
    print("\nPlease enter your Binance API credentials:")
    api_key = input("API Key: ").strip()
    if not api_key:
        print("❌ API Key cannot be empty")
        return False
    
    api_secret = input("API Secret: ").strip()
    if not api_secret:
        print("❌ API Secret cannot be empty")
        return False
    
    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Create simple config
    config = {
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
    
    # Save main config
    config_file = config_dir / "bot_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save credentials separately (unencrypted for now)
    credentials = {
        "api_key": api_key,
        "api_secret": api_secret
    }
    
    cred_file = config_dir / "credentials.json"
    with open(cred_file, 'w') as f:
        json.dump(credentials, f, indent=2)
    
    print(f"✅ Configuration saved to {config_file}")
    print(f"✅ Credentials saved to {cred_file}")
    print("⚠️  Note: Credentials are stored unencrypted for now")
    
    return True


def test_config():
    """Test the created configuration."""
    print("\n📋 Testing Configuration...")
    
    try:
        config_file = Path("config/bot_config.json")
        cred_file = Path("config/credentials.json")
        
        if not config_file.exists():
            print("❌ Configuration file not found")
            return False
        
        if not cred_file.exists():
            print("❌ Credentials file not found")
            return False
        
        # Load and validate config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        with open(cred_file, 'r') as f:
            credentials = json.load(f)
        
        print(f"✅ Configuration loaded: {len(config)} sections")
        print(f"✅ Credentials loaded: API key length {len(credentials['api_key'])}")
        print(f"   Testnet mode: {config['testnet']}")
        print(f"   Symbols: {config['symbols']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def create_requirements_installer():
    """Create a batch file to install requirements."""
    print("\n📦 Creating Requirements Installer...")
    
    # Create a batch file for Windows
    batch_content = """@echo off
echo Installing Python dependencies...
echo.

REM Try different Python commands
py -m pip install --upgrade pip
if %errorlevel% neq 0 (
    python -m pip install --upgrade pip
    if %errorlevel% neq 0 (
        python3 -m pip install --upgrade pip
    )
)

echo.
echo Installing required packages...
py -m pip install cryptography aiohttp websockets pandas numpy
if %errorlevel% neq 0 (
    python -m pip install cryptography aiohttp websockets pandas numpy
    if %errorlevel% neq 0 (
        python3 -m pip install cryptography aiohttp websockets pandas numpy
    )
)

echo.
echo Installation complete!
pause
"""
    
    with open("install_requirements.bat", 'w') as f:
        f.write(batch_content)
    
    print("✅ Created install_requirements.bat")
    print("   Run this file to install Python dependencies")
    
    return True


def main():
    """Main function."""
    print("🚀 Simple Crypto Trading Bot Setup\n")
    
    # Create configuration
    config_success = create_simple_config()
    
    if config_success:
        # Test configuration
        test_success = test_config()
        
        # Create installer
        installer_success = create_requirements_installer()
        
        if test_success:
            print("\n🎉 Setup Complete!")
            print("\n📝 Next Steps:")
            print("   1. Run install_requirements.bat to install dependencies")
            print("   2. Run: py simple_test.py (to test core functionality)")
            print("   3. Run: py test_binance_api.py (to test API connection)")
            print("   4. Continue with bot development")
            
            print("\n⚠️  Security Note:")
            print("   Your credentials are currently stored unencrypted.")
            print("   After installing dependencies, the bot will use encrypted storage.")
            
            return True
    
    print("\n❌ Setup failed. Please try again.")
    return False


if __name__ == "__main__":
    try:
        success = main()
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        print("\n⏹️ Setup cancelled by user")
    except Exception as e:
        print(f"\n💥 Setup crashed: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")