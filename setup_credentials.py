#!/usr/bin/env python3
"""
Credential setup script for the crypto trading bot.

This script helps you securely store your Binance API credentials.
"""

import sys
import getpass
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def setup_credentials():
    """Interactive credential setup."""
    print("🔐 Crypto Trading Bot - Credential Setup")
    print("=" * 50)
    
    print("\n⚠️  IMPORTANT SECURITY NOTES:")
    print("   • Your API credentials will be encrypted and stored locally")
    print("   • Never share your API secret with anyone")
    print("   • Use testnet credentials for initial testing")
    print("   • Ensure your API key has only necessary permissions")
    
    print("\n📝 Please enter your Binance API credentials:")
    
    # Get API key
    api_key = input("API Key: ").strip()
    if not api_key:
        print("❌ API Key cannot be empty")
        return False
    
    # Get API secret (hidden input)
    api_secret = getpass.getpass("API Secret: ").strip()
    if not api_secret:
        print("❌ API Secret cannot be empty")
        return False
    
    # Confirm testnet usage
    use_testnet = input("\nUse Binance Testnet? (y/N): ").strip().lower()
    testnet = use_testnet in ['y', 'yes']
    
    if testnet:
        print("✅ Using Binance Testnet (recommended for testing)")
    else:
        confirm = input("⚠️  You're using LIVE trading! Are you sure? (type 'LIVE' to confirm): ")
        if confirm != 'LIVE':
            print("❌ Setup cancelled for safety")
            return False
    
    try:
        # Try to import and use the credential manager
        from crypto_trading_bot.utils.config import ConfigManager
        
        config_manager = ConfigManager()
        
        # Load existing config
        config = config_manager.load_config()
        
        # Update testnet setting
        config['testnet'] = testnet
        
        # Add credentials to config for saving
        config['api_key'] = api_key
        config['api_secret'] = api_secret
        
        # Save config with credentials
        success = config_manager.save_config(config)
        
        if success:
            print("\n✅ Credentials saved successfully!")
            print(f"   Testnet mode: {'Enabled' if testnet else 'Disabled'}")
            print("   Credentials are encrypted and stored in config/credentials.enc")
            
            # Test credential loading
            try:
                loaded_key, loaded_secret = config_manager.get_api_credentials()
                if loaded_key == api_key and loaded_secret == api_secret:
                    print("✅ Credential verification successful")
                else:
                    print("⚠️  Credential verification failed")
                    return False
            except Exception as e:
                print(f"⚠️  Credential verification error: {e}")
                return False
            
            return True
        else:
            print("❌ Failed to save credentials")
            return False
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install dependencies first: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


def main():
    """Main setup function."""
    try:
        success = setup_credentials()
        
        if success:
            print("\n🎉 Setup Complete!")
            print("\n📝 Next Steps:")
            print("   1. Run: py simple_test.py (to verify core functionality)")
            print("   2. Install dependencies: pip install -r requirements.txt")
            print("   3. Test API connection with Binance")
            print("   4. Continue with bot development")
            
            return True
        else:
            print("\n❌ Setup failed. Please try again.")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  Setup cancelled by user")
        return False
    except Exception as e:
        print(f"\n💥 Setup crashed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)