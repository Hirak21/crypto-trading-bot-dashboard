#!/usr/bin/env python3
"""
Simple credential test without external dependencies.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_credentials():
    """Test credential loading without external dependencies."""
    print("🔐 Testing Credential Management...")
    
    try:
        # Test basic imports first
        from crypto_trading_bot.utils.config import ConfigManager
        print("✅ ConfigManager imported successfully")
        
        # Create config manager
        config_manager = ConfigManager()
        print("✅ ConfigManager created")
        
        # Try to load credentials
        try:
            api_key, api_secret = config_manager.get_api_credentials()
            
            # Don't print the actual credentials for security
            if api_key and api_secret:
                print(f"✅ Credentials loaded successfully")
                print(f"   API Key length: {len(api_key)} characters")
                print(f"   API Secret length: {len(api_secret)} characters")
                print(f"   API Key starts with: {api_key[:8]}...")
                return True
            else:
                print("❌ Credentials are empty")
                return False
                
        except FileNotFoundError:
            print("❌ Credentials file not found")
            print("   You need to set up your credentials first")
            return False
        except Exception as e:
            print(f"❌ Failed to load credentials: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\n⚙️ Testing Configuration...")
    
    try:
        from crypto_trading_bot.utils.config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        print(f"✅ Configuration loaded with {len(config)} settings")
        print(f"   Testnet mode: {config.get('testnet', 'Not set')}")
        print(f"   Symbols: {config.get('symbols', 'Not set')}")
        print(f"   Strategies: {list(config.get('strategies', {}).keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def setup_credentials_interactive():
    """Interactive credential setup."""
    print("\n🔧 Setting up credentials...")
    print("Please enter your Binance API credentials:")
    
    api_key = input("API Key: ").strip()
    if not api_key:
        print("❌ API Key cannot be empty")
        return False
    
    api_secret = input("API Secret: ").strip()
    if not api_secret:
        print("❌ API Secret cannot be empty")
        return False
    
    try:
        from crypto_trading_bot.utils.config import ConfigManager
        
        config_manager = ConfigManager()
        
        # Load existing config
        config = config_manager.load_config()
        
        # Add credentials
        config['api_key'] = api_key
        config['api_secret'] = api_secret
        config['testnet'] = True  # Always use testnet for safety
        
        # Save config
        success = config_manager.save_config(config)
        
        if success:
            print("✅ Credentials saved successfully!")
            return True
        else:
            print("❌ Failed to save credentials")
            return False
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


def main():
    """Main function."""
    print("🚀 Crypto Trading Bot - Credential Test\n")
    
    # Test if credentials already exist
    cred_test = test_credentials()
    config_test = test_config()
    
    if not cred_test:
        print("\n" + "="*50)
        print("Credentials not found or invalid.")
        print("Let's set them up now!")
        print("="*50)
        
        setup_success = setup_credentials_interactive()
        
        if setup_success:
            print("\n✅ Setup complete! Testing again...")
            cred_test = test_credentials()
            config_test = test_config()
    
    print(f"\n📊 Results:")
    print(f"   Credentials: {'✅ Working' if cred_test else '❌ Failed'}")
    print(f"   Configuration: {'✅ Working' if config_test else '❌ Failed'}")
    
    if cred_test and config_test:
        print("\n🎉 Everything is working!")
        print("\n📝 Next Steps:")
        print("   1. Install dependencies: pip install aiohttp websockets")
        print("   2. Run: py test_binance_api.py")
        print("   3. Continue with bot development")
        return True
    else:
        print("\n⚠️ Some issues need to be resolved.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)