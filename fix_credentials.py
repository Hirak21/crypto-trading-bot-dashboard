#!/usr/bin/env python3
"""
Fix credentials issue by ensuring proper file paths and format
"""

import json
import os
from pathlib import Path

def fix_credentials():
    """Fix the credentials file issue"""
    
    # Ensure config directory exists
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Check if credentials exist in bot_config.json
    bot_config_path = config_dir / "bot_config.json"
    credentials_path = config_dir / "credentials.json"
    
    if bot_config_path.exists():
        print("✅ Found bot_config.json")
        
        # If credentials.json doesn't exist, create it from user input
        if not credentials_path.exists():
            print("⚠️ credentials.json not found, creating it...")
            
            # Get API credentials
            print("\nPlease enter your Binance API credentials:")
            api_key = input("API Key: ").strip()
            api_secret = input("API Secret: ").strip()
            
            # Create credentials file
            credentials = {
                "binance": {
                    "api_key": api_key,
                    "api_secret": api_secret
                }
            }
            
            with open(credentials_path, 'w') as f:
                json.dump(credentials, f, indent=2)
            
            print(f"✅ Credentials saved to {credentials_path}")
        else:
            print("✅ credentials.json already exists")
    
    else:
        print("❌ bot_config.json not found")
        print("Please run simple_credential_setup.py first")
        return False
    
    # Verify credentials file
    try:
        with open(credentials_path, 'r') as f:
            creds = json.load(f)
        
        if 'binance' in creds and 'api_key' in creds['binance']:
            print("✅ Credentials file is valid")
            return True
        else:
            print("❌ Invalid credentials format")
            return False
            
    except Exception as e:
        print(f"❌ Error reading credentials: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing credentials configuration...")
    if fix_credentials():
        print("\n🎉 Credentials fixed successfully!")
        print("\nYou can now run:")
        print("python enhanced_scanner.py")
    else:
        print("\n❌ Failed to fix credentials")
        print("Please run simple_credential_setup.py first")