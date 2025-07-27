#!/bin/bash
# Termux Installation Script for Crypto Trading Bot
# Optimized for Android mobile devices

set -e

echo "📱 CRYPTO TRADING BOT - TERMUX INSTALLER"
echo "========================================"
echo "Installing on Android via Termux..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in Termux
check_termux() {
    print_status "Checking Termux environment..."
    
    if [ -z "$PREFIX" ] || [ ! -d "$PREFIX" ]; then
        print_error "This script must be run in Termux!"
        print_error "Please install Termux from F-Droid or Google Play"
        exit 1
    fi
    
    print_success "Termux environment detected"
}

# Update Termux packages
update_packages() {
    print_status "Updating Termux packages..."
    
    # Update package lists
    pkg update -y
    
    # Upgrade existing packages
    pkg upgrade -y
    
    print_success "Packages updated"
}

# Install required packages
install_packages() {
    print_status "Installing required packages..."
    
    # Essential packages for Python and compilation
    pkg install -y \
        python \
        python-pip \
        git \
        curl \
        wget \
        build-essential \
        libffi \
        openssl \
        rust \
        binutils
    
    print_success "Packages installed"
}

# Setup Python environment
setup_python() {
    print_status "Setting up Python environment..."
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install essential Python packages
    pip install --upgrade setuptools wheel
    
    print_success "Python environment ready"
}

# Install bot dependencies with mobile optimizations
install_bot_dependencies() {
    print_status "Installing bot dependencies (this may take 10-15 minutes)..."
    
    # Install dependencies one by one for better error handling
    print_status "Installing core dependencies..."
    pip install python-binance websockets aiohttp
    
    print_status "Installing data processing libraries..."
    pip install pandas numpy scipy
    
    print_status "Installing security and config libraries..."
    pip install cryptography pydantic python-dotenv pyyaml
    
    print_status "Installing logging library..."
    pip install structlog
    
    print_success "Bot dependencies installed"
}

# Create mobile-optimized configuration
create_mobile_config() {
    print_status "Creating mobile-optimized configuration..."
    
    # Create config directory
    mkdir -p config
    
    # Create mobile-optimized bot config
    cat > config/bot_config.json << 'EOF'
{
  "testnet": true,
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "mobile_optimized": true,
  "batch_size": 50,
  "scan_delay": 2,
  "strategies": {
    "liquidity": {"enabled": true, "weight": 1.0},
    "momentum": {"enabled": true, "weight": 1.0},
    "chart_patterns": {"enabled": true, "weight": 1.0},
    "candlestick_patterns": {"enabled": true, "weight": 1.0}
  },
  "risk_config": {
    "max_position_size": 0.01,
    "daily_loss_limit": 0.03,
    "max_drawdown": 0.10,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.03
  },
  "notification_config": {
    "enabled": true,
    "channels": ["console"]
  },
  "logging_config": {
    "level": "INFO",
    "log_dir": "logs"
  }
}
EOF
    
    print_success "Mobile configuration created"
}

# Setup credentials
setup_credentials() {
    print_status "Setting up API credentials..."
    
    if [ -f "config/credentials.json" ]; then
        print_warning "Credentials file already exists"
        return
    fi
    
    echo ""
    echo "📋 Please enter your Binance API credentials:"
    echo "   (You can get these from Binance > API Management)"
    echo ""
    
    read -p "API Key: " api_key
    read -p "API Secret: " api_secret
    
    if [ -z "$api_key" ] || [ -z "$api_secret" ]; then
        print_error "API credentials cannot be empty"
        return 1
    fi
    
    # Create credentials file
    cat > config/credentials.json << EOF
{
  "binance": {
    "api_key": "$api_key",
    "api_secret": "$api_secret"
  }
}
EOF
    
    print_success "Credentials saved"
}

# Create mobile helper scripts
create_mobile_scripts() {
    print_status "Creating mobile helper scripts..."
    
    # Bot starter script
    cat > start_bot.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Crypto Trading Bot on Mobile..."

# Enable wake lock to prevent sleep
if command -v termux-wake-lock &> /dev/null; then
    termux-wake-lock
    echo "✅ Wake lock enabled"
fi

# Start bot with logging
python scalping_scanner.py 2>&1 | tee bot.log
EOF
    
    # Background runner
    cat > run_background.sh << 'EOF'
#!/bin/bash
echo "🔄 Starting bot in background..."

# Enable wake lock
if command -v termux-wake-lock &> /dev/null; then
    termux-wake-lock
fi

# Run in background
nohup python scalping_scanner.py > bot.log 2>&1 &

echo "✅ Bot started in background"
echo "📊 Check status: ps aux | grep python"
echo "📋 View logs: tail -f bot.log"
EOF
    
    # Status checker
    cat > check_status.sh << 'EOF'
#!/bin/bash
echo "📊 Bot Status Check"
echo "=================="

# Check if bot is running
if pgrep -f "scalping_scanner.py" > /dev/null; then
    echo "✅ Bot is running"
    echo "📊 Process ID: $(pgrep -f scalping_scanner.py)"
else
    echo "❌ Bot is not running"
fi

# Show recent logs
if [ -f "bot.log" ]; then
    echo ""
    echo "📋 Recent logs:"
    tail -10 bot.log
fi

# Show results
echo ""
echo "📁 Result files:"
ls -la *.json 2>/dev/null || echo "No result files found"

# System resources
echo ""
echo "💾 Memory usage:"
free -h

echo ""
echo "🔋 Battery status:"
if command -v termux-battery-status &> /dev/null; then
    termux-battery-status | head -5
else
    echo "Install termux-api for battery info: pkg install termux-api"
fi
EOF
    
    # Make scripts executable
    chmod +x start_bot.sh run_background.sh check_status.sh
    
    print_success "Helper scripts created"
}

# Install Termux API (optional)
install_termux_api() {
    print_status "Installing Termux API for enhanced mobile features..."
    
    pkg install -y termux-api
    
    print_success "Termux API installed"
    print_warning "For full functionality, also install 'Termux:API' app from F-Droid"
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test Python imports
    python -c "
import json
from binance.client import Client
import pandas as pd
import numpy as np
print('✅ All imports successful')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
        return 0
    else
        print_error "Installation test failed"
        return 1
    fi
}

# Show mobile-specific instructions
show_mobile_instructions() {
    echo ""
    print_success "🎉 TERMUX INSTALLATION COMPLETE!"
    echo "=================================="
    echo ""
    echo "📱 Mobile Trading Bot Ready!"
    echo ""
    echo "🚀 Quick Start:"
    echo "   ./start_bot.sh                 # Start bot with console output"
    echo "   ./run_background.sh            # Run bot in background"
    echo "   ./check_status.sh              # Check bot status"
    echo ""
    echo "📊 Manual Commands:"
    echo "   python scalping_scanner.py     # Run adaptive scanner"
    echo "   python strategy_scanner.py     # Run strategy scanner"
    echo "   python trading_executor.py     # Execute trades"
    echo ""
    echo "📋 Mobile Tips:"
    echo "   • Keep Termux app open or use background mode"
    echo "   • Enable wake lock to prevent sleep"
    echo "   • Monitor battery usage"
    echo "   • Use WiFi for better stability"
    echo ""
    echo "🔧 Useful Commands:"
    echo "   tail -f bot.log                # View live logs"
    echo "   ps aux | grep python           # Check running processes"
    echo "   pkill -f python                # Stop all bots"
    echo ""
    echo "📚 Documentation:"
    echo "   Read TERMUX_SETUP.md for detailed mobile guide"
    echo ""
    echo "🛡️ Safety Reminder:"
    echo "   Bot is configured for TESTNET by default"
    echo "   Change 'testnet': false in config/bot_config.json for live trading"
    echo ""
    print_success "Happy Mobile Trading! 📱💰"
}

# Main installation function
main() {
    echo "Starting Termux installation..."
    
    check_termux
    update_packages
    install_packages
    setup_python
    install_bot_dependencies
    create_mobile_config
    setup_credentials
    create_mobile_scripts
    install_termux_api
    
    if test_installation; then
        show_mobile_instructions
        return 0
    else
        print_error "Installation failed. Please check the errors above."
        return 1
    fi
}

# Run main function
main "$@"