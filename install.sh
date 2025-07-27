#!/bin/bash
# Universal Installation Script for Crypto Trading Bot
# Works on Ubuntu, CentOS, macOS

set -e

echo "🚀 CRYPTO TRADING BOT - UNIVERSAL INSTALLER"
echo "============================================"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -f /etc/debian_version ]; then
        OS="ubuntu"
        echo "📱 Detected: Ubuntu/Debian"
    elif [ -f /etc/redhat-release ]; then
        OS="centos"
        echo "📱 Detected: CentOS/RHEL"
    else
        OS="linux"
        echo "📱 Detected: Generic Linux"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "📱 Detected: macOS"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

# Install Python and Git based on OS
install_dependencies() {
    echo "📦 Installing dependencies..."
    
    case $OS in
        "ubuntu")
            sudo apt update
            sudo apt install -y python3 python3-pip git
            ;;
        "centos")
            sudo yum update -y
            sudo yum install -y python3 python3-pip git
            ;;
        "macos")
            # Check if Homebrew is installed
            if ! command -v brew &> /dev/null; then
                echo "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install python git
            ;;
    esac
    
    echo "✅ Dependencies installed"
}

# Check if Python is available
check_python() {
    echo "🐍 Checking Python..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "❌ Python not found. Installing..."
        install_dependencies
        PYTHON_CMD="python3"
    fi
    
    echo "✅ Python found: $PYTHON_CMD"
}

# Install pip packages
install_python_packages() {
    echo "📦 Installing Python packages..."
    
    $PYTHON_CMD -m pip install --upgrade pip
    $PYTHON_CMD -m pip install -r requirements.txt
    
    echo "✅ Python packages installed"
}

# Run setup
run_setup() {
    echo "⚙️ Running bot setup..."
    
    $PYTHON_CMD setup_bot.py
    
    echo "✅ Bot setup complete"
}

# Main installation
main() {
    echo "Starting installation..."
    
    # Check if we're in the right directory
    if [ ! -f "requirements.txt" ]; then
        echo "❌ requirements.txt not found. Please run this script from the bot directory."
        exit 1
    fi
    
    check_python
    install_python_packages
    run_setup
    
    echo ""
    echo "🎉 INSTALLATION COMPLETE!"
    echo "========================="
    echo ""
    echo "🚀 Quick Start:"
    echo "   $PYTHON_CMD scalping_scanner.py"
    echo ""
    echo "📚 Documentation:"
    echo "   - Read QUICK_START.md for detailed instructions"
    echo "   - Check README.md for full documentation"
    echo ""
    echo "Happy Trading! 🎯"
}

# Run main function
main "$@"