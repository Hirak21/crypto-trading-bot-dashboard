#!/bin/bash

# 🐧 Crypto Trading Bot - Ubuntu Setup Script
# This script automates the setup process for Ubuntu Linux

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu
check_ubuntu() {
    if [[ ! -f /etc/os-release ]] || ! grep -q "Ubuntu" /etc/os-release; then
        log_error "This script is designed for Ubuntu Linux"
        exit 1
    fi
    
    local version=$(lsb_release -rs)
    log_info "Detected Ubuntu $version"
    
    if [[ $(echo "$version >= 20.04" | bc -l) -eq 0 ]]; then
        log_warning "Ubuntu 20.04 or newer is recommended"
    fi
}

# Update system packages
update_system() {
    log_info "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    log_success "System packages updated"
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    sudo apt install -y \
        build-essential \
        curl \
        wget \
        git \
        vim \
        htop \
        unzip \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        bc
    
    log_success "System dependencies installed"
}

# Install Python 3.11+
install_python() {
    log_info "Installing Python 3.11+..."
    
    # Add deadsnakes PPA for newer Python versions
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    
    sudo apt install -y \
        python3.11 \
        python3.11-pip \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        python3-venv \
        python3-dev
    
    # Set python3.11 as default python3 if available
    if command -v python3.11 &> /dev/null; then
        sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
        log_success "Python 3.11 installed and set as default"
    else
        log_warning "Python 3.11 not available, using system Python3"
    fi
    
    python3 --version
}

# Install Node.js and npm
install_nodejs() {
    log_info "Installing Node.js and npm..."
    
    # Install Node.js 18.x
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt install -y nodejs
    
    # Verify installation
    node --version
    npm --version
    
    log_success "Node.js and npm installed"
}

# Create project structure
setup_project() {
    log_info "Setting up project structure..."
    
    # Create directories
    mkdir -p config logs state
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    log_success "Project structure created"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install core dependencies
    pip install \
        cryptography \
        pydantic \
        python-dotenv \
        pyyaml \
        aiohttp \
        websockets \
        fastapi \
        uvicorn
    
    # Install data analysis libraries
    pip install pandas numpy scipy || log_warning "Some data analysis libraries failed to install"
    
    # Install optional dependencies
    pip install python-binance structlog || log_warning "Some optional dependencies failed to install"
    
    # Install testing dependencies
    pip install pytest pytest-asyncio pytest-mock
    
    # Install development tools
    pip install black flake8 mypy
    
    log_success "Python dependencies installed"
}

# Install Node.js dependencies for dashboard
install_dashboard_deps() {
    log_info "Installing dashboard dependencies..."
    
    if [[ -d "dashboard" ]]; then
        cd dashboard
        npm install
        cd ..
        log_success "Dashboard dependencies installed"
    else
        log_warning "Dashboard directory not found, skipping dashboard setup"
    fi
}

# Create configuration files
create_config() {
    log_info "Creating configuration files..."
    
    # Create .env file
    cat > .env << 'EOF'
# API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
TESTNET=true

# Bot Configuration
TRADING_ENABLED=false
DRY_RUN=true
LOG_LEVEL=INFO

# Database Configuration (optional)
DATABASE_URL=sqlite:///trading_bot.db

# Notification Configuration
NOTIFICATIONS_ENABLED=true
CONSOLE_NOTIFICATIONS=true
EOF

    # Create bot config
    cat > config/bot_config.yaml << 'EOF'
# Trading Bot Configuration
symbols:
  - BTCUSDT
  - ETHUSDT
  - ADAUSDT

strategies:
  liquidity:
    enabled: true
    weight: 1.0
    min_confidence: 0.6
  momentum:
    enabled: true
    weight: 1.2
    min_confidence: 0.7
  chart_patterns:
    enabled: false
    weight: 0.8
    min_confidence: 0.8

risk_management:
  max_position_size: 0.02
  daily_loss_limit: 0.05
  max_drawdown: 0.15
  stop_loss_pct: 0.02
  take_profit_pct: 0.04

logging:
  level: INFO
  log_dir: logs
  console_logging: true
EOF

    # Set proper permissions
    chmod 600 .env
    chmod 600 config/bot_config.yaml
    
    log_success "Configuration files created"
}

# Create systemd service (optional)
create_service() {
    log_info "Creating systemd service..."
    
    local service_file="/etc/systemd/system/crypto-trading-bot.service"
    local current_dir=$(pwd)
    local current_user=$(whoami)
    
    sudo tee $service_file > /dev/null << EOF
[Unit]
Description=Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=$current_user
WorkingDirectory=$current_dir
Environment=PATH=$current_dir/venv/bin
ExecStart=$current_dir/venv/bin/python -m crypto_trading_bot.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    
    log_success "Systemd service created (not enabled by default)"
    log_info "To enable: sudo systemctl enable crypto-trading-bot"
    log_info "To start: sudo systemctl start crypto-trading-bot"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    source venv/bin/activate
    
    # Run basic test
    if python3 test_basic.py; then
        log_success "Basic tests passed"
    else
        log_warning "Some basic tests failed"
    fi
    
    # Run component tests
    if python3 run_tests.py --component strategies; then
        log_success "Strategy tests passed"
    else
        log_warning "Some strategy tests failed"
    fi
}

# Create startup scripts
create_scripts() {
    log_info "Creating startup scripts..."
    
    # Create start script
    cat > start_bot.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Crypto Trading Bot..."

# Activate virtual environment
source venv/bin/activate

# Start the trading bot
python3 -m crypto_trading_bot.main
EOF

    # Create API server start script
    cat > start_api.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting API Server..."

# Activate virtual environment
source venv/bin/activate

# Install FastAPI if not installed
pip install fastapi uvicorn

# Start the API server
python3 api_server.py
EOF

    # Create dashboard start script
    cat > start_dashboard.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Dashboard..."

# Navigate to dashboard directory
cd dashboard

# Install dependencies if needed
npm install

# Start the dashboard
npm start
EOF

    # Make scripts executable
    chmod +x start_bot.sh start_api.sh start_dashboard.sh
    
    log_success "Startup scripts created"
}

# Display final instructions
show_instructions() {
    log_success "🎉 Setup completed successfully!"
    echo
    echo "📋 Next Steps:"
    echo "1. Configure your API keys in .env file:"
    echo "   nano .env"
    echo
    echo "2. Test the installation:"
    echo "   source venv/bin/activate"
    echo "   python3 test_basic.py"
    echo
    echo "3. Start the API server:"
    echo "   ./start_api.sh"
    echo "   # API will be available at http://localhost:8000"
    echo
    echo "4. Start the dashboard (in a new terminal):"
    echo "   ./start_dashboard.sh"
    echo "   # Dashboard will be available at http://localhost:3000"
    echo
    echo "5. Start the trading bot (in a new terminal):"
    echo "   ./start_bot.sh"
    echo
    echo "📚 Documentation:"
    echo "   - API docs: http://localhost:8000/docs"
    echo "   - Setup guide: UBUNTU_SETUP_GUIDE.md"
    echo
    echo "⚠️  Important:"
    echo "   - Always test with TESTNET=true first"
    echo "   - Set DRY_RUN=true for paper trading"
    echo "   - Configure your Binance API keys before live trading"
    echo
}

# Main execution
main() {
    echo "🐧 Crypto Trading Bot - Ubuntu Setup"
    echo "===================================="
    echo
    
    check_ubuntu
    update_system
    install_system_deps
    install_python
    install_nodejs
    setup_project
    install_python_deps
    install_dashboard_deps
    create_config
    create_service
    create_scripts
    run_tests
    show_instructions
}

# Run main function
main "$@"