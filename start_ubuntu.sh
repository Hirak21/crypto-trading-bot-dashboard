#!/bin/bash

# Crypto Trading Bot - Ubuntu Startup Script
# This script sets up and starts the trading bot with dashboard on Ubuntu

set -e  # Exit on any error

echo "🚀 Starting Crypto Trading Bot on Ubuntu"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if running on Ubuntu
if ! command -v lsb_release &> /dev/null || [[ $(lsb_release -si) != "Ubuntu" ]]; then
    print_warning "This script is designed for Ubuntu. Proceeding anyway..."
fi

# Check if Python 3.11+ is installed
if ! command -v python3.11 &> /dev/null; then
    print_error "Python 3.11 is required but not installed."
    print_info "Please install Python 3.11 first:"
    echo "sudo apt update"
    echo "sudo apt install python3.11 python3.11-pip python3.11-venv -y"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is required but not installed."
    print_info "Please install Node.js first:"
    echo "curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -"
    echo "sudo apt-get install -y nodejs"
    exit 1
fi

print_status "System requirements check passed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating Python virtual environment..."
    python3.11 -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
print_info "Installing Python dependencies..."
pip install --upgrade pip

# Install core dependencies
pip install cryptography pydantic python-dotenv pyyaml aiohttp websockets

# Install FastAPI for API server
pip install fastapi uvicorn

# Install optional dependencies (with error handling)
print_info "Installing optional dependencies..."
pip install pandas numpy scipy || print_warning "Some optional dependencies failed to install"

print_status "Python dependencies installed"

# Install Node.js dependencies for dashboard
if [ -d "dashboard" ]; then
    print_info "Installing dashboard dependencies..."
    cd dashboard
    
    # Install dependencies
    npm install
    
    # Build the dashboard
    print_info "Building dashboard..."
    npm run build
    
    cd ..
    print_status "Dashboard built successfully"
else
    print_warning "Dashboard directory not found, skipping dashboard setup"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env configuration file..."
    cat > .env << EOF
# API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
TESTNET=true

# Bot Configuration
TRADING_ENABLED=false
DRY_RUN=true
LOG_LEVEL=INFO

# Database (optional)
DATABASE_URL=sqlite:///trading_bot.db

# Dashboard Configuration
DASHBOARD_PORT=3000
API_PORT=8000

# Security
SECRET_KEY=your_secret_key_here
EOF
    print_status ".env file created"
    print_warning "Please edit .env file with your actual API credentials"
else
    print_info ".env file already exists"
fi

# Create logs directory
mkdir -p logs
print_status "Logs directory created"

# Create data directory
mkdir -p data
print_status "Data directory created"

# Make scripts executable
chmod +x *.sh
chmod +x *.py

# Function to start the bot
start_bot() {
    print_info "Starting trading bot API server..."
    python api_server.py &
    API_PID=$!
    echo $API_PID > api_server.pid
    print_status "API server started (PID: $API_PID)"
}

# Function to start the dashboard
start_dashboard() {
    if [ -d "dashboard" ]; then
        print_info "Starting dashboard server..."
        cd dashboard
        npm start &
        DASHBOARD_PID=$!
        echo $DASHBOARD_PID > ../dashboard.pid
        cd ..
        print_status "Dashboard started (PID: $DASHBOARD_PID)"
    fi
}

# Function to stop services
stop_services() {
    print_info "Stopping services..."
    
    if [ -f "api_server.pid" ]; then
        kill $(cat api_server.pid) 2>/dev/null || true
        rm -f api_server.pid
    fi
    
    if [ -f "dashboard.pid" ]; then
        kill $(cat dashboard.pid) 2>/dev/null || true
        rm -f dashboard.pid
    fi
    
    print_status "Services stopped"
}

# Handle script termination
trap stop_services EXIT

# Parse command line arguments
case "${1:-start}" in
    "start")
        print_info "Starting all services..."
        start_bot
        start_dashboard
        
        print_status "All services started successfully!"
        echo ""
        print_info "Access points:"
        echo "  📊 Dashboard: http://localhost:3000"
        echo "  🔌 API: http://localhost:8000"
        echo "  📚 API Docs: http://localhost:8000/docs"
        echo ""
        print_info "Press Ctrl+C to stop all services"
        
        # Wait for services
        wait
        ;;
    
    "stop")
        stop_services
        ;;
    
    "restart")
        stop_services
        sleep 2
        start_bot
        start_dashboard
        print_status "Services restarted"
        ;;
    
    "test")
        print_info "Running tests..."
        python test_basic.py
        if [ $? -eq 0 ]; then
            print_status "Basic tests passed"
        else
            print_error "Tests failed"
            exit 1
        fi
        ;;
    
    "install")
        print_status "Installation completed successfully!"
        print_info "Next steps:"
        echo "1. Edit .env file with your API credentials"
        echo "2. Run: ./start_ubuntu.sh start"
        ;;
    
    "status")
        print_info "Checking service status..."
        
        if [ -f "api_server.pid" ] && kill -0 $(cat api_server.pid) 2>/dev/null; then
            print_status "API server is running (PID: $(cat api_server.pid))"
        else
            print_warning "API server is not running"
        fi
        
        if [ -f "dashboard.pid" ] && kill -0 $(cat dashboard.pid) 2>/dev/null; then
            print_status "Dashboard is running (PID: $(cat dashboard.pid))"
        else
            print_warning "Dashboard is not running"
        fi
        ;;
    
    "logs")
        print_info "Showing recent logs..."
        if [ -f "logs/trading_bot.log" ]; then
            tail -f logs/trading_bot.log
        else
            print_warning "No log file found"
        fi
        ;;
    
    "help"|*)
        echo "Usage: $0 {start|stop|restart|test|install|status|logs|help}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the trading bot and dashboard"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  test     - Run basic tests"
        echo "  install  - Complete installation setup"
        echo "  status   - Check service status"
        echo "  logs     - Show recent logs"
        echo "  help     - Show this help message"
        ;;
esac