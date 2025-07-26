# 🐧 Crypto Trading Bot - Ubuntu Linux Setup Guide

This guide will walk you through setting up and running the crypto trading bot on Ubuntu Linux.

## 📋 Prerequisites

- Ubuntu 20.04 LTS or newer
- Internet connection
- Terminal access
- At least 4GB RAM and 10GB free disk space

## 🚀 Step 1: System Update and Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential curl wget git vim

# Install Python 3.11+ and pip
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Install Node.js and npm for the dashboard
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installations
python3 --version
pip3 --version
node --version
npm --version
```

## 📁 Step 2: Project Setup

```bash
# Clone or navigate to your project directory
cd /path/to/your/crypto-trading-bot

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## 📦 Step 3: Install Python Dependencies

```bash
# Install core dependencies
pip install cryptography pydantic python-dotenv pyyaml aiohttp websockets

# Install data analysis libraries
pip install pandas numpy scipy

# Install optional dependencies for full functionality
pip install python-binance structlog

# Install testing dependencies
pip install pytest pytest-asyncio pytest-mock

# Install development tools
pip install black flake8 mypy

# Verify installation
python3 -c "import crypto_trading_bot; print('✅ Core modules imported successfully')"
```

## ⚙️ Step 4: Configuration Setup

```bash
# Create configuration directory
mkdir -p config

# Create environment file
cat > .env << EOF
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

# Create basic config file
cat > config/bot_config.yaml << EOF
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

# Create logs directory
mkdir -p logs
```

## 🧪 Step 5: Run Tests

```bash
# Run basic functionality test
python3 test_basic.py

# Run comprehensive test suite
python3 run_tests.py

# Run specific component tests
python3 run_tests.py --component strategies
python3 run_tests.py --component technical
python3 run_tests.py --component managers

# Run with coverage (optional)
pip install coverage
python3 run_tests.py --coverage
```

## 🚀 Step 6: Start the Trading Bot

```bash
# Start in dry-run mode (recommended for first run)
python3 -m crypto_trading_bot.main

# Or run with specific configuration
python3 -m crypto_trading_bot.main --config config/bot_config.yaml

# Run in background with logging
nohup python3 -m crypto_trading_bot.main > logs/bot.log 2>&1 &

# Check if running
ps aux | grep python3
```

## 📊 Step 7: Setup React Dashboard

```bash
# Navigate to dashboard directory
cd dashboard

# Install Node.js dependencies
npm install

# Install additional dependencies if needed
npm install @types/react @types/react-dom typescript

# Start development server
npm start

# The dashboard will be available at http://localhost:3000
```

## 🔧 Step 8: API Server Setup

```bash
# Install FastAPI for the API server
pip install fastapi uvicorn

# Start the API server (in a new terminal)
python3 api_server.py

# The API will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

## 📱 Step 9: Access the Dashboard

1. **Open your web browser**
2. **Navigate to**: `http://localhost:3000`
3. **API Backend**: `http://localhost:8000`

The dashboard provides:
- Real-time trading status
- Portfolio overview
- Strategy performance
- Risk metrics
- Trade history
- System health monitoring

## 🔍 Step 10: Monitoring and Logs

```bash
# View real-time logs
tail -f logs/bot.log

# Check system resources
htop

# Monitor network connections
netstat -tulpn | grep python3

# View trading bot status
curl http://localhost:8000/status

# Check portfolio
curl http://localhost:8000/portfolio
```

## 🛠️ Troubleshooting

### Common Issues:

1. **Permission Denied**:
   ```bash
   chmod +x run_tests.py
   chmod +x api_server.py
   ```

2. **Port Already in Use**:
   ```bash
   # Kill process using port 8000
   sudo lsof -t -i:8000 | xargs kill -9
   
   # Kill process using port 3000
   sudo lsof -t -i:3000 | xargs kill -9
   ```

3. **Python Module Not Found**:
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   
   # Add project to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

4. **API Connection Issues**:
   ```bash
   # Check if API server is running
   curl http://localhost:8000/health
   
   # Restart API server
   pkill -f api_server.py
   python3 api_server.py
   ```

## 🔐 Security Considerations

```bash
# Set proper file permissions
chmod 600 .env
chmod 600 config/bot_config.yaml

# Create firewall rules (optional)
sudo ufw allow 3000/tcp  # Dashboard
sudo ufw allow 8000/tcp  # API (only if needed externally)

# Use environment variables for sensitive data
export BINANCE_API_KEY="your_actual_api_key"
export BINANCE_API_SECRET="your_actual_api_secret"
```

## 📈 Performance Optimization

```bash
# Install performance monitoring tools
pip install psutil memory_profiler

# Monitor resource usage
python3 -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\"/\").percent}%')
"

# Optimize Python performance
export PYTHONOPTIMIZE=1
```

## 🔄 Auto-Start Setup (Optional)

```bash
# Create systemd service
sudo tee /etc/systemd/system/crypto-trading-bot.service > /dev/null << EOF
[Unit]
Description=Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python -m crypto_trading_bot.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable crypto-trading-bot
sudo systemctl start crypto-trading-bot

# Check status
sudo systemctl status crypto-trading-bot
```

## 📚 Next Steps

1. **Configure API Keys**: Add your real Binance API credentials
2. **Customize Strategies**: Modify strategy parameters in config
3. **Set Risk Limits**: Adjust risk management settings
4. **Monitor Performance**: Use the dashboard to track results
5. **Scale Up**: Consider running on a VPS or cloud instance

## 🆘 Getting Help

- **Check logs**: `tail -f logs/bot.log`
- **Test components**: `python3 run_tests.py`
- **API documentation**: `http://localhost:8000/docs`
- **Dashboard**: `http://localhost:3000`

---

**⚠️ Important**: Always start with `DRY_RUN=true` and `TESTNET=true` for testing!