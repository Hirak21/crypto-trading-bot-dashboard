# 🚀 Crypto Trading Bot - Complete Setup Guide

## 🎯 Overview

This is a comprehensive crypto trading bot with a React dashboard for monitoring and control. The system includes:

- **Trading Bot**: Python-based algorithmic trading system
- **React Dashboard**: Modern web interface for monitoring and control
- **REST API**: FastAPI server connecting the dashboard to the bot
- **Comprehensive Testing**: Full test suite with 3,497+ lines of test code

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React         │    │   FastAPI       │    │   Trading Bot   │
│   Dashboard     │◄──►│   API Server    │◄──►│   Core System   │
│   (Port 3000)   │    │   (Port 8000)   │    │   (Background)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🐧 Ubuntu Linux Setup (Automated)

### Quick Setup (Recommended)

```bash
# Make setup script executable
chmod +x setup_ubuntu.sh

# Run automated setup
./setup_ubuntu.sh
```

This script will:
- ✅ Update system packages
- ✅ Install Python 3.11+, Node.js, and dependencies
- ✅ Create virtual environment
- ✅ Install all Python and Node.js packages
- ✅ Create configuration files
- ✅ Set up systemd service
- ✅ Create startup scripts
- ✅ Run basic tests

### Manual Setup (Step by Step)

If you prefer manual setup, follow the detailed guide in `UBUNTU_SETUP_GUIDE.md`.

## 🚀 Quick Start

### 1. Configure API Keys

```bash
# Edit environment file
nano .env

# Add your Binance API credentials:
BINANCE_API_KEY=your_actual_api_key
BINANCE_API_SECRET=your_actual_api_secret
TESTNET=true  # Keep true for testing
```

### 2. Start the System

**Terminal 1 - API Server:**
```bash
./start_api.sh
# API available at: http://localhost:8000
# Documentation at: http://localhost:8000/docs
```

**Terminal 2 - Dashboard:**
```bash
./start_dashboard.sh
# Dashboard available at: http://localhost:3000
```

**Terminal 3 - Trading Bot (Optional):**
```bash
./start_bot.sh
# Bot runs in background
```

### 3. Access the Dashboard

Open your browser and navigate to: **http://localhost:3000**

## 📊 Dashboard Features

### 🏠 Dashboard Overview
- Real-time portfolio value and P&L
- Active positions summary
- Recent trades and performance metrics
- System health indicators

### 💼 Portfolio Management
- Position tracking with real-time P&L
- Asset allocation visualization
- Performance analytics and charts
- Risk metrics and drawdown analysis

### 🎯 Strategy Management
- Enable/disable trading strategies
- Strategy performance comparison
- Parameter adjustment interface
- Backtesting results

### 📈 Trading Control
- Start/stop trading with one click
- Paper trading vs live trading modes
- Emergency stop functionality
- Real-time market signals

### ⚙️ Settings & Configuration
- API key management
- Risk management parameters
- Notification preferences
- System configuration

### 📋 System Logs
- Real-time log monitoring
- Filter by component and log level
- Download logs functionality
- Error tracking and alerts

## 🧪 Testing

### Run All Tests
```bash
# Activate virtual environment
source venv/bin/activate

# Run comprehensive test suite
python3 run_tests.py

# Run specific component tests
python3 run_tests.py --component strategies
python3 run_tests.py --component technical
python3 run_tests.py --component managers

# Run with coverage analysis
python3 run_tests.py --coverage
```

### Test Coverage
- **3,497+ lines** of test code
- **Unit tests** for all components
- **Integration tests** for workflows
- **End-to-end tests** for complete scenarios
- **Mock data generators** for realistic testing

## 🔧 Configuration

### Bot Configuration (`config/bot_config.yaml`)
```yaml
symbols:
  - BTCUSDT
  - ETHUSDT
  - ADAUSDT

strategies:
  liquidity:
    enabled: true
    weight: 1.0
  momentum:
    enabled: true
    weight: 1.2

risk_management:
  max_position_size: 0.02  # 2% per trade
  daily_loss_limit: 0.05   # 5% daily limit
  max_drawdown: 0.15       # 15% max drawdown
```

### Environment Variables (`.env`)
```bash
# API Configuration
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
TESTNET=true

# Trading Configuration
TRADING_ENABLED=false
DRY_RUN=true
LOG_LEVEL=INFO
```

## 🔐 Security Best Practices

### API Key Security
```bash
# Set proper file permissions
chmod 600 .env
chmod 600 config/bot_config.yaml

# Use environment variables
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

### Firewall Configuration
```bash
# Allow dashboard access (optional)
sudo ufw allow 3000/tcp

# Allow API access (only if needed externally)
sudo ufw allow 8000/tcp
```

## 📈 Monitoring & Maintenance

### System Service Management
```bash
# Enable auto-start
sudo systemctl enable crypto-trading-bot

# Start/stop service
sudo systemctl start crypto-trading-bot
sudo systemctl stop crypto-trading-bot

# Check status
sudo systemctl status crypto-trading-bot
```

### Log Monitoring
```bash
# View real-time logs
tail -f logs/bot.log

# Check API server logs
tail -f logs/api.log

# Monitor system resources
htop
```

### Health Checks
```bash
# Check API health
curl http://localhost:8000/health

# Check bot status
curl http://localhost:8000/status

# View portfolio
curl http://localhost:8000/portfolio
```

## 🛠️ Development

### Project Structure
```
crypto-trading-bot/
├── crypto_trading_bot/          # Core trading bot
│   ├── strategies/              # Trading strategies
│   ├── managers/                # Component managers
│   ├── models/                  # Data models
│   ├── utils/                   # Utilities
│   └── api/                     # API clients
├── dashboard/                   # React dashboard
│   ├── src/                     # Source code
│   ├── public/                  # Static files
│   └── package.json             # Dependencies
├── tests/                       # Test suite
│   ├── test_strategies.py       # Strategy tests
│   ├── test_technical_analysis.py # TA tests
│   └── test_managers.py         # Manager tests
├── config/                      # Configuration files
├── logs/                        # Log files
└── api_server.py               # FastAPI server
```

### Adding New Features

1. **New Strategy:**
   ```bash
   # Create strategy file
   touch crypto_trading_bot/strategies/my_strategy.py
   
   # Add tests
   # Update configuration
   # Test thoroughly
   ```

2. **Dashboard Component:**
   ```bash
   # Create React component
   touch dashboard/src/components/MyComponent.tsx
   
   # Add to routing
   # Style with Tailwind CSS
   # Connect to API
   ```

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use:**
   ```bash
   # Kill process using port
   sudo lsof -t -i:8000 | xargs kill -9
   sudo lsof -t -i:3000 | xargs kill -9
   ```

2. **Python Module Not Found:**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   
   # Add to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **API Connection Issues:**
   ```bash
   # Check if API server is running
   curl http://localhost:8000/health
   
   # Restart API server
   pkill -f api_server.py
   python3 api_server.py
   ```

4. **Dashboard Build Issues:**
   ```bash
   # Clear npm cache
   cd dashboard
   npm cache clean --force
   rm -rf node_modules package-lock.json
   npm install
   ```

### Getting Help

- **Check logs:** `tail -f logs/bot.log`
- **Test components:** `python3 run_tests.py`
- **API documentation:** `http://localhost:8000/docs`
- **Dashboard:** `http://localhost:3000`

## 📚 Additional Resources

- **Ubuntu Setup Guide:** `UBUNTU_SETUP_GUIDE.md`
- **API Documentation:** Available at `http://localhost:8000/docs`
- **Test Results:** Run `python3 test_summary.py`

## ⚠️ Important Warnings

- **Always start with `TESTNET=true`**
- **Use `DRY_RUN=true` for paper trading**
- **Test thoroughly before live trading**
- **Never share your API keys**
- **Monitor your positions regularly**
- **Set appropriate risk limits**

## 🎉 Success Indicators

✅ **API Server:** `http://localhost:8000/health` returns "healthy"  
✅ **Dashboard:** `http://localhost:3000` loads successfully  
✅ **Tests:** `python3 test_basic.py` passes all tests  
✅ **Configuration:** API keys configured and testnet enabled  
✅ **Logs:** System logs show no critical errors  

---

**🚀 You're now ready to start algorithmic crypto trading with a professional dashboard!**

Remember to start with paper trading and gradually move to live trading as you gain confidence in the system.