# 🚀 Crypto Trading Bot - Complete System

An advanced cryptocurrency trading bot with adaptive scanning, multiple strategies, and automated execution. Features both Python trading engine and React dashboard for monitoring.

## ⚡ Quick Start (5 Minutes)

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/crypto-trading-bot-dashboard.git
cd crypto-trading-bot-dashboard

# 2. Automated setup
python setup_bot.py

# 3. Start trading
python scalping_scanner.py
```

**That's it!** The bot will start scanning for trading opportunities.

## 🎯 Key Features

### 🔍 **Adaptive Scanning System**
- **15-minute timeframe** → Quick scalping opportunities
- **30-minute timeframe** → If no 15m signals found  
- **5-minute continuous** → 24/7 scanning until signals found
- **1000+ symbols** scanned in batches

### 🤖 **Trading Strategies**
- **Scalping Scanner**: RSI, EMA crossover, volume breakout, momentum
- **Strategy Scanner**: Liquidity, momentum, pattern, candlestick (your original bot logic)
- **Risk Management**: Position sizing, stop loss, take profit, daily limits

### 📊 **Automated Execution**
- **Signal validation** with confidence thresholds
- **Position monitoring** with automatic stop loss/take profit
- **Trade logging** to JSON files
- **Real-time P&L tracking**

### 🛡️ **Safety Features**
- **Testnet by default** - Safe testing environment
- **Position limits** - Max 5 concurrent, 2% position size
- **Daily limits** - Max 10 trades, 5% loss limit
- **Quality filters** - Only high-confidence signals

## 📋 System Components

### Core Trading Engine (Python)
- `scalping_scanner.py` - Adaptive timeframe scanning
- `strategy_scanner.py` - Original bot strategies  
- `trading_executor.py` - Trade execution and monitoring

### Dashboard (React) - Optional
- Real-time monitoring interface
- Portfolio tracking
- Strategy performance analysis

## 🚀 Usage Examples

### Scalping Mode (Recommended)
```bash
python scalping_scanner.py
# Finds quick scalping opportunities
# Adapts timeframes automatically
# Runs 24/7 until signals found
```

### Strategy Mode  
```bash
python strategy_scanner.py
# Uses your original 4 strategies
# Better for swing trades
# 1-hour timeframe analysis
```

### Execute Trades
```bash
python trading_executor.py
# Reads signals from scanner results
# Executes with risk management
# Monitors positions automatically
```

## 📁 Generated Files

- `scalping_results.json` - Scalping signals
- `strategy_results.json` - Strategy signals
- `trade_executions.json` - All executed trades  
- `position_closes.json` - All closed positions

## ⚙️ Configuration

Edit `config/bot_config.json`:
```json
{
  "testnet": true,
  "risk_config": {
    "max_position_size": 0.02,
    "daily_loss_limit": 0.05,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04
  }
}
```

## 🔧 Deployment Options

### Local Development
```bash
python deploy.py --type local
```

### Server/VPS
```bash
python deploy.py --type server
# Creates systemd service for 24/7 operation
```

### Docker
```bash
python deploy.py --type docker
# Creates Dockerfile and docker-compose.yml
```

### Cloud (AWS/GCP/Azure)
```bash
python deploy.py --type cloud
# Creates cloud deployment scripts
```

## Project Structure

```
crypto-trading-bot-dashboard/
├── dashboard/                 # React frontend application
│   ├── public/               # Static assets
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   │   ├── Navbar.tsx
│   │   │   └── Sidebar.tsx
│   │   ├── pages/           # Page components
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Portfolio.tsx
│   │   │   └── Strategies.tsx
│   │   ├── App.tsx          # Main app component
│   │   ├── index.tsx        # Entry point
│   │   ├── index.css        # Global styles
│   │   └── config.ts        # Configuration
│   ├── package.json
│   └── tailwind.config.js
├── api_server.py            # Python backend (optional)
├── tests/                   # Test files
└── README.md
```

## Available Scripts

In the `dashboard` directory:

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

## Configuration

The dashboard can be configured via environment variables:

```bash
# .env file in dashboard directory
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000/ws
```

## Features Overview

### Dashboard
- Bot status monitoring
- Real-time P&L tracking
- Performance metrics
- Interactive price charts

### Portfolio
- Asset allocation visualization
- Portfolio performance tracking
- Balance monitoring

### Strategies
- Strategy configuration
- Performance comparison
- Enable/disable strategies
- Parameter tuning

## Development

### Adding New Components

1. Create component in `src/components/` or `src/pages/`
2. Follow TypeScript best practices
3. Use Tailwind CSS for styling
4. Add proper prop types and interfaces

### Styling Guidelines

- Use Tailwind CSS utility classes
- Follow dark theme color scheme
- Maintain responsive design
- Use consistent spacing and typography

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review the code examples

---

Built with ❤️ for the crypto trading community