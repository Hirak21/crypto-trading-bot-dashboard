# Crypto Trading Bot

A comprehensive cryptocurrency trading bot designed for Binance Futures markets with advanced market analysis capabilities and multiple trading strategies.

## Features

- **Multiple Trading Strategies**:
  - Liquidity analysis for optimal entry/exit points
  - Momentum-based trading across multiple timeframes
  - Chart pattern recognition (triangles, head & shoulders, flags, wedges)
  - Candlestick pattern analysis as fallback strategy

- **Advanced Risk Management**:
  - Automatic stop-loss and take-profit orders
  - Position sizing based on risk parameters
  - Daily loss limits and maximum drawdown protection
  - Emergency stop functionality

- **Real-time Market Analysis**:
  - WebSocket integration for live market data
  - Technical indicator calculations (RSI, MACD, ROC, ADX)
  - Order book depth analysis
  - Multi-timeframe analysis

- **Performance Tracking**:
  - Detailed trade logging and performance metrics
  - Strategy-specific performance breakdown
  - Risk-adjusted return calculations
  - Backtesting capabilities

- **Security & Configuration**:
  - Encrypted API credential storage
  - Configurable parameters for all strategies
  - Structured logging with rotation
  - Comprehensive error handling

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd crypto-trading-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Configuration

1. Set up your Binance API credentials:
   - Create API keys in your Binance account
   - Enable Futures trading permissions
   - Configure IP restrictions for security

2. Run the bot to create initial configuration:
```bash
python -m crypto_trading_bot.main
```

3. Update the configuration files in the `config/` directory:
   - `bot_config.json`: Main bot configuration
   - `credentials.enc`: Encrypted API credentials (created automatically)

## Usage

### Running the Bot

```bash
# Run with default configuration
python -m crypto_trading_bot.main

# Or use the console script
crypto-trading-bot
```

### Configuration Options

The bot can be configured through the `config/bot_config.json` file:

```json
{
  "testnet": true,
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "strategies": {
    "liquidity": true,
    "momentum": true,
    "chart_patterns": true,
    "candlestick_patterns": true
  },
  "risk_config": {
    "max_position_size": 0.02,
    "daily_loss_limit": 0.05,
    "max_drawdown": 0.15,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04
  }
}
```

## Project Structure

```
crypto_trading_bot/
├── __init__.py
├── main.py                 # Main application entry point
├── interfaces.py           # Core interfaces and contracts
├── models/                 # Data models
├── strategies/             # Trading strategies
│   ├── base_strategy.py    # Base strategy class
│   ├── liquidity.py        # Liquidity analysis strategy
│   ├── momentum.py         # Momentum trading strategy
│   ├── chart_patterns.py   # Chart pattern recognition
│   └── candlestick.py      # Candlestick pattern analysis
├── managers/               # Core system managers
│   ├── market_manager.py   # Market data management
│   ├── strategy_manager.py # Strategy coordination
│   ├── risk_manager.py     # Risk management
│   ├── trade_manager.py    # Trade execution
│   └── portfolio_manager.py # Portfolio tracking
└── utils/                  # Utility modules
    ├── config.py           # Configuration management
    ├── logging_config.py   # Logging setup
    └── technical_analysis.py # Technical indicators
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black crypto_trading_bot/
flake8 crypto_trading_bot/
```

### Type Checking

```bash
mypy crypto_trading_bot/
```

## Risk Disclaimer

This trading bot is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Never trade with money you cannot afford to lose. Always test strategies thoroughly on testnet before using real funds.

## License

MIT License - see LICENSE file for details.