# 🚀 Quick Start Guide - Crypto Trading Bot

Get the bot running in 5 minutes!

## 📋 Prerequisites

- **Python 3.8+** installed
- **Git** installed
- **Binance account** with API keys

## ⚡ Quick Setup (Windows/Linux/Mac)

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/crypto-trading-bot-dashboard.git
cd crypto-trading-bot-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Credentials
```bash
python setup_credentials.py
```
Enter your Binance API keys when prompted.

### 4. Run the Bot

#### Option A: Adaptive Scalping Scanner (Recommended)
```bash
python scalping_scanner.py
```
- Scans 15m → 30m → continuous 5m until signals found
- Perfect for quick scalping trades

#### Option B: Strategy Scanner (Your Original Bot Logic)
```bash
python strategy_scanner.py
```
- Uses your 4 original strategies: liquidity, momentum, pattern, candlestick
- Better for swing trades

#### Option C: Execute Trades from Signals
```bash
python trading_executor.py
```
- Reads signals from scanner results
- Executes trades with risk management
- Monitors positions automatically

## 🎯 Complete Workflow

```bash
# 1. Get signals
python scalping_scanner.py

# 2. Execute trades
python trading_executor.py
# Choose option 1 for scalping signals

# 3. Monitor (automatic)
# The executor monitors positions and closes them at target/stop loss
```

## 📊 What You Get

- ✅ **Adaptive scanning** - Finds opportunities across timeframes
- ✅ **Risk management** - Automatic position sizing and stops
- ✅ **24/7 operation** - Continuous scanning when needed
- ✅ **Trade logging** - All trades saved to JSON files
- ✅ **Real-time monitoring** - Automatic position management

## 🛡️ Safety Features

- **Testnet by default** - Safe testing environment
- **Position limits** - Max 5 concurrent positions
- **Daily limits** - Max 10 trades per day, 5% loss limit
- **Quality filters** - Only high-confidence signals executed

## 📁 Generated Files

- `scalping_results.json` - Scalping signals
- `strategy_results.json` - Strategy signals
- `trade_executions.json` - All executed trades
- `position_closes.json` - All closed positions

## 🔧 Configuration

Edit `config/bot_config.json` to customize:
- Risk limits
- Position sizes
- Trading symbols
- Strategy weights

## 🆘 Troubleshooting

### Common Issues:

1. **"python not recognized"** → Use `py` instead of `python`
2. **API errors** → Check your Binance API keys and permissions
3. **No signals found** → Normal in some market conditions, try different timeframes
4. **Permission errors** → Run as administrator or check file permissions

### Get Help:
- Check the logs in console output
- Review `config/credentials.json` format
- Ensure API keys have futures trading permissions

---

**Ready to trade? Start with the scalping scanner!** 🎯

```bash
python scalping_scanner.py
```