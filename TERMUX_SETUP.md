# 📱 Termux Setup Guide - Run Trading Bot on Mobile

Run the crypto trading bot directly on your Android phone using Termux!

## 📋 Prerequisites

- **Android phone** (Android 7.0+)
- **Termux app** installed from F-Droid (recommended) or Google Play
- **Stable internet connection**
- **Binance API keys**

## ⚡ Quick Setup (10 Minutes)

### Step 1: Install Termux
Download Termux from:
- **F-Droid** (recommended): https://f-droid.org/packages/com.termux/
- **Google Play**: https://play.google.com/store/apps/details?id=com.termux

### Step 2: Update Termux Packages
```bash
# Update package lists
pkg update && pkg upgrade -y

# Install essential packages
pkg install -y python git curl wget
```

### Step 3: Install Python Dependencies
```bash
# Install Python build tools
pkg install -y python-pip build-essential libffi openssl

# Upgrade pip
pip install --upgrade pip
```

### Step 4: Clone Repository
```bash
# Clone the trading bot repository
git clone https://github.com/Hirak21/crypto-trading-bot-dashboard.git

# Navigate to project directory
cd crypto-trading-bot-dashboard
```

### Step 5: Install Bot Dependencies
```bash
# Install Python packages (this may take 5-10 minutes)
pip install -r requirements.txt
```

### Step 6: Setup Credentials
```bash
# Run automated setup
python setup_bot.py
```

Enter your Binance API keys when prompted.

### Step 7: Run the Bot
```bash
# Start adaptive scalping scanner
python scalping_scanner.py
```

## 🔧 Termux-Specific Optimizations

### Enable Wake Lock (Prevent Sleep)
```bash
# Install termux-api for wake lock
pkg install termux-api

# Keep screen awake while bot runs
termux-wake-lock
```

### Background Execution
```bash
# Run bot in background
nohup python scalping_scanner.py > bot.log 2>&1 &

# Check if bot is running
ps aux | grep python

# View logs
tail -f bot.log
```

### Storage Access (Optional)
```bash
# Allow Termux to access phone storage
termux-setup-storage

# This creates ~/storage/ directory with access to:
# - Downloads
# - Pictures
# - Music
# - etc.
```

## 📊 Mobile-Optimized Commands

### Quick Status Check
```bash
# Check bot status
ps aux | grep scalping_scanner

# View recent logs
tail -20 bot.log

# Check results
ls -la *.json
```

### Resource Monitoring
```bash
# Check memory usage
free -h

# Check CPU usage
top

# Check disk space
df -h
```

### Network Status
```bash
# Check internet connection
ping -c 4 google.com

# Check Binance API connectivity
curl -s https://api.binance.com/api/v3/ping
```

## 🛡️ Mobile-Specific Safety Tips

### Battery Optimization
```bash
# Check battery status
termux-battery-status

# Reduce scanning frequency for battery saving
# Edit scalping_scanner.py and increase sleep times
```

### Data Usage Management
```bash
# Monitor data usage (approximate)
# The bot uses ~1-5MB per hour depending on activity

# For limited data plans, consider:
# - Running only during WiFi
# - Reducing scan frequency
# - Using strategy scanner instead of continuous scalping
```

### Notification Setup
```bash
# Install notification support
pkg install termux-api

# Add to your bot script for notifications:
# termux-notification --title "Trading Bot" --content "Signal found!"
```

## 🚀 Advanced Mobile Setup

### Auto-Start on Boot
Create a startup script:
```bash
# Create startup script
cat > ~/start_bot.sh << 'EOF'
#!/bin/bash
cd ~/crypto-trading-bot-dashboard
python scalping_scanner.py > bot.log 2>&1 &
EOF

# Make executable
chmod +x ~/start_bot.sh

# Add to .bashrc for auto-start
echo "~/start_bot.sh" >> ~/.bashrc
```

### Multiple Bot Instances
```bash
# Run scalping scanner
python scalping_scanner.py > scalping.log 2>&1 &

# Run strategy scanner
python strategy_scanner.py > strategy.log 2>&1 &

# Run executor
python trading_executor.py > executor.log 2>&1 &
```

### Remote Access (SSH)
```bash
# Install SSH server
pkg install openssh

# Start SSH server
sshd

# Find your phone's IP
ip addr show

# Now you can SSH from computer:
# ssh -p 8022 u0_a123@192.168.1.100
```

## 📱 Mobile UI Tips

### Better Terminal Experience
```bash
# Install better terminal
pkg install zsh
chsh -s zsh

# Install oh-my-zsh (optional)
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

### File Management
```bash
# Install file manager
pkg install mc

# Run midnight commander
mc
```

### Text Editing
```bash
# Install nano editor
pkg install nano

# Edit configuration
nano config/bot_config.json
```

## 🔧 Troubleshooting

### Common Termux Issues

1. **Package installation fails**:
```bash
pkg update
pkg upgrade
pkg install python git -y
```

2. **Python modules won't install**:
```bash
pkg install build-essential libffi openssl
pip install --upgrade pip setuptools wheel
```

3. **Bot stops when screen locks**:
```bash
# Install wake lock
pkg install termux-api
termux-wake-lock

# Or run in background
nohup python scalping_scanner.py &
```

4. **Out of memory**:
```bash
# Check memory
free -h

# Reduce batch size in scanner
# Edit scalping_scanner.py: self.batch_size = 50
```

5. **Network timeouts**:
```bash
# Check connection
ping api.binance.com

# Increase timeout in code if needed
```

## 📊 Performance on Mobile

### Expected Performance:
- **RAM Usage**: 50-150MB
- **CPU Usage**: 5-15%
- **Battery**: 3-8% per hour
- **Data**: 1-5MB per hour
- **Scan Speed**: 3-5 minutes per full scan

### Optimization Tips:
- Use WiFi when possible
- Enable battery optimization exceptions for Termux
- Close other apps while bot runs
- Use power saving mode if available

## 🎯 Mobile Trading Workflow

```bash
# 1. Start bot
python scalping_scanner.py

# 2. Monitor in another session
tail -f scalping_results.json

# 3. Execute trades (if signals found)
python trading_executor.py

# 4. Check results
cat trade_executions.json
```

## 🆘 Emergency Commands

```bash
# Stop all bots
pkill -f python

# Check what's running
ps aux | grep python

# Clear logs
> bot.log

# Restart Termux
exit
# (reopen Termux app)
```

---

**🎉 You're ready to trade from your phone!**

The bot will run continuously on your mobile device, scanning for opportunities and executing trades automatically. Perfect for monitoring markets on the go! 📱💰