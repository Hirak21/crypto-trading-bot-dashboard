#!/bin/bash

# Crypto Trading Bot - Ubuntu Installation Script
# This script installs all dependencies and sets up the environment

set -e  # Exit on any error

echo "🐧 Crypto Trading Bot - Ubuntu Installation"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Update system packages
print_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_status "System updated"

# Install Python 3.11
print_info "Installing Python 3.11..."
sudo apt install python3.11 python3.11-pip python3.11-venv python3.11-dev -y

# Install build essentials (needed for some Python packages)
sudo apt install build-essential -y

# Install Node.js 18
print_info "Installing Node.js 18..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install additional system dependencies
print_info "Installing system dependencies..."
sudo apt install -y \
    git \
    curl \
    wget \
    htop \
    nginx \
    postgresql \
    postgresql-contrib \
    redis-server \
    supervisor \
    ufw

# Install Python dependencies globally (for system tools)
print_info "Installing global Python tools..."
sudo python3.11 -m pip install --upgrade pip
sudo python3.11 -m pip install virtualenv

# Install PM2 for process management
print_info "Installing PM2 process manager..."
sudo npm install -g pm2

# Create project directory structure
print_info "Setting up project structure..."
mkdir -p logs data config backups

# Set up firewall
print_info "Configuring firewall..."
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 3000  # Dashboard
sudo ufw allow 8000  # API
print_status "Firewall configured"

# Create systemd service files
print_info "Creating systemd service files..."

# Trading bot API service
sudo tee /etc/systemd/system/trading-bot-api.service > /dev/null <<EOF
[Unit]
Description=Crypto Trading Bot API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Dashboard service
sudo tee /etc/systemd/system/trading-bot-dashboard.service > /dev/null <<EOF
[Unit]
Description=Crypto Trading Bot Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)/dashboard
ExecStart=/usr/bin/npm start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload
print_status "Systemd services created"

# Create nginx configuration
print_info "Setting up Nginx reverse proxy..."
sudo tee /etc/nginx/sites-available/trading-bot > /dev/null <<EOF
server {
    listen 80;
    server_name localhost;

    # Dashboard
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }

    # API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # WebSocket
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable nginx site
sudo ln -sf /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
print_status "Nginx configured"

# Create log rotation configuration
print_info "Setting up log rotation..."
sudo tee /etc/logrotate.d/trading-bot > /dev/null <<EOF
$(pwd)/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload trading-bot-api || true
    endscript
}
EOF
print_status "Log rotation configured"

# Create backup script
print_info "Creating backup script..."
tee backup.sh > /dev/null <<'EOF'
#!/bin/bash
# Backup script for trading bot

BACKUP_DIR="backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="trading_bot_backup_$DATE.tar.gz"

mkdir -p $BACKUP_DIR

# Create backup
tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
    --exclude='venv' \
    --exclude='node_modules' \
    --exclude='backups' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    .

echo "Backup created: $BACKUP_DIR/$BACKUP_FILE"

# Keep only last 7 backups
cd $BACKUP_DIR
ls -t trading_bot_backup_*.tar.gz | tail -n +8 | xargs -r rm
EOF

chmod +x backup.sh
print_status "Backup script created"

# Create monitoring script
print_info "Creating monitoring script..."
tee monitor.sh > /dev/null <<'EOF'
#!/bin/bash
# Monitoring script for trading bot

check_service() {
    if systemctl is-active --quiet $1; then
        echo "✅ $1 is running"
    else
        echo "❌ $1 is not running"
        return 1
    fi
}

echo "🔍 Trading Bot System Status"
echo "============================"

# Check services
check_service trading-bot-api
check_service trading-bot-dashboard
check_service nginx
check_service postgresql
check_service redis-server

# Check disk space
echo ""
echo "💾 Disk Usage:"
df -h / | tail -1

# Check memory
echo ""
echo "🧠 Memory Usage:"
free -h

# Check recent logs
echo ""
echo "📋 Recent Logs (last 5 lines):"
if [ -f "logs/trading_bot.log" ]; then
    tail -5 logs/trading_bot.log
else
    echo "No log file found"
fi
EOF

chmod +x monitor.sh
print_status "Monitoring script created"

# Create update script
print_info "Creating update script..."
tee update.sh > /dev/null <<'EOF'
#!/bin/bash
# Update script for trading bot

echo "🔄 Updating Trading Bot..."

# Stop services
sudo systemctl stop trading-bot-api trading-bot-dashboard

# Backup current version
./backup.sh

# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python dependencies
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# Update Node.js dependencies
cd dashboard
npm update
npm run build
cd ..

# Restart services
sudo systemctl start trading-bot-api trading-bot-dashboard

echo "✅ Update completed"
EOF

chmod +x update.sh
print_status "Update script created"

# Set up cron jobs
print_info "Setting up cron jobs..."
(crontab -l 2>/dev/null; echo "0 2 * * * $(pwd)/backup.sh") | crontab -
(crontab -l 2>/dev/null; echo "*/5 * * * * $(pwd)/monitor.sh >> logs/monitor.log 2>&1") | crontab -
print_status "Cron jobs configured"

# Create requirements.txt
print_info "Creating requirements.txt..."
tee requirements.txt > /dev/null <<EOF
# Core dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0
aiohttp>=3.9.0
pydantic>=2.5.0
python-dotenv>=1.0.0
pyyaml>=6.0.1
cryptography>=41.0.0

# Optional dependencies
pandas>=2.1.0
numpy>=1.24.0
scipy>=1.11.0

# Development dependencies
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
flake8>=6.0.0
EOF
print_status "Requirements file created"

# Final setup message
echo ""
print_status "Installation completed successfully!"
echo ""
print_info "Next steps:"
echo "1. Run: ./start_ubuntu.sh install  # Complete project setup"
echo "2. Edit .env file with your API credentials"
echo "3. Run: ./start_ubuntu.sh start    # Start the services"
echo ""
print_info "Useful commands:"
echo "• ./start_ubuntu.sh status   # Check service status"
echo "• ./monitor.sh              # System monitoring"
echo "• ./backup.sh               # Create backup"
echo "• ./update.sh               # Update system"
echo ""
print_info "Service management:"
echo "• sudo systemctl start trading-bot-api"
echo "• sudo systemctl stop trading-bot-api"
echo "• sudo systemctl status trading-bot-api"
echo ""
print_warning "Remember to configure your firewall and API keys before starting!"