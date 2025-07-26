# 🐧 Complete Ubuntu Deployment Guide for Crypto Trading Bot

## 🎯 **Quick Start for Ubuntu**

### **Step 1: System Preparation**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-pip python3.11-venv -y

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install build tools
sudo apt install build-essential git -y
```

### **Step 2: Project Setup**
```bash
# Clone or navigate to project
cd /path/to/crypto-trading-bot

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install fastapi uvicorn websockets aiohttp pydantic python-dotenv pyyaml cryptography

# Set up dashboard
cd dashboard
npm install
npm run build
cd ..
```

### **Step 3: Configuration**
```bash
# Create environment file
cat > .env << EOF
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
TESTNET=true
TRADING_ENABLED=false
DRY_RUN=true
LOG_LEVEL=INFO
DASHBOARD_PORT=3000
API_PORT=8000
EOF

# Create directories
mkdir -p logs data config
```

### **Step 4: Start Services**
```bash
# Terminal 1: Start API server
source venv/bin/activate
python api_server.py

# Terminal 2: Start dashboard
cd dashboard
npm start
```

### **Step 5: Access Dashboard**
- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 🚀 **React Dashboard Features**

### **Dashboard Pages**
1. **📊 Dashboard** - Overview and real-time metrics
2. **💼 Portfolio** - Position tracking and P&L
3. **🎯 Strategies** - Strategy management and configuration
4. **📈 Trading** - Manual trading and market data
5. **⚙️ Settings** - Bot configuration and risk management
6. **📋 Logs** - System logs and monitoring

### **Key Features**
- ✅ Real-time WebSocket updates
- ✅ Interactive charts and graphs
- ✅ Strategy configuration interface
- ✅ Risk management controls
- ✅ Trade execution monitoring
- ✅ System health monitoring
- ✅ Dark/light theme support
- ✅ Responsive design

---

## 🔧 **Production Deployment**

### **Using PM2 Process Manager**
```bash
# Install PM2
npm install -g pm2

# Start services with PM2
pm2 start api_server.py --name "trading-bot-api" --interpreter python
pm2 start "npm start" --name "trading-bot-dashboard" --cwd ./dashboard

# Save PM2 configuration
pm2 save
pm2 startup
```

### **Using Systemd Services**
```bash
# Create API service
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

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable trading-bot-api
sudo systemctl start trading-bot-api
```

### **Nginx Reverse Proxy**
```bash
# Install Nginx
sudo apt install nginx -y

# Create configuration
sudo tee /etc/nginx/sites-available/trading-bot > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## 🔒 **Security Configuration**

### **Firewall Setup**
```bash
# Enable UFW firewall
sudo ufw enable

# Allow necessary ports
sudo ufw allow ssh
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS (if using SSL)
sudo ufw allow 3000  # Dashboard (if direct access needed)
sudo ufw allow 8000  # API (if direct access needed)
```

### **SSL Certificate (Let's Encrypt)**
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### **Environment Security**
```bash
# Secure .env file
chmod 600 .env

# Create secure backup
tar -czf backup.tar.gz --exclude='venv' --exclude='node_modules' .
gpg -c backup.tar.gz  # Encrypt backup
```

---

## 📊 **Monitoring and Maintenance**

### **Log Management**
```bash
# View logs
tail -f logs/trading_bot.log

# Log rotation
sudo tee /etc/logrotate.d/trading-bot > /dev/null <<EOF
/path/to/project/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
}
EOF
```

### **Health Monitoring**
```bash
# Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
echo "=== Trading Bot Status ==="
systemctl status trading-bot-api --no-pager
echo "=== Disk Usage ==="
df -h /
echo "=== Memory Usage ==="
free -h
echo "=== Recent Logs ==="
tail -5 logs/trading_bot.log
EOF

chmod +x monitor.sh

# Add to cron for regular checks
crontab -e
# Add: */5 * * * * /path/to/project/monitor.sh >> logs/monitor.log
```

### **Backup Strategy**
```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "backup_$DATE.tar.gz" \
    --exclude='venv' \
    --exclude='node_modules' \
    --exclude='*.pyc' \
    .
echo "Backup created: backup_$DATE.tar.gz"
EOF

chmod +x backup.sh

# Schedule daily backups
crontab -e
# Add: 0 2 * * * /path/to/project/backup.sh
```

---

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Port Already in Use**
```bash
# Find process using port
sudo lsof -i :8000
sudo lsof -i :3000

# Kill process
sudo kill -9 <PID>
```

#### **Permission Denied**
```bash
# Fix file permissions
chmod +x *.sh
chmod 600 .env
chown -R $USER:$USER .
```

#### **Python Module Not Found**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### **Node.js Issues**
```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### **Service Management**
```bash
# Check service status
sudo systemctl status trading-bot-api

# View service logs
sudo journalctl -u trading-bot-api -f

# Restart service
sudo systemctl restart trading-bot-api
```

---

## 📈 **Performance Optimization**

### **System Optimization**
```bash
# Increase file limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize network settings
echo "net.core.somaxconn = 65536" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### **Database Optimization (if using PostgreSQL)**
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Create database
sudo -u postgres createdb trading_bot

# Optimize PostgreSQL
sudo nano /etc/postgresql/*/main/postgresql.conf
# Adjust: shared_buffers, effective_cache_size, work_mem
```

---

## 🎯 **Next Steps**

### **Development**
1. **Add Real API Integration**: Replace mock data with real Binance API calls
2. **Implement Strategy Logic**: Complete the strategy implementations
3. **Add Database**: Integrate PostgreSQL for data persistence
4. **Enhance Security**: Add authentication and authorization
5. **Add Testing**: Implement comprehensive test coverage

### **Production Readiness**
1. **Load Testing**: Test with high-frequency data
2. **Error Handling**: Implement comprehensive error recovery
3. **Monitoring**: Set up advanced monitoring with Grafana/Prometheus
4. **Scaling**: Implement horizontal scaling capabilities
5. **Documentation**: Create user and API documentation

---

## 🆘 **Support and Resources**

### **Useful Commands**
```bash
# Quick status check
./monitor.sh

# View all logs
tail -f logs/*.log

# Restart everything
sudo systemctl restart trading-bot-api nginx

# Check system resources
htop
iotop
```

### **Configuration Files**
- `.env` - Environment variables
- `config/bot_config.yaml` - Bot configuration
- `dashboard/src/config.ts` - Dashboard configuration
- `/etc/nginx/sites-available/trading-bot` - Nginx configuration

### **Important Directories**
- `logs/` - Application logs
- `data/` - Trading data and state
- `backups/` - System backups
- `venv/` - Python virtual environment
- `dashboard/build/` - Built dashboard files

---

**⚠️ IMPORTANT SECURITY NOTICE**
- Never commit API keys to version control
- Always use HTTPS in production
- Regularly update dependencies
- Monitor logs for suspicious activity
- Test thoroughly before live trading

**🚀 Your crypto trading bot is now ready for Ubuntu deployment!**