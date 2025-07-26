#!/bin/bash

# 🚀 Crypto Trading Bot Dashboard Setup Script
# This script creates a React dashboard for monitoring the trading bot

echo "🚀 Creating Crypto Trading Bot Dashboard..."
echo "=========================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Installing Node.js..."
    
    # Install Node.js on Ubuntu
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
    
    echo "✅ Node.js installed successfully"
else
    echo "✅ Node.js is already installed: $(node --version)"
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not available"
    exit 1
else
    echo "✅ npm is available: $(npm --version)"
fi

# Create dashboard directory
DASHBOARD_DIR="trading-bot-dashboard"
echo "📁 Creating dashboard directory: $DASHBOARD_DIR"

# Remove existing directory if it exists
if [ -d "$DASHBOARD_DIR" ]; then
    echo "⚠️  Directory $DASHBOARD_DIR already exists. Removing..."
    rm -rf "$DASHBOARD_DIR"
fi

# Create React app
echo "⚛️  Creating React application..."
npx create-react-app "$DASHBOARD_DIR" --template typescript

# Navigate to dashboard directory
cd "$DASHBOARD_DIR"

# Install additional dependencies
echo "📦 Installing additional dependencies..."
npm install --save \
    @mui/material \
    @emotion/react \
    @emotion/styled \
    @mui/icons-material \
    @mui/x-charts \
    axios \
    socket.io-client \
    recharts \
    react-router-dom \
    @types/react-router-dom

# Install development dependencies
npm install --save-dev \
    @types/socket.io-client

echo "✅ Dashboard setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. cd $DASHBOARD_DIR"
echo "2. npm start"
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "🔧 The dashboard components will be created next..."