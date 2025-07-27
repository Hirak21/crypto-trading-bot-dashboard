#!/usr/bin/env python3
"""
Deployment Script for Crypto Trading Bot
Handles different deployment scenarios
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

def deploy_local():
    """Deploy for local development"""
    print("🏠 Setting up for local development...")
    
    # Run setup
    subprocess.run([sys.executable, "setup_bot.py"])
    
    print("✅ Local deployment complete!")
    print("Run: python scalping_scanner.py")

def deploy_server():
    """Deploy for server/VPS"""
    print("🖥️ Setting up for server deployment...")
    
    # Create systemd service file
    service_content = """[Unit]
Description=Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/crypto-trading-bot
ExecStart=/usr/bin/python3 scalping_scanner.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    with open("crypto-trading-bot.service", "w") as f:
        f.write(service_content)
    
    print("✅ Created systemd service file")
    print("To install:")
    print("1. sudo cp crypto-trading-bot.service /etc/systemd/system/")
    print("2. sudo systemctl enable crypto-trading-bot")
    print("3. sudo systemctl start crypto-trading-bot")

def deploy_docker():
    """Create Docker deployment"""
    print("🐳 Creating Docker deployment...")
    
    # Create Dockerfile
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create config directory
RUN mkdir -p config logs results

# Run setup
RUN python setup_bot.py --non-interactive

# Default command
CMD ["python", "scalping_scanner.py"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Create docker-compose.yml
    compose_content = """version: '3.8'

services:
  trading-bot:
    build: .
    container_name: crypto-trading-bot
    restart: unless-stopped
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
      - ./results:/app/results
    environment:
      - PYTHONUNBUFFERED=1
    command: python scalping_scanner.py
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)
    
    print("✅ Docker files created")
    print("To deploy:")
    print("1. docker-compose build")
    print("2. docker-compose up -d")

def deploy_cloud():
    """Create cloud deployment scripts"""
    print("☁️ Creating cloud deployment scripts...")
    
    # AWS deployment script
    aws_script = """#!/bin/bash
# AWS EC2 Deployment Script

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and Git
sudo apt install python3 python3-pip git -y

# Clone repository
git clone https://github.com/YOUR_USERNAME/crypto-trading-bot.git
cd crypto-trading-bot

# Run setup
python3 setup_bot.py

# Install as service
sudo cp crypto-trading-bot.service /etc/systemd/system/
sudo systemctl enable crypto-trading-bot
sudo systemctl start crypto-trading-bot

echo "✅ AWS deployment complete!"
"""
    
    with open("deploy_aws.sh", "w") as f:
        f.write(aws_script)
    
    # Make executable
    os.chmod("deploy_aws.sh", 0o755)
    
    print("✅ Cloud deployment scripts created")

def main():
    parser = argparse.ArgumentParser(description="Deploy Crypto Trading Bot")
    parser.add_argument("--type", choices=["local", "server", "docker", "cloud"], 
                       default="local", help="Deployment type")
    
    args = parser.parse_args()
    
    print("🚀 CRYPTO TRADING BOT DEPLOYMENT")
    print("=" * 50)
    
    if args.type == "local":
        deploy_local()
    elif args.type == "server":
        deploy_server()
    elif args.type == "docker":
        deploy_docker()
    elif args.type == "cloud":
        deploy_cloud()

if __name__ == "__main__":
    main()