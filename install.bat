@echo off
REM Windows Installation Script for Crypto Trading Bot

echo 🚀 CRYPTO TRADING BOT - WINDOWS INSTALLER
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ from python.org
    echo    Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found

REM Check if pip is available
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip not found. Please reinstall Python with pip
    pause
    exit /b 1
)

echo ✅ pip found

REM Install requirements
echo 📦 Installing Python packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

echo ✅ Python packages installed

REM Run setup
echo ⚙️ Running bot setup...
python setup_bot.py

if %errorlevel% neq 0 (
    echo ❌ Setup failed
    pause
    exit /b 1
)

echo.
echo 🎉 INSTALLATION COMPLETE!
echo =========================
echo.
echo 🚀 Quick Start:
echo    python scalping_scanner.py
echo.
echo 📚 Documentation:
echo    - Read QUICK_START.md for detailed instructions
echo    - Check README.md for full documentation
echo.
echo Happy Trading! 🎯
echo.
pause