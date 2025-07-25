@echo off
echo Installing Python dependencies...
echo.

REM Try different Python commands
py -m pip install --upgrade pip
if %errorlevel% neq 0 (
    python -m pip install --upgrade pip
    if %errorlevel% neq 0 (
        python3 -m pip install --upgrade pip
    )
)

echo.
echo Installing required packages...
py -m pip install cryptography aiohttp websockets pandas numpy
if %errorlevel% neq 0 (
    python -m pip install cryptography aiohttp websockets pandas numpy
    if %errorlevel% neq 0 (
        python3 -m pip install cryptography aiohttp websockets pandas numpy
    )
)

echo.
echo Installation complete!
pause
