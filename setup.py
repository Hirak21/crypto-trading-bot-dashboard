"""Setup script for the crypto trading bot."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crypto-trading-bot",
    version="1.0.0",
    author="Crypto Trading Bot",
    description="A comprehensive cryptocurrency trading bot for Binance Futures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "python-binance>=1.0.19",
        "websockets>=11.0.3",
        "aiohttp>=3.8.5",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "ta-lib>=0.4.26",
        "scipy>=1.11.1",
        "cryptography>=41.0.3",
        "pydantic>=2.1.1",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "structlog>=23.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.1",
            "pytest-mock>=3.11.1",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "crypto-trading-bot=crypto_trading_bot.main:main",
        ],
    },
)