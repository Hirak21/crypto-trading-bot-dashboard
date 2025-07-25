"""
Logging configuration for the crypto trading bot.

This module sets up structured logging with appropriate formatters,
handlers, and log levels for different components.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        """Format log record with structured data."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'trade_id'):
            log_entry['trade_id'] = record.trade_id
        if hasattr(record, 'symbol'):
            log_entry['symbol'] = record.symbol
        if hasattr(record, 'strategy'):
            log_entry['strategy'] = record.strategy
        if hasattr(record, 'error_code'):
            log_entry['error_code'] = record.error_code
        
        return json.dumps(log_entry)


class TradingBotLogger:
    """Centralized logging configuration for the trading bot."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.log_dir = Path(self.config.get('log_dir', 'logs'))
        self.log_dir.mkdir(exist_ok=True)
        
        self._setup_logging()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            'level': 'INFO',
            'log_dir': 'logs',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,
            'structured_logging': True,
            'console_logging': True
        }
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config['level'].upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Set up formatters
        if self.config.get('structured_logging', True):
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        if self.config.get('console_logging', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = self.log_dir / 'trading_bot.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.get('max_file_size', 10 * 1024 * 1024),
            backupCount=self.config.get('backup_count', 5)
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Separate handlers for different log types
        self._setup_trade_logger()
        self._setup_error_logger()
        self._setup_performance_logger()
    
    def _setup_trade_logger(self) -> None:
        """Set up dedicated trade logging."""
        trade_logger = logging.getLogger('trading_bot.trades')
        
        trade_file = self.log_dir / 'trades.log'
        trade_handler = logging.handlers.RotatingFileHandler(
            trade_file,
            maxBytes=self.config.get('max_file_size', 10 * 1024 * 1024),
            backupCount=self.config.get('backup_count', 5)
        )
        
        if self.config.get('structured_logging', True):
            trade_handler.setFormatter(StructuredFormatter())
        else:
            trade_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
        
        trade_logger.addHandler(trade_handler)
        trade_logger.propagate = False  # Don't propagate to root logger
    
    def _setup_error_logger(self) -> None:
        """Set up dedicated error logging."""
        error_logger = logging.getLogger('trading_bot.errors')
        
        error_file = self.log_dir / 'errors.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=self.config.get('max_file_size', 10 * 1024 * 1024),
            backupCount=self.config.get('backup_count', 5)
        )
        
        if self.config.get('structured_logging', True):
            error_handler.setFormatter(StructuredFormatter())
        else:
            error_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
        
        error_handler.setLevel(logging.ERROR)
        error_logger.addHandler(error_handler)
        error_logger.propagate = False
    
    def _setup_performance_logger(self) -> None:
        """Set up dedicated performance logging."""
        perf_logger = logging.getLogger('trading_bot.performance')
        
        perf_file = self.log_dir / 'performance.log'
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_file,
            maxBytes=self.config.get('max_file_size', 10 * 1024 * 1024),
            backupCount=self.config.get('backup_count', 5)
        )
        
        if self.config.get('structured_logging', True):
            perf_handler.setFormatter(StructuredFormatter())
        else:
            perf_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
        
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False


def setup_logging(config: Dict[str, Any] = None) -> None:
    """Set up logging for the trading bot."""
    TradingBotLogger(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


def log_trade(trade_id: str, symbol: str, action: str, price: float, 
              quantity: float, strategy: str) -> None:
    """Log trade execution with structured data."""
    logger = logging.getLogger('trading_bot.trades')
    logger.info(
        f"Trade executed: {action} {quantity} {symbol} at {price}",
        extra={
            'trade_id': trade_id,
            'symbol': symbol,
            'action': action,
            'price': price,
            'quantity': quantity,
            'strategy': strategy
        }
    )


def log_error(error_code: str, message: str, exception: Exception = None) -> None:
    """Log error with structured data."""
    logger = logging.getLogger('trading_bot.errors')
    extra_data = {'error_code': error_code}
    
    if exception:
        logger.error(f"{message}: {str(exception)}", extra=extra_data, exc_info=True)
    else:
        logger.error(message, extra=extra_data)


def log_performance(strategy: str, metrics: Dict[str, Any]) -> None:
    """Log performance metrics with structured data."""
    logger = logging.getLogger('trading_bot.performance')
    logger.info(
        f"Performance update for {strategy}",
        extra={
            'strategy': strategy,
            **metrics
        }
    )