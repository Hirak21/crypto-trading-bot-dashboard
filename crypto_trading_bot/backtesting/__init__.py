"""
Backtesting and optimization framework for crypto trading bot.

This module provides comprehensive backtesting capabilities including:
- Historical data backtesting system
- Parameter optimization using historical performance
- Strategy performance comparison tools
- Comprehensive backtesting tests
"""

from .backtest_engine import BacktestEngine, BacktestResult
from .data_provider import HistoricalDataProvider, DataSource
from .optimizer import ParameterOptimizer, OptimizationResult
from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from .strategy_comparator import StrategyComparator, ComparisonResult

__all__ = [
    'BacktestEngine',
    'BacktestResult', 
    'HistoricalDataProvider',
    'DataSource',
    'ParameterOptimizer',
    'OptimizationResult',
    'PerformanceAnalyzer',
    'PerformanceMetrics',
    'StrategyComparator',
    'ComparisonResult'
]