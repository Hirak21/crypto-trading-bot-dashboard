"""
Comprehensive tests for the backtesting and optimization framework.

This module tests all components of the backtesting system including
the backtest engine, data provider, optimizer, performance analyzer,
and strategy comparator.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import backtesting components
from crypto_trading_bot.backtesting.backtest_engine import (
    BacktestEngine, BacktestConfig, BacktestResult, BacktestStatus
)
from crypto_trading_bot.backtesting.data_provider import (
    HistoricalDataProvider, DataSource, DataRequest
)
from crypto_trading_bot.backtesting.optimizer import (
    ParameterOptimizer, OptimizationConfig, OptimizationMethod, 
    OptimizationObjective, ParameterRange, OptimizationResult
)
from crypto_trading_bot.backtesting.performance_analyzer import (
    PerformanceAnalyzer, PerformanceMetrics
)
from crypto_trading_bot.backtesting.strategy_comparator import (
    StrategyComparator, StrategyConfig, ComparisonResult, ComparisonMetric
)

# Mock strategy for testing
class MockStrategy:
    """Mock strategy for testing."""
    
    def __init__(self, signal_probability=0.1, signal_action='BUY', **kwargs):
        self.signal_probability = signal_probability
        self.signal_action = signal_action
        self.parameters = kwargs
    
    async def analyze_market_data(self, market_data):
        """Generate mock trading signals."""
        from crypto_trading_bot.models.trading import TradingSignal
        
        # Generate signals based on probability
        if np.random.random() < self.signal_probability:
            return TradingSignal(
                symbol=market_data.symbol,
                action=self.signal_action,
                confidence=0.8,
                strategy="mock_strategy",
                timestamp=market_data.timestamp,
                target_price=market_data.close * 1.02,
                stop_loss=market_data.close * 0.98
            )
        return None


class TestBacktestEngine:
    """Test suite for BacktestEngine."""
    
    @pytest.fixture
    def mock_data_provider(self):
        """Create mock data provider."""
        provider = Mock()
        provider.get_historical_data = AsyncMock()
        return provider
    
    @pytest.fixture
    def backtest_engine(self, mock_data_provider):
        """Create BacktestEngine instance."""
        return BacktestEngine(data_provider=mock_data_provider)
    
    @pytest.fixture
    def sample_backtest_config(self):
        """Create sample backtest configuration."""
        return BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            initial_balance=10000.0,
            symbols=["BTCUSDT"],
            timeframe="1h",
            commission_rate=0.001,
            slippage_rate=0.0005
        )
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='H')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 40000
        returns = np.random.normal(0.0001, 0.02, len(dates))
        
        prices = [base_price]
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 100))  # Ensure positive prices
        
        data = []
        for i, date in enumerate(dates):
            open_price = prices[i]
            close_price = prices[i] if i == len(prices) - 1 else prices[i + 1]
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.uniform(1000000, 5000000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return {"BTCUSDT": pd.DataFrame(data)}
    
    def test_backtest_config_validation(self):
        """Test BacktestConfig validation."""
        # Valid config
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            initial_balance=10000.0,
            symbols=["BTCUSDT"]
        )
        assert config.start_date < config.end_date
        
        # Invalid date range
        with pytest.raises(ValueError):
            BacktestConfig(
                start_date=datetime(2023, 1, 31),
                end_date=datetime(2023, 1, 1),
                initial_balance=10000.0,
                symbols=["BTCUSDT"]
            )
        
        # Invalid balance
        with pytest.raises(ValueError):
            BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 31),
                initial_balance=-1000.0,
                symbols=["BTCUSDT"]
            )
        
        # Empty symbols
        with pytest.raises(ValueError):
            BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 31),
                initial_balance=10000.0,
                symbols=[]
            )
    
    @pytest.mark.asyncio
    async def test_backtest_execution(self, backtest_engine, sample_backtest_config, 
                                    sample_historical_data, mock_data_provider):
        """Test basic backtest execution."""
        # Setup mock data provider
        mock_data_provider.get_historical_data.return_value = sample_historical_data
        
        # Create mock strategy
        strategy = MockStrategy(signal_probability=0.05)
        
        # Run backtest
        result = await backtest_engine.run_backtest(strategy, sample_backtest_config)
        
        # Verify result
        assert isinstance(result, BacktestResult)
        assert result.status == BacktestStatus.COMPLETED
        assert result.config == sample_backtest_config
        assert len(result.portfolio_values) > 0
        assert result.total_return != 0  # Should have some return
    
    @pytest.mark.asyncio
    async def test_backtest_with_no_data(self, backtest_engine, sample_backtest_config, 
                                       mock_data_provider):
        """Test backtest with no historical data."""
        # Setup mock to return empty data
        mock_data_provider.get_historical_data.return_value = {}
        
        strategy = MockStrategy()
        
        # Should raise error for no data
        with pytest.raises(ValueError, match="No historical data available"):
            await backtest_engine.run_backtest(strategy, sample_backtest_config)
    
    def test_backtest_result_serialization(self, sample_backtest_config):
        """Test BacktestResult serialization."""
        result = BacktestResult(
            config=sample_backtest_config,
            status=BacktestStatus.COMPLETED,
            start_time=datetime.now(),
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.08
        )
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'config' in result_dict
        assert 'performance' in result_dict
        assert result_dict['performance']['total_return'] == 0.15
        assert result_dict['performance']['sharpe_ratio'] == 1.2


class TestHistoricalDataProvider:
    """Test suite for HistoricalDataProvider."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def data_provider(self, temp_cache_dir):
        """Create HistoricalDataProvider instance."""
        return HistoricalDataProvider(cache_dir=temp_cache_dir)
    
    @pytest.mark.asyncio
    async def test_mock_data_generation(self, data_provider):
        """Test mock data generation."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)
        
        data = await data_provider.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe="1h",
            source=DataSource.MOCK
        )
        
        assert len(data) == 2
        assert "BTCUSDT" in data
        assert "ETHUSDT" in data
        
        # Check data structure
        btc_data = data["BTCUSDT"]
        assert not btc_data.empty
        assert all(col in btc_data.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Check data consistency
        for _, row in btc_data.iterrows():
            assert row['high'] >= max(row['open'], row['close'])
            assert row['low'] <= min(row['open'], row['close'])
            assert row['volume'] > 0
    
    def test_timeframe_parsing(self, data_provider):
        """Test timeframe parsing."""
        assert data_provider._parse_timeframe("1m") == 1
        assert data_provider._parse_timeframe("5m") == 5
        assert data_provider._parse_timeframe("1h") == 60
        assert data_provider._parse_timeframe("4h") == 240
        assert data_provider._parse_timeframe("1d") == 1440
        
        with pytest.raises(ValueError):
            data_provider._parse_timeframe("invalid")
    
    @pytest.mark.asyncio
    async def test_data_caching(self, data_provider):
        """Test data caching functionality."""
        symbols = ["BTCUSDT"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        
        # First call should generate data
        data1 = await data_provider.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            source=DataSource.MOCK
        )
        
        # Second call should use cached data
        data2 = await data_provider.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            source=DataSource.MOCK
        )
        
        # Data should be identical (from cache)
        pd.testing.assert_frame_equal(data1["BTCUSDT"], data2["BTCUSDT"])
    
    def test_data_validation(self, data_provider):
        """Test data validation functionality."""
        # Create valid data
        valid_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='H'),
            'open': np.random.uniform(100, 200, 10),
            'high': np.random.uniform(150, 250, 10),
            'low': np.random.uniform(50, 150, 10),
            'close': np.random.uniform(100, 200, 10),
            'volume': np.random.uniform(1000, 5000, 10)
        })
        
        # Ensure price consistency
        for i in range(len(valid_data)):
            valid_data.loc[i, 'high'] = max(valid_data.loc[i, 'open'], valid_data.loc[i, 'close'], valid_data.loc[i, 'high'])
            valid_data.loc[i, 'low'] = min(valid_data.loc[i, 'open'], valid_data.loc[i, 'close'], valid_data.loc[i, 'low'])
        
        validation_result = data_provider.validate_data(valid_data, "BTCUSDT")
        assert validation_result['valid'] is True
        assert len(validation_result['errors']) == 0
        
        # Create invalid data (high < low)
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'high'] = 50
        invalid_data.loc[0, 'low'] = 100
        
        validation_result = data_provider.validate_data(invalid_data, "BTCUSDT")
        assert validation_result['valid'] is False
        assert len(validation_result['errors']) > 0


class TestParameterOptimizer:
    """Test suite for ParameterOptimizer."""
    
    @pytest.fixture
    def mock_backtest_engine(self):
        """Create mock backtest engine."""
        engine = Mock()
        engine.run_backtest = AsyncMock()
        return engine
    
    @pytest.fixture
    def mock_data_provider(self):
        """Create mock data provider."""
        provider = Mock()
        return provider
    
    @pytest.fixture
    def optimizer(self, mock_backtest_engine, mock_data_provider):
        """Create ParameterOptimizer instance."""
        return ParameterOptimizer(mock_backtest_engine, mock_data_provider)
    
    @pytest.fixture
    def sample_optimization_config(self):
        """Create sample optimization configuration."""
        return OptimizationConfig(
            method=OptimizationMethod.GRID_SEARCH,
            objective=OptimizationObjective.SHARPE_RATIO,
            parameter_ranges=[
                ParameterRange(
                    name="signal_probability",
                    min_value=0.01,
                    max_value=0.1,
                    step=0.02,
                    parameter_type="float"
                ),
                ParameterRange(
                    name="signal_action",
                    values=["BUY", "SELL"],
                    parameter_type="choice"
                )
            ],
            max_iterations=10,
            max_workers=2
        )
    
    def test_parameter_range_validation(self):
        """Test ParameterRange validation."""
        # Valid range
        param_range = ParameterRange(
            name="test_param",
            min_value=0.0,
            max_value=1.0,
            parameter_type="float"
        )
        assert param_range.min_value < param_range.max_value
        
        # Invalid range
        with pytest.raises(ValueError):
            ParameterRange(
                name="test_param",
                min_value=1.0,
                max_value=0.0,
                parameter_type="float"
            )
        
        # Choice parameter without values
        with pytest.raises(ValueError):
            ParameterRange(
                name="test_param",
                parameter_type="choice"
            )
    
    def test_parameter_value_generation(self):
        """Test parameter value generation."""
        # Float parameter with step
        float_range = ParameterRange(
            name="float_param",
            min_value=0.0,
            max_value=1.0,
            step=0.2,
            parameter_type="float"
        )
        values = float_range.generate_values()
        expected = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        assert values == expected
        
        # Integer parameter
        int_range = ParameterRange(
            name="int_param",
            min_value=1,
            max_value=5,
            parameter_type="int"
        )
        values = int_range.generate_values()
        assert values == [1, 2, 3, 4, 5]
        
        # Choice parameter
        choice_range = ParameterRange(
            name="choice_param",
            values=["A", "B", "C"],
            parameter_type="choice"
        )
        values = choice_range.generate_values()
        assert values == ["A", "B", "C"]
        
        # Boolean parameter
        bool_range = ParameterRange(
            name="bool_param",
            parameter_type="bool"
        )
        values = bool_range.generate_values()
        assert values == [True, False]
    
    @pytest.mark.asyncio
    async def test_grid_search_optimization(self, optimizer, sample_optimization_config, 
                                          mock_backtest_engine):
        """Test grid search optimization."""
        # Setup mock backtest results
        def create_mock_result(score):
            result = Mock()
            result.status.value = "completed"
            result.sharpe_ratio = score
            result.total_return = score * 0.1
            result.max_drawdown = 0.1
            return result
        
        # Mock backtest engine to return different scores
        mock_backtest_engine.run_backtest.side_effect = [
            create_mock_result(1.0),
            create_mock_result(1.5),  # Best result
            create_mock_result(0.8),
            create_mock_result(1.2)
        ]
        
        backtest_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            initial_balance=10000.0,
            symbols=["BTCUSDT"]
        )
        
        # Run optimization
        result = await optimizer.optimize_parameters(
            MockStrategy,
            backtest_config,
            sample_optimization_config
        )
        
        # Verify result
        assert isinstance(result, OptimizationResult)
        assert result.best_score == 1.5
        assert result.best_parameters is not None
        assert result.total_evaluations > 0
        assert len(result.all_results) > 0
    
    def test_optimization_config_validation(self):
        """Test OptimizationConfig validation."""
        # Valid config
        config = OptimizationConfig(
            parameter_ranges=[
                ParameterRange(name="param1", min_value=0, max_value=1, parameter_type="float")
            ]
        )
        assert len(config.parameter_ranges) > 0
        
        # No parameter ranges
        with pytest.raises(ValueError):
            OptimizationConfig(parameter_ranges=[])
        
        # Invalid validation split
        with pytest.raises(ValueError):
            OptimizationConfig(
                parameter_ranges=[
                    ParameterRange(name="param1", min_value=0, max_value=1, parameter_type="float")
                ],
                validation_split=1.5
            )


class TestPerformanceAnalyzer:
    """Test suite for PerformanceAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create PerformanceAnalyzer instance."""
        return PerformanceAnalyzer(risk_free_rate=0.02)
    
    @pytest.fixture
    def sample_backtest_result(self):
        """Create sample backtest result."""
        # Generate sample portfolio values
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        initial_value = 10000
        
        # Generate realistic portfolio evolution
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        
        portfolio_values = []
        current_value = initial_value
        
        for i, date in enumerate(dates):
            if i > 0:
                current_value *= (1 + returns[i])
            portfolio_values.append((date.to_pydatetime(), current_value))
        
        # Generate sample trades
        from crypto_trading_bot.models.trading import Trade
        trades = []
        for i in range(10):
            trade = Trade(
                id=f"trade_{i}",
                symbol="BTCUSDT",
                side="SELL" if i % 2 == 0 else "BUY",
                size=0.1,
                price=40000 + i * 100,
                commission=10.0,
                timestamp=dates[i * 10].to_pydatetime(),
                strategy="test_strategy",
                pnl=np.random.normal(50, 100) if i % 2 == 0 else None
            )
            trades.append(trade)
        
        config = BacktestConfig(
            start_date=dates[0].to_pydatetime(),
            end_date=dates[-1].to_pydatetime(),
            initial_balance=initial_value,
            symbols=["BTCUSDT"]
        )
        
        result = BacktestResult(
            config=config,
            status=BacktestStatus.COMPLETED,
            start_time=datetime.now(),
            portfolio_values=portfolio_values,
            trades=trades
        )
        
        return result
    
    def test_performance_metrics_calculation(self, analyzer, sample_backtest_result):
        """Test performance metrics calculation."""
        metrics = analyzer.analyze_backtest_result(sample_backtest_result)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return != 0
        assert metrics.annualized_return != 0
        assert metrics.volatility >= 0
        assert 0 <= metrics.max_drawdown <= 1
        assert 0 <= metrics.win_rate <= 1
        assert metrics.total_trades > 0
    
    def test_risk_metrics_calculation(self, analyzer, sample_backtest_result):
        """Test risk metrics calculation."""
        metrics = analyzer.analyze_backtest_result(sample_backtest_result)
        
        # Risk metrics should be calculated
        assert metrics.volatility >= 0
        assert metrics.max_drawdown >= 0
        assert metrics.value_at_risk_95 <= 0  # VaR should be negative
        assert metrics.conditional_value_at_risk_95 <= metrics.value_at_risk_95
    
    def test_risk_adjusted_returns(self, analyzer, sample_backtest_result):
        """Test risk-adjusted return calculations."""
        metrics = analyzer.analyze_backtest_result(sample_backtest_result)
        
        # Sharpe ratio should be calculated
        assert isinstance(metrics.sharpe_ratio, float)
        
        # Sortino ratio should be calculated
        assert isinstance(metrics.sortino_ratio, float)
        
        # Calmar ratio should be calculated if max drawdown > 0
        if metrics.max_drawdown > 0:
            assert isinstance(metrics.calmar_ratio, float)
    
    def test_trade_analysis(self, analyzer, sample_backtest_result):
        """Test trade analysis."""
        metrics = analyzer.analyze_backtest_result(sample_backtest_result)
        
        # Trade statistics should be calculated
        assert metrics.total_trades > 0
        assert metrics.winning_trades + metrics.losing_trades <= metrics.total_trades
        
        if metrics.total_trades > 0:
            assert 0 <= metrics.win_rate <= 1
        
        if metrics.winning_trades > 0:
            assert metrics.avg_winning_trade != 0
        
        if metrics.losing_trades > 0:
            assert metrics.avg_losing_trade != 0
    
    def test_performance_report_generation(self, analyzer, sample_backtest_result):
        """Test performance report generation."""
        metrics = analyzer.analyze_backtest_result(sample_backtest_result)
        report = analyzer.generate_performance_report(metrics, "Test Strategy")
        
        assert isinstance(report, str)
        assert "PERFORMANCE REPORT" in report
        assert "Test Strategy" in report
        assert "RETURN METRICS" in report
        assert "RISK METRICS" in report
        assert "TRADING STATISTICS" in report
    
    def test_strategy_comparison(self, analyzer):
        """Test strategy comparison functionality."""
        # Create mock metrics for different strategies
        metrics1 = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            win_rate=0.6
        )
        
        metrics2 = PerformanceMetrics(
            total_return=0.12,
            sharpe_ratio=1.5,
            max_drawdown=0.05,
            win_rate=0.65
        )
        
        metrics_list = [
            ("Strategy A", metrics1),
            ("Strategy B", metrics2)
        ]
        
        comparison = analyzer.compare_strategies(metrics_list)
        
        assert isinstance(comparison, dict)
        assert 'strategies' in comparison
        assert 'rankings' in comparison
        assert 'summary' in comparison
        
        # Check rankings
        assert len(comparison['rankings']['by_total_return']) == 2
        assert len(comparison['rankings']['by_sharpe_ratio']) == 2


class TestStrategyComparator:
    """Test suite for StrategyComparator."""
    
    @pytest.fixture
    def mock_backtest_engine(self):
        """Create mock backtest engine."""
        engine = Mock()
        engine.run_backtest = AsyncMock()
        return engine
    
    @pytest.fixture
    def mock_data_provider(self):
        """Create mock data provider."""
        return Mock()
    
    @pytest.fixture
    def mock_performance_analyzer(self):
        """Create mock performance analyzer."""
        analyzer = Mock()
        analyzer.analyze_backtest_result = Mock()
        return analyzer
    
    @pytest.fixture
    def comparator(self, mock_backtest_engine, mock_data_provider, mock_performance_analyzer):
        """Create StrategyComparator instance."""
        return StrategyComparator(mock_backtest_engine, mock_data_provider, mock_performance_analyzer)
    
    @pytest.fixture
    def sample_strategies(self):
        """Create sample strategy configurations."""
        return [
            StrategyConfig(
                name="Strategy A",
                strategy_class=MockStrategy,
                parameters={"signal_probability": 0.05},
                enabled=True
            ),
            StrategyConfig(
                name="Strategy B",
                strategy_class=MockStrategy,
                parameters={"signal_probability": 0.1},
                enabled=True
            )
        ]
    
    @pytest.mark.asyncio
    async def test_strategy_comparison(self, comparator, sample_strategies, 
                                     mock_backtest_engine, mock_performance_analyzer):
        """Test strategy comparison."""
        # Setup mock results
        def create_mock_backtest_result(return_value):
            result = Mock()
            result.status.value = "completed"
            result.total_return = return_value
            result.portfolio_values = [(datetime.now(), 10000 * (1 + return_value))]
            return result
        
        def create_mock_metrics(return_value):
            metrics = Mock()
            metrics.total_return = return_value
            metrics.sharpe_ratio = return_value * 5
            metrics.max_drawdown = 0.1
            metrics.win_rate = 0.6
            return metrics
        
        # Setup mocks
        mock_backtest_engine.run_backtest.side_effect = [
            create_mock_backtest_result(0.15),
            create_mock_backtest_result(0.12)
        ]
        
        mock_performance_analyzer.analyze_backtest_result.side_effect = [
            create_mock_metrics(0.15),
            create_mock_metrics(0.12)
        ]
        
        backtest_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            initial_balance=10000.0,
            symbols=["BTCUSDT"]
        )
        
        # Run comparison
        result = await comparator.compare_strategies(
            sample_strategies,
            backtest_config,
            ComparisonMetric.TOTAL_RETURN
        )
        
        # Verify result
        assert isinstance(result, ComparisonResult)
        assert len(result.strategy_results) == 2
        assert len(result.strategy_metrics) == 2
        assert result.best_strategy is not None
        assert result.best_metric_value is not None
        assert result.execution_time > 0
    
    def test_comparison_result_serialization(self, sample_strategies):
        """Test ComparisonResult serialization."""
        backtest_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            initial_balance=10000.0,
            symbols=["BTCUSDT"]
        )
        
        result = ComparisonResult(
            comparison_config={"test": "config"},
            backtest_config=backtest_config,
            strategies=sample_strategies,
            best_strategy="Strategy A",
            best_metric_value=0.15
        )
        
        # Test serialization
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'comparison_config' in result_dict
        assert 'backtest_config' in result_dict
        assert 'strategies' in result_dict
        assert result_dict['best_strategy'] == "Strategy A"
        assert result_dict['best_metric_value'] == 0.15


# Integration tests
class TestBacktestingIntegration:
    """Integration tests for the complete backtesting framework."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_end_to_end_backtest(self, temp_dir):
        """Test complete end-to-end backtesting workflow."""
        # Setup components
        data_provider = HistoricalDataProvider(cache_dir=temp_dir)
        backtest_engine = BacktestEngine(data_provider=data_provider)
        performance_analyzer = PerformanceAnalyzer()
        
        # Configuration
        backtest_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 7),
            initial_balance=10000.0,
            symbols=["BTCUSDT"],
            timeframe="1h"
        )
        
        # Strategy
        strategy = MockStrategy(signal_probability=0.02)
        
        # Run backtest
        backtest_result = await backtest_engine.run_backtest(strategy, backtest_config)
        
        # Analyze performance
        metrics = performance_analyzer.analyze_backtest_result(backtest_result)
        
        # Verify results
        assert backtest_result.status == BacktestStatus.COMPLETED
        assert isinstance(metrics, PerformanceMetrics)
        assert len(backtest_result.portfolio_values) > 0
        
        # Generate report
        report = performance_analyzer.generate_performance_report(metrics, "Test Strategy")
        assert isinstance(report, str)
        assert "Test Strategy" in report
    
    @pytest.mark.asyncio
    async def test_parameter_optimization_workflow(self, temp_dir):
        """Test parameter optimization workflow."""
        # Setup components
        data_provider = HistoricalDataProvider(cache_dir=temp_dir)
        backtest_engine = BacktestEngine(data_provider=data_provider)
        optimizer = ParameterOptimizer(backtest_engine, data_provider)
        
        # Configuration
        backtest_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 3),  # Short period for testing
            initial_balance=10000.0,
            symbols=["BTCUSDT"],
            timeframe="1h"
        )
        
        optimization_config = OptimizationConfig(
            method=OptimizationMethod.GRID_SEARCH,
            objective=OptimizationObjective.TOTAL_RETURN,
            parameter_ranges=[
                ParameterRange(
                    name="signal_probability",
                    min_value=0.01,
                    max_value=0.05,
                    step=0.02,
                    parameter_type="float"
                )
            ],
            max_workers=1  # Single worker for testing
        )
        
        # Run optimization
        result = await optimizer.optimize_parameters(
            MockStrategy,
            backtest_config,
            optimization_config
        )
        
        # Verify results
        assert isinstance(result, OptimizationResult)
        assert result.best_parameters is not None
        assert result.total_evaluations > 0
        assert len(result.all_results) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])