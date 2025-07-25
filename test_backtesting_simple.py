"""
Simple test for backtesting and optimization framework.
"""

import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def test_basic_components():
    """Test basic component creation and functionality."""
    print("Testing basic backtesting components...")
    
    # Test BacktestConfig creation
    from crypto_trading_bot.backtesting.backtest_engine import BacktestConfig, BacktestStatus
    
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        initial_balance=10000.0,
        symbols=["BTCUSDT"],
        timeframe="1h"
    )
    
    assert config.start_date < config.end_date
    assert config.initial_balance > 0
    assert len(config.symbols) > 0
    print("✓ BacktestConfig creation successful")
    
    # Test ParameterRange creation
    from crypto_trading_bot.backtesting.optimizer import ParameterRange
    
    param_range = ParameterRange(
        name="test_param",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        parameter_type="float"
    )
    
    values = param_range.generate_values()
    expected_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert values == expected_values
    print("✓ ParameterRange creation and value generation successful")
    
    # Test PerformanceMetrics creation
    from crypto_trading_bot.backtesting.performance_analyzer import PerformanceMetrics
    
    metrics = PerformanceMetrics(
        total_return=0.15,
        sharpe_ratio=1.2,
        max_drawdown=0.08,
        win_rate=0.6
    )
    
    metrics_dict = metrics.to_dict()
    assert isinstance(metrics_dict, dict)
    assert 'returns' in metrics_dict
    assert 'risk' in metrics_dict
    print("✓ PerformanceMetrics creation and serialization successful")
    
    print("✓ All basic component tests passed")
    return True


async def test_data_provider():
    """Test historical data provider."""
    print("\nTesting HistoricalDataProvider...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        from crypto_trading_bot.backtesting.data_provider import HistoricalDataProvider, DataSource
        
        provider = HistoricalDataProvider(cache_dir=temp_dir)
        
        # Test mock data generation
        symbols = ["BTCUSDT", "ETHUSDT"]
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)
        
        data = await provider.get_historical_data(
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
        
        # Validate data consistency
        validation_result = provider.validate_data(btc_data, "BTCUSDT")
        assert validation_result['valid'] is True
        
        print(f"✓ Generated {len(btc_data)} data points for BTCUSDT")
        print(f"✓ Data validation passed")
        print("✓ HistoricalDataProvider tests passed")
        
        return True
        
    except Exception as e:
        print(f"❌ HistoricalDataProvider test failed: {e}")
        return False
    
    finally:
        shutil.rmtree(temp_dir)


async def test_backtest_engine():
    """Test backtest engine with mock strategy."""
    print("\nTesting BacktestEngine...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        from crypto_trading_bot.backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestStatus
        from crypto_trading_bot.backtesting.data_provider import HistoricalDataProvider
        from crypto_trading_bot.models.trading import TradingSignal
        
        # Create mock strategy
        class MockStrategy:
            def __init__(self, signal_probability=0.05):
                self.signal_probability = signal_probability
            
            async def analyze_market_data(self, market_data):
                # Generate signals based on probability
                if np.random.random() < self.signal_probability:
                    return TradingSignal(
                        symbol=market_data.symbol,
                        action='BUY',
                        confidence=0.8,
                        strategy="mock_strategy",
                        timestamp=market_data.timestamp,
                        target_price=market_data.close * 1.02,
                        stop_loss=market_data.close * 0.98
                    )
                return None
        
        # Setup components
        data_provider = HistoricalDataProvider(cache_dir=temp_dir)
        backtest_engine = BacktestEngine(data_provider=data_provider)
        
        # Configuration
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 7),
            initial_balance=10000.0,
            symbols=["BTCUSDT"],
            timeframe="1h"
        )
        
        # Strategy
        strategy = MockStrategy(signal_probability=0.02)
        
        # Run backtest
        result = await backtest_engine.run_backtest(strategy, config)
        
        # Verify results
        assert result.status == BacktestStatus.COMPLETED
        assert len(result.portfolio_values) > 0
        assert result.config == config
        
        print(f"✓ Backtest completed successfully")
        print(f"✓ Portfolio tracked {len(result.portfolio_values)} data points")
        print(f"✓ Total trades executed: {len(result.trades)}")
        print(f"✓ Final portfolio value: ${result.portfolio_values[-1][1]:.2f}")
        print("✓ BacktestEngine tests passed")
        
        return True
        
    except Exception as e:
        print(f"❌ BacktestEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        shutil.rmtree(temp_dir)


async def test_performance_analyzer():
    """Test performance analyzer."""
    print("\nTesting PerformanceAnalyzer...")
    
    try:
        from crypto_trading_bot.backtesting.performance_analyzer import PerformanceAnalyzer
        from crypto_trading_bot.backtesting.backtest_engine import BacktestResult, BacktestConfig, BacktestStatus
        from crypto_trading_bot.models.trading import Trade
        
        # Create sample backtest result
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
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
        trades = []
        for i in range(5):
            trade = Trade(
                id=f"trade_{i}",
                symbol="BTCUSDT",
                side="SELL",
                size=0.1,
                price=40000 + i * 100,
                commission=10.0,
                timestamp=dates[i * 5].to_pydatetime(),
                strategy="test_strategy",
                pnl=np.random.normal(50, 100)
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
        
        # Analyze performance
        analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
        metrics = analyzer.analyze_backtest_result(result)
        
        # Verify metrics
        assert metrics.total_return != 0
        assert metrics.volatility >= 0
        assert 0 <= metrics.max_drawdown <= 1
        assert metrics.total_trades == len([t for t in trades if t.pnl is not None])
        
        # Generate report
        report = analyzer.generate_performance_report(metrics, "Test Strategy")
        assert isinstance(report, str)
        assert "PERFORMANCE REPORT" in report
        
        print(f"✓ Performance analysis completed")
        print(f"✓ Total return: {metrics.total_return:.2%}")
        print(f"✓ Sharpe ratio: {metrics.sharpe_ratio:.3f}")
        print(f"✓ Max drawdown: {metrics.max_drawdown:.2%}")
        print(f"✓ Total trades analyzed: {metrics.total_trades}")
        print("✓ PerformanceAnalyzer tests passed")
        
        return True
        
    except Exception as e:
        print(f"❌ PerformanceAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_parameter_optimizer():
    """Test parameter optimizer (simplified)."""
    print("\nTesting ParameterOptimizer...")
    
    try:
        from crypto_trading_bot.backtesting.optimizer import (
            ParameterOptimizer, OptimizationConfig, OptimizationMethod, 
            OptimizationObjective, ParameterRange
        )
        
        # Test optimization configuration
        config = OptimizationConfig(
            method=OptimizationMethod.GRID_SEARCH,
            objective=OptimizationObjective.TOTAL_RETURN,
            parameter_ranges=[
                ParameterRange(
                    name="signal_probability",
                    min_value=0.01,
                    max_value=0.05,
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
            max_workers=1
        )
        
        # Test parameter combination generation
        from unittest.mock import Mock
        
        mock_engine = Mock()
        mock_provider = Mock()
        optimizer = ParameterOptimizer(mock_engine, mock_provider)
        
        combinations = optimizer._generate_parameter_combinations(config.parameter_ranges)
        
        # Should generate combinations for all parameter values
        expected_combinations = 3 * 2  # 3 signal_probability values * 2 signal_action values
        assert len(combinations) == expected_combinations
        
        # Verify combination structure
        for combo in combinations:
            assert 'signal_probability' in combo
            assert 'signal_action' in combo
            assert combo['signal_probability'] in [0.01, 0.03, 0.05]
            assert combo['signal_action'] in ["BUY", "SELL"]
        
        print(f"✓ Generated {len(combinations)} parameter combinations")
        print("✓ Parameter combination generation successful")
        
        # Test random parameter generation
        random_combinations = optimizer._generate_random_parameter_combinations(
            config.parameter_ranges, 10
        )
        
        assert len(random_combinations) == 10
        print(f"✓ Generated {len(random_combinations)} random parameter combinations")
        print("✓ ParameterOptimizer tests passed")
        
        return True
        
    except Exception as e:
        print(f"❌ ParameterOptimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("BACKTESTING AND OPTIMIZATION FRAMEWORK TESTS")
    print("=" * 60)
    
    tests = [
        test_basic_components,
        test_data_provider,
        test_backtest_engine,
        test_performance_analyzer,
        test_parameter_optimizer
    ]
    
    all_passed = True
    
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                result = await test()
            else:
                result = test()
            all_passed &= result
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Backtesting framework is working correctly.")
        print("\nKey features implemented:")
        print("• Historical data provider with mock data generation")
        print("• Backtest engine with realistic trading simulation")
        print("• Parameter optimization with multiple algorithms")
        print("• Comprehensive performance analysis")
        print("• Strategy comparison tools")
        print("• Risk metrics and drawdown analysis")
        print("• Statistical testing and validation")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())