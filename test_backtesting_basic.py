"""
Basic test for backtesting framework without external dependencies.
"""

from datetime import datetime, timedelta
import json
import tempfile
import shutil


def test_backtest_config():
    """Test BacktestConfig creation and validation."""
    print("Testing BacktestConfig...")
    
    from crypto_trading_bot.backtesting.backtest_engine import BacktestConfig, BacktestStatus
    
    # Test valid configuration
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        initial_balance=10000.0,
        symbols=["BTCUSDT"],
        timeframe="1h",
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    assert config.start_date < config.end_date
    assert config.initial_balance > 0
    assert len(config.symbols) > 0
    assert config.timeframe == "1h"
    assert 0 <= config.commission_rate <= 1
    assert 0 <= config.slippage_rate <= 1
    
    print("✓ Valid BacktestConfig created successfully")
    
    # Test validation errors
    try:
        # Invalid date range
        BacktestConfig(
            start_date=datetime(2023, 1, 31),
            end_date=datetime(2023, 1, 1),
            initial_balance=10000.0,
            symbols=["BTCUSDT"]
        )
        assert False, "Should have raised ValueError for invalid date range"
    except ValueError:
        print("✓ Date range validation working")
    
    try:
        # Invalid balance
        BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            initial_balance=-1000.0,
            symbols=["BTCUSDT"]
        )
        assert False, "Should have raised ValueError for negative balance"
    except ValueError:
        print("✓ Balance validation working")
    
    try:
        # Empty symbols
        BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            initial_balance=10000.0,
            symbols=[]
        )
        assert False, "Should have raised ValueError for empty symbols"
    except ValueError:
        print("✓ Symbols validation working")
    
    print("✓ BacktestConfig tests passed")
    return True


def test_parameter_range():
    """Test ParameterRange functionality."""
    print("\nTesting ParameterRange...")
    
    from crypto_trading_bot.backtesting.optimizer import ParameterRange
    
    # Test float parameter with step
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
    print("✓ Float parameter range generation working")
    
    # Test integer parameter
    int_range = ParameterRange(
        name="int_param",
        min_value=1,
        max_value=5,
        parameter_type="int"
    )
    
    values = int_range.generate_values()
    assert values == [1, 2, 3, 4, 5]
    print("✓ Integer parameter range generation working")
    
    # Test choice parameter
    choice_range = ParameterRange(
        name="choice_param",
        values=["A", "B", "C"],
        parameter_type="choice"
    )
    
    values = choice_range.generate_values()
    assert values == ["A", "B", "C"]
    print("✓ Choice parameter range generation working")
    
    # Test boolean parameter
    bool_range = ParameterRange(
        name="bool_param",
        parameter_type="bool"
    )
    
    values = bool_range.generate_values()
    assert values == [True, False]
    print("✓ Boolean parameter range generation working")
    
    # Test validation
    try:
        # Invalid range
        ParameterRange(
            name="invalid_param",
            min_value=1.0,
            max_value=0.0,
            parameter_type="float"
        )
        assert False, "Should have raised ValueError for invalid range"
    except ValueError:
        print("✓ Parameter range validation working")
    
    print("✓ ParameterRange tests passed")
    return True


def test_performance_metrics():
    """Test PerformanceMetrics functionality."""
    print("\nTesting PerformanceMetrics...")
    
    from crypto_trading_bot.backtesting.performance_analyzer import PerformanceMetrics
    
    # Create metrics instance
    metrics = PerformanceMetrics(
        total_return=0.15,
        annualized_return=0.12,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        max_drawdown=0.08,
        win_rate=0.6,
        profit_factor=1.8,
        total_trades=100,
        winning_trades=60,
        losing_trades=40
    )
    
    # Test basic properties
    assert metrics.total_return == 0.15
    assert metrics.sharpe_ratio == 1.2
    assert metrics.max_drawdown == 0.08
    assert metrics.win_rate == 0.6
    assert metrics.total_trades == 100
    
    print("✓ PerformanceMetrics creation successful")
    
    # Test serialization
    metrics_dict = metrics.to_dict()
    
    assert isinstance(metrics_dict, dict)
    assert 'returns' in metrics_dict
    assert 'risk' in metrics_dict
    assert 'trading' in metrics_dict
    assert 'risk_adjusted' in metrics_dict
    
    assert metrics_dict['returns']['total_return'] == 0.15
    assert metrics_dict['risk']['max_drawdown'] == 0.08
    assert metrics_dict['trading']['win_rate'] == 0.6
    assert metrics_dict['risk_adjusted']['sharpe_ratio'] == 1.2
    
    print("✓ PerformanceMetrics serialization working")
    print("✓ PerformanceMetrics tests passed")
    return True


def test_optimization_config():
    """Test OptimizationConfig functionality."""
    print("\nTesting OptimizationConfig...")
    
    from crypto_trading_bot.backtesting.optimizer import (
        OptimizationConfig, OptimizationMethod, OptimizationObjective, ParameterRange
    )
    
    # Create valid configuration
    config = OptimizationConfig(
        method=OptimizationMethod.GRID_SEARCH,
        objective=OptimizationObjective.SHARPE_RATIO,
        parameter_ranges=[
            ParameterRange(
                name="param1",
                min_value=0.0,
                max_value=1.0,
                parameter_type="float"
            ),
            ParameterRange(
                name="param2",
                values=["A", "B"],
                parameter_type="choice"
            )
        ],
        max_iterations=100,
        validation_split=0.3,
        cross_validation_folds=3
    )
    
    assert config.method == OptimizationMethod.GRID_SEARCH
    assert config.objective == OptimizationObjective.SHARPE_RATIO
    assert len(config.parameter_ranges) == 2
    assert config.max_iterations == 100
    assert config.validation_split == 0.3
    assert config.cross_validation_folds == 3
    
    print("✓ Valid OptimizationConfig created successfully")
    
    # Test validation errors
    try:
        # No parameter ranges
        OptimizationConfig(parameter_ranges=[])
        assert False, "Should have raised ValueError for empty parameter ranges"
    except ValueError:
        print("✓ Parameter ranges validation working")
    
    try:
        # Invalid validation split
        OptimizationConfig(
            parameter_ranges=[
                ParameterRange(name="param1", min_value=0, max_value=1, parameter_type="float")
            ],
            validation_split=1.5
        )
        assert False, "Should have raised ValueError for invalid validation split"
    except ValueError:
        print("✓ Validation split validation working")
    
    print("✓ OptimizationConfig tests passed")
    return True


def test_data_provider_basic():
    """Test basic HistoricalDataProvider functionality."""
    print("\nTesting HistoricalDataProvider basics...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        from crypto_trading_bot.backtesting.data_provider import HistoricalDataProvider, DataSource
        
        # Create provider
        provider = HistoricalDataProvider(cache_dir=temp_dir)
        
        # Test timeframe parsing
        assert provider._parse_timeframe("1m") == 1
        assert provider._parse_timeframe("5m") == 5
        assert provider._parse_timeframe("1h") == 60
        assert provider._parse_timeframe("4h") == 240
        assert provider._parse_timeframe("1d") == 1440
        
        print("✓ Timeframe parsing working")
        
        # Test invalid timeframe
        try:
            provider._parse_timeframe("invalid")
            assert False, "Should have raised ValueError for invalid timeframe"
        except ValueError:
            print("✓ Timeframe validation working")
        
        # Test cache key generation
        cache_key = provider._get_cache_key(
            "BTCUSDT",
            datetime(2023, 1, 1),
            datetime(2023, 1, 31),
            "1h",
            DataSource.MOCK
        )
        
        assert isinstance(cache_key, str)
        assert "BTCUSDT" in cache_key
        assert "20230101" in cache_key
        assert "20230131" in cache_key
        assert "1h" in cache_key
        assert "mock" in cache_key
        
        print("✓ Cache key generation working")
        
        # Test cache info
        cache_info = provider.get_cache_info()
        assert isinstance(cache_info, dict)
        assert 'memory_cache_size' in cache_info
        assert 'cache_directory' in cache_info
        assert 'cached_files' in cache_info
        
        print("✓ Cache info generation working")
        print("✓ HistoricalDataProvider basic tests passed")
        
        return True
        
    except Exception as e:
        print(f"❌ HistoricalDataProvider test failed: {e}")
        return False
    
    finally:
        shutil.rmtree(temp_dir)


def test_strategy_config():
    """Test StrategyConfig functionality."""
    print("\nTesting StrategyConfig...")
    
    from crypto_trading_bot.backtesting.strategy_comparator import StrategyConfig
    
    # Mock strategy class
    class MockStrategy:
        def __init__(self, **kwargs):
            self.parameters = kwargs
    
    # Create strategy configuration
    config = StrategyConfig(
        name="Test Strategy",
        strategy_class=MockStrategy,
        parameters={"param1": 0.5, "param2": "value"},
        enabled=True,
        description="Test strategy for validation"
    )
    
    assert config.name == "Test Strategy"
    assert config.strategy_class == MockStrategy
    assert config.parameters["param1"] == 0.5
    assert config.parameters["param2"] == "value"
    assert config.enabled is True
    assert config.description == "Test strategy for validation"
    
    print("✓ StrategyConfig creation successful")
    print("✓ StrategyConfig tests passed")
    return True


def test_serialization():
    """Test serialization of various components."""
    print("\nTesting serialization...")
    
    # Test BacktestResult serialization
    from crypto_trading_bot.backtesting.backtest_engine import BacktestResult, BacktestConfig, BacktestStatus
    
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        initial_balance=10000.0,
        symbols=["BTCUSDT"]
    )
    
    result = BacktestResult(
        config=config,
        status=BacktestStatus.COMPLETED,
        start_time=datetime.now(),
        total_return=0.15,
        sharpe_ratio=1.2,
        max_drawdown=0.08
    )
    
    result_dict = result.to_dict()
    
    assert isinstance(result_dict, dict)
    assert 'config' in result_dict
    assert 'performance' in result_dict
    assert result_dict['performance']['total_return'] == 0.15
    
    print("✓ BacktestResult serialization working")
    
    # Test OptimizationResult serialization
    from crypto_trading_bot.backtesting.optimizer import OptimizationResult, OptimizationConfig, OptimizationMethod, OptimizationObjective, ParameterRange
    
    opt_config = OptimizationConfig(
        method=OptimizationMethod.GRID_SEARCH,
        objective=OptimizationObjective.SHARPE_RATIO,
        parameter_ranges=[
            ParameterRange(name="param1", min_value=0, max_value=1, parameter_type="float")
        ]
    )
    
    opt_result = OptimizationResult(
        config=opt_config,
        best_parameters={"param1": 0.5},
        best_score=1.2,
        total_evaluations=10,
        execution_time=30.5
    )
    
    opt_result_dict = opt_result.to_dict()
    
    assert isinstance(opt_result_dict, dict)
    assert 'best_parameters' in opt_result_dict
    assert 'best_score' in opt_result_dict
    assert opt_result_dict['best_parameters']['param1'] == 0.5
    assert opt_result_dict['best_score'] == 1.2
    
    print("✓ OptimizationResult serialization working")
    print("✓ Serialization tests passed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("BACKTESTING FRAMEWORK BASIC TESTS")
    print("=" * 60)
    
    tests = [
        test_backtest_config,
        test_parameter_range,
        test_performance_metrics,
        test_optimization_config,
        test_data_provider_basic,
        test_strategy_config,
        test_serialization
    ]
    
    all_passed = True
    
    for test in tests:
        try:
            result = test()
            all_passed &= result
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL BASIC TESTS PASSED! Backtesting framework structure is correct.")
        print("\nKey components validated:")
        print("• BacktestConfig creation and validation")
        print("• ParameterRange functionality")
        print("• PerformanceMetrics structure")
        print("• OptimizationConfig validation")
        print("• HistoricalDataProvider basics")
        print("• StrategyConfig structure")
        print("• Serialization capabilities")
        print("\nThe backtesting and optimization framework is ready for use!")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    main()