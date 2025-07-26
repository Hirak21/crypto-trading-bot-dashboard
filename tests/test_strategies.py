"""
Unit tests for trading strategy implementations.

Tests all strategy classes including base strategy functionality,
liquidity strategy, momentum strategy, pattern strategy, and candlestick strategy.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

from crypto_trading_bot.strategies.base_strategy import BaseStrategy
from crypto_trading_bot.strategies.liquidity_strategy import LiquidityStrategy
from crypto_trading_bot.strategies.momentum_strategy import MomentumStrategy
from crypto_trading_bot.strategies.pattern_strategy import PatternStrategy
from crypto_trading_bot.strategies.candlestick_strategy import CandlestickStrategy
from crypto_trading_bot.models.trading import TradingSignal, MarketData, SignalAction, Trade, OrderSide
from tests.test_mock_data import MockDataGenerator, create_mock_market_data, create_mock_trading_signal


class TestBaseStrategy(unittest.TestCase):
    """Test cases for BaseStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_generator = MockDataGenerator()
        
        # Create a concrete implementation of BaseStrategy for testing
        class TestStrategy(BaseStrategy):
            def _generate_signal(self, market_data: MarketData):
                if market_data.price > 50000:
                    return TradingSignal(
                        symbol=market_data.symbol,
                        action=SignalAction.BUY,
                        confidence=0.8,
                        strategy=self.name,
                        timestamp=datetime.now()
                    )
                return None
            
            def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
                return 'test_param' in parameters
        
        self.strategy = TestStrategy("test_strategy", {"test_param": 100})
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.get_name(), "test_strategy")
        self.assertEqual(self.strategy.get_confidence(), 0.0)
        self.assertTrue(self.strategy.is_active)
        self.assertIsNone(self.strategy.last_signal_time)
        self.assertEqual(self.strategy.performance_metrics['total_trades'], 0)
    
    def test_parameter_management(self):
        """Test parameter setting and validation."""
        # Valid parameters
        valid_params = {"test_param": 200, "new_param": "value"}
        self.assertTrue(self.strategy.set_parameters(valid_params))
        self.assertEqual(self.strategy.get_parameters()["test_param"], 200)
        
        # Invalid parameters
        invalid_params = {"invalid_param": 300}
        self.assertFalse(self.strategy.set_parameters(invalid_params))
    
    def test_signal_generation(self):
        """Test signal generation logic."""
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        market_data.price = 55000  # Above threshold
        
        signal = self.strategy.analyze(market_data)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.action, SignalAction.BUY)
        self.assertEqual(signal.confidence, 0.8)
        self.assertEqual(signal.strategy, "test_strategy")
    
    def test_cooldown_period(self):
        """Test signal cooldown functionality."""
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        market_data.price = 55000
        
        # First signal should work
        signal1 = self.strategy.analyze(market_data)
        self.assertIsNotNone(signal1)
        
        # Second signal immediately should be blocked by cooldown
        signal2 = self.strategy.analyze(market_data)
        self.assertIsNone(signal2)
        
        # Mock time passage
        self.strategy.last_signal_time = datetime.now() - timedelta(minutes=2)
        signal3 = self.strategy.analyze(market_data)
        self.assertIsNotNone(signal3)
    
    def test_market_data_validation(self):
        """Test market data validation."""
        # Valid market data
        valid_data = self.mock_generator.generate_market_data(count=1)[0]
        self.assertTrue(self.strategy._validate_market_data(valid_data))
        
        # Invalid market data (negative price)
        invalid_data = self.mock_generator.generate_market_data(count=1)[0]
        invalid_data.price = -100
        self.assertFalse(self.strategy._validate_market_data(invalid_data))
        
        # Invalid market data (negative volume)
        invalid_data2 = self.mock_generator.generate_market_data(count=1)[0]
        invalid_data2.volume = -1000
        self.assertFalse(self.strategy._validate_market_data(invalid_data2))
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        # Create mock trades
        winning_trade = Trade(
            trade_id="trade1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=0.1,
            price=50000,
            commission=5,
            timestamp=datetime.now(),
            strategy="test_strategy",
            pnl=100
        )
        
        losing_trade = Trade(
            trade_id="trade2",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            size=0.1,
            price=49000,
            commission=5,
            timestamp=datetime.now(),
            strategy="test_strategy",
            pnl=-50
        )
        
        # Update performance
        self.strategy.update_performance(winning_trade)
        self.strategy.update_performance(losing_trade)
        
        metrics = self.strategy.get_performance_metrics()
        self.assertEqual(metrics['total_trades'], 2)
        self.assertEqual(metrics['winning_trades'], 1)
        self.assertEqual(metrics['losing_trades'], 1)
        self.assertEqual(metrics['total_pnl'], 50)
        self.assertEqual(metrics['win_rate'], 0.5)
    
    def test_strategy_activation(self):
        """Test strategy activation/deactivation."""
        self.assertTrue(self.strategy.is_active)
        
        self.strategy.deactivate()
        self.assertFalse(self.strategy.is_active)
        
        # Should not generate signals when inactive
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        market_data.price = 55000
        signal = self.strategy.analyze(market_data)
        self.assertIsNone(signal)
        
        self.strategy.activate()
        self.assertTrue(self.strategy.is_active)
        
        # Should generate signals when active
        signal = self.strategy.analyze(market_data)
        self.assertIsNotNone(signal)
    
    def test_performance_reset(self):
        """Test performance metrics reset."""
        # Add some performance data
        trade = self.mock_generator.generate_trade()
        trade.strategy = "test_strategy"
        self.strategy.update_performance(trade)
        
        self.assertGreater(self.strategy.performance_metrics['total_trades'], 0)
        
        # Reset performance
        self.strategy.reset_performance()
        self.assertEqual(self.strategy.performance_metrics['total_trades'], 0)
        self.assertEqual(self.strategy.confidence, 0.0)
        self.assertEqual(len(self.strategy.trade_history), 0)


class TestLiquidityStrategy(unittest.TestCase):
    """Test cases for LiquidityStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_generator = MockDataGenerator()
        self.strategy = LiquidityStrategy(
            name="liquidity_test",
            parameters={
                'min_liquidity_threshold': 0.6,
                'high_liquidity_threshold': 0.8,
                'volume_threshold': 1000000,
                'spread_threshold': 0.005
            }
        )
    
    def test_initialization(self):
        """Test liquidity strategy initialization."""
        self.assertEqual(self.strategy.get_name(), "liquidity_test")
        self.assertIn('min_liquidity_threshold', self.strategy.get_parameters())
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        valid_params = {
            'min_liquidity_threshold': 0.5,
            'high_liquidity_threshold': 0.7,
            'volume_threshold': 500000,
            'spread_threshold': 0.01
        }
        self.assertTrue(self.strategy.validate_parameters(valid_params))
        
        # Invalid threshold (negative)
        invalid_params = {'min_liquidity_threshold': -0.1}
        self.assertFalse(self.strategy.validate_parameters(invalid_params))
        
        # Invalid threshold (too high)
        invalid_params2 = {'high_liquidity_threshold': 1.5}
        self.assertFalse(self.strategy.validate_parameters(invalid_params2))
    
    @patch('crypto_trading_bot.strategies.liquidity_strategy.LiquidityStrategy._calculate_liquidity_score')
    def test_signal_generation_high_liquidity(self, mock_liquidity):
        """Test signal generation with high liquidity."""
        mock_liquidity.return_value = 0.9  # High liquidity
        
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        market_data.volume = 2000000  # High volume
        
        with patch.object(self.strategy, '_detect_buying_pressure', return_value=True):
            signal = self.strategy.analyze(market_data)
            
        self.assertIsNotNone(signal)
        self.assertEqual(signal.action, SignalAction.BUY)
        self.assertGreater(signal.confidence, 0.8)
    
    @patch('crypto_trading_bot.strategies.liquidity_strategy.LiquidityStrategy._calculate_liquidity_score')
    def test_signal_generation_low_liquidity(self, mock_liquidity):
        """Test signal generation with low liquidity."""
        mock_liquidity.return_value = 0.3  # Low liquidity
        
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        signal = self.strategy.analyze(market_data)
        
        self.assertIsNone(signal)  # Should not generate signal with low liquidity
    
    def test_liquidity_score_calculation(self):
        """Test liquidity score calculation."""
        order_book = self.mock_generator.generate_order_book()
        
        # Mock the method to test internal calculation
        with patch.object(self.strategy, '_get_order_book', return_value=order_book):
            score = self.strategy._calculate_liquidity_score(order_book)
            
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestMomentumStrategy(unittest.TestCase):
    """Test cases for MomentumStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_generator = MockDataGenerator()
        self.strategy = MomentumStrategy(
            name="momentum_test",
            parameters={
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'momentum_threshold': 0.6
            }
        )
    
    def test_initialization(self):
        """Test momentum strategy initialization."""
        self.assertEqual(self.strategy.get_name(), "momentum_test")
        self.assertIn('rsi_period', self.strategy.get_parameters())
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        valid_params = {
            'rsi_period': 21,
            'rsi_overbought': 75,
            'rsi_oversold': 25,
            'momentum_threshold': 0.7
        }
        self.assertTrue(self.strategy.validate_parameters(valid_params))
        
        # Invalid RSI period
        invalid_params = {'rsi_period': 0}
        self.assertFalse(self.strategy.validate_parameters(invalid_params))
        
        # Invalid threshold
        invalid_params2 = {'momentum_threshold': 1.5}
        self.assertFalse(self.strategy.validate_parameters(invalid_params2))
    
    @patch('crypto_trading_bot.utils.technical_analysis.calculate_rsi')
    @patch('crypto_trading_bot.utils.technical_analysis.calculate_macd')
    def test_signal_generation_bullish_momentum(self, mock_macd, mock_rsi):
        """Test signal generation with bullish momentum."""
        # Mock RSI oversold condition
        mock_rsi.return_value = [None] * 13 + [25.0]  # Oversold
        
        # Mock MACD bullish crossover
        mock_macd.return_value = {
            'macd': [None] * 25 + [0.1],
            'signal': [None] * 25 + [-0.1],
            'histogram': [None] * 25 + [0.2]
        }
        
        market_data_list = self.mock_generator.generate_market_data(count=50)
        
        with patch.object(self.strategy, '_get_market_history', return_value=market_data_list):
            signal = self.strategy.analyze(market_data_list[-1])
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.action, SignalAction.BUY)
    
    @patch('crypto_trading_bot.utils.technical_analysis.calculate_rsi')
    @patch('crypto_trading_bot.utils.technical_analysis.calculate_macd')
    def test_signal_generation_bearish_momentum(self, mock_macd, mock_rsi):
        """Test signal generation with bearish momentum."""
        # Mock RSI overbought condition
        mock_rsi.return_value = [None] * 13 + [75.0]  # Overbought
        
        # Mock MACD bearish crossover
        mock_macd.return_value = {
            'macd': [None] * 25 + [-0.1],
            'signal': [None] * 25 + [0.1],
            'histogram': [None] * 25 + [-0.2]
        }
        
        market_data_list = self.mock_generator.generate_market_data(count=50)
        
        with patch.object(self.strategy, '_get_market_history', return_value=market_data_list):
            signal = self.strategy.analyze(market_data_list[-1])
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.action, SignalAction.SELL)
    
    def test_momentum_alignment_detection(self):
        """Test multi-timeframe momentum alignment."""
        # Create mock data for different timeframes
        market_data = self.mock_generator.generate_market_data(count=100)
        
        with patch.object(self.strategy, '_get_market_history', return_value=market_data):
            with patch.object(self.strategy, '_calculate_momentum_indicators') as mock_calc:
                mock_calc.return_value = {
                    'rsi_1m': 65,
                    'rsi_5m': 68,
                    'rsi_15m': 70,
                    'macd_alignment': 'bullish'
                }
                
                alignment = self.strategy._check_momentum_alignment(market_data[-1])
                
        self.assertIsInstance(alignment, dict)
        self.assertIn('strength', alignment)
        self.assertIn('direction', alignment)


class TestPatternStrategy(unittest.TestCase):
    """Test cases for PatternStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_generator = MockDataGenerator()
        self.strategy = PatternStrategy(
            name="pattern_test",
            parameters={
                'min_pattern_confidence': 0.7,
                'lookback_periods': 50,
                'breakout_confirmation': True,
                'volume_confirmation': True
            }
        )
    
    def test_initialization(self):
        """Test pattern strategy initialization."""
        self.assertEqual(self.strategy.get_name(), "pattern_test")
        self.assertIn('min_pattern_confidence', self.strategy.get_parameters())
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        valid_params = {
            'min_pattern_confidence': 0.8,
            'lookback_periods': 100,
            'breakout_confirmation': False
        }
        self.assertTrue(self.strategy.validate_parameters(valid_params))
        
        # Invalid confidence
        invalid_params = {'min_pattern_confidence': 1.5}
        self.assertFalse(self.strategy.validate_parameters(invalid_params))
        
        # Invalid lookback
        invalid_params2 = {'lookback_periods': 0}
        self.assertFalse(self.strategy.validate_parameters(invalid_params2))
    
    @patch('crypto_trading_bot.utils.pattern_recognition.detect_triangle_patterns')
    def test_triangle_pattern_detection(self, mock_detect):
        """Test triangle pattern detection."""
        mock_detect.return_value = [{
            'type': 'ascending_triangle',
            'confidence': 0.8,
            'breakout_direction': 'up',
            'target_price': 55000,
            'stop_loss': 48000
        }]
        
        market_data_list = self.mock_generator.generate_market_data(count=100)
        
        with patch.object(self.strategy, '_get_market_history', return_value=market_data_list):
            signal = self.strategy.analyze(market_data_list[-1])
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.action, SignalAction.BUY)
        self.assertGreaterEqual(signal.confidence, 0.7)
    
    @patch('crypto_trading_bot.utils.pattern_recognition.detect_head_and_shoulders')
    def test_head_and_shoulders_detection(self, mock_detect):
        """Test head and shoulders pattern detection."""
        mock_detect.return_value = [{
            'type': 'head_and_shoulders',
            'confidence': 0.9,
            'breakout_direction': 'down',
            'target_price': 45000,
            'stop_loss': 52000
        }]
        
        market_data_list = self.mock_generator.generate_market_data(count=100)
        
        with patch.object(self.strategy, '_get_market_history', return_value=market_data_list):
            signal = self.strategy.analyze(market_data_list[-1])
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.action, SignalAction.SELL)
        self.assertGreaterEqual(signal.confidence, 0.7)
    
    def test_pattern_confidence_filtering(self):
        """Test that low confidence patterns are filtered out."""
        with patch('crypto_trading_bot.utils.pattern_recognition.detect_triangle_patterns') as mock_detect:
            mock_detect.return_value = [{
                'type': 'triangle',
                'confidence': 0.5,  # Below threshold
                'breakout_direction': 'up'
            }]
            
            market_data_list = self.mock_generator.generate_market_data(count=100)
            
            with patch.object(self.strategy, '_get_market_history', return_value=market_data_list):
                signal = self.strategy.analyze(market_data_list[-1])
            
            self.assertIsNone(signal)  # Should be filtered out


class TestCandlestickStrategy(unittest.TestCase):
    """Test cases for CandlestickStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_generator = MockDataGenerator()
        self.strategy = CandlestickStrategy(
            name="candlestick_test",
            parameters={
                'min_pattern_confidence': 0.6,
                'volume_confirmation': True,
                'lookback_periods': 20,
                'fallback_threshold': 0.6
            }
        )
    
    def test_initialization(self):
        """Test candlestick strategy initialization."""
        self.assertEqual(self.strategy.get_name(), "candlestick_test")
        self.assertIn('min_pattern_confidence', self.strategy.get_parameters())
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        valid_params = {
            'min_pattern_confidence': 0.7,
            'volume_confirmation': False,
            'lookback_periods': 30
        }
        self.assertTrue(self.strategy.validate_parameters(valid_params))
        
        # Invalid confidence
        invalid_params = {'min_pattern_confidence': -0.1}
        self.assertFalse(self.strategy.validate_parameters(invalid_params))
    
    @patch('crypto_trading_bot.utils.pattern_recognition.detect_candlestick_patterns')
    def test_bullish_candlestick_detection(self, mock_detect):
        """Test bullish candlestick pattern detection."""
        mock_detect.return_value = [{
            'type': 'hammer',
            'confidence': 0.8,
            'direction': 'bullish',
            'strength': 0.7
        }]
        
        market_data_list = self.mock_generator.generate_market_data(count=50)
        
        with patch.object(self.strategy, '_get_market_history', return_value=market_data_list):
            with patch.object(self.strategy, '_check_volume_confirmation', return_value=True):
                signal = self.strategy.analyze(market_data_list[-1])
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.action, SignalAction.BUY)
    
    @patch('crypto_trading_bot.utils.pattern_recognition.detect_candlestick_patterns')
    def test_bearish_candlestick_detection(self, mock_detect):
        """Test bearish candlestick pattern detection."""
        mock_detect.return_value = [{
            'type': 'shooting_star',
            'confidence': 0.9,
            'direction': 'bearish',
            'strength': 0.8
        }]
        
        market_data_list = self.mock_generator.generate_market_data(count=50)
        
        with patch.object(self.strategy, '_get_market_history', return_value=market_data_list):
            with patch.object(self.strategy, '_check_volume_confirmation', return_value=True):
                signal = self.strategy.analyze(market_data_list[-1])
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.action, SignalAction.SELL)
    
    def test_fallback_activation(self):
        """Test fallback activation when liquidity is insufficient."""
        # Mock low liquidity condition
        with patch.object(self.strategy, '_check_liquidity_confidence', return_value=0.4):
            self.assertTrue(self.strategy._should_activate_fallback())
        
        # Mock high liquidity condition
        with patch.object(self.strategy, '_check_liquidity_confidence', return_value=0.8):
            self.assertFalse(self.strategy._should_activate_fallback())
    
    def test_volume_confirmation(self):
        """Test volume confirmation logic."""
        market_data_list = self.mock_generator.generate_market_data(count=20)
        
        # High volume should confirm
        market_data_list[-1].volume = 2000000
        for i in range(len(market_data_list) - 5, len(market_data_list) - 1):
            market_data_list[i].volume = 1000000
        
        confirmed = self.strategy._check_volume_confirmation(market_data_list)
        self.assertTrue(confirmed)
        
        # Low volume should not confirm
        market_data_list[-1].volume = 500000
        confirmed = self.strategy._check_volume_confirmation(market_data_list)
        self.assertFalse(confirmed)


if __name__ == '__main__':
    unittest.main()