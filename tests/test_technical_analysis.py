"""
Unit tests for technical analysis utilities.

Tests all technical indicator calculations including moving averages,
momentum indicators, trend indicators, volatility indicators, and volume indicators.
"""

import unittest
import math
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from crypto_trading_bot.utils.technical_analysis import (
    MovingAverages, MomentumIndicators, TrendIndicators, VolatilityIndicators,
    VolumeIndicators, TechnicalAnalyzer, TechnicalIndicatorError,
    calculate_rsi, calculate_macd, calculate_sma, calculate_ema,
    calculate_bollinger_bands, analyze_market_data
)
from crypto_trading_bot.models.trading import MarketData
from tests.test_mock_data import MockDataGenerator, create_price_series


class TestMovingAverages(unittest.TestCase):
    """Test cases for MovingAverages class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_generator = MockDataGenerator()
        self.test_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation."""
        # Test with period 3
        sma_values = MovingAverages.sma(self.test_prices, 3)
        
        # First two values should be None
        self.assertIsNone(sma_values[0])
        self.assertIsNone(sma_values[1])
        
        # Third value should be average of first 3 prices
        expected_sma_3 = (100 + 102 + 101) / 3
        self.assertAlmostEqual(sma_values[2], expected_sma_3, places=6)
        
        # Fourth value should be average of prices 1-3
        expected_sma_4 = (102 + 101 + 103) / 3
        self.assertAlmostEqual(sma_values[3], expected_sma_4, places=6)
    
    def test_sma_edge_cases(self):
        """Test SMA edge cases."""
        # Empty list
        self.assertEqual(MovingAverages.sma([], 5), [])
        
        # Period larger than data
        self.assertEqual(MovingAverages.sma([1, 2, 3], 5), [])
        
        # Zero period
        self.assertEqual(MovingAverages.sma([1, 2, 3], 0), [])
        
        # Negative period
        self.assertEqual(MovingAverages.sma([1, 2, 3], -1), [])
    
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation."""
        ema_values = MovingAverages.ema(self.test_prices, 3)
        
        # First value should equal first price
        self.assertEqual(ema_values[0], self.test_prices[0])
        
        # Check EMA calculation formula
        alpha = 2.0 / (3 + 1)  # 0.5 for period 3
        expected_ema_2 = alpha * self.test_prices[1] + (1 - alpha) * ema_values[0]
        self.assertAlmostEqual(ema_values[1], expected_ema_2, places=6)
    
    def test_ema_edge_cases(self):
        """Test EMA edge cases."""
        # Empty list
        self.assertEqual(MovingAverages.ema([], 5), [])
        
        # Zero period
        self.assertEqual(MovingAverages.ema([1, 2, 3], 0), [])
        
        # Single value
        self.assertEqual(MovingAverages.ema([100], 1), [100])
    
    def test_wma_calculation(self):
        """Test Weighted Moving Average calculation."""
        wma_values = MovingAverages.wma(self.test_prices, 3)
        
        # First two values should be None
        self.assertIsNone(wma_values[0])
        self.assertIsNone(wma_values[1])
        
        # Third value calculation: (100*1 + 102*2 + 101*3) / (1+2+3)
        expected_wma = (100*1 + 102*2 + 101*3) / 6
        self.assertAlmostEqual(wma_values[2], expected_wma, places=6)
    
    def test_vwap_calculation(self):
        """Test Volume Weighted Average Price calculation."""
        prices = [100, 101, 102, 103, 104]
        volumes = [1000, 1500, 1200, 1800, 1100]
        
        vwap_values = MovingAverages.vwap(prices, volumes)
        
        # First VWAP should equal first price
        self.assertEqual(vwap_values[0], prices[0])
        
        # Second VWAP calculation
        cumulative_pv = prices[0] * volumes[0] + prices[1] * volumes[1]
        cumulative_volume = volumes[0] + volumes[1]
        expected_vwap = cumulative_pv / cumulative_volume
        self.assertAlmostEqual(vwap_values[1], expected_vwap, places=6)
    
    def test_vwap_edge_cases(self):
        """Test VWAP edge cases."""
        # Mismatched lengths
        self.assertEqual(MovingAverages.vwap([1, 2], [100]), [])
        
        # Empty lists
        self.assertEqual(MovingAverages.vwap([], []), [])
        
        # Zero volume handling
        prices = [100, 101]
        volumes = [0, 1000]
        vwap_values = MovingAverages.vwap(prices, volumes)
        self.assertEqual(vwap_values[0], prices[0])  # Should use price when volume is 0


class TestMomentumIndicators(unittest.TestCase):
    """Test cases for MomentumIndicators class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create trending price data for RSI testing
        self.trending_up = [50, 52, 54, 53, 55, 57, 56, 58, 60, 59, 61, 63, 62, 64, 66]
        self.trending_down = [66, 64, 62, 63, 61, 59, 60, 58, 56, 57, 55, 53, 54, 52, 50]
        self.sideways = [50, 51, 49, 50, 52, 48, 50, 51, 49, 50, 52, 48, 50, 51, 49]
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        rsi_values = MomentumIndicators.rsi(self.trending_up, 14)
        
        # Should have None values for first 14 periods
        for i in range(14):
            self.assertIsNone(rsi_values[i])
        
        # RSI should be above 50 for uptrending data
        if len(rsi_values) > 14:
            self.assertGreater(rsi_values[14], 50)
    
    def test_rsi_edge_cases(self):
        """Test RSI edge cases."""
        # Insufficient data
        self.assertEqual(MomentumIndicators.rsi([1, 2, 3], 14), [])
        
        # Empty data
        self.assertEqual(MomentumIndicators.rsi([], 14), [])
        
        # Zero period
        self.assertEqual(MomentumIndicators.rsi([1, 2, 3], 0), [])
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        prices = create_price_series(50, trend=0.001)  # Slight uptrend
        macd_result = MomentumIndicators.macd(prices)
        
        # Should return dictionary with required keys
        self.assertIn('macd', macd_result)
        self.assertIn('signal', macd_result)
        self.assertIn('histogram', macd_result)
        
        # Check lengths
        self.assertEqual(len(macd_result['macd']), len(prices))
        self.assertEqual(len(macd_result['signal']), len(prices))
        self.assertEqual(len(macd_result['histogram']), len(prices))
        
        # First 25 values should be None (slow period - 1)
        for i in range(25):
            self.assertIsNone(macd_result['macd'][i])
    
    def test_macd_edge_cases(self):
        """Test MACD edge cases."""
        # Insufficient data
        result = MomentumIndicators.macd([1, 2, 3])
        self.assertEqual(result['macd'], [])
        self.assertEqual(result['signal'], [])
        self.assertEqual(result['histogram'], [])
        
        # Empty data
        result = MomentumIndicators.macd([])
        self.assertEqual(result['macd'], [])
    
    def test_roc_calculation(self):
        """Test Rate of Change calculation."""
        prices = [100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112]
        roc_values = MomentumIndicators.roc(prices, 5)
        
        # First 5 values should be None
        for i in range(5):
            self.assertIsNone(roc_values[i])
        
        # Check ROC calculation: ((current - previous) / previous) * 100
        if len(roc_values) > 5:
            expected_roc = ((prices[5] - prices[0]) / prices[0]) * 100
            self.assertAlmostEqual(roc_values[5], expected_roc, places=6)
    
    def test_stochastic_calculation(self):
        """Test Stochastic Oscillator calculation."""
        highs = [105, 107, 109, 108, 110, 112, 111, 113, 115, 114, 116, 118, 117, 119, 121]
        lows = [95, 97, 99, 98, 100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111]
        closes = [100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 116]
        
        stoch_result = MomentumIndicators.stochastic(highs, lows, closes, 14, 3)
        
        # Should return dictionary with %K and %D
        self.assertIn('%K', stoch_result)
        self.assertIn('%D', stoch_result)
        
        # Check that %K values are between 0 and 100
        for value in stoch_result['%K']:
            if value is not None:
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 100)
    
    def test_stochastic_edge_cases(self):
        """Test Stochastic edge cases."""
        # Mismatched lengths
        result = MomentumIndicators.stochastic([1, 2], [1], [1, 2])
        self.assertEqual(result['%K'], [])
        self.assertEqual(result['%D'], [])
        
        # Insufficient data
        result = MomentumIndicators.stochastic([1, 2], [1, 2], [1, 2], 14)
        self.assertEqual(result['%K'], [])


class TestTrendIndicators(unittest.TestCase):
    """Test cases for TrendIndicators class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_generator = MockDataGenerator()
        # Create sample OHLC data
        self.highs = [105, 107, 109, 108, 110, 112, 111, 113, 115, 114, 116, 118, 117, 119, 121, 120]
        self.lows = [95, 97, 99, 98, 100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110]
        self.closes = [100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 116, 115]
    
    def test_adx_calculation(self):
        """Test ADX calculation."""
        adx_result = TrendIndicators.adx(self.highs, self.lows, self.closes, 14)
        
        # Should return dictionary with ADX, +DI, -DI
        self.assertIn('ADX', adx_result)
        self.assertIn('+DI', adx_result)
        self.assertIn('-DI', adx_result)
        
        # Check lengths
        self.assertEqual(len(adx_result['ADX']), len(self.closes))
        
        # First 14 values should be None
        for i in range(14):
            self.assertIsNone(adx_result['ADX'][i])
    
    def test_adx_edge_cases(self):
        """Test ADX edge cases."""
        # Insufficient data
        result = TrendIndicators.adx([1, 2], [1, 2], [1, 2], 14)
        self.assertEqual(result['ADX'], [])
        
        # Mismatched lengths
        result = TrendIndicators.adx([1, 2], [1], [1, 2], 14)
        self.assertEqual(result['ADX'], [])
    
    def test_wilder_smooth(self):
        """Test Wilder's smoothing method."""
        values = [10, 12, 11, 13, 15, 14, 16, 18, 17, 19]
        smoothed = TrendIndicators._wilder_smooth(values, 5)
        
        # Should start with simple average of first 5 values
        expected_first = sum(values[:5]) / 5
        self.assertAlmostEqual(smoothed[0], expected_first, places=6)
        
        # Check Wilder's formula for second value
        if len(smoothed) > 1:
            expected_second = ((smoothed[0] * 4) + values[5]) / 5
            self.assertAlmostEqual(smoothed[1], expected_second, places=6)
    
    def test_parabolic_sar_calculation(self):
        """Test Parabolic SAR calculation."""
        sar_values = TrendIndicators.parabolic_sar(self.highs, self.lows)
        
        # Should have same length as input
        self.assertEqual(len(sar_values), len(self.highs))
        
        # First value should be None
        self.assertIsNone(sar_values[0])
        
        # Subsequent values should be numbers
        for i in range(1, len(sar_values)):
            self.assertIsInstance(sar_values[i], (int, float))
    
    def test_parabolic_sar_edge_cases(self):
        """Test Parabolic SAR edge cases."""
        # Insufficient data
        self.assertEqual(TrendIndicators.parabolic_sar([1], [1]), [])
        
        # Mismatched lengths
        self.assertEqual(TrendIndicators.parabolic_sar([1, 2], [1]), [])
        
        # Empty data
        self.assertEqual(TrendIndicators.parabolic_sar([], []), [])


class TestVolatilityIndicators(unittest.TestCase):
    """Test cases for VolatilityIndicators class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115, 117, 116, 118, 120]
        self.highs = [h + 2 for h in self.test_prices]
        self.lows = [l - 2 for l in self.test_prices]
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        bb_result = VolatilityIndicators.bollinger_bands(self.test_prices, 10, 2.0)
        
        # Should return dictionary with upper, middle, lower
        self.assertIn('upper', bb_result)
        self.assertIn('middle', bb_result)
        self.assertIn('lower', bb_result)
        
        # Check lengths
        self.assertEqual(len(bb_result['upper']), len(self.test_prices))
        self.assertEqual(len(bb_result['middle']), len(self.test_prices))
        self.assertEqual(len(bb_result['lower']), len(self.test_prices))
        
        # First 9 values should be None
        for i in range(9):
            self.assertIsNone(bb_result['upper'][i])
            self.assertIsNone(bb_result['middle'][i])
            self.assertIsNone(bb_result['lower'][i])
        
        # Upper band should be above middle, middle above lower
        for i in range(10, len(self.test_prices)):
            if all(x is not None for x in [bb_result['upper'][i], bb_result['middle'][i], bb_result['lower'][i]]):
                self.assertGreater(bb_result['upper'][i], bb_result['middle'][i])
                self.assertGreater(bb_result['middle'][i], bb_result['lower'][i])
    
    def test_bollinger_bands_edge_cases(self):
        """Test Bollinger Bands edge cases."""
        # Insufficient data
        result = VolatilityIndicators.bollinger_bands([1, 2, 3], 10)
        self.assertEqual(result['upper'], [])
        
        # Empty data
        result = VolatilityIndicators.bollinger_bands([], 10)
        self.assertEqual(result['upper'], [])
    
    def test_atr_calculation(self):
        """Test Average True Range calculation."""
        atr_values = VolatilityIndicators.atr(self.highs, self.lows, self.test_prices, 14)
        
        # Should have same length as input
        self.assertEqual(len(atr_values), len(self.test_prices))
        
        # First 14 values should be None
        for i in range(14):
            self.assertIsNone(atr_values[i])
        
        # ATR values should be positive
        for i in range(14, len(atr_values)):
            if atr_values[i] is not None:
                self.assertGreater(atr_values[i], 0)
    
    def test_atr_edge_cases(self):
        """Test ATR edge cases."""
        # Insufficient data
        self.assertEqual(VolatilityIndicators.atr([1], [1], [1], 14), [])
        
        # Mismatched lengths
        self.assertEqual(VolatilityIndicators.atr([1, 2], [1], [1, 2], 14), [])


class TestVolumeIndicators(unittest.TestCase):
    """Test cases for VolumeIndicators class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.closes = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        self.volumes = [1000, 1200, 1100, 1300, 1500, 1400, 1600, 1800, 1700, 1900]
        self.highs = [h + 1 for h in self.closes]
        self.lows = [l - 1 for l in self.closes]
    
    def test_obv_calculation(self):
        """Test On-Balance Volume calculation."""
        obv_values = VolumeIndicators.obv(self.closes, self.volumes)
        
        # Should start with 0
        self.assertEqual(obv_values[0], 0)
        
        # Check OBV logic
        for i in range(1, len(self.closes)):
            if self.closes[i] > self.closes[i-1]:
                # Price up: add volume
                expected = obv_values[i-1] + self.volumes[i]
                self.assertEqual(obv_values[i], expected)
            elif self.closes[i] < self.closes[i-1]:
                # Price down: subtract volume
                expected = obv_values[i-1] - self.volumes[i]
                self.assertEqual(obv_values[i], expected)
            else:
                # Price unchanged: OBV unchanged
                self.assertEqual(obv_values[i], obv_values[i-1])
    
    def test_obv_edge_cases(self):
        """Test OBV edge cases."""
        # Mismatched lengths
        self.assertEqual(VolumeIndicators.obv([1, 2], [100]), [])
        
        # Empty data
        self.assertEqual(VolumeIndicators.obv([], []), [])
    
    def test_ad_line_calculation(self):
        """Test Accumulation/Distribution Line calculation."""
        ad_values = VolumeIndicators.ad_line(self.highs, self.lows, self.closes, self.volumes)
        
        # Should start with calculated value (not 0)
        self.assertIsInstance(ad_values[0], (int, float))
        
        # Should have same length as input
        self.assertEqual(len(ad_values), len(self.closes))
    
    def test_ad_line_edge_cases(self):
        """Test A/D Line edge cases."""
        # Mismatched lengths
        self.assertEqual(VolumeIndicators.ad_line([1, 2], [1], [1, 2], [100, 200]), [])
        
        # Empty data
        self.assertEqual(VolumeIndicators.ad_line([], [], [], []), [])


class TestTechnicalAnalyzer(unittest.TestCase):
    """Test cases for TechnicalAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TechnicalAnalyzer()
        self.mock_generator = MockDataGenerator()
        self.market_data = self.mock_generator.generate_market_data(count=50)
    
    def test_analyze_market_data(self):
        """Test market data analysis."""
        indicators = ['sma_20', 'ema_12', 'rsi_14', 'macd', 'bollinger_bands']
        results = self.analyzer.analyze_market_data(self.market_data, indicators)
        
        # Should return results for all requested indicators
        for indicator in indicators:
            self.assertIn(indicator, results)
        
        # Check that results have correct length
        for indicator, values in results.items():
            if isinstance(values, list):
                self.assertEqual(len(values), len(self.market_data))
            elif isinstance(values, dict):
                for key, value_list in values.items():
                    self.assertEqual(len(value_list), len(self.market_data))
    
    def test_analyze_market_data_default_indicators(self):
        """Test market data analysis with default indicators."""
        results = self.analyzer.analyze_market_data(self.market_data)
        
        # Should include default indicators
        default_indicators = ['sma_20', 'ema_12', 'rsi_14', 'macd', 'bollinger_bands']
        for indicator in default_indicators:
            self.assertIn(indicator, results)
    
    def test_analyze_market_data_edge_cases(self):
        """Test market data analysis edge cases."""
        # Empty market data
        results = self.analyzer.analyze_market_data([])
        self.assertEqual(results, {})
        
        # Invalid indicator
        results = self.analyzer.analyze_market_data(self.market_data, ['invalid_indicator'])
        self.assertNotIn('invalid_indicator', results)
    
    def test_get_latest_values(self):
        """Test getting latest values from analysis results."""
        results = self.analyzer.analyze_market_data(self.market_data)
        latest_values = self.analyzer.get_latest_values(results)
        
        # Should return dictionary with latest values
        self.assertIsInstance(latest_values, dict)
        
        # Check that complex indicators are handled correctly
        if 'macd' in results and results['macd']:
            self.assertIn('macd', latest_values)
            self.assertIsInstance(latest_values['macd'], dict)
    
    def test_detect_signals(self):
        """Test signal detection from indicators."""
        # Create mock analysis results
        analysis_results = {
            'rsi_14': [None] * 13 + [75.0],  # Overbought
            'macd': {
                'macd': [None] * 25 + [0.1, -0.1],
                'signal': [None] * 25 + [-0.1, 0.1],
                'histogram': [None] * 25 + [0.2, -0.2]
            },
            'sma_20': [None] * 19 + [50000, 50100],
            'sma_50': [None] * 49 + [50200, 50000]
        }
        
        signals = self.analyzer.detect_signals(analysis_results)
        
        # Should detect RSI overbought signal
        self.assertIn('rsi', signals)
        self.assertEqual(signals['rsi'], 'SELL')
        
        # Should detect MACD bearish crossover
        self.assertIn('macd', signals)
        self.assertEqual(signals['macd'], 'SELL')
        
        # Should detect death cross
        self.assertIn('ma_cross', signals)
        self.assertEqual(signals['ma_cross'], 'SELL')
    
    def test_calculate_indicator_strength(self):
        """Test indicator strength calculation."""
        analysis_results = {
            'rsi_14': [None] * 13 + [80.0],  # Strong signal (far from 50)
            'macd': {
                'histogram': [None] * 25 + [5.0]  # Strong histogram
            },
            'adx': {
                'ADX': [None] * 14 + [40.0]  # Strong trend
            }
        }
        
        strengths = self.analyzer.calculate_indicator_strength(analysis_results)
        
        # Should calculate strength for available indicators
        self.assertIn('rsi', strengths)
        self.assertIn('macd', strengths)
        self.assertIn('adx', strengths)
        
        # Strengths should be between 0 and 1
        for strength in strengths.values():
            self.assertGreaterEqual(strength, 0.0)
            self.assertLessEqual(strength, 1.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_prices = create_price_series(50)
        self.mock_generator = MockDataGenerator()
    
    def test_calculate_rsi_function(self):
        """Test calculate_rsi convenience function."""
        rsi_values = calculate_rsi(self.test_prices, 14)
        
        # Should return list
        self.assertIsInstance(rsi_values, list)
        
        # Should have same length as input
        self.assertEqual(len(rsi_values), len(self.test_prices))
    
    def test_calculate_macd_function(self):
        """Test calculate_macd convenience function."""
        macd_result = calculate_macd(self.test_prices)
        
        # Should return dictionary
        self.assertIsInstance(macd_result, dict)
        
        # Should have required keys
        self.assertIn('macd', macd_result)
        self.assertIn('signal', macd_result)
        self.assertIn('histogram', macd_result)
    
    def test_calculate_sma_function(self):
        """Test calculate_sma convenience function."""
        sma_values = calculate_sma(self.test_prices, 20)
        
        # Should return list
        self.assertIsInstance(sma_values, list)
        
        # Should have same length as input
        self.assertEqual(len(sma_values), len(self.test_prices))
    
    def test_calculate_ema_function(self):
        """Test calculate_ema convenience function."""
        ema_values = calculate_ema(self.test_prices, 12)
        
        # Should return list
        self.assertIsInstance(ema_values, list)
        
        # Should have same length as input
        self.assertEqual(len(ema_values), len(self.test_prices))
    
    def test_calculate_bollinger_bands_function(self):
        """Test calculate_bollinger_bands convenience function."""
        bb_result = calculate_bollinger_bands(self.test_prices, 20)
        
        # Should return dictionary
        self.assertIsInstance(bb_result, dict)
        
        # Should have required keys
        self.assertIn('upper', bb_result)
        self.assertIn('middle', bb_result)
        self.assertIn('lower', bb_result)
    
    def test_analyze_market_data_function(self):
        """Test analyze_market_data convenience function."""
        market_data = self.mock_generator.generate_market_data(count=50)
        results = analyze_market_data(market_data)
        
        # Should return dictionary
        self.assertIsInstance(results, dict)
        
        # Should include default indicators
        default_indicators = ['sma_20', 'ema_12', 'rsi_14', 'macd', 'bollinger_bands']
        for indicator in default_indicators:
            self.assertIn(indicator, results)


class TestTechnicalIndicatorError(unittest.TestCase):
    """Test cases for TechnicalIndicatorError exception."""
    
    def test_exception_creation(self):
        """Test creating TechnicalIndicatorError."""
        error = TechnicalIndicatorError("Test error message")
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Test error message")
    
    def test_exception_inheritance(self):
        """Test that TechnicalIndicatorError inherits from Exception."""
        self.assertTrue(issubclass(TechnicalIndicatorError, Exception))


if __name__ == '__main__':
    unittest.main()