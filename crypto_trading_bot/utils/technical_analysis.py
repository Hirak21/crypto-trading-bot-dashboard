"""
Technical analysis indicators for trading strategies.

This module provides comprehensive technical indicator calculations
including trend, momentum, volatility, and volume indicators.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import deque
from datetime import datetime

from ..models.trading import MarketData


class TechnicalIndicatorError(Exception):
    """Custom exception for technical indicator errors."""
    pass


class MovingAverages:
    """Moving average calculations."""
    
    @staticmethod
    def sma(values: List[float], period: int) -> List[float]:
        """Simple Moving Average."""
        if not values or period <= 0 or period > len(values):
            return []
        
        sma_values = []
        for i in range(len(values)):
            if i < period - 1:
                sma_values.append(None)
            else:
                window_sum = sum(values[i - period + 1:i + 1])
                sma_values.append(window_sum / period)
        
        return sma_values
    
    @staticmethod
    def ema(values: List[float], period: int) -> List[float]:
        """Exponential Moving Average."""
        if not values or period <= 0:
            return []
        
        alpha = 2.0 / (period + 1)
        ema_values = []
        
        # First EMA value is the first price
        ema_values.append(values[0])
        
        for i in range(1, len(values)):
            ema_value = alpha * values[i] + (1 - alpha) * ema_values[i - 1]
            ema_values.append(ema_value)
        
        return ema_values
    
    @staticmethod
    def wma(values: List[float], period: int) -> List[float]:
        """Weighted Moving Average."""
        if not values or period <= 0 or period > len(values):
            return []
        
        wma_values = []
        weights = list(range(1, period + 1))
        weight_sum = sum(weights)
        
        for i in range(len(values)):
            if i < period - 1:
                wma_values.append(None)
            else:
                weighted_sum = sum(values[i - period + 1 + j] * weights[j] 
                                 for j in range(period))
                wma_values.append(weighted_sum / weight_sum)
        
        return wma_values
    
    @staticmethod
    def vwap(prices: List[float], volumes: List[float]) -> List[float]:
        """Volume Weighted Average Price."""
        if not prices or not volumes or len(prices) != len(volumes):
            return []
        
        vwap_values = []
        cumulative_pv = 0
        cumulative_volume = 0
        
        for i in range(len(prices)):
            cumulative_pv += prices[i] * volumes[i]
            cumulative_volume += volumes[i]
            
            if cumulative_volume > 0:
                vwap_values.append(cumulative_pv / cumulative_volume)
            else:
                vwap_values.append(prices[i])
        
        return vwap_values


class MomentumIndicators:
    """Momentum-based technical indicators."""
    
    @staticmethod
    def rsi(values: List[float], period: int = 14) -> List[float]:
        """Relative Strength Index."""
        if not values or period <= 0 or len(values) < period + 1:
            return []
        
        gains = []
        losses = []
        
        # Calculate price changes
        for i in range(1, len(values)):
            change = values[i] - values[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = [None] * period  # First 'period' values are None
        
        # Calculate RSI
        for i in range(period, len(gains)):
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
            
            # Update averages using Wilder's smoothing
            if i < len(gains) - 1:
                avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
                avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        
        return rsi_values
    
    @staticmethod
    def macd(values: List[float], fast_period: int = 12, slow_period: int = 26, 
             signal_period: int = 9) -> Dict[str, List[float]]:
        """Moving Average Convergence Divergence."""
        if not values or len(values) < slow_period:
            return {'macd': [], 'signal': [], 'histogram': []}
        
        # Calculate EMAs
        ema_fast = MovingAverages.ema(values, fast_period)
        ema_slow = MovingAverages.ema(values, slow_period)
        
        # Calculate MACD line
        macd_line = []
        for i in range(len(values)):
            if i < slow_period - 1:
                macd_line.append(None)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        # Calculate signal line (EMA of MACD)
        macd_values = [x for x in macd_line if x is not None]
        signal_line_values = MovingAverages.ema(macd_values, signal_period)
        
        # Pad signal line with None values
        signal_line = [None] * (slow_period - 1 + signal_period - 1)
        signal_line.extend(signal_line_values)
        
        # Calculate histogram
        histogram = []
        for i in range(len(macd_line)):
            if macd_line[i] is None or i >= len(signal_line) or signal_line[i] is None:
                histogram.append(None)
            else:
                histogram.append(macd_line[i] - signal_line[i])
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def roc(values: List[float], period: int = 12) -> List[float]:
        """Rate of Change."""
        if not values or period <= 0 or len(values) <= period:
            return []
        
        roc_values = [None] * period
        
        for i in range(period, len(values)):
            if values[i - period] != 0:
                roc = ((values[i] - values[i - period]) / values[i - period]) * 100
                roc_values.append(roc)
            else:
                roc_values.append(0)
        
        return roc_values
    
    @staticmethod
    def stochastic(highs: List[float], lows: List[float], closes: List[float], 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, List[float]]:
        """Stochastic Oscillator."""
        if not all([highs, lows, closes]) or len(highs) != len(lows) != len(closes):
            return {'%K': [], '%D': []}
        
        if len(closes) < k_period:
            return {'%K': [], '%D': []}
        
        k_values = [None] * (k_period - 1)
        
        for i in range(k_period - 1, len(closes)):
            highest_high = max(highs[i - k_period + 1:i + 1])
            lowest_low = min(lows[i - k_period + 1:i + 1])
            
            if highest_high == lowest_low:
                k_values.append(50)  # Neutral value when no range
            else:
                k = ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100
                k_values.append(k)
        
        # Calculate %D (SMA of %K)
        k_valid = [x for x in k_values if x is not None]
        d_values = MovingAverages.sma(k_valid, d_period)
        
        # Pad %D with None values
        d_padded = [None] * (k_period - 1 + d_period - 1)
        d_padded.extend(d_values)
        
        return {'%K': k_values, '%D': d_padded}


class TrendIndicators:
    """Trend-following indicators."""
    
    @staticmethod
    def adx(highs: List[float], lows: List[float], closes: List[float], 
            period: int = 14) -> Dict[str, List[float]]:
        """Average Directional Index."""
        if not all([highs, lows, closes]) or len(highs) != len(lows) != len(closes):
            return {'ADX': [], '+DI': [], '-DI': []}
        
        if len(closes) < period + 1:
            return {'ADX': [], '+DI': [], '-DI': []}
        
        # Calculate True Range and Directional Movement
        tr_values = []
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(closes)):
            # True Range
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i - 1])
            tr3 = abs(lows[i] - closes[i - 1])
            tr_values.append(max(tr1, tr2, tr3))
            
            # Directional Movement
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            
            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
        
        # Calculate smoothed values
        atr = TrendIndicators._wilder_smooth(tr_values, period)
        plus_di_smooth = TrendIndicators._wilder_smooth(plus_dm, period)
        minus_di_smooth = TrendIndicators._wilder_smooth(minus_dm, period)
        
        # Calculate DI values
        plus_di = []
        minus_di = []
        dx_values = []
        
        for i in range(len(atr)):
            if atr[i] != 0:
                plus_di.append((plus_di_smooth[i] / atr[i]) * 100)
                minus_di.append((minus_di_smooth[i] / atr[i]) * 100)
                
                # Calculate DX
                di_sum = plus_di[i] + minus_di[i]
                if di_sum != 0:
                    dx = (abs(plus_di[i] - minus_di[i]) / di_sum) * 100
                    dx_values.append(dx)
                else:
                    dx_values.append(0)
            else:
                plus_di.append(0)
                minus_di.append(0)
                dx_values.append(0)
        
        # Calculate ADX (smoothed DX)
        adx_values = TrendIndicators._wilder_smooth(dx_values, period)
        
        # Pad with None values
        padding = [None] * period
        
        return {
            'ADX': padding + adx_values,
            '+DI': padding + plus_di,
            '-DI': padding + minus_di
        }
    
    @staticmethod
    def _wilder_smooth(values: List[float], period: int) -> List[float]:
        """Wilder's smoothing method."""
        if not values or period <= 0:
            return []
        
        smoothed = []
        
        # First value is simple average
        if len(values) >= period:
            first_avg = sum(values[:period]) / period
            smoothed.append(first_avg)
            
            # Subsequent values use Wilder's smoothing
            for i in range(period, len(values)):
                wilder_value = ((smoothed[-1] * (period - 1)) + values[i]) / period
                smoothed.append(wilder_value)
        
        return smoothed
    
    @staticmethod
    def parabolic_sar(highs: List[float], lows: List[float], 
                      acceleration: float = 0.02, maximum: float = 0.2) -> List[float]:
        """Parabolic SAR."""
        if not highs or not lows or len(highs) != len(lows) or len(highs) < 2:
            return []
        
        sar_values = [None]  # First value is None
        
        # Initialize
        trend = 1  # 1 for uptrend, -1 for downtrend
        af = acceleration
        ep = highs[0] if trend == 1 else lows[0]  # Extreme point
        sar = lows[0] if trend == 1 else highs[0]
        
        for i in range(1, len(highs)):
            # Calculate SAR
            sar = sar + af * (ep - sar)
            
            # Check for trend reversal
            if trend == 1:  # Uptrend
                if lows[i] <= sar:
                    # Trend reversal to downtrend
                    trend = -1
                    sar = ep
                    ep = lows[i]
                    af = acceleration
                else:
                    # Continue uptrend
                    if highs[i] > ep:
                        ep = highs[i]
                        af = min(af + acceleration, maximum)
                    
                    # SAR cannot be above previous two lows
                    if i >= 2:
                        sar = min(sar, lows[i-1], lows[i-2])
                    elif i >= 1:
                        sar = min(sar, lows[i-1])
            
            else:  # Downtrend
                if highs[i] >= sar:
                    # Trend reversal to uptrend
                    trend = 1
                    sar = ep
                    ep = highs[i]
                    af = acceleration
                else:
                    # Continue downtrend
                    if lows[i] < ep:
                        ep = lows[i]
                        af = min(af + acceleration, maximum)
                    
                    # SAR cannot be below previous two highs
                    if i >= 2:
                        sar = max(sar, highs[i-1], highs[i-2])
                    elif i >= 1:
                        sar = max(sar, highs[i-1])
            
            sar_values.append(sar)
        
        return sar_values


class VolatilityIndicators:
    """Volatility-based indicators."""
    
    @staticmethod
    def bollinger_bands(values: List[float], period: int = 20, 
                       std_dev: float = 2.0) -> Dict[str, List[float]]:
        """Bollinger Bands."""
        if not values or period <= 0 or len(values) < period:
            return {'upper': [], 'middle': [], 'lower': []}
        
        # Calculate middle band (SMA)
        middle_band = MovingAverages.sma(values, period)
        
        # Calculate standard deviation
        upper_band = []
        lower_band = []
        
        for i in range(len(values)):
            if i < period - 1:
                upper_band.append(None)
                lower_band.append(None)
            else:
                # Calculate standard deviation for the period
                window_values = values[i - period + 1:i + 1]
                mean = sum(window_values) / period
                variance = sum((x - mean) ** 2 for x in window_values) / period
                std = math.sqrt(variance)
                
                upper_band.append(middle_band[i] + (std_dev * std))
                lower_band.append(middle_band[i] - (std_dev * std))
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }
    
    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], 
            period: int = 14) -> List[float]:
        """Average True Range."""
        if not all([highs, lows, closes]) or len(highs) != len(lows) != len(closes):
            return []
        
        if len(closes) < 2:
            return []
        
        # Calculate True Range
        tr_values = []
        for i in range(1, len(closes)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i - 1])
            tr3 = abs(lows[i] - closes[i - 1])
            tr_values.append(max(tr1, tr2, tr3))
        
        # Calculate ATR using Wilder's smoothing
        atr_values = TrendIndicators._wilder_smooth(tr_values, period)
        
        # Pad with None values
        padding = [None] * (period)
        return padding + atr_values


class VolumeIndicators:
    """Volume-based indicators."""
    
    @staticmethod
    def obv(closes: List[float], volumes: List[float]) -> List[float]:
        """On-Balance Volume."""
        if not closes or not volumes or len(closes) != len(volumes):
            return []
        
        obv_values = [0]  # Start with 0
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv_values.append(obv_values[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv_values.append(obv_values[-1] - volumes[i])
            else:
                obv_values.append(obv_values[-1])
        
        return obv_values
    
    @staticmethod
    def ad_line(highs: List[float], lows: List[float], closes: List[float], 
                volumes: List[float]) -> List[float]:
        """Accumulation/Distribution Line."""
        if not all([highs, lows, closes, volumes]):
            return []
        
        if not all(len(x) == len(closes) for x in [highs, lows, volumes]):
            return []
        
        ad_values = [0]  # Start with 0
        
        for i in range(len(closes)):
            if highs[i] == lows[i]:
                clv = 0  # Close Location Value
            else:
                clv = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i])
            
            if i == 0:
                ad_values[0] = clv * volumes[i]
            else:
                ad_values.append(ad_values[-1] + (clv * volumes[i]))
        
        return ad_values


class TechnicalAnalyzer:
    """Main technical analysis class that combines all indicators."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Indicator instances
        self.ma = MovingAverages()
        self.momentum = MomentumIndicators()
        self.trend = TrendIndicators()
        self.volatility = VolatilityIndicators()
        self.volume = VolumeIndicators()
    
    def analyze_market_data(self, market_data: List[MarketData], 
                           indicators: List[str] = None) -> Dict[str, Any]:
        """Analyze market data with specified indicators."""
        if not market_data:
            return {}
        
        try:
            # Extract price and volume data
            closes = [data.price for data in market_data]
            highs = [data.high_24h or data.price for data in market_data]
            lows = [data.low_24h or data.price for data in market_data]
            volumes = [data.volume for data in market_data]
            
            # Default indicators if none specified
            if indicators is None:
                indicators = ['sma_20', 'ema_12', 'rsi_14', 'macd', 'bollinger_bands']
            
            results = {}
            
            for indicator in indicators:
                try:
                    if indicator == 'sma_20':
                        results['sma_20'] = self.ma.sma(closes, 20)
                    elif indicator == 'sma_50':
                        results['sma_50'] = self.ma.sma(closes, 50)
                    elif indicator == 'ema_12':
                        results['ema_12'] = self.ma.ema(closes, 12)
                    elif indicator == 'ema_26':
                        results['ema_26'] = self.ma.ema(closes, 26)
                    elif indicator == 'rsi_14':
                        results['rsi_14'] = self.momentum.rsi(closes, 14)
                    elif indicator == 'macd':
                        results['macd'] = self.momentum.macd(closes)
                    elif indicator == 'roc_12':
                        results['roc_12'] = self.momentum.roc(closes, 12)
                    elif indicator == 'adx':
                        results['adx'] = self.trend.adx(highs, lows, closes)
                    elif indicator == 'bollinger_bands':
                        results['bollinger_bands'] = self.volatility.bollinger_bands(closes)
                    elif indicator == 'atr':
                        results['atr'] = self.volatility.atr(highs, lows, closes)
                    elif indicator == 'obv':
                        results['obv'] = self.volume.obv(closes, volumes)
                    elif indicator == 'vwap':
                        results['vwap'] = self.ma.vwap(closes, volumes)
                    
                except Exception as e:
                    self.logger.error(f"Error calculating {indicator}: {e}")
                    results[indicator] = []
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in market data analysis: {e}")
            return {}
    
    def get_latest_values(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Get the latest values from analysis results."""
        latest_values = {}
        
        for indicator, values in analysis_results.items():
            try:
                if isinstance(values, dict):
                    # Handle complex indicators like MACD, Bollinger Bands
                    latest_values[indicator] = {}
                    for key, value_list in values.items():
                        if value_list and value_list[-1] is not None:
                            latest_values[indicator][key] = value_list[-1]
                elif isinstance(values, list) and values:
                    # Handle simple indicators
                    if values[-1] is not None:
                        latest_values[indicator] = values[-1]
                        
            except Exception as e:
                self.logger.error(f"Error getting latest value for {indicator}: {e}")
        
        return latest_values
    
    def detect_signals(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Detect basic trading signals from technical indicators."""
        signals = {}
        
        try:
            # RSI signals
            if 'rsi_14' in analysis_results:
                rsi_values = analysis_results['rsi_14']
                if rsi_values and rsi_values[-1] is not None:
                    rsi = rsi_values[-1]
                    if rsi > 70:
                        signals['rsi'] = 'SELL'  # Overbought
                    elif rsi < 30:
                        signals['rsi'] = 'BUY'   # Oversold
                    else:
                        signals['rsi'] = 'HOLD'
            
            # MACD signals
            if 'macd' in analysis_results:
                macd_data = analysis_results['macd']
                if (macd_data.get('macd') and macd_data.get('signal') and
                    len(macd_data['macd']) >= 2 and len(macd_data['signal']) >= 2):
                    
                    macd_current = macd_data['macd'][-1]
                    macd_previous = macd_data['macd'][-2]
                    signal_current = macd_data['signal'][-1]
                    signal_previous = macd_data['signal'][-2]
                    
                    if (macd_current is not None and signal_current is not None and
                        macd_previous is not None and signal_previous is not None):
                        
                        # Bullish crossover
                        if macd_previous <= signal_previous and macd_current > signal_current:
                            signals['macd'] = 'BUY'
                        # Bearish crossover
                        elif macd_previous >= signal_previous and macd_current < signal_current:
                            signals['macd'] = 'SELL'
                        else:
                            signals['macd'] = 'HOLD'
            
            # Moving Average signals
            if 'sma_20' in analysis_results and 'sma_50' in analysis_results:
                sma_20 = analysis_results['sma_20']
                sma_50 = analysis_results['sma_50']
                
                if (sma_20 and sma_50 and len(sma_20) >= 2 and len(sma_50) >= 2 and
                    sma_20[-1] is not None and sma_50[-1] is not None and
                    sma_20[-2] is not None and sma_50[-2] is not None):
                    
                    # Golden cross
                    if sma_20[-2] <= sma_50[-2] and sma_20[-1] > sma_50[-1]:
                        signals['ma_cross'] = 'BUY'
                    # Death cross
                    elif sma_20[-2] >= sma_50[-2] and sma_20[-1] < sma_50[-1]:
                        signals['ma_cross'] = 'SELL'
                    else:
                        signals['ma_cross'] = 'HOLD'
            
        except Exception as e:
            self.logger.error(f"Error detecting signals: {e}")
        
        return signals
    
    def calculate_indicator_strength(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the strength of various indicators (0.0 to 1.0)."""
        strengths = {}
        
        try:
            # RSI strength
            if 'rsi_14' in analysis_results:
                rsi_values = analysis_results['rsi_14']
                if rsi_values and rsi_values[-1] is not None:
                    rsi = rsi_values[-1]
                    # Convert RSI to strength (distance from neutral 50)
                    strengths['rsi'] = abs(rsi - 50) / 50
            
            # MACD strength
            if 'macd' in analysis_results:
                macd_data = analysis_results['macd']
                if macd_data.get('histogram') and macd_data['histogram'][-1] is not None:
                    histogram = abs(macd_data['histogram'][-1])
                    # Normalize histogram value (this is a simplified approach)
                    strengths['macd'] = min(histogram / 10, 1.0)
            
            # ADX strength
            if 'adx' in analysis_results:
                adx_data = analysis_results['adx']
                if adx_data.get('ADX') and adx_data['ADX'][-1] is not None:
                    adx = adx_data['ADX'][-1]
                    # ADX above 25 indicates strong trend
                    strengths['adx'] = min(adx / 50, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicator strength: {e}")
        
        return strengths


# Global analyzer instance
analyzer = TechnicalAnalyzer()


# Convenience functions
def calculate_rsi(values: List[float], period: int = 14) -> List[float]:
    """Calculate RSI."""
    return MomentumIndicators.rsi(values, period)


def calculate_macd(values: List[float]) -> Dict[str, List[float]]:
    """Calculate MACD."""
    return MomentumIndicators.macd(values)


def calculate_sma(values: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average."""
    return MovingAverages.sma(values, period)


def calculate_ema(values: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average."""
    return MovingAverages.ema(values, period)


def calculate_bollinger_bands(values: List[float], period: int = 20) -> Dict[str, List[float]]:
    """Calculate Bollinger Bands."""
    return VolatilityIndicators.bollinger_bands(values, period)


def analyze_market_data(market_data: List[MarketData]) -> Dict[str, Any]:
    """Analyze market data with default indicators."""
    return analyzer.analyze_market_data(market_data)