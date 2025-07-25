"""
Pattern recognition utilities for chart and candlestick patterns.

This module provides comprehensive pattern detection algorithms
for technical analysis and trading signal generation.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from datetime import datetime
from enum import Enum

from ..models.trading import MarketData


class PatternType(Enum):
    """Pattern type enumeration."""
    # Chart Patterns
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    PENNANT = "pennant"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    
    # Candlestick Patterns
    DOJI = "doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    INVERTED_HAMMER = "inverted_hammer"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"


class PatternSignal(Enum):
    """Pattern signal enumeration."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PatternMatch(NamedTuple):
    """Pattern match result."""
    pattern_type: PatternType
    signal: PatternSignal
    confidence: float
    start_index: int
    end_index: int
    key_levels: Dict[str, float]
    metadata: Dict[str, Any]


class PeakTroughDetector:
    """Detects peaks and troughs in price data."""
    
    def __init__(self, min_distance: int = 5, threshold_pct: float = 1.0):
        self.min_distance = min_distance
        self.threshold_pct = threshold_pct / 100.0
        self.logger = logging.getLogger(__name__)
    
    def find_peaks_and_troughs(self, prices: List[float]) -> Dict[str, List[int]]:
        """Find peaks and troughs in price data."""
        if len(prices) < self.min_distance * 2 + 1:
            return {'peaks': [], 'troughs': []}
        
        peaks = []
        troughs = []
        
        for i in range(self.min_distance, len(prices) - self.min_distance):
            is_peak = True
            is_trough = True
            
            # Check if current point is a peak
            for j in range(i - self.min_distance, i + self.min_distance + 1):
                if j != i:
                    if prices[j] >= prices[i]:
                        is_peak = False
                    if prices[j] <= prices[i]:
                        is_trough = False
            
            # Verify significance
            if is_peak:
                # Check if peak is significant enough
                left_min = min(prices[max(0, i - self.min_distance):i])
                right_min = min(prices[i + 1:min(len(prices), i + self.min_distance + 1)])
                min_nearby = min(left_min, right_min)
                
                if (prices[i] - min_nearby) / min_nearby >= self.threshold_pct:
                    peaks.append(i)
            
            elif is_trough:
                # Check if trough is significant enough
                left_max = max(prices[max(0, i - self.min_distance):i])
                right_max = max(prices[i + 1:min(len(prices), i + self.min_distance + 1)])
                max_nearby = max(left_max, right_max)
                
                if (max_nearby - prices[i]) / max_nearby >= self.threshold_pct:
                    troughs.append(i)
        
        return {'peaks': peaks, 'troughs': troughs}
    
    def find_support_resistance(self, prices: List[float], 
                               tolerance_pct: float = 2.0) -> Dict[str, List[float]]:
        """Find support and resistance levels."""
        peaks_troughs = self.find_peaks_and_troughs(prices)
        
        # Combine all significant levels
        all_levels = []
        for peak_idx in peaks_troughs['peaks']:
            all_levels.append(prices[peak_idx])
        for trough_idx in peaks_troughs['troughs']:
            all_levels.append(prices[trough_idx])
        
        if not all_levels:
            return {'support': [], 'resistance': []}
        
        # Group similar levels
        tolerance = tolerance_pct / 100.0
        support_levels = []
        resistance_levels = []
        
        # Sort levels
        sorted_levels = sorted(all_levels)
        current_price = prices[-1]
        
        # Identify support (below current price) and resistance (above current price)
        for level in sorted_levels:
            if level < current_price * (1 - tolerance):
                support_levels.append(level)
            elif level > current_price * (1 + tolerance):
                resistance_levels.append(level)
        
        return {
            'support': support_levels[-3:] if len(support_levels) > 3 else support_levels,
            'resistance': resistance_levels[:3] if len(resistance_levels) > 3 else resistance_levels
        }


class ChartPatternDetector:
    """Detects chart patterns in price data."""
    
    def __init__(self):
        self.peak_trough_detector = PeakTroughDetector()
        self.logger = logging.getLogger(__name__)
    
    def detect_triangle_patterns(self, highs: List[float], lows: List[float], 
                                min_touches: int = 4) -> List[PatternMatch]:
        """Detect triangle patterns."""
        patterns = []
        
        if len(highs) < 20 or len(lows) < 20:
            return patterns
        
        try:
            # Find peaks and troughs
            high_peaks = self.peak_trough_detector.find_peaks_and_troughs(highs)['peaks']
            low_troughs = self.peak_trough_detector.find_peaks_and_troughs(lows)['troughs']
            
            if len(high_peaks) < 2 or len(low_troughs) < 2:
                return patterns
            
            # Analyze recent peaks and troughs
            recent_peaks = high_peaks[-4:] if len(high_peaks) >= 4 else high_peaks
            recent_troughs = low_troughs[-4:] if len(low_troughs) >= 4 else low_troughs
            
            if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
                # Calculate trend lines
                peak_slope = self._calculate_slope(recent_peaks, [highs[i] for i in recent_peaks])
                trough_slope = self._calculate_slope(recent_troughs, [lows[i] for i in recent_troughs])
                
                # Determine triangle type
                if abs(peak_slope) < 0.001 and trough_slope > 0.001:
                    # Ascending triangle
                    pattern = PatternMatch(
                        pattern_type=PatternType.TRIANGLE_ASCENDING,
                        signal=PatternSignal.BULLISH,
                        confidence=self._calculate_triangle_confidence(recent_peaks, recent_troughs, highs, lows),
                        start_index=min(recent_peaks[0], recent_troughs[0]),
                        end_index=max(recent_peaks[-1], recent_troughs[-1]),
                        key_levels={
                            'resistance': max(highs[i] for i in recent_peaks),
                            'support_slope': trough_slope
                        },
                        metadata={'peaks': recent_peaks, 'troughs': recent_troughs}
                    )
                    patterns.append(pattern)
                
                elif peak_slope < -0.001 and abs(trough_slope) < 0.001:
                    # Descending triangle
                    pattern = PatternMatch(
                        pattern_type=PatternType.TRIANGLE_DESCENDING,
                        signal=PatternSignal.BEARISH,
                        confidence=self._calculate_triangle_confidence(recent_peaks, recent_troughs, highs, lows),
                        start_index=min(recent_peaks[0], recent_troughs[0]),
                        end_index=max(recent_peaks[-1], recent_troughs[-1]),
                        key_levels={
                            'support': min(lows[i] for i in recent_troughs),
                            'resistance_slope': peak_slope
                        },
                        metadata={'peaks': recent_peaks, 'troughs': recent_troughs}
                    )
                    patterns.append(pattern)
                
                elif peak_slope < -0.001 and trough_slope > 0.001:
                    # Symmetrical triangle
                    pattern = PatternMatch(
                        pattern_type=PatternType.TRIANGLE_SYMMETRICAL,
                        signal=PatternSignal.NEUTRAL,
                        confidence=self._calculate_triangle_confidence(recent_peaks, recent_troughs, highs, lows),
                        start_index=min(recent_peaks[0], recent_troughs[0]),
                        end_index=max(recent_peaks[-1], recent_troughs[-1]),
                        key_levels={
                            'resistance_slope': peak_slope,
                            'support_slope': trough_slope
                        },
                        metadata={'peaks': recent_peaks, 'troughs': recent_troughs}
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error detecting triangle patterns: {e}")
        
        return patterns
    
    def detect_head_and_shoulders(self, highs: List[float], lows: List[float]) -> List[PatternMatch]:
        """Detect head and shoulders patterns."""
        patterns = []
        
        if len(highs) < 30:
            return patterns
        
        try:
            peaks = self.peak_trough_detector.find_peaks_and_troughs(highs)['peaks']
            troughs = self.peak_trough_detector.find_peaks_and_troughs(lows)['troughs']
            
            if len(peaks) < 3 or len(troughs) < 2:
                return patterns
            
            # Look for head and shoulders pattern in recent data
            for i in range(len(peaks) - 2):
                left_shoulder = peaks[i]
                head = peaks[i + 1]
                right_shoulder = peaks[i + 2]
                
                # Check if head is higher than shoulders
                if (highs[head] > highs[left_shoulder] and 
                    highs[head] > highs[right_shoulder]):
                    
                    # Find neckline (troughs between shoulders and head)
                    relevant_troughs = [t for t in troughs 
                                      if left_shoulder < t < right_shoulder]
                    
                    if len(relevant_troughs) >= 1:
                        neckline_level = min(lows[t] for t in relevant_troughs)
                        
                        # Calculate confidence
                        confidence = self._calculate_hs_confidence(
                            left_shoulder, head, right_shoulder, 
                            relevant_troughs, highs, lows
                        )
                        
                        if confidence > 0.5:
                            pattern = PatternMatch(
                                pattern_type=PatternType.HEAD_AND_SHOULDERS,
                                signal=PatternSignal.BEARISH,
                                confidence=confidence,
                                start_index=left_shoulder,
                                end_index=right_shoulder,
                                key_levels={
                                    'neckline': neckline_level,
                                    'head_level': highs[head],
                                    'target': neckline_level - (highs[head] - neckline_level)
                                },
                                metadata={
                                    'left_shoulder': left_shoulder,
                                    'head': head,
                                    'right_shoulder': right_shoulder,
                                    'troughs': relevant_troughs
                                }
                            )
                            patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error detecting head and shoulders: {e}")
        
        return patterns
    
    def detect_double_patterns(self, highs: List[float], lows: List[float]) -> List[PatternMatch]:
        """Detect double top and double bottom patterns."""
        patterns = []
        
        try:
            peaks = self.peak_trough_detector.find_peaks_and_troughs(highs)['peaks']
            troughs = self.peak_trough_detector.find_peaks_and_troughs(lows)['troughs']
            
            # Double top detection
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    peak1 = peaks[i]
                    peak2 = peaks[i + 1]
                    
                    # Check if peaks are similar in height
                    height_diff = abs(highs[peak1] - highs[peak2]) / max(highs[peak1], highs[peak2])
                    
                    if height_diff < 0.03:  # Within 3%
                        # Find valley between peaks
                        valley_troughs = [t for t in troughs if peak1 < t < peak2]
                        
                        if valley_troughs:
                            valley_level = min(lows[t] for t in valley_troughs)
                            peak_level = (highs[peak1] + highs[peak2]) / 2
                            
                            confidence = self._calculate_double_pattern_confidence(
                                peak1, peak2, valley_troughs, highs, lows, True
                            )
                            
                            if confidence > 0.6:
                                pattern = PatternMatch(
                                    pattern_type=PatternType.DOUBLE_TOP,
                                    signal=PatternSignal.BEARISH,
                                    confidence=confidence,
                                    start_index=peak1,
                                    end_index=peak2,
                                    key_levels={
                                        'resistance': peak_level,
                                        'support': valley_level,
                                        'target': valley_level - (peak_level - valley_level)
                                    },
                                    metadata={'peaks': [peak1, peak2], 'valley': valley_troughs}
                                )
                                patterns.append(pattern)
            
            # Double bottom detection
            if len(troughs) >= 2:
                for i in range(len(troughs) - 1):
                    trough1 = troughs[i]
                    trough2 = troughs[i + 1]
                    
                    # Check if troughs are similar in depth
                    depth_diff = abs(lows[trough1] - lows[trough2]) / max(lows[trough1], lows[trough2])
                    
                    if depth_diff < 0.03:  # Within 3%
                        # Find peak between troughs
                        peak_highs = [p for p in peaks if trough1 < p < trough2]
                        
                        if peak_highs:
                            peak_level = max(highs[p] for p in peak_highs)
                            trough_level = (lows[trough1] + lows[trough2]) / 2
                            
                            confidence = self._calculate_double_pattern_confidence(
                                trough1, trough2, peak_highs, lows, highs, False
                            )
                            
                            if confidence > 0.6:
                                pattern = PatternMatch(
                                    pattern_type=PatternType.DOUBLE_BOTTOM,
                                    signal=PatternSignal.BULLISH,
                                    confidence=confidence,
                                    start_index=trough1,
                                    end_index=trough2,
                                    key_levels={
                                        'support': trough_level,
                                        'resistance': peak_level,
                                        'target': peak_level + (peak_level - trough_level)
                                    },
                                    metadata={'troughs': [trough1, trough2], 'peak': peak_highs}
                                )
                                patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error detecting double patterns: {e}")
        
        return patterns
    
    def _calculate_slope(self, x_values: List[int], y_values: List[float]) -> float:
        """Calculate slope of trend line."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _calculate_triangle_confidence(self, peaks: List[int], troughs: List[int],
                                     highs: List[float], lows: List[float]) -> float:
        """Calculate confidence for triangle pattern."""
        try:
            # Base confidence on number of touches and trend line fit
            total_touches = len(peaks) + len(troughs)
            touch_score = min(total_touches / 6.0, 1.0)  # Max score at 6 touches
            
            # Calculate R-squared for trend lines (simplified)
            peak_r2 = self._calculate_r_squared(peaks, [highs[i] for i in peaks])
            trough_r2 = self._calculate_r_squared(troughs, [lows[i] for i in troughs])
            
            fit_score = (peak_r2 + trough_r2) / 2
            
            return (touch_score * 0.6 + fit_score * 0.4)
            
        except Exception:
            return 0.5
    
    def _calculate_hs_confidence(self, left_shoulder: int, head: int, right_shoulder: int,
                               troughs: List[int], highs: List[float], lows: List[float]) -> float:
        """Calculate confidence for head and shoulders pattern."""
        try:
            # Check symmetry
            left_distance = head - left_shoulder
            right_distance = right_shoulder - head
            symmetry_score = 1.0 - abs(left_distance - right_distance) / max(left_distance, right_distance)
            
            # Check shoulder height similarity
            shoulder_diff = abs(highs[left_shoulder] - highs[right_shoulder])
            avg_shoulder = (highs[left_shoulder] + highs[right_shoulder]) / 2
            shoulder_score = 1.0 - (shoulder_diff / avg_shoulder)
            
            # Check head prominence
            head_prominence = (highs[head] - avg_shoulder) / avg_shoulder
            prominence_score = min(head_prominence / 0.1, 1.0)  # 10% prominence gives max score
            
            return (symmetry_score * 0.4 + shoulder_score * 0.3 + prominence_score * 0.3)
            
        except Exception:
            return 0.5
    
    def _calculate_double_pattern_confidence(self, point1: int, point2: int, middle_points: List[int],
                                           primary_values: List[float], secondary_values: List[float],
                                           is_top: bool) -> float:
        """Calculate confidence for double top/bottom patterns."""
        try:
            # Check similarity of the two peaks/troughs
            similarity = 1.0 - abs(primary_values[point1] - primary_values[point2]) / max(primary_values[point1], primary_values[point2])
            
            # Check distance between points (should be reasonable)
            distance = abs(point2 - point1)
            distance_score = min(distance / 20, 1.0) if distance > 5 else distance / 5
            
            # Check valley/peak between the two points
            if middle_points:
                middle_value = max(secondary_values[p] for p in middle_points) if is_top else min(secondary_values[p] for p in middle_points)
                avg_primary = (primary_values[point1] + primary_values[point2]) / 2
                separation = abs(middle_value - avg_primary) / avg_primary
                separation_score = min(separation / 0.05, 1.0)  # 5% separation gives max score
            else:
                separation_score = 0.0
            
            return (similarity * 0.5 + distance_score * 0.2 + separation_score * 0.3)
            
        except Exception:
            return 0.5
    
    def _calculate_r_squared(self, x_values: List[int], y_values: List[float]) -> float:
        """Calculate R-squared for trend line fit."""
        if len(x_values) < 2:
            return 0.0
        
        try:
            # Calculate slope and intercept
            slope = self._calculate_slope(x_values, y_values)
            mean_y = sum(y_values) / len(y_values)
            mean_x = sum(x_values) / len(x_values)
            intercept = mean_y - slope * mean_x
            
            # Calculate R-squared
            ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))
            ss_tot = sum((y - mean_y) ** 2 for y in y_values)
            
            if ss_tot == 0:
                return 1.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, r_squared)
            
        except Exception:
            return 0.0


class CandlestickPatternDetector:
    """Detects candlestick patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_single_candlestick_patterns(self, market_data: List[MarketData]) -> List[PatternMatch]:
        """Detect single candlestick patterns."""
        patterns = []
        
        if not market_data:
            return patterns
        
        for i, data in enumerate(market_data):
            try:
                # Extract OHLC data (using price as close, estimating others)
                open_price = data.price * 0.999  # Estimate
                high_price = data.high_24h or data.price * 1.001
                low_price = data.low_24h or data.price * 0.999
                close_price = data.price
                
                # Calculate body and shadow sizes
                body_size = abs(close_price - open_price)
                upper_shadow = high_price - max(open_price, close_price)
                lower_shadow = min(open_price, close_price) - low_price
                total_range = high_price - low_price
                
                if total_range == 0:
                    continue
                
                # Doji pattern
                if body_size / total_range < 0.1:
                    confidence = 1.0 - (body_size / total_range) / 0.1
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.DOJI,
                        signal=PatternSignal.NEUTRAL,
                        confidence=confidence,
                        start_index=i,
                        end_index=i,
                        key_levels={'price': close_price},
                        metadata={'body_ratio': body_size / total_range}
                    ))
                
                # Hammer pattern
                elif (lower_shadow > body_size * 2 and 
                      upper_shadow < body_size * 0.5 and
                      body_size / total_range > 0.1):
                    
                    confidence = min(lower_shadow / (body_size * 2), 1.0) * 0.8
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.HAMMER,
                        signal=PatternSignal.BULLISH,
                        confidence=confidence,
                        start_index=i,
                        end_index=i,
                        key_levels={'support': low_price},
                        metadata={'lower_shadow_ratio': lower_shadow / total_range}
                    ))
                
                # Shooting Star pattern
                elif (upper_shadow > body_size * 2 and 
                      lower_shadow < body_size * 0.5 and
                      body_size / total_range > 0.1):
                    
                    confidence = min(upper_shadow / (body_size * 2), 1.0) * 0.8
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.SHOOTING_STAR,
                        signal=PatternSignal.BEARISH,
                        confidence=confidence,
                        start_index=i,
                        end_index=i,
                        key_levels={'resistance': high_price},
                        metadata={'upper_shadow_ratio': upper_shadow / total_range}
                    ))
            
            except Exception as e:
                self.logger.error(f"Error detecting single candlestick pattern at index {i}: {e}")
        
        return patterns
    
    def detect_two_candlestick_patterns(self, market_data: List[MarketData]) -> List[PatternMatch]:
        """Detect two-candlestick patterns."""
        patterns = []
        
        if len(market_data) < 2:
            return patterns
        
        for i in range(1, len(market_data)):
            try:
                # Get current and previous candle data
                prev_data = market_data[i - 1]
                curr_data = market_data[i]
                
                # Extract OHLC for both candles
                prev_open = prev_data.price * 0.999
                prev_close = prev_data.price
                prev_high = prev_data.high_24h or prev_data.price * 1.001
                prev_low = prev_data.low_24h or prev_data.price * 0.999
                
                curr_open = curr_data.price * 0.999
                curr_close = curr_data.price
                curr_high = curr_data.high_24h or curr_data.price * 1.001
                curr_low = curr_data.low_24h or curr_data.price * 0.999
                
                # Bullish Engulfing
                if (prev_close < prev_open and  # Previous candle is bearish
                    curr_close > curr_open and  # Current candle is bullish
                    curr_open < prev_close and  # Current opens below previous close
                    curr_close > prev_open):    # Current closes above previous open
                    
                    engulfing_ratio = (curr_close - curr_open) / (prev_open - prev_close)
                    confidence = min(engulfing_ratio / 1.5, 1.0) * 0.8
                    
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.ENGULFING_BULLISH,
                        signal=PatternSignal.BULLISH,
                        confidence=confidence,
                        start_index=i - 1,
                        end_index=i,
                        key_levels={'support': min(prev_low, curr_low)},
                        metadata={'engulfing_ratio': engulfing_ratio}
                    ))
                
                # Bearish Engulfing
                elif (prev_close > prev_open and  # Previous candle is bullish
                      curr_close < curr_open and  # Current candle is bearish
                      curr_open > prev_close and  # Current opens above previous close
                      curr_close < prev_open):    # Current closes below previous open
                    
                    engulfing_ratio = (curr_open - curr_close) / (prev_close - prev_open)
                    confidence = min(engulfing_ratio / 1.5, 1.0) * 0.8
                    
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.ENGULFING_BEARISH,
                        signal=PatternSignal.BEARISH,
                        confidence=confidence,
                        start_index=i - 1,
                        end_index=i,
                        key_levels={'resistance': max(prev_high, curr_high)},
                        metadata={'engulfing_ratio': engulfing_ratio}
                    ))
                
                # Piercing Line
                elif (prev_close < prev_open and  # Previous candle is bearish
                      curr_close > curr_open and  # Current candle is bullish
                      curr_open < prev_low and    # Current opens below previous low
                      curr_close > (prev_open + prev_close) / 2):  # Closes above midpoint
                    
                    penetration = (curr_close - prev_close) / (prev_open - prev_close)
                    confidence = min(penetration / 0.7, 1.0) * 0.7
                    
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.PIERCING_LINE,
                        signal=PatternSignal.BULLISH,
                        confidence=confidence,
                        start_index=i - 1,
                        end_index=i,
                        key_levels={'support': curr_open},
                        metadata={'penetration_ratio': penetration}
                    ))
            
            except Exception as e:
                self.logger.error(f"Error detecting two-candlestick pattern at index {i}: {e}")
        
        return patterns
    
    def detect_three_candlestick_patterns(self, market_data: List[MarketData]) -> List[PatternMatch]:
        """Detect three-candlestick patterns."""
        patterns = []
        
        if len(market_data) < 3:
            return patterns
        
        for i in range(2, len(market_data)):
            try:
                # Get three candles
                candles = market_data[i-2:i+1]
                
                # Extract basic price data
                opens = [c.price * 0.999 for c in candles]
                closes = [c.price for c in candles]
                highs = [c.high_24h or c.price * 1.001 for c in candles]
                lows = [c.low_24h or c.price * 0.999 for c in candles]
                
                # Morning Star pattern
                if (closes[0] < opens[0] and  # First candle is bearish
                    abs(closes[1] - opens[1]) < (closes[0] - opens[0]) * 0.3 and  # Second is small
                    closes[2] > opens[2] and  # Third is bullish
                    closes[2] > (opens[0] + closes[0]) / 2):  # Third closes above first's midpoint
                    
                    confidence = self._calculate_star_confidence(opens, closes, highs, lows, True)
                    
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.MORNING_STAR,
                        signal=PatternSignal.BULLISH,
                        confidence=confidence,
                        start_index=i - 2,
                        end_index=i,
                        key_levels={'support': min(lows)},
                        metadata={'star_body_ratio': abs(closes[1] - opens[1]) / abs(closes[0] - opens[0])}
                    ))
                
                # Evening Star pattern
                elif (closes[0] > opens[0] and  # First candle is bullish
                      abs(closes[1] - opens[1]) < (opens[0] - closes[0]) * 0.3 and  # Second is small
                      closes[2] < opens[2] and  # Third is bearish
                      closes[2] < (opens[0] + closes[0]) / 2):  # Third closes below first's midpoint
                    
                    confidence = self._calculate_star_confidence(opens, closes, highs, lows, False)
                    
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.EVENING_STAR,
                        signal=PatternSignal.BEARISH,
                        confidence=confidence,
                        start_index=i - 2,
                        end_index=i,
                        key_levels={'resistance': max(highs)},
                        metadata={'star_body_ratio': abs(closes[1] - opens[1]) / abs(opens[0] - closes[0])}
                    ))
            
            except Exception as e:
                self.logger.error(f"Error detecting three-candlestick pattern at index {i}: {e}")
        
        return patterns
    
    def _calculate_star_confidence(self, opens: List[float], closes: List[float],
                                 highs: List[float], lows: List[float], is_morning: bool) -> float:
        """Calculate confidence for star patterns."""
        try:
            # Check gap between first and second candle
            if is_morning:
                gap_score = 1.0 if max(opens[1], closes[1]) < min(opens[0], closes[0]) else 0.5
            else:
                gap_score = 1.0 if min(opens[1], closes[1]) > max(opens[0], closes[0]) else 0.5
            
            # Check size of star (should be small)
            star_size = abs(closes[1] - opens[1])
            first_size = abs(closes[0] - opens[0])
            size_score = 1.0 - min(star_size / first_size, 1.0)
            
            # Check third candle penetration
            if is_morning:
                penetration = (closes[2] - min(opens[0], closes[0])) / abs(opens[0] - closes[0])
            else:
                penetration = (max(opens[0], closes[0]) - closes[2]) / abs(opens[0] - closes[0])
            
            penetration_score = min(penetration / 0.6, 1.0)
            
            return (gap_score * 0.3 + size_score * 0.3 + penetration_score * 0.4) * 0.8
            
        except Exception:
            return 0.5


class PatternRecognizer:
    """Main pattern recognition class."""
    
    def __init__(self):
        self.chart_detector = ChartPatternDetector()
        self.candlestick_detector = CandlestickPatternDetector()
        self.logger = logging.getLogger(__name__)
    
    def analyze_patterns(self, market_data: List[MarketData], 
                        pattern_types: List[str] = None) -> List[PatternMatch]:
        """Analyze market data for patterns."""
        if not market_data:
            return []
        
        all_patterns = []
        
        try:
            # Extract price data for chart patterns
            highs = [data.high_24h or data.price for data in market_data]
            lows = [data.low_24h or data.price for data in market_data]
            
            # Detect chart patterns
            if not pattern_types or any('triangle' in pt for pt in pattern_types):
                all_patterns.extend(self.chart_detector.detect_triangle_patterns(highs, lows))
            
            if not pattern_types or 'head_and_shoulders' in pattern_types:
                all_patterns.extend(self.chart_detector.detect_head_and_shoulders(highs, lows))
            
            if not pattern_types or any('double' in pt for pt in pattern_types):
                all_patterns.extend(self.chart_detector.detect_double_patterns(highs, lows))
            
            # Detect candlestick patterns
            if not pattern_types or any(pt in ['doji', 'hammer', 'shooting_star'] for pt in pattern_types):
                all_patterns.extend(self.candlestick_detector.detect_single_candlestick_patterns(market_data))
            
            if not pattern_types or any('engulfing' in pt or 'piercing' in pt for pt in pattern_types):
                all_patterns.extend(self.candlestick_detector.detect_two_candlestick_patterns(market_data))
            
            if not pattern_types or any('star' in pt for pt in pattern_types):
                all_patterns.extend(self.candlestick_detector.detect_three_candlestick_patterns(market_data))
            
            # Sort by confidence and recency
            all_patterns.sort(key=lambda p: (p.end_index, p.confidence), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {e}")
        
        return all_patterns
    
    def get_pattern_signals(self, patterns: List[PatternMatch], 
                           min_confidence: float = 0.6) -> Dict[str, Any]:
        """Get trading signals from detected patterns."""
        signals = {
            'bullish_patterns': [],
            'bearish_patterns': [],
            'neutral_patterns': [],
            'overall_signal': PatternSignal.NEUTRAL,
            'confidence': 0.0
        }
        
        try:
            # Filter patterns by confidence
            high_confidence_patterns = [p for p in patterns if p.confidence >= min_confidence]
            
            # Categorize patterns
            bullish_score = 0.0
            bearish_score = 0.0
            
            for pattern in high_confidence_patterns:
                if pattern.signal == PatternSignal.BULLISH:
                    signals['bullish_patterns'].append(pattern)
                    bullish_score += pattern.confidence
                elif pattern.signal == PatternSignal.BEARISH:
                    signals['bearish_patterns'].append(pattern)
                    bearish_score += pattern.confidence
                else:
                    signals['neutral_patterns'].append(pattern)
            
            # Determine overall signal
            if bullish_score > bearish_score * 1.2:
                signals['overall_signal'] = PatternSignal.BULLISH
                signals['confidence'] = bullish_score / (bullish_score + bearish_score) if bearish_score > 0 else 1.0
            elif bearish_score > bullish_score * 1.2:
                signals['overall_signal'] = PatternSignal.BEARISH
                signals['confidence'] = bearish_score / (bullish_score + bearish_score) if bullish_score > 0 else 1.0
            else:
                signals['overall_signal'] = PatternSignal.NEUTRAL
                signals['confidence'] = 0.5
            
        except Exception as e:
            self.logger.error(f"Error getting pattern signals: {e}")
        
        return signals


# Global recognizer instance
recognizer = PatternRecognizer()


# Convenience functions
def detect_patterns(market_data: List[MarketData]) -> List[PatternMatch]:
    """Detect patterns in market data."""
    return recognizer.analyze_patterns(market_data)


def get_pattern_signals(market_data: List[MarketData]) -> Dict[str, Any]:
    """Get pattern-based trading signals."""
    patterns = recognizer.analyze_patterns(market_data)
    return recognizer.get_pattern_signals(patterns)