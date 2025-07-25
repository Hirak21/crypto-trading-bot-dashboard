"""
Chart pattern recognition trading strategy.

This strategy identifies and trades based on classic chart patterns including
triangles, head and shoulders, flags, pennants, and wedges.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from datetime import datetime, timedelta
from collections import deque
import statistics
import math

from ..strategies.base_strategy import BaseStrategy
from ..models.trading import TradingSignal, MarketData, SignalAction
from ..utils.technical_analysis import MovingAverages, TrendIndicators


class PatternPoint(NamedTuple):
    """Represents a significant point in price data."""
    timestamp: datetime
    price: float
    index: int
    point_type: str  # 'peak', 'trough', 'support', 'resistance'


class PatternMatch:
    """Represents a detected chart pattern."""
    
    def __init__(self, pattern_type: str, points: List[PatternPoint], 
                 confidence: float, breakout_target: float, stop_loss: float):
        self.pattern_type = pattern_type
        self.points = points
        self.confidence = confidence
        self.breakout_target = breakout_target
        self.stop_loss = stop_loss
        self.detected_at = datetime.now()
        self.is_active = True
        
        # Calculate pattern metrics
        self.height = self._calculate_pattern_height()
        self.width = self._calculate_pattern_width()
        self.slope = self._calculate_pattern_slope()
    
    def _calculate_pattern_height(self) -> float:
        """Calculate the height of the pattern."""
        if not self.points:
            return 0.0
        
        prices = [p.price for p in self.points]
        return max(prices) - min(prices)
    
    def _calculate_pattern_width(self) -> int:
        """Calculate the width (duration) of the pattern."""
        if len(self.points) < 2:
            return 0
        
        return self.points[-1].index - self.points[0].index
    
    def _calculate_pattern_slope(self) -> float:
        """Calculate the overall slope of the pattern."""
        if len(self.points) < 2:
            return 0.0
        
        start_point = self.points[0]
        end_point = self.points[-1]
        
        if end_point.index == start_point.index:
            return 0.0
        
        return (end_point.price - start_point.price) / (end_point.index - start_point.index)


class TrianglePatternDetector:
    """Detects triangle patterns (ascending, descending, symmetrical)."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_pattern_length = 20
        self.min_touches = 4  # Minimum touches for trend lines
        self.tolerance = 0.02  # 2% tolerance for trend line validation
    
    def detect_triangles(self, price_data: List[float], 
                        timestamps: List[datetime]) -> List[PatternMatch]:
        """Detect triangle patterns in price data."""
        patterns = []
        
        try:
            if len(price_data) < self.min_pattern_length:
                return patterns
            
            # Find significant peaks and troughs
            peaks, troughs = self._find_peaks_and_troughs(price_data)
            
            # Detect ascending triangles
            ascending = self._detect_ascending_triangle(price_data, peaks, troughs, timestamps)
            if ascending:
                patterns.append(ascending)
            
            # Detect descending triangles
            descending = self._detect_descending_triangle(price_data, peaks, troughs, timestamps)
            if descending:
                patterns.append(descending)
            
            # Detect symmetrical triangles
            symmetrical = self._detect_symmetrical_triangle(price_data, peaks, troughs, timestamps)
            if symmetrical:
                patterns.append(symmetrical)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting triangle patterns: {e}")
            return []
    
    def _find_peaks_and_troughs(self, price_data: List[float]) -> Tuple[List[int], List[int]]:
        """Find significant peaks and troughs in price data."""
        peaks = []
        troughs = []
        
        try:
            window = 5  # Look-ahead/behind window
            
            for i in range(window, len(price_data) - window):
                # Check for peak
                is_peak = all(price_data[i] >= price_data[j] for j in range(i - window, i + window + 1) if j != i)
                if is_peak and price_data[i] > price_data[i-1] and price_data[i] > price_data[i+1]:
                    peaks.append(i)
                
                # Check for trough
                is_trough = all(price_data[i] <= price_data[j] for j in range(i - window, i + window + 1) if j != i)
                if is_trough and price_data[i] < price_data[i-1] and price_data[i] < price_data[i+1]:
                    troughs.append(i)
            
            return peaks, troughs
            
        except Exception as e:
            self.logger.error(f"Error finding peaks and troughs: {e}")
            return [], []
    
    def _detect_ascending_triangle(self, price_data: List[float], peaks: List[int], 
                                  troughs: List[int], timestamps: List[datetime]) -> Optional[PatternMatch]:
        """Detect ascending triangle pattern."""
        try:
            if len(peaks) < 2 or len(troughs) < 2:
                return None
            
            # Get recent peaks and troughs
            recent_peaks = peaks[-4:] if len(peaks) >= 4 else peaks
            recent_troughs = troughs[-4:] if len(troughs) >= 4 else troughs
            
            # Check for horizontal resistance (peaks at similar level)
            peak_prices = [price_data[i] for i in recent_peaks]
            resistance_level = statistics.mean(peak_prices)
            
            # Validate horizontal resistance
            resistance_valid = all(
                abs(price - resistance_level) / resistance_level <= self.tolerance 
                for price in peak_prices
            )
            
            if not resistance_valid:
                return None
            
            # Check for ascending support (rising troughs)
            if len(recent_troughs) < 2:
                return None
            
            trough_prices = [price_data[i] for i in recent_troughs]
            
            # Calculate support line slope
            support_slope = self._calculate_trend_line_slope(recent_troughs, trough_prices)
            
            # Ascending triangle should have positive support slope
            if support_slope <= 0:
                return None
            
            # Create pattern points
            pattern_points = []
            for i, peak_idx in enumerate(recent_peaks):
                pattern_points.append(PatternPoint(
                    timestamps[peak_idx], price_data[peak_idx], peak_idx, 'resistance'
                ))
            
            for i, trough_idx in enumerate(recent_troughs):
                pattern_points.append(PatternPoint(
                    timestamps[trough_idx], price_data[trough_idx], trough_idx, 'support'
                ))
            
            # Sort by index
            pattern_points.sort(key=lambda x: x.index)
            
            # Calculate breakout target and stop loss
            pattern_height = resistance_level - min(trough_prices)
            breakout_target = resistance_level + pattern_height
            stop_loss = min(trough_prices) - (pattern_height * 0.1)
            
            # Calculate confidence based on pattern quality
            confidence = self._calculate_triangle_confidence(
                pattern_points, resistance_level, support_slope, 'ascending'
            )
            
            return PatternMatch(
                'ascending_triangle',
                pattern_points,
                confidence,
                breakout_target,
                stop_loss
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting ascending triangle: {e}")
            return None
    
    def _detect_descending_triangle(self, price_data: List[float], peaks: List[int], 
                                   troughs: List[int], timestamps: List[datetime]) -> Optional[PatternMatch]:
        """Detect descending triangle pattern."""
        try:
            if len(peaks) < 2 or len(troughs) < 2:
                return None
            
            # Get recent peaks and troughs
            recent_peaks = peaks[-4:] if len(peaks) >= 4 else peaks
            recent_troughs = troughs[-4:] if len(troughs) >= 4 else troughs
            
            # Check for horizontal support (troughs at similar level)
            trough_prices = [price_data[i] for i in recent_troughs]
            support_level = statistics.mean(trough_prices)
            
            # Validate horizontal support
            support_valid = all(
                abs(price - support_level) / support_level <= self.tolerance 
                for price in trough_prices
            )
            
            if not support_valid:
                return None
            
            # Check for descending resistance (falling peaks)
            if len(recent_peaks) < 2:
                return None
            
            peak_prices = [price_data[i] for i in recent_peaks]
            
            # Calculate resistance line slope
            resistance_slope = self._calculate_trend_line_slope(recent_peaks, peak_prices)
            
            # Descending triangle should have negative resistance slope
            if resistance_slope >= 0:
                return None
            
            # Create pattern points
            pattern_points = []
            for i, peak_idx in enumerate(recent_peaks):
                pattern_points.append(PatternPoint(
                    timestamps[peak_idx], price_data[peak_idx], peak_idx, 'resistance'
                ))
            
            for i, trough_idx in enumerate(recent_troughs):
                pattern_points.append(PatternPoint(
                    timestamps[trough_idx], price_data[trough_idx], trough_idx, 'support'
                ))
            
            # Sort by index
            pattern_points.sort(key=lambda x: x.index)
            
            # Calculate breakout target and stop loss
            pattern_height = max(peak_prices) - support_level
            breakout_target = support_level - pattern_height
            stop_loss = max(peak_prices) + (pattern_height * 0.1)
            
            # Calculate confidence
            confidence = self._calculate_triangle_confidence(
                pattern_points, support_level, resistance_slope, 'descending'
            )
            
            return PatternMatch(
                'descending_triangle',
                pattern_points,
                confidence,
                breakout_target,
                stop_loss
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting descending triangle: {e}")
            return None    

    def _detect_symmetrical_triangle(self, price_data: List[float], peaks: List[int], 
                                    troughs: List[int], timestamps: List[datetime]) -> Optional[PatternMatch]:
        """Detect symmetrical triangle pattern."""
        try:
            if len(peaks) < 2 or len(troughs) < 2:
                return None
            
            # Get recent peaks and troughs
            recent_peaks = peaks[-4:] if len(peaks) >= 4 else peaks
            recent_troughs = troughs[-4:] if len(troughs) >= 4 else troughs
            
            # Calculate trend line slopes
            peak_prices = [price_data[i] for i in recent_peaks]
            trough_prices = [price_data[i] for i in recent_troughs]
            
            resistance_slope = self._calculate_trend_line_slope(recent_peaks, peak_prices)
            support_slope = self._calculate_trend_line_slope(recent_troughs, trough_prices)
            
            # Symmetrical triangle: descending resistance + ascending support
            if resistance_slope >= 0 or support_slope <= 0:
                return None
            
            # Check if slopes are converging (similar magnitude, opposite direction)
            slope_ratio = abs(resistance_slope / support_slope) if support_slope != 0 else 0
            if not (0.5 <= slope_ratio <= 2.0):  # Slopes should be reasonably similar
                return None
            
            # Create pattern points
            pattern_points = []
            for i, peak_idx in enumerate(recent_peaks):
                pattern_points.append(PatternPoint(
                    timestamps[peak_idx], price_data[peak_idx], peak_idx, 'resistance'
                ))
            
            for i, trough_idx in enumerate(recent_troughs):
                pattern_points.append(PatternPoint(
                    timestamps[trough_idx], price_data[trough_idx], trough_idx, 'support'
                ))
            
            # Sort by index
            pattern_points.sort(key=lambda x: x.index)
            
            # Calculate breakout target and stop loss
            pattern_height = max(peak_prices) - min(trough_prices)
            current_price = price_data[-1]
            
            # Symmetrical triangle can break either direction
            breakout_target = current_price + (pattern_height * 0.618)  # Fibonacci target
            stop_loss = current_price - (pattern_height * 0.382)  # Fibonacci stop
            
            # Calculate confidence
            confidence = self._calculate_triangle_confidence(
                pattern_points, current_price, (resistance_slope + support_slope) / 2, 'symmetrical'
            )
            
            return PatternMatch(
                'symmetrical_triangle',
                pattern_points,
                confidence,
                breakout_target,
                stop_loss
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting symmetrical triangle: {e}")
            return None
    
    def _calculate_trend_line_slope(self, indices: List[int], prices: List[float]) -> float:
        """Calculate the slope of a trend line through given points."""
        if len(indices) < 2 or len(prices) < 2:
            return 0.0
        
        try:
            # Use linear regression to find best fit line
            n = len(indices)
            sum_x = sum(indices)
            sum_y = sum(prices)
            sum_xy = sum(indices[i] * prices[i] for i in range(n))
            sum_x2 = sum(x * x for x in indices)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
            
        except Exception as e:
            self.logger.error(f"Error calculating trend line slope: {e}")
            return 0.0
    
    def _calculate_triangle_confidence(self, points: List[PatternPoint], 
                                     key_level: float, slope: float, 
                                     triangle_type: str) -> float:
        """Calculate confidence score for triangle pattern."""
        try:
            confidence = 0.5  # Base confidence
            
            # Factor 1: Number of touch points
            touch_points = len(points)
            if touch_points >= 6:
                confidence += 0.2
            elif touch_points >= 4:
                confidence += 0.1
            
            # Factor 2: Pattern duration
            if len(points) >= 2:
                duration = points[-1].index - points[0].index
                if duration >= 30:  # Good duration
                    confidence += 0.15
                elif duration >= 20:
                    confidence += 0.1
            
            # Factor 3: Slope quality
            if triangle_type in ['ascending', 'descending']:
                if abs(slope) > 0.001:  # Meaningful slope
                    confidence += 0.1
            elif triangle_type == 'symmetrical':
                if 0.001 < abs(slope) < 0.01:  # Moderate convergence
                    confidence += 0.15
            
            # Factor 4: Price action quality
            if len(points) >= 4:
                price_variance = statistics.variance([p.price for p in points])
                if price_variance > 0:  # Good price movement
                    confidence += 0.05
            
            return min(confidence, 0.95)  # Cap at 95%
            
        except Exception as e:
            self.logger.error(f"Error calculating triangle confidence: {e}")
            return 0.5


class HeadAndShouldersDetector:
    """Detects head and shoulders patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_pattern_length = 15
        self.shoulder_tolerance = 0.05  # 5% tolerance for shoulder symmetry
    
    def detect_head_and_shoulders(self, price_data: List[float], 
                                 timestamps: List[datetime]) -> List[PatternMatch]:
        """Detect head and shoulders patterns."""
        patterns = []
        
        try:
            if len(price_data) < self.min_pattern_length:
                return patterns
            
            # Find peaks for potential head and shoulders
            peaks, troughs = self._find_significant_points(price_data)
            
            # Regular head and shoulders (bearish)
            regular_hs = self._detect_regular_head_shoulders(price_data, peaks, troughs, timestamps)
            if regular_hs:
                patterns.append(regular_hs)
            
            # Inverse head and shoulders (bullish)
            inverse_hs = self._detect_inverse_head_shoulders(price_data, peaks, troughs, timestamps)
            if inverse_hs:
                patterns.append(inverse_hs)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting head and shoulders: {e}")
            return []
    
    def _find_significant_points(self, price_data: List[float]) -> Tuple[List[int], List[int]]:
        """Find significant peaks and troughs."""
        peaks = []
        troughs = []
        
        try:
            window = 3
            
            for i in range(window, len(price_data) - window):
                # Peak detection
                if (price_data[i] > price_data[i-1] and price_data[i] > price_data[i+1] and
                    price_data[i] > price_data[i-window] and price_data[i] > price_data[i+window]):
                    peaks.append(i)
                
                # Trough detection
                if (price_data[i] < price_data[i-1] and price_data[i] < price_data[i+1] and
                    price_data[i] < price_data[i-window] and price_data[i] < price_data[i+window]):
                    troughs.append(i)
            
            return peaks, troughs
            
        except Exception as e:
            self.logger.error(f"Error finding significant points: {e}")
            return [], []
    
    def _detect_regular_head_shoulders(self, price_data: List[float], peaks: List[int], 
                                      troughs: List[int], timestamps: List[datetime]) -> Optional[PatternMatch]:
        """Detect regular (bearish) head and shoulders pattern."""
        try:
            if len(peaks) < 3 or len(troughs) < 2:
                return None
            
            # Look for three consecutive peaks: left shoulder, head, right shoulder
            for i in range(len(peaks) - 2):
                left_shoulder_idx = peaks[i]
                head_idx = peaks[i + 1]
                right_shoulder_idx = peaks[i + 2]
                
                left_shoulder = price_data[left_shoulder_idx]
                head = price_data[head_idx]
                right_shoulder = price_data[right_shoulder_idx]
                
                # Head should be higher than both shoulders
                if not (head > left_shoulder and head > right_shoulder):
                    continue
                
                # Shoulders should be roughly equal
                shoulder_diff = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
                if shoulder_diff > self.shoulder_tolerance:
                    continue
                
                # Find neckline (troughs between shoulders and head)
                neckline_troughs = [t for t in troughs if left_shoulder_idx < t < right_shoulder_idx]
                if len(neckline_troughs) < 1:
                    continue
                
                # Calculate neckline level
                neckline_prices = [price_data[t] for t in neckline_troughs]
                neckline_level = statistics.mean(neckline_prices)
                
                # Create pattern points
                pattern_points = [
                    PatternPoint(timestamps[left_shoulder_idx], left_shoulder, left_shoulder_idx, 'left_shoulder'),
                    PatternPoint(timestamps[head_idx], head, head_idx, 'head'),
                    PatternPoint(timestamps[right_shoulder_idx], right_shoulder, right_shoulder_idx, 'right_shoulder')
                ]
                
                # Add neckline points
                for t_idx in neckline_troughs:
                    pattern_points.append(PatternPoint(timestamps[t_idx], price_data[t_idx], t_idx, 'neckline'))
                
                # Calculate targets
                pattern_height = head - neckline_level
                breakout_target = neckline_level - pattern_height  # Bearish target
                stop_loss = head + (pattern_height * 0.1)
                
                # Calculate confidence
                confidence = self._calculate_hs_confidence(pattern_points, shoulder_diff, pattern_height)
                
                return PatternMatch(
                    'head_and_shoulders',
                    pattern_points,
                    confidence,
                    breakout_target,
                    stop_loss
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting regular head and shoulders: {e}")
            return None
    
    def _detect_inverse_head_shoulders(self, price_data: List[float], peaks: List[int], 
                                      troughs: List[int], timestamps: List[datetime]) -> Optional[PatternMatch]:
        """Detect inverse (bullish) head and shoulders pattern."""
        try:
            if len(troughs) < 3 or len(peaks) < 2:
                return None
            
            # Look for three consecutive troughs: left shoulder, head, right shoulder
            for i in range(len(troughs) - 2):
                left_shoulder_idx = troughs[i]
                head_idx = troughs[i + 1]
                right_shoulder_idx = troughs[i + 2]
                
                left_shoulder = price_data[left_shoulder_idx]
                head = price_data[head_idx]
                right_shoulder = price_data[right_shoulder_idx]
                
                # Head should be lower than both shoulders
                if not (head < left_shoulder and head < right_shoulder):
                    continue
                
                # Shoulders should be roughly equal
                shoulder_diff = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
                if shoulder_diff > self.shoulder_tolerance:
                    continue
                
                # Find neckline (peaks between shoulders and head)
                neckline_peaks = [p for p in peaks if left_shoulder_idx < p < right_shoulder_idx]
                if len(neckline_peaks) < 1:
                    continue
                
                # Calculate neckline level
                neckline_prices = [price_data[p] for p in neckline_peaks]
                neckline_level = statistics.mean(neckline_prices)
                
                # Create pattern points
                pattern_points = [
                    PatternPoint(timestamps[left_shoulder_idx], left_shoulder, left_shoulder_idx, 'left_shoulder'),
                    PatternPoint(timestamps[head_idx], head, head_idx, 'head'),
                    PatternPoint(timestamps[right_shoulder_idx], right_shoulder, right_shoulder_idx, 'right_shoulder')
                ]
                
                # Add neckline points
                for p_idx in neckline_peaks:
                    pattern_points.append(PatternPoint(timestamps[p_idx], price_data[p_idx], p_idx, 'neckline'))
                
                # Calculate targets
                pattern_height = neckline_level - head
                breakout_target = neckline_level + pattern_height  # Bullish target
                stop_loss = head - (pattern_height * 0.1)
                
                # Calculate confidence
                confidence = self._calculate_hs_confidence(pattern_points, shoulder_diff, pattern_height)
                
                return PatternMatch(
                    'inverse_head_and_shoulders',
                    pattern_points,
                    confidence,
                    breakout_target,
                    stop_loss
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting inverse head and shoulders: {e}")
            return None
    
    def _calculate_hs_confidence(self, points: List[PatternPoint], 
                                shoulder_diff: float, pattern_height: float) -> float:
        """Calculate confidence for head and shoulders pattern."""
        try:
            confidence = 0.6  # Base confidence
            
            # Factor 1: Shoulder symmetry
            if shoulder_diff < 0.02:  # Very symmetric
                confidence += 0.2
            elif shoulder_diff < 0.03:
                confidence += 0.1
            
            # Factor 2: Pattern height (significance)
            if pattern_height > 0.05:  # Significant pattern
                confidence += 0.15
            elif pattern_height > 0.03:
                confidence += 0.1
            
            # Factor 3: Neckline quality
            neckline_points = [p for p in points if p.point_type == 'neckline']
            if len(neckline_points) >= 2:
                confidence += 0.05
            
            return min(confidence, 0.95)
            
        except Exception as e:
            self.logger.error(f"Error calculating H&S confidence: {e}")
            return 0.6


class FlagPennantDetector:
    """Detects flag and pennant continuation patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_flagpole_height = 0.03  # 3% minimum flagpole
        self.max_consolidation_time = 20  # Maximum consolidation periods
    
    def detect_flags_pennants(self, price_data: List[float], 
                             timestamps: List[datetime], 
                             volume_data: List[float] = None) -> List[PatternMatch]:
        """Detect flag and pennant patterns."""
        patterns = []
        
        try:
            if len(price_data) < 15:
                return patterns
            
            # Detect bullish flags
            bull_flag = self._detect_bullish_flag(price_data, timestamps, volume_data)
            if bull_flag:
                patterns.append(bull_flag)
            
            # Detect bearish flags
            bear_flag = self._detect_bearish_flag(price_data, timestamps, volume_data)
            if bear_flag:
                patterns.append(bear_flag)
            
            # Detect bullish pennants
            bull_pennant = self._detect_bullish_pennant(price_data, timestamps, volume_data)
            if bull_pennant:
                patterns.append(bull_pennant)
            
            # Detect bearish pennants
            bear_pennant = self._detect_bearish_pennant(price_data, timestamps, volume_data)
            if bear_pennant:
                patterns.append(bear_pennant)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting flags and pennants: {e}")
            return []
    
    def _detect_bullish_flag(self, price_data: List[float], timestamps: List[datetime], 
                            volume_data: List[float] = None) -> Optional[PatternMatch]:
        """Detect bullish flag pattern."""
        try:
            # Find strong upward move (flagpole)
            flagpole_start, flagpole_end = self._find_flagpole(price_data, 'bullish')
            if not flagpole_start or not flagpole_end:
                return None
            
            flagpole_height = price_data[flagpole_end] - price_data[flagpole_start]
            if flagpole_height / price_data[flagpole_start] < self.min_flagpole_height:
                return None
            
            # Find consolidation period (flag)
            consolidation_start = flagpole_end
            consolidation_end = min(consolidation_start + self.max_consolidation_time, len(price_data) - 1)
            
            # Check for slight downward drift in consolidation
            consolidation_data = price_data[consolidation_start:consolidation_end + 1]
            if len(consolidation_data) < 5:
                return None
            
            # Flag should have slight downward slope
            flag_slope = (consolidation_data[-1] - consolidation_data[0]) / len(consolidation_data)
            if flag_slope > 0:  # Should be negative or neutral for bullish flag
                return None
            
            # Check volume pattern (should decrease during consolidation)
            volume_confirmation = True
            if volume_data and len(volume_data) > consolidation_end:
                flagpole_volume = statistics.mean(volume_data[flagpole_start:flagpole_end])
                flag_volume = statistics.mean(volume_data[consolidation_start:consolidation_end])
                volume_confirmation = flag_volume < flagpole_volume * 0.8
            
            # Create pattern points
            pattern_points = [
                PatternPoint(timestamps[flagpole_start], price_data[flagpole_start], flagpole_start, 'flagpole_start'),
                PatternPoint(timestamps[flagpole_end], price_data[flagpole_end], flagpole_end, 'flagpole_end'),
                PatternPoint(timestamps[consolidation_end], price_data[consolidation_end], consolidation_end, 'flag_end')
            ]
            
            # Calculate targets
            breakout_target = price_data[consolidation_end] + flagpole_height
            stop_loss = min(consolidation_data) - (flagpole_height * 0.1)
            
            # Calculate confidence
            confidence = self._calculate_flag_confidence(
                flagpole_height, len(consolidation_data), volume_confirmation, 'bullish'
            )
            
            return PatternMatch(
                'bullish_flag',
                pattern_points,
                confidence,
                breakout_target,
                stop_loss
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting bullish flag: {e}")
            return None
    
    def _detect_bearish_flag(self, price_data: List[float], timestamps: List[datetime], 
                            volume_data: List[float] = None) -> Optional[PatternMatch]:
        """Detect bearish flag pattern."""
        try:
            # Find strong downward move (flagpole)
            flagpole_start, flagpole_end = self._find_flagpole(price_data, 'bearish')
            if not flagpole_start or not flagpole_end:
                return None
            
            flagpole_height = price_data[flagpole_start] - price_data[flagpole_end]
            if flagpole_height / price_data[flagpole_start] < self.min_flagpole_height:
                return None
            
            # Find consolidation period (flag)
            consolidation_start = flagpole_end
            consolidation_end = min(consolidation_start + self.max_consolidation_time, len(price_data) - 1)
            
            # Check for slight upward drift in consolidation
            consolidation_data = price_data[consolidation_start:consolidation_end + 1]
            if len(consolidation_data) < 5:
                return None
            
            # Flag should have slight upward slope
            flag_slope = (consolidation_data[-1] - consolidation_data[0]) / len(consolidation_data)
            if flag_slope < 0:  # Should be positive or neutral for bearish flag
                return None
            
            # Check volume pattern
            volume_confirmation = True
            if volume_data and len(volume_data) > consolidation_end:
                flagpole_volume = statistics.mean(volume_data[flagpole_start:flagpole_end])
                flag_volume = statistics.mean(volume_data[consolidation_start:consolidation_end])
                volume_confirmation = flag_volume < flagpole_volume * 0.8
            
            # Create pattern points
            pattern_points = [
                PatternPoint(timestamps[flagpole_start], price_data[flagpole_start], flagpole_start, 'flagpole_start'),
                PatternPoint(timestamps[flagpole_end], price_data[flagpole_end], flagpole_end, 'flagpole_end'),
                PatternPoint(timestamps[consolidation_end], price_data[consolidation_end], consolidation_end, 'flag_end')
            ]
            
            # Calculate targets
            breakout_target = price_data[consolidation_end] - flagpole_height
            stop_loss = max(consolidation_data) + (flagpole_height * 0.1)
            
            # Calculate confidence
            confidence = self._calculate_flag_confidence(
                flagpole_height, len(consolidation_data), volume_confirmation, 'bearish'
            )
            
            return PatternMatch(
                'bearish_flag',
                pattern_points,
                confidence,
                breakout_target,
                stop_loss
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting bearish flag: {e}")
            return None
    
    def _detect_bullish_pennant(self, price_data: List[float], timestamps: List[datetime], 
                               volume_data: List[float] = None) -> Optional[PatternMatch]:
        """Detect bullish pennant pattern."""
        try:
            # Find strong upward move (flagpole)
            flagpole_start, flagpole_end = self._find_flagpole(price_data, 'bullish')
            if not flagpole_start or not flagpole_end:
                return None
            
            flagpole_height = price_data[flagpole_end] - price_data[flagpole_start]
            if flagpole_height / price_data[flagpole_start] < self.min_flagpole_height:
                return None
            
            # Find consolidation period (pennant)
            consolidation_start = flagpole_end
            consolidation_end = min(consolidation_start + self.max_consolidation_time, len(price_data) - 1)
            
            consolidation_data = price_data[consolidation_start:consolidation_end + 1]
            if len(consolidation_data) < 8:
                return None
            
            # Pennant should show converging price action
            highs, lows = self._find_consolidation_bounds(consolidation_data)
            if not self._is_converging_pattern(highs, lows):
                return None
            
            # Create pattern points
            pattern_points = [
                PatternPoint(timestamps[flagpole_start], price_data[flagpole_start], flagpole_start, 'flagpole_start'),
                PatternPoint(timestamps[flagpole_end], price_data[flagpole_end], flagpole_end, 'flagpole_end'),
                PatternPoint(timestamps[consolidation_end], price_data[consolidation_end], consolidation_end, 'pennant_end')
            ]
            
            # Calculate targets
            breakout_target = price_data[consolidation_end] + flagpole_height
            stop_loss = min(consolidation_data) - (flagpole_height * 0.1)
            
            # Calculate confidence
            volume_confirmation = True
            if volume_data:
                volume_confirmation = self._check_volume_pattern(volume_data, flagpole_start, consolidation_end)
            
            confidence = self._calculate_pennant_confidence(
                flagpole_height, len(consolidation_data), volume_confirmation
            )
            
            return PatternMatch(
                'bullish_pennant',
                pattern_points,
                confidence,
                breakout_target,
                stop_loss
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting bullish pennant: {e}")
            return None
    
    def _detect_bearish_pennant(self, price_data: List[float], timestamps: List[datetime], 
                               volume_data: List[float] = None) -> Optional[PatternMatch]:
        """Detect bearish pennant pattern."""
        try:
            # Find strong downward move (flagpole)
            flagpole_start, flagpole_end = self._find_flagpole(price_data, 'bearish')
            if not flagpole_start or not flagpole_end:
                return None
            
            flagpole_height = price_data[flagpole_start] - price_data[flagpole_end]
            if flagpole_height / price_data[flagpole_start] < self.min_flagpole_height:
                return None
            
            # Find consolidation period (pennant)
            consolidation_start = flagpole_end
            consolidation_end = min(consolidation_start + self.max_consolidation_time, len(price_data) - 1)
            
            consolidation_data = price_data[consolidation_start:consolidation_end + 1]
            if len(consolidation_data) < 8:
                return None
            
            # Pennant should show converging price action
            highs, lows = self._find_consolidation_bounds(consolidation_data)
            if not self._is_converging_pattern(highs, lows):
                return None
            
            # Create pattern points
            pattern_points = [
                PatternPoint(timestamps[flagpole_start], price_data[flagpole_start], flagpole_start, 'flagpole_start'),
                PatternPoint(timestamps[flagpole_end], price_data[flagpole_end], flagpole_end, 'flagpole_end'),
                PatternPoint(timestamps[consolidation_end], price_data[consolidation_end], consolidation_end, 'pennant_end')
            ]
            
            # Calculate targets
            breakout_target = price_data[consolidation_end] - flagpole_height
            stop_loss = max(consolidation_data) + (flagpole_height * 0.1)
            
            # Calculate confidence
            volume_confirmation = True
            if volume_data:
                volume_confirmation = self._check_volume_pattern(volume_data, flagpole_start, consolidation_end)
            
            confidence = self._calculate_pennant_confidence(
                flagpole_height, len(consolidation_data), volume_confirmation
            )
            
            return PatternMatch(
                'bearish_pennant',
                pattern_points,
                confidence,
                breakout_target,
                stop_loss
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting bearish pennant: {e}")
            return None 
   
    def _find_flagpole(self, price_data: List[float], direction: str) -> Tuple[Optional[int], Optional[int]]:
        """Find the flagpole (strong directional move)."""
        try:
            min_move = 0.03  # 3% minimum move
            max_lookback = 20
            
            if direction == 'bullish':
                # Look for strong upward move
                for i in range(len(price_data) - 5, max(0, len(price_data) - max_lookback), -1):
                    for j in range(i + 3, min(len(price_data), i + 15)):
                        move_pct = (price_data[j] - price_data[i]) / price_data[i]
                        if move_pct >= min_move:
                            # Check if move is relatively straight
                            if self._is_strong_directional_move(price_data[i:j+1], 'up'):
                                return i, j
            
            elif direction == 'bearish':
                # Look for strong downward move
                for i in range(len(price_data) - 5, max(0, len(price_data) - max_lookback), -1):
                    for j in range(i + 3, min(len(price_data), i + 15)):
                        move_pct = (price_data[i] - price_data[j]) / price_data[i]
                        if move_pct >= min_move:
                            # Check if move is relatively straight
                            if self._is_strong_directional_move(price_data[i:j+1], 'down'):
                                return i, j
            
            return None, None
            
        except Exception as e:
            self.logger.error(f"Error finding flagpole: {e}")
            return None, None
    
    def _is_strong_directional_move(self, price_segment: List[float], direction: str) -> bool:
        """Check if price segment represents a strong directional move."""
        try:
            if len(price_segment) < 3:
                return False
            
            # Calculate how much of the move is in the intended direction
            total_move = abs(price_segment[-1] - price_segment[0])
            if total_move == 0:
                return False
            
            # Count directional consistency
            directional_moves = 0
            total_moves = 0
            
            for i in range(1, len(price_segment)):
                move = price_segment[i] - price_segment[i-1]
                total_moves += 1
                
                if direction == 'up' and move > 0:
                    directional_moves += 1
                elif direction == 'down' and move < 0:
                    directional_moves += 1
            
            # At least 60% of moves should be in the right direction
            consistency = directional_moves / total_moves if total_moves > 0 else 0
            return consistency >= 0.6
            
        except Exception as e:
            self.logger.error(f"Error checking directional move: {e}")
            return False
    
    def _find_consolidation_bounds(self, consolidation_data: List[float]) -> Tuple[List[float], List[float]]:
        """Find the upper and lower bounds of consolidation."""
        try:
            window = 2
            highs = []
            lows = []
            
            for i in range(window, len(consolidation_data) - window):
                # Local high
                if all(consolidation_data[i] >= consolidation_data[j] 
                      for j in range(i - window, i + window + 1) if j != i):
                    highs.append(consolidation_data[i])
                
                # Local low
                if all(consolidation_data[i] <= consolidation_data[j] 
                      for j in range(i - window, i + window + 1) if j != i):
                    lows.append(consolidation_data[i])
            
            return highs, lows
            
        except Exception as e:
            self.logger.error(f"Error finding consolidation bounds: {e}")
            return [], []
    
    def _is_converging_pattern(self, highs: List[float], lows: List[float]) -> bool:
        """Check if highs and lows are converging (pennant pattern)."""
        try:
            if len(highs) < 2 or len(lows) < 2:
                return False
            
            # Check if range is decreasing
            initial_range = max(highs[:2]) - min(lows[:2]) if len(highs) >= 2 and len(lows) >= 2 else 0
            final_range = max(highs[-2:]) - min(lows[-2:]) if len(highs) >= 2 and len(lows) >= 2 else 0
            
            return final_range < initial_range * 0.7  # Range should decrease by at least 30%
            
        except Exception as e:
            self.logger.error(f"Error checking convergence: {e}")
            return False
    
    def _check_volume_pattern(self, volume_data: List[float], 
                             flagpole_start: int, consolidation_end: int) -> bool:
        """Check if volume pattern supports the flag/pennant."""
        try:
            if len(volume_data) <= consolidation_end:
                return True  # Can't verify, assume OK
            
            flagpole_volume = statistics.mean(volume_data[flagpole_start:flagpole_start + 5])
            consolidation_volume = statistics.mean(volume_data[consolidation_end - 5:consolidation_end])
            
            # Volume should decrease during consolidation
            return consolidation_volume < flagpole_volume * 0.8
            
        except Exception as e:
            self.logger.error(f"Error checking volume pattern: {e}")
            return True
    
    def _calculate_flag_confidence(self, flagpole_height: float, consolidation_length: int, 
                                  volume_confirmation: bool, flag_type: str) -> float:
        """Calculate confidence for flag pattern."""
        try:
            confidence = 0.6  # Base confidence
            
            # Factor 1: Flagpole strength
            if flagpole_height > 0.05:  # Strong move
                confidence += 0.15
            elif flagpole_height > 0.04:
                confidence += 0.1
            
            # Factor 2: Consolidation duration
            if 5 <= consolidation_length <= 15:  # Ideal duration
                confidence += 0.1
            elif consolidation_length <= 20:
                confidence += 0.05
            
            # Factor 3: Volume confirmation
            if volume_confirmation:
                confidence += 0.1
            
            return min(confidence, 0.9)
            
        except Exception as e:
            self.logger.error(f"Error calculating flag confidence: {e}")
            return 0.6
    
    def _calculate_pennant_confidence(self, flagpole_height: float, 
                                     consolidation_length: int, volume_confirmation: bool) -> float:
        """Calculate confidence for pennant pattern."""
        try:
            confidence = 0.65  # Base confidence (slightly higher than flags)
            
            # Factor 1: Flagpole strength
            if flagpole_height > 0.05:
                confidence += 0.15
            elif flagpole_height > 0.04:
                confidence += 0.1
            
            # Factor 2: Consolidation duration
            if 8 <= consolidation_length <= 15:
                confidence += 0.1
            elif consolidation_length <= 20:
                confidence += 0.05
            
            # Factor 3: Volume confirmation
            if volume_confirmation:
                confidence += 0.1
            
            return min(confidence, 0.9)
            
        except Exception as e:
            self.logger.error(f"Error calculating pennant confidence: {e}")
            return 0.65


class WedgeDetector:
    """Detects rising and falling wedge patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_pattern_length = 15
        self.convergence_threshold = 0.7  # Lines should converge
    
    def detect_wedges(self, price_data: List[float], 
                     timestamps: List[datetime]) -> List[PatternMatch]:
        """Detect wedge patterns."""
        patterns = []
        
        try:
            if len(price_data) < self.min_pattern_length:
                return patterns
            
            # Find peaks and troughs
            peaks, troughs = self._find_wedge_points(price_data)
            
            # Detect rising wedge (bearish)
            rising_wedge = self._detect_rising_wedge(price_data, peaks, troughs, timestamps)
            if rising_wedge:
                patterns.append(rising_wedge)
            
            # Detect falling wedge (bullish)
            falling_wedge = self._detect_falling_wedge(price_data, peaks, troughs, timestamps)
            if falling_wedge:
                patterns.append(falling_wedge)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting wedges: {e}")
            return []
    
    def _find_wedge_points(self, price_data: List[float]) -> Tuple[List[int], List[int]]:
        """Find points for wedge pattern analysis."""
        peaks = []
        troughs = []
        
        try:
            window = 3
            
            for i in range(window, len(price_data) - window):
                # Peak detection
                if all(price_data[i] >= price_data[j] 
                      for j in range(i - window, i + window + 1) if j != i):
                    if price_data[i] > price_data[i-1] and price_data[i] > price_data[i+1]:
                        peaks.append(i)
                
                # Trough detection
                if all(price_data[i] <= price_data[j] 
                      for j in range(i - window, i + window + 1) if j != i):
                    if price_data[i] < price_data[i-1] and price_data[i] < price_data[i+1]:
                        troughs.append(i)
            
            return peaks, troughs
            
        except Exception as e:
            self.logger.error(f"Error finding wedge points: {e}")
            return [], []
    
    def _detect_rising_wedge(self, price_data: List[float], peaks: List[int], 
                            troughs: List[int], timestamps: List[datetime]) -> Optional[PatternMatch]:
        """Detect rising wedge (bearish reversal) pattern."""
        try:
            if len(peaks) < 2 or len(troughs) < 2:
                return None
            
            # Get recent points
            recent_peaks = peaks[-4:] if len(peaks) >= 4 else peaks
            recent_troughs = troughs[-4:] if len(troughs) >= 4 else troughs
            
            # Calculate trend line slopes
            peak_prices = [price_data[i] for i in recent_peaks]
            trough_prices = [price_data[i] for i in recent_troughs]
            
            peak_slope = self._calculate_slope(recent_peaks, peak_prices)
            trough_slope = self._calculate_slope(recent_troughs, trough_prices)
            
            # Rising wedge: both slopes positive, but support slope > resistance slope
            if peak_slope <= 0 or trough_slope <= 0:
                return None
            
            if trough_slope <= peak_slope:  # Support should rise faster than resistance
                return None
            
            # Check convergence
            if not self._check_convergence(recent_peaks, peak_prices, recent_troughs, trough_prices):
                return None
            
            # Create pattern points
            pattern_points = []
            for i, peak_idx in enumerate(recent_peaks):
                pattern_points.append(PatternPoint(
                    timestamps[peak_idx], price_data[peak_idx], peak_idx, 'resistance'
                ))
            
            for i, trough_idx in enumerate(recent_troughs):
                pattern_points.append(PatternPoint(
                    timestamps[trough_idx], price_data[trough_idx], trough_idx, 'support'
                ))
            
            pattern_points.sort(key=lambda x: x.index)
            
            # Calculate targets
            pattern_height = max(peak_prices) - min(trough_prices)
            current_price = price_data[-1]
            breakout_target = current_price - pattern_height  # Bearish target
            stop_loss = max(peak_prices) + (pattern_height * 0.1)
            
            # Calculate confidence
            confidence = self._calculate_wedge_confidence(
                pattern_points, peak_slope, trough_slope, 'rising'
            )
            
            return PatternMatch(
                'rising_wedge',
                pattern_points,
                confidence,
                breakout_target,
                stop_loss
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting rising wedge: {e}")
            return None
    
    def _detect_falling_wedge(self, price_data: List[float], peaks: List[int], 
                             troughs: List[int], timestamps: List[datetime]) -> Optional[PatternMatch]:
        """Detect falling wedge (bullish reversal) pattern."""
        try:
            if len(peaks) < 2 or len(troughs) < 2:
                return None
            
            # Get recent points
            recent_peaks = peaks[-4:] if len(peaks) >= 4 else peaks
            recent_troughs = troughs[-4:] if len(troughs) >= 4 else troughs
            
            # Calculate trend line slopes
            peak_prices = [price_data[i] for i in recent_peaks]
            trough_prices = [price_data[i] for i in recent_troughs]
            
            peak_slope = self._calculate_slope(recent_peaks, peak_prices)
            trough_slope = self._calculate_slope(recent_troughs, trough_prices)
            
            # Falling wedge: both slopes negative, but resistance slope > support slope
            if peak_slope >= 0 or trough_slope >= 0:
                return None
            
            if peak_slope <= trough_slope:  # Resistance should fall slower than support
                return None
            
            # Check convergence
            if not self._check_convergence(recent_peaks, peak_prices, recent_troughs, trough_prices):
                return None
            
            # Create pattern points
            pattern_points = []
            for i, peak_idx in enumerate(recent_peaks):
                pattern_points.append(PatternPoint(
                    timestamps[peak_idx], price_data[peak_idx], peak_idx, 'resistance'
                ))
            
            for i, trough_idx in enumerate(recent_troughs):
                pattern_points.append(PatternPoint(
                    timestamps[trough_idx], price_data[trough_idx], trough_idx, 'support'
                ))
            
            pattern_points.sort(key=lambda x: x.index)
            
            # Calculate targets
            pattern_height = max(peak_prices) - min(trough_prices)
            current_price = price_data[-1]
            breakout_target = current_price + pattern_height  # Bullish target
            stop_loss = min(trough_prices) - (pattern_height * 0.1)
            
            # Calculate confidence
            confidence = self._calculate_wedge_confidence(
                pattern_points, peak_slope, trough_slope, 'falling'
            )
            
            return PatternMatch(
                'falling_wedge',
                pattern_points,
                confidence,
                breakout_target,
                stop_loss
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting falling wedge: {e}")
            return None
    
    def _calculate_slope(self, indices: List[int], prices: List[float]) -> float:
        """Calculate slope of trend line."""
        if len(indices) < 2:
            return 0.0
        
        try:
            n = len(indices)
            sum_x = sum(indices)
            sum_y = sum(prices)
            sum_xy = sum(indices[i] * prices[i] for i in range(n))
            sum_x2 = sum(x * x for x in indices)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return 0.0
            
            return (n * sum_xy - sum_x * sum_y) / denominator
            
        except Exception as e:
            self.logger.error(f"Error calculating slope: {e}")
            return 0.0
    
    def _check_convergence(self, peak_indices: List[int], peak_prices: List[float],
                          trough_indices: List[int], trough_prices: List[float]) -> bool:
        """Check if trend lines are converging."""
        try:
            if len(peak_indices) < 2 or len(trough_indices) < 2:
                return False
            
            # Calculate initial and final spreads
            initial_spread = abs(peak_prices[0] - trough_prices[0])
            final_spread = abs(peak_prices[-1] - trough_prices[-1])
            
            # Lines should be converging
            return final_spread < initial_spread * self.convergence_threshold
            
        except Exception as e:
            self.logger.error(f"Error checking convergence: {e}")
            return False
    
    def _calculate_wedge_confidence(self, points: List[PatternPoint], 
                                   peak_slope: float, trough_slope: float, 
                                   wedge_type: str) -> float:
        """Calculate confidence for wedge pattern."""
        try:
            confidence = 0.6  # Base confidence
            
            # Factor 1: Number of touch points
            if len(points) >= 6:
                confidence += 0.15
            elif len(points) >= 4:
                confidence += 0.1
            
            # Factor 2: Slope quality
            slope_diff = abs(peak_slope - trough_slope)
            if slope_diff > 0.001:  # Good slope difference
                confidence += 0.1
            
            # Factor 3: Pattern duration
            if len(points) >= 2:
                duration = points[-1].index - points[0].index
                if duration >= 20:
                    confidence += 0.1
                elif duration >= 15:
                    confidence += 0.05
            
            return min(confidence, 0.9)
            
        except Exception as e:
            self.logger.error(f"Error calculating wedge confidence: {e}")
            return 0.6


class PatternStrategy(BaseStrategy):
    """Chart pattern recognition trading strategy."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__("PatternStrategy", parameters)
        
        # Default parameters
        default_params = {
            'lookback_periods': 50,
            'min_pattern_confidence': 0.6,
            'enable_triangles': True,
            'enable_head_shoulders': True,
            'enable_flags_pennants': True,
            'enable_wedges': True,
            'breakout_confirmation': True,
            'volume_confirmation': False,
            'position_size_factor': 1.0,
            'max_active_patterns': 3,
            'pattern_timeout_hours': 24
        }
        
        self.parameters.update(default_params)
        if parameters:
            self.parameters.update(parameters)
        
        # Initialize pattern detectors
        self.triangle_detector = TrianglePatternDetector()
        self.hs_detector = HeadAndShouldersDetector()
        self.flag_pennant_detector = FlagPennantDetector()
        self.wedge_detector = WedgeDetector()
        
        # Strategy state
        self.price_history = deque(maxlen=self.parameters['lookback_periods'])
        self.timestamp_history = deque(maxlen=self.parameters['lookback_periods'])
        self.volume_history = deque(maxlen=self.parameters['lookback_periods'])
        
        self.active_patterns = []
        self.pattern_history = deque(maxlen=100)
        
        self.performance_metrics = {
            'total_patterns': 0,
            'successful_patterns': 0,
            'failed_patterns': 0,
            'pattern_type_performance': {},
            'avg_confidence': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_market(self, market_data: MarketData) -> Optional[TradingSignal]:
        """Analyze market data for chart patterns and generate trading signals."""
        try:
            # Update price history
            self._update_history(market_data)
            
            # Need sufficient history for pattern detection
            if len(self.price_history) < 20:
                return None
            
            # Clean up expired patterns
            self._cleanup_expired_patterns()
            
            # Detect new patterns
            new_patterns = self._detect_patterns()
            
            # Update active patterns
            self._update_active_patterns(new_patterns)
            
            # Check for breakout signals
            breakout_signal = self._check_breakout_signals(market_data)
            
            if breakout_signal:
                self._update_performance_metrics(breakout_signal)
                return breakout_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {e}")
            return None
    
    def _update_history(self, market_data: MarketData) -> None:
        """Update price and volume history."""
        try:
            self.price_history.append(market_data.price)
            self.timestamp_history.append(market_data.timestamp)
            self.volume_history.append(market_data.volume or 0)
        except Exception as e:
            self.logger.error(f"Error updating history: {e}")
    
    def _detect_patterns(self) -> List[PatternMatch]:
        """Detect all enabled chart patterns."""
        patterns = []
        
        try:
            price_data = list(self.price_history)
            timestamps = list(self.timestamp_history)
            volume_data = list(self.volume_history) if any(self.volume_history) else None
            
            # Triangle patterns
            if self.parameters['enable_triangles']:
                triangles = self.triangle_detector.detect_triangles(price_data, timestamps)
                patterns.extend(triangles)
            
            # Head and shoulders patterns
            if self.parameters['enable_head_shoulders']:
                hs_patterns = self.hs_detector.detect_head_and_shoulders(price_data, timestamps)
                patterns.extend(hs_patterns)
            
            # Flag and pennant patterns
            if self.parameters['enable_flags_pennants']:
                flag_pennants = self.flag_pennant_detector.detect_flags_pennants(
                    price_data, timestamps, volume_data
                )
                patterns.extend(flag_pennants)
            
            # Wedge patterns
            if self.parameters['enable_wedges']:
                wedges = self.wedge_detector.detect_wedges(price_data, timestamps)
                patterns.extend(wedges)
            
            # Filter by confidence
            filtered_patterns = [
                p for p in patterns 
                if p.confidence >= self.parameters['min_pattern_confidence']
            ]
            
            return filtered_patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _update_active_patterns(self, new_patterns: List[PatternMatch]) -> None:
        """Update the list of active patterns."""
        try:
            # Add new patterns
            for pattern in new_patterns:
                if len(self.active_patterns) < self.parameters['max_active_patterns']:
                    self.active_patterns.append(pattern)
                    self.pattern_history.append({
                        'pattern': pattern,
                        'detected_at': pattern.detected_at,
                        'status': 'active'
                    })
            
            # Sort by confidence (keep highest confidence patterns)
            self.active_patterns.sort(key=lambda x: x.confidence, reverse=True)
            self.active_patterns = self.active_patterns[:self.parameters['max_active_patterns']]
            
        except Exception as e:
            self.logger.error(f"Error updating active patterns: {e}")
    
    def _cleanup_expired_patterns(self) -> None:
        """Remove expired patterns."""
        try:
            timeout_delta = timedelta(hours=self.parameters['pattern_timeout_hours'])
            current_time = datetime.now()
            
            self.active_patterns = [
                p for p in self.active_patterns 
                if current_time - p.detected_at < timeout_delta and p.is_active
            ]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up patterns: {e}")
    
    def _check_breakout_signals(self, market_data: MarketData) -> Optional[TradingSignal]:
        """Check active patterns for breakout signals."""
        try:
            if not self.active_patterns:
                return None
            
            current_price = market_data.price
            
            for pattern in self.active_patterns:
                breakout_signal = self._evaluate_pattern_breakout(pattern, current_price, market_data)
                if breakout_signal:
                    pattern.is_active = False  # Deactivate pattern after breakout
                    return breakout_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking breakout signals: {e}")
            return None
    
    def _evaluate_pattern_breakout(self, pattern: PatternMatch, current_price: float, 
                                  market_data: MarketData) -> Optional[TradingSignal]:
        """Evaluate if a pattern has broken out."""
        try:
            # Determine breakout direction and thresholds
            breakout_info = self._get_breakout_info(pattern, current_price)
            if not breakout_info:
                return None
            
            direction, breakout_level, confirmation_needed = breakout_info
            
            # Check if price has broken the level
            if direction == 'bullish' and current_price > breakout_level:
                # Bullish breakout
                if self.parameters['breakout_confirmation'] and not confirmation_needed:
                    return None
                
                # Calculate position size
                position_size = self._calculate_position_size(pattern, market_data)
                
                return TradingSignal(
                    symbol=market_data.symbol,
                    action=SignalAction.BUY,
                    price=current_price,
                    quantity=position_size,
                    confidence=pattern.confidence,
                    timestamp=market_data.timestamp,
                    strategy_name=self.name,
                    metadata={
                        'pattern_type': pattern.pattern_type,
                        'pattern_points': [
                            {'timestamp': p.timestamp, 'price': p.price, 'type': p.point_type}
                            for p in pattern.points
                        ],
                        'breakout_target': pattern.breakout_target,
                        'stop_loss': pattern.stop_loss,
                        'pattern_height': pattern.height,
                        'pattern_width': pattern.width,
                        'breakout_level': breakout_level
                    }
                )
            
            elif direction == 'bearish' and current_price < breakout_level:
                # Bearish breakout
                if self.parameters['breakout_confirmation'] and not confirmation_needed:
                    return None
                
                # Calculate position size
                position_size = self._calculate_position_size(pattern, market_data)
                
                return TradingSignal(
                    symbol=market_data.symbol,
                    action=SignalAction.SELL,
                    price=current_price,
                    quantity=position_size,
                    confidence=pattern.confidence,
                    timestamp=market_data.timestamp,
                    strategy_name=self.name,
                    metadata={
                        'pattern_type': pattern.pattern_type,
                        'pattern_points': [
                            {'timestamp': p.timestamp, 'price': p.price, 'type': p.point_type}
                            for p in pattern.points
                        ],
                        'breakout_target': pattern.breakout_target,
                        'stop_loss': pattern.stop_loss,
                        'pattern_height': pattern.height,
                        'pattern_width': pattern.width,
                        'breakout_level': breakout_level
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error evaluating pattern breakout: {e}")
            return None
    
    def _get_breakout_info(self, pattern: PatternMatch, current_price: float) -> Optional[Tuple[str, float, bool]]:
        """Get breakout information for a pattern."""
        try:
            pattern_type = pattern.pattern_type
            
            # Bullish patterns
            if pattern_type in ['ascending_triangle', 'inverse_head_and_shoulders', 
                              'bullish_flag', 'bullish_pennant', 'falling_wedge']:
                # Find resistance level to break
                resistance_points = [p for p in pattern.points if p.point_type in ['resistance', 'neckline']]
                if resistance_points:
                    breakout_level = max(p.price for p in resistance_points)
                    return 'bullish', breakout_level, True
            
            # Bearish patterns
            elif pattern_type in ['descending_triangle', 'head_and_shoulders', 
                                'bearish_flag', 'bearish_pennant', 'rising_wedge']:
                # Find support level to break
                support_points = [p for p in pattern.points if p.point_type in ['support', 'neckline']]
                if support_points:
                    breakout_level = min(p.price for p in support_points)
                    return 'bearish', breakout_level, True
            
            # Symmetrical patterns (can break either way)
            elif pattern_type == 'symmetrical_triangle':
                # Use current price trend to determine likely direction
                recent_prices = list(self.price_history)[-5:]
                if len(recent_prices) >= 2:
                    trend = recent_prices[-1] - recent_prices[0]
                    if trend > 0:
                        # Bullish bias
                        resistance_points = [p for p in pattern.points if p.point_type == 'resistance']
                        if resistance_points:
                            breakout_level = max(p.price for p in resistance_points)
                            return 'bullish', breakout_level, True
                    else:
                        # Bearish bias
                        support_points = [p for p in pattern.points if p.point_type == 'support']
                        if support_points:
                            breakout_level = min(p.price for p in support_points)
                            return 'bearish', breakout_level, True
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting breakout info: {e}")
            return None
    
    def _calculate_position_size(self, pattern: PatternMatch, market_data: MarketData) -> float:
        """Calculate position size based on pattern characteristics."""
        try:
            base_size = self.parameters['position_size_factor']
            
            # Adjust for pattern confidence
            confidence_multiplier = pattern.confidence
            
            # Adjust for pattern type reliability
            type_multiplier = {
                'ascending_triangle': 1.1,
                'descending_triangle': 1.1,
                'head_and_shoulders': 1.2,
                'inverse_head_and_shoulders': 1.2,
                'bullish_flag': 1.0,
                'bearish_flag': 1.0,
                'bullish_pennant': 1.0,
                'bearish_pennant': 1.0,
                'rising_wedge': 0.9,
                'falling_wedge': 0.9,
                'symmetrical_triangle': 0.8
            }.get(pattern.pattern_type, 1.0)
            
            # Adjust for pattern size (larger patterns = higher confidence)
            size_multiplier = min(pattern.height / market_data.price * 10, 1.2)
            
            position_size = base_size * confidence_multiplier * type_multiplier * size_multiplier
            
            # Cap position size
            max_size = base_size * 2.0
            return min(position_size, max_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self.parameters['position_size_factor']
    
    def _update_performance_metrics(self, signal: TradingSignal) -> None:
        """Update strategy performance metrics."""
        try:
            self.performance_metrics['total_patterns'] += 1
            
            # Update pattern type performance
            pattern_type = signal.metadata.get('pattern_type', 'unknown')
            if pattern_type not in self.performance_metrics['pattern_type_performance']:
                self.performance_metrics['pattern_type_performance'][pattern_type] = {
                    'count': 0, 'avg_confidence': 0.0
                }
            
            type_perf = self.performance_metrics['pattern_type_performance'][pattern_type]
            type_perf['count'] += 1
            type_total_conf = type_perf['avg_confidence'] * (type_perf['count'] - 1) + signal.confidence
            type_perf['avg_confidence'] = type_total_conf / type_perf['count']
            
            # Update overall average confidence
            total_conf = (self.performance_metrics['avg_confidence'] * 
                         (self.performance_metrics['total_patterns'] - 1) + signal.confidence)
            self.performance_metrics['avg_confidence'] = total_conf / self.performance_metrics['total_patterns']
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate strategy parameters."""
        try:
            required_params = [
                'lookback_periods', 'min_pattern_confidence', 'position_size_factor'
            ]
            
            for param in required_params:
                if param not in parameters:
                    return False
            
            # Validate ranges
            if not (10 <= parameters['lookback_periods'] <= 200):
                return False
            
            if not (0.0 <= parameters['min_pattern_confidence'] <= 1.0):
                return False
            
            if parameters['position_size_factor'] <= 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and current state."""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'confidence': self.confidence,
            'performance_metrics': self.performance_metrics,
            'state': {
                'price_history_length': len(self.price_history),
                'active_patterns': len(self.active_patterns),
                'pattern_history_length': len(self.pattern_history),
                'active_pattern_types': [p.pattern_type for p in self.active_patterns]
            }
        }