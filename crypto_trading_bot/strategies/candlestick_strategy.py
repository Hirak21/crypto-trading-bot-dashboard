"""
Candlestick pattern recognition trading strategy.

This strategy identifies and trades based on candlestick patterns including
single, double, and triple candlestick formations with fallback activation logic.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from datetime import datetime, timedelta
from collections import deque
import statistics
import math

from ..strategies.base_strategy import BaseStrategy
from ..models.trading import TradingSignal, MarketData, SignalAction
from ..utils.technical_analysis import MovingAverages, VolatilityIndicators


class Candlestick(NamedTuple):
    """Represents a single candlestick."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def body_size(self) -> float:
        """Size of the candlestick body."""
        return abs(self.close - self.open)
    
    @property
    def upper_shadow(self) -> float:
        """Size of the upper shadow."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        """Size of the lower shadow."""
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        """Total range of the candlestick."""
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        """True if candlestick is bullish (close > open)."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """True if candlestick is bearish (close < open)."""
        return self.close < self.open
    
    @property
    def is_doji(self) -> bool:
        """True if candlestick is a doji (small body)."""
        if self.total_range == 0:
            return True
        return self.body_size / self.total_range < 0.1


class CandlestickPattern:
    """Represents a detected candlestick pattern."""
    
    def __init__(self, pattern_type: str, candlesticks: List[Candlestick], 
                 confidence: float, signal_direction: str, strength: float):
        self.pattern_type = pattern_type
        self.candlesticks = candlesticks
        self.confidence = confidence
        self.signal_direction = signal_direction  # 'bullish', 'bearish', 'neutral'
        self.strength = strength
        self.detected_at = datetime.now()
        
        # Calculate pattern metrics
        self.pattern_range = self._calculate_pattern_range()
        self.volume_profile = self._calculate_volume_profile()
    
    def _calculate_pattern_range(self) -> float:
        """Calculate the price range of the pattern."""
        if not self.candlesticks:
            return 0.0
        
        highs = [c.high for c in self.candlesticks]
        lows = [c.low for c in self.candlesticks]
        return max(highs) - min(lows)
    
    def _calculate_volume_profile(self) -> Dict[str, float]:
        """Calculate volume characteristics of the pattern."""
        if not self.candlesticks:
            return {'avg_volume': 0.0, 'volume_trend': 0.0}
        
        volumes = [c.volume for c in self.candlesticks]
        avg_volume = statistics.mean(volumes)
        
        # Calculate volume trend (increasing/decreasing)
        if len(volumes) > 1:
            volume_trend = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0.0
        else:
            volume_trend = 0.0
        
        return {
            'avg_volume': avg_volume,
            'volume_trend': volume_trend
        }


class SingleCandlestickDetector:
    """Detects single candlestick patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_body_ratio = 0.1  # Minimum body size ratio
        self.doji_threshold = 0.1  # Doji body threshold
        self.shadow_ratio_threshold = 2.0  # Shadow to body ratio
    
    def detect_patterns(self, candlesticks: List[Candlestick], 
                       market_context: Dict[str, Any] = None) -> List[CandlestickPattern]:
        """Detect single candlestick patterns."""
        patterns = []
        
        try:
            if not candlesticks:
                return patterns
            
            current_candle = candlesticks[-1]
            
            # Doji patterns
            doji_pattern = self._detect_doji(current_candle, market_context)
            if doji_pattern:
                patterns.append(doji_pattern)
            
            # Hammer patterns
            hammer_pattern = self._detect_hammer(current_candle, market_context)
            if hammer_pattern:
                patterns.append(hammer_pattern)
            
            # Shooting star patterns
            shooting_star_pattern = self._detect_shooting_star(current_candle, market_context)
            if shooting_star_pattern:
                patterns.append(shooting_star_pattern)
            
            # Spinning top patterns
            spinning_top_pattern = self._detect_spinning_top(current_candle, market_context)
            if spinning_top_pattern:
                patterns.append(spinning_top_pattern)
            
            # Marubozu patterns
            marubozu_pattern = self._detect_marubozu(current_candle, market_context)
            if marubozu_pattern:
                patterns.append(marubozu_pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting single candlestick patterns: {e}")
            return []
    
    def _detect_doji(self, candle: Candlestick, context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect doji pattern."""
        try:
            if not candle.is_doji:
                return None
            
            # Determine doji type and significance
            confidence = 0.6  # Base confidence
            signal_direction = 'neutral'
            strength = 0.5
            
            # Context-based adjustments
            if context:
                trend = context.get('trend', 'neutral')
                if trend == 'uptrend':
                    signal_direction = 'bearish'  # Doji in uptrend suggests reversal
                    confidence += 0.1
                elif trend == 'downtrend':
                    signal_direction = 'bullish'  # Doji in downtrend suggests reversal
                    confidence += 0.1
            
            # Long-legged doji (long shadows) is more significant
            if candle.total_range > 0:
                shadow_ratio = (candle.upper_shadow + candle.lower_shadow) / candle.total_range
                if shadow_ratio > 0.8:  # Long shadows
                    confidence += 0.15
                    strength += 0.2
            
            return CandlestickPattern(
                'doji',
                [candle],
                confidence,
                signal_direction,
                strength
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting doji: {e}")
            return None
    
    def _detect_hammer(self, candle: Candlestick, context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect hammer pattern."""
        try:
            if candle.total_range == 0:
                return None
            
            # Hammer criteria:
            # 1. Small body at the top of the range
            # 2. Long lower shadow (at least 2x body size)
            # 3. Little or no upper shadow
            
            body_ratio = candle.body_size / candle.total_range
            lower_shadow_ratio = candle.lower_shadow / candle.total_range
            upper_shadow_ratio = candle.upper_shadow / candle.total_range
            
            # Check hammer criteria
            if (body_ratio < 0.3 and  # Small body
                lower_shadow_ratio > 0.6 and  # Long lower shadow
                upper_shadow_ratio < 0.1 and  # Small upper shadow
                candle.lower_shadow > candle.body_size * 2):  # Lower shadow > 2x body
                
                confidence = 0.7
                strength = 0.7
                signal_direction = 'bullish'
                
                # Context adjustments
                if context:
                    trend = context.get('trend', 'neutral')
                    if trend == 'downtrend':
                        confidence += 0.15  # Hammer in downtrend is more significant
                        strength += 0.2
                    elif trend == 'uptrend':
                        confidence -= 0.1  # Less significant in uptrend
                
                # Volume confirmation
                if context and context.get('volume_above_average', False):
                    confidence += 0.1
                    strength += 0.1
                
                return CandlestickPattern(
                    'hammer',
                    [candle],
                    confidence,
                    signal_direction,
                    strength
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting hammer: {e}")
            return None
    
    def _detect_shooting_star(self, candle: Candlestick, context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect shooting star pattern."""
        try:
            if candle.total_range == 0:
                return None
            
            # Shooting star criteria:
            # 1. Small body at the bottom of the range
            # 2. Long upper shadow (at least 2x body size)
            # 3. Little or no lower shadow
            
            body_ratio = candle.body_size / candle.total_range
            upper_shadow_ratio = candle.upper_shadow / candle.total_range
            lower_shadow_ratio = candle.lower_shadow / candle.total_range
            
            # Check shooting star criteria
            if (body_ratio < 0.3 and  # Small body
                upper_shadow_ratio > 0.6 and  # Long upper shadow
                lower_shadow_ratio < 0.1 and  # Small lower shadow
                candle.upper_shadow > candle.body_size * 2):  # Upper shadow > 2x body
                
                confidence = 0.7
                strength = 0.7
                signal_direction = 'bearish'
                
                # Context adjustments
                if context:
                    trend = context.get('trend', 'neutral')
                    if trend == 'uptrend':
                        confidence += 0.15  # Shooting star in uptrend is more significant
                        strength += 0.2
                    elif trend == 'downtrend':
                        confidence -= 0.1  # Less significant in downtrend
                
                # Volume confirmation
                if context and context.get('volume_above_average', False):
                    confidence += 0.1
                    strength += 0.1
                
                return CandlestickPattern(
                    'shooting_star',
                    [candle],
                    confidence,
                    signal_direction,
                    strength
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting shooting star: {e}")
            return None
    
    def _detect_spinning_top(self, candle: Candlestick, context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect spinning top pattern."""
        try:
            if candle.total_range == 0:
                return None
            
            # Spinning top criteria:
            # 1. Small body
            # 2. Long upper and lower shadows
            # 3. Body in the middle of the range
            
            body_ratio = candle.body_size / candle.total_range
            upper_shadow_ratio = candle.upper_shadow / candle.total_range
            lower_shadow_ratio = candle.lower_shadow / candle.total_range
            
            # Check spinning top criteria
            if (body_ratio < 0.3 and  # Small body
                upper_shadow_ratio > 0.25 and  # Decent upper shadow
                lower_shadow_ratio > 0.25 and  # Decent lower shadow
                abs(upper_shadow_ratio - lower_shadow_ratio) < 0.3):  # Balanced shadows
                
                confidence = 0.6
                strength = 0.5
                signal_direction = 'neutral'  # Indecision pattern
                
                # Context adjustments
                if context:
                    trend = context.get('trend', 'neutral')
                    if trend in ['uptrend', 'downtrend']:
                        confidence += 0.1  # More significant in trending markets
                        signal_direction = 'reversal'  # Suggests potential reversal
                
                return CandlestickPattern(
                    'spinning_top',
                    [candle],
                    confidence,
                    signal_direction,
                    strength
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting spinning top: {e}")
            return None
    
    def _detect_marubozu(self, candle: Candlestick, context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect marubozu pattern."""
        try:
            if candle.total_range == 0:
                return None
            
            # Marubozu criteria:
            # 1. Large body (>70% of total range)
            # 2. Very small or no shadows
            
            body_ratio = candle.body_size / candle.total_range
            shadow_ratio = (candle.upper_shadow + candle.lower_shadow) / candle.total_range
            
            # Check marubozu criteria
            if body_ratio > 0.7 and shadow_ratio < 0.1:
                confidence = 0.75
                strength = 0.8
                signal_direction = 'bullish' if candle.is_bullish else 'bearish'
                
                # Context adjustments
                if context:
                    trend = context.get('trend', 'neutral')
                    if ((signal_direction == 'bullish' and trend == 'uptrend') or
                        (signal_direction == 'bearish' and trend == 'downtrend')):
                        confidence += 0.1  # Continuation pattern
                        strength += 0.1
                
                # Volume confirmation
                if context and context.get('volume_above_average', False):
                    confidence += 0.15
                    strength += 0.1
                
                pattern_type = 'bullish_marubozu' if candle.is_bullish else 'bearish_marubozu'
                
                return CandlestickPattern(
                    pattern_type,
                    [candle],
                    confidence,
                    signal_direction,
                    strength
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting marubozu: {e}")
            return None


class TwoCandlestickDetector:
    """Detects two-candlestick patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.engulfing_threshold = 1.0  # Second candle should engulf first
        self.harami_threshold = 0.8  # First candle should contain second
    
    def detect_patterns(self, candlesticks: List[Candlestick], 
                       market_context: Dict[str, Any] = None) -> List[CandlestickPattern]:
        """Detect two-candlestick patterns."""
        patterns = []
        
        try:
            if len(candlesticks) < 2:
                return patterns
            
            first_candle = candlesticks[-2]
            second_candle = candlesticks[-1]
            
            # Engulfing patterns
            engulfing_pattern = self._detect_engulfing(first_candle, second_candle, market_context)
            if engulfing_pattern:
                patterns.append(engulfing_pattern)
            
            # Harami patterns
            harami_pattern = self._detect_harami(first_candle, second_candle, market_context)
            if harami_pattern:
                patterns.append(harami_pattern)
            
            # Piercing line / Dark cloud cover
            piercing_pattern = self._detect_piercing_line(first_candle, second_candle, market_context)
            if piercing_pattern:
                patterns.append(piercing_pattern)
            
            dark_cloud_pattern = self._detect_dark_cloud_cover(first_candle, second_candle, market_context)
            if dark_cloud_pattern:
                patterns.append(dark_cloud_pattern)
            
            # Tweezer patterns
            tweezer_pattern = self._detect_tweezer(first_candle, second_candle, market_context)
            if tweezer_pattern:
                patterns.append(tweezer_pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting two-candlestick patterns: {e}")
            return []
    
    def _detect_engulfing(self, first: Candlestick, second: Candlestick, 
                         context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect bullish/bearish engulfing patterns."""
        try:
            # Engulfing criteria:
            # 1. Opposite colors
            # 2. Second candle's body completely engulfs first candle's body
            # 3. First candle should have a reasonable body size
            
            if first.body_size == 0 or second.body_size == 0:
                return None
            
            # Check for bullish engulfing
            if (first.is_bearish and second.is_bullish and
                second.open < first.close and second.close > first.open):
                
                # Calculate engulfing ratio
                engulfing_ratio = second.body_size / first.body_size
                
                if engulfing_ratio >= self.engulfing_threshold:
                    confidence = 0.75
                    strength = 0.8
                    signal_direction = 'bullish'
                    
                    # Context adjustments
                    if context:
                        trend = context.get('trend', 'neutral')
                        if trend == 'downtrend':
                            confidence += 0.15  # More significant in downtrend
                            strength += 0.15
                    
                    # Volume confirmation
                    if second.volume > first.volume * 1.2:
                        confidence += 0.1
                        strength += 0.1
                    
                    return CandlestickPattern(
                        'bullish_engulfing',
                        [first, second],
                        confidence,
                        signal_direction,
                        strength
                    )
            
            # Check for bearish engulfing
            elif (first.is_bullish and second.is_bearish and
                  second.open > first.close and second.close < first.open):
                
                # Calculate engulfing ratio
                engulfing_ratio = second.body_size / first.body_size
                
                if engulfing_ratio >= self.engulfing_threshold:
                    confidence = 0.75
                    strength = 0.8
                    signal_direction = 'bearish'
                    
                    # Context adjustments
                    if context:
                        trend = context.get('trend', 'neutral')
                        if trend == 'uptrend':
                            confidence += 0.15  # More significant in uptrend
                            strength += 0.15
                    
                    # Volume confirmation
                    if second.volume > first.volume * 1.2:
                        confidence += 0.1
                        strength += 0.1
                    
                    return CandlestickPattern(
                        'bearish_engulfing',
                        [first, second],
                        confidence,
                        signal_direction,
                        strength
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting engulfing pattern: {e}")
            return None
    
    def _detect_harami(self, first: Candlestick, second: Candlestick, 
                      context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect harami patterns."""
        try:
            # Harami criteria:
            # 1. Opposite colors
            # 2. First candle has large body
            # 3. Second candle's body is contained within first candle's body
            
            if first.body_size == 0 or second.body_size == 0:
                return None
            
            # First candle should be significant
            if first.total_range > 0 and first.body_size / first.total_range < 0.5:
                return None
            
            # Check if second candle is contained within first
            first_body_high = max(first.open, first.close)
            first_body_low = min(first.open, first.close)
            second_body_high = max(second.open, second.close)
            second_body_low = min(second.open, second.close)
            
            if (second_body_high < first_body_high and second_body_low > first_body_low):
                
                # Calculate harami ratio
                harami_ratio = second.body_size / first.body_size
                
                if harami_ratio < self.harami_threshold:
                    confidence = 0.65
                    strength = 0.6
                    
                    # Determine signal direction
                    if first.is_bearish and second.is_bullish:
                        signal_direction = 'bullish'
                        pattern_type = 'bullish_harami'
                    elif first.is_bullish and second.is_bearish:
                        signal_direction = 'bearish'
                        pattern_type = 'bearish_harami'
                    else:
                        return None  # Same color harami is less significant
                    
                    # Context adjustments
                    if context:
                        trend = context.get('trend', 'neutral')
                        if ((signal_direction == 'bullish' and trend == 'downtrend') or
                            (signal_direction == 'bearish' and trend == 'uptrend')):
                            confidence += 0.15
                            strength += 0.1
                    
                    # Special case: Doji harami (second candle is doji)
                    if second.is_doji:
                        confidence += 0.1
                        pattern_type = f'{signal_direction}_harami_doji'
                    
                    return CandlestickPattern(
                        pattern_type,
                        [first, second],
                        confidence,
                        signal_direction,
                        strength
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting harami pattern: {e}")
            return None
    
    def _detect_piercing_line(self, first: Candlestick, second: Candlestick, 
                             context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect piercing line pattern (bullish)."""
        try:
            # Piercing line criteria:
            # 1. First candle is bearish with good body size
            # 2. Second candle is bullish
            # 3. Second candle opens below first's low
            # 4. Second candle closes above midpoint of first candle's body
            
            if not (first.is_bearish and second.is_bullish):
                return None
            
            if first.body_size == 0:
                return None
            
            # Check opening gap down
            if second.open >= first.low:
                return None
            
            # Check if second candle closes above midpoint of first body
            first_midpoint = (first.open + first.close) / 2
            if second.close <= first_midpoint:
                return None
            
            # Calculate penetration ratio
            penetration = (second.close - first.close) / first.body_size
            
            if penetration > 0.5:  # Should penetrate at least 50%
                confidence = 0.7
                strength = 0.75
                signal_direction = 'bullish'
                
                # Context adjustments
                if context:
                    trend = context.get('trend', 'neutral')
                    if trend == 'downtrend':
                        confidence += 0.15
                        strength += 0.1
                
                # Volume confirmation
                if second.volume > first.volume:
                    confidence += 0.1
                
                return CandlestickPattern(
                    'piercing_line',
                    [first, second],
                    confidence,
                    signal_direction,
                    strength
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting piercing line: {e}")
            return None
    
    def _detect_dark_cloud_cover(self, first: Candlestick, second: Candlestick, 
                                context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect dark cloud cover pattern (bearish)."""
        try:
            # Dark cloud cover criteria:
            # 1. First candle is bullish with good body size
            # 2. Second candle is bearish
            # 3. Second candle opens above first's high
            # 4. Second candle closes below midpoint of first candle's body
            
            if not (first.is_bullish and second.is_bearish):
                return None
            
            if first.body_size == 0:
                return None
            
            # Check opening gap up
            if second.open <= first.high:
                return None
            
            # Check if second candle closes below midpoint of first body
            first_midpoint = (first.open + first.close) / 2
            if second.close >= first_midpoint:
                return None
            
            # Calculate penetration ratio
            penetration = (first.close - second.close) / first.body_size
            
            if penetration > 0.5:  # Should penetrate at least 50%
                confidence = 0.7
                strength = 0.75
                signal_direction = 'bearish'
                
                # Context adjustments
                if context:
                    trend = context.get('trend', 'neutral')
                    if trend == 'uptrend':
                        confidence += 0.15
                        strength += 0.1
                
                # Volume confirmation
                if second.volume > first.volume:
                    confidence += 0.1
                
                return CandlestickPattern(
                    'dark_cloud_cover',
                    [first, second],
                    confidence,
                    signal_direction,
                    strength
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting dark cloud cover: {e}")
            return None
    
    def _detect_tweezer(self, first: Candlestick, second: Candlestick, 
                       context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect tweezer top/bottom patterns."""
        try:
            # Tweezer criteria:
            # 1. Similar highs (tweezer top) or lows (tweezer bottom)
            # 2. Opposite colors preferred
            # 3. Should occur at potential reversal points
            
            price_tolerance = 0.002  # 0.2% tolerance for "same" level
            
            # Check for tweezer top
            high_diff = abs(first.high - second.high) / max(first.high, second.high)
            if high_diff < price_tolerance:
                confidence = 0.6
                strength = 0.6
                signal_direction = 'bearish'
                pattern_type = 'tweezer_top'
                
                # Stronger if opposite colors
                if first.is_bullish and second.is_bearish:
                    confidence += 0.1
                    strength += 0.1
                
                # Context adjustments
                if context:
                    trend = context.get('trend', 'neutral')
                    if trend == 'uptrend':
                        confidence += 0.15
                        strength += 0.1
                
                return CandlestickPattern(
                    pattern_type,
                    [first, second],
                    confidence,
                    signal_direction,
                    strength
                )
            
            # Check for tweezer bottom
            low_diff = abs(first.low - second.low) / max(first.low, second.low)
            if low_diff < price_tolerance:
                confidence = 0.6
                strength = 0.6
                signal_direction = 'bullish'
                pattern_type = 'tweezer_bottom'
                
                # Stronger if opposite colors
                if first.is_bearish and second.is_bullish:
                    confidence += 0.1
                    strength += 0.1
                
                # Context adjustments
                if context:
                    trend = context.get('trend', 'neutral')
                    if trend == 'downtrend':
                        confidence += 0.15
                        strength += 0.1
                
                return CandlestickPattern(
                    pattern_type,
                    [first, second],
                    confidence,
                    signal_direction,
                    strength
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting tweezer pattern: {e}")
            return None


class ThreeCandlestickDetector:
    """Detects three-candlestick patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.star_gap_threshold = 0.001  # Minimum gap for star patterns
    
    def detect_patterns(self, candlesticks: List[Candlestick], 
                       market_context: Dict[str, Any] = None) -> List[CandlestickPattern]:
        """Detect three-candlestick patterns."""
        patterns = []
        
        try:
            if len(candlesticks) < 3:
                return patterns
            
            first_candle = candlesticks[-3]
            second_candle = candlesticks[-2]
            third_candle = candlesticks[-1]
            
            # Morning star pattern
            morning_star = self._detect_morning_star(first_candle, second_candle, third_candle, market_context)
            if morning_star:
                patterns.append(morning_star)
            
            # Evening star pattern
            evening_star = self._detect_evening_star(first_candle, second_candle, third_candle, market_context)
            if evening_star:
                patterns.append(evening_star)
            
            # Three white soldiers
            three_soldiers = self._detect_three_white_soldiers(first_candle, second_candle, third_candle, market_context)
            if three_soldiers:
                patterns.append(three_soldiers)
            
            # Three black crows
            three_crows = self._detect_three_black_crows(first_candle, second_candle, third_candle, market_context)
            if three_crows:
                patterns.append(three_crows)
            
            # Three inside up/down
            three_inside = self._detect_three_inside(first_candle, second_candle, third_candle, market_context)
            if three_inside:
                patterns.append(three_inside)
            
            # Three outside up/down
            three_outside = self._detect_three_outside(first_candle, second_candle, third_candle, market_context)
            if three_outside:
                patterns.append(three_outside)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting three-candlestick patterns: {e}")
            return []
    
    def _detect_morning_star(self, first: Candlestick, second: Candlestick, third: Candlestick, 
                            context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect morning star pattern (bullish reversal)."""
        try:
            # Morning star criteria:
            # 1. First candle is bearish with good body
            # 2. Second candle is small (star) with gap down
            # 3. Third candle is bullish and closes above midpoint of first candle
            
            if not first.is_bearish or not third.is_bullish:
                return None
            
            if first.body_size == 0:
                return None
            
            # Check for gap down to star
            if second.high >= first.close:
                return None
            
            # Check for gap up from star
            if third.open <= second.high:
                return None
            
            # Third candle should close above midpoint of first
            first_midpoint = (first.open + first.close) / 2
            if third.close <= first_midpoint:
                return None
            
            # Star should be small relative to first candle
            if second.body_size > first.body_size * 0.5:
                return None
            
            confidence = 0.8
            strength = 0.85
            signal_direction = 'bullish'
            
            # Context adjustments
            if context:
                trend = context.get('trend', 'neutral')
                if trend == 'downtrend':
                    confidence += 0.15
                    strength += 0.1
            
            # Doji star is stronger
            if second.is_doji:
                confidence += 0.05
                pattern_type = 'morning_doji_star'
            else:
                pattern_type = 'morning_star'
            
            # Volume confirmation
            if third.volume > (first.volume + second.volume) / 2:
                confidence += 0.05
            
            return CandlestickPattern(
                pattern_type,
                [first, second, third],
                confidence,
                signal_direction,
                strength
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting morning star: {e}")
            return None
    
    def _detect_evening_star(self, first: Candlestick, second: Candlestick, third: Candlestick, 
                            context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect evening star pattern (bearish reversal)."""
        try:
            # Evening star criteria:
            # 1. First candle is bullish with good body
            # 2. Second candle is small (star) with gap up
            # 3. Third candle is bearish and closes below midpoint of first candle
            
            if not first.is_bullish or not third.is_bearish:
                return None
            
            if first.body_size == 0:
                return None
            
            # Check for gap up to star
            if second.low <= first.close:
                return None
            
            # Check for gap down from star
            if third.open >= second.low:
                return None
            
            # Third candle should close below midpoint of first
            first_midpoint = (first.open + first.close) / 2
            if third.close >= first_midpoint:
                return None
            
            # Star should be small relative to first candle
            if second.body_size > first.body_size * 0.5:
                return None
            
            confidence = 0.8
            strength = 0.85
            signal_direction = 'bearish'
            
            # Context adjustments
            if context:
                trend = context.get('trend', 'neutral')
                if trend == 'uptrend':
                    confidence += 0.15
                    strength += 0.1
            
            # Doji star is stronger
            if second.is_doji:
                confidence += 0.05
                pattern_type = 'evening_doji_star'
            else:
                pattern_type = 'evening_star'
            
            # Volume confirmation
            if third.volume > (first.volume + second.volume) / 2:
                confidence += 0.05
            
            return CandlestickPattern(
                pattern_type,
                [first, second, third],
                confidence,
                signal_direction,
                strength
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting evening star: {e}")
            return None
    
    def _detect_three_white_soldiers(self, first: Candlestick, second: Candlestick, third: Candlestick, 
                                    context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect three white soldiers pattern (bullish continuation)."""
        try:
            # Three white soldiers criteria:
            # 1. Three consecutive bullish candles
            # 2. Each candle opens within previous candle's body
            # 3. Each candle closes higher than previous
            # 4. Bodies should be of similar size
            
            if not (first.is_bullish and second.is_bullish and third.is_bullish):
                return None
            
            # Check opening within previous body
            if not (first.open < second.open < first.close and
                    second.open < third.open < second.close):
                return None
            
            # Check progressive closes
            if not (first.close < second.close < third.close):
                return None
            
            # Check body sizes are reasonable and similar
            bodies = [first.body_size, second.body_size, third.body_size]
            avg_body = statistics.mean(bodies)
            
            if avg_body == 0:
                return None
            
            # Bodies shouldn't vary too much
            body_variance = statistics.variance(bodies) / (avg_body ** 2)
            if body_variance > 0.5:  # Too much variance
                return None
            
            confidence = 0.75
            strength = 0.8
            signal_direction = 'bullish'
            
            # Context adjustments
            if context:
                trend = context.get('trend', 'neutral')
                if trend == 'uptrend':
                    confidence += 0.1  # Continuation pattern
                elif trend == 'downtrend':
                    confidence += 0.15  # Reversal pattern
            
            # Volume should ideally increase
            volumes = [first.volume, second.volume, third.volume]
            if volumes[2] > volumes[0]:  # Increasing volume
                confidence += 0.1
            
            return CandlestickPattern(
                'three_white_soldiers',
                [first, second, third],
                confidence,
                signal_direction,
                strength
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting three white soldiers: {e}")
            return None
    
    def _detect_three_black_crows(self, first: Candlestick, second: Candlestick, third: Candlestick, 
                                 context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect three black crows pattern (bearish continuation)."""
        try:
            # Three black crows criteria:
            # 1. Three consecutive bearish candles
            # 2. Each candle opens within previous candle's body
            # 3. Each candle closes lower than previous
            # 4. Bodies should be of similar size
            
            if not (first.is_bearish and second.is_bearish and third.is_bearish):
                return None
            
            # Check opening within previous body
            if not (first.close < second.open < first.open and
                    second.close < third.open < second.open):
                return None
            
            # Check progressive closes
            if not (first.close > second.close > third.close):
                return None
            
            # Check body sizes are reasonable and similar
            bodies = [first.body_size, second.body_size, third.body_size]
            avg_body = statistics.mean(bodies)
            
            if avg_body == 0:
                return None
            
            # Bodies shouldn't vary too much
            body_variance = statistics.variance(bodies) / (avg_body ** 2)
            if body_variance > 0.5:  # Too much variance
                return None
            
            confidence = 0.75
            strength = 0.8
            signal_direction = 'bearish'
            
            # Context adjustments
            if context:
                trend = context.get('trend', 'neutral')
                if trend == 'downtrend':
                    confidence += 0.1  # Continuation pattern
                elif trend == 'uptrend':
                    confidence += 0.15  # Reversal pattern
            
            # Volume should ideally increase
            volumes = [first.volume, second.volume, third.volume]
            if volumes[2] > volumes[0]:  # Increasing volume
                confidence += 0.1
            
            return CandlestickPattern(
                'three_black_crows',
                [first, second, third],
                confidence,
                signal_direction,
                strength
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting three black crows: {e}")
            return None
    
    def _detect_three_inside(self, first: Candlestick, second: Candlestick, third: Candlestick, 
                            context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect three inside up/down patterns."""
        try:
            # Three inside criteria:
            # 1. First two candles form harami pattern
            # 2. Third candle confirms the direction
            
            # Check for harami in first two candles
            if first.body_size == 0 or second.body_size == 0:
                return None
            
            first_body_high = max(first.open, first.close)
            first_body_low = min(first.open, first.close)
            second_body_high = max(second.open, second.close)
            second_body_low = min(second.open, second.close)
            
            # Second candle should be inside first
            if not (second_body_high < first_body_high and second_body_low > first_body_low):
                return None
            
            # Three inside up
            if (first.is_bearish and second.is_bullish and third.is_bullish and
                third.close > first.close):
                
                confidence = 0.75
                strength = 0.75
                signal_direction = 'bullish'
                pattern_type = 'three_inside_up'
                
                # Context adjustments
                if context:
                    trend = context.get('trend', 'neutral')
                    if trend == 'downtrend':
                        confidence += 0.15
                        strength += 0.1
                
                return CandlestickPattern(
                    pattern_type,
                    [first, second, third],
                    confidence,
                    signal_direction,
                    strength
                )
            
            # Three inside down
            elif (first.is_bullish and second.is_bearish and third.is_bearish and
                  third.close < first.close):
                
                confidence = 0.75
                strength = 0.75
                signal_direction = 'bearish'
                pattern_type = 'three_inside_down'
                
                # Context adjustments
                if context:
                    trend = context.get('trend', 'neutral')
                    if trend == 'uptrend':
                        confidence += 0.15
                        strength += 0.1
                
                return CandlestickPattern(
                    pattern_type,
                    [first, second, third],
                    confidence,
                    signal_direction,
                    strength
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting three inside pattern: {e}")
            return None
    
    def _detect_three_outside(self, first: Candlestick, second: Candlestick, third: Candlestick, 
                             context: Dict[str, Any] = None) -> Optional[CandlestickPattern]:
        """Detect three outside up/down patterns."""
        try:
            # Three outside criteria:
            # 1. First two candles form engulfing pattern
            # 2. Third candle confirms the direction
            
            # Check for engulfing in first two candles
            if first.body_size == 0 or second.body_size == 0:
                return None
            
            # Three outside up (bullish engulfing + confirmation)
            if (first.is_bearish and second.is_bullish and third.is_bullish and
                second.open < first.close and second.close > first.open and
                third.close > second.close):
                
                confidence = 0.8
                strength = 0.8
                signal_direction = 'bullish'
                pattern_type = 'three_outside_up'
                
                # Context adjustments
                if context:
                    trend = context.get('trend', 'neutral')
                    if trend == 'downtrend':
                        confidence += 0.15
                        strength += 0.1
                
                # Volume confirmation
                if third.volume > second.volume:
                    confidence += 0.05
                
                return CandlestickPattern(
                    pattern_type,
                    [first, second, third],
                    confidence,
                    signal_direction,
                    strength
                )
            
            # Three outside down (bearish engulfing + confirmation)
            elif (first.is_bullish and second.is_bearish and third.is_bearish and
                  second.open > first.close and second.close < first.open and
                  third.close < second.close):
                
                confidence = 0.8
                strength = 0.8
                signal_direction = 'bearish'
                pattern_type = 'three_outside_down'
                
                # Context adjustments
                if context:
                    trend = context.get('trend', 'neutral')
                    if trend == 'uptrend':
                        confidence += 0.15
                        strength += 0.1
                
                # Volume confirmation
                if third.volume > second.volume:
                    confidence += 0.05
                
                return CandlestickPattern(
                    pattern_type,
                    [first, second, third],
                    confidence,
                    signal_direction,
                    strength
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting three outside pattern: {e}")
            return None


class CandlestickStrategy(BaseStrategy):
    """Candlestick pattern recognition trading strategy with fallback activation."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__("CandlestickStrategy", parameters)
        
        # Default parameters
        default_params = {
            'lookback_periods': 50,
            'min_pattern_confidence': 0.65,
            'enable_single_patterns': True,
            'enable_two_patterns': True,
            'enable_three_patterns': True,
            'volume_confirmation': True,
            'trend_confirmation': True,
            'position_size_factor': 1.0,
            'liquidity_fallback_threshold': 0.4,  # Activate when liquidity confidence < 40%
            'fallback_confidence_boost': 0.15,    # Boost confidence when in fallback mode
            'max_active_patterns': 5,
            'pattern_timeout_minutes': 60
        }
        
        self.parameters.update(default_params)
        if parameters:
            self.parameters.update(parameters)
        
        # Initialize pattern detectors
        self.single_detector = SingleCandlestickDetector()
        self.two_detector = TwoCandlestickDetector()
        self.three_detector = ThreeCandlestickDetector()
        
        # Strategy state
        self.candlestick_history = deque(maxlen=self.parameters['lookback_periods'])
        self.active_patterns = deque(maxlen=self.parameters['max_active_patterns'])
        self.pattern_history = deque(maxlen=200)
        
        # Fallback activation state
        self.is_fallback_active = False
        self.fallback_activation_time = None
        self.liquidity_confidence_history = deque(maxlen=10)
        
        # Performance metrics
        self.performance_metrics = {
            'total_patterns': 0,
            'successful_patterns': 0,
            'failed_patterns': 0,
            'pattern_type_performance': {},
            'fallback_activations': 0,
            'fallback_success_rate': 0.0,
            'avg_confidence': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_market(self, market_data: MarketData) -> Optional[TradingSignal]:
        """Analyze market data for candlestick patterns and generate trading signals."""
        try:
            # Create candlestick from market data
            candlestick = self._create_candlestick(market_data)
            if not candlestick:
                return None
            
            # Update candlestick history
            self.candlestick_history.append(candlestick)
            
            # Need at least 3 candlesticks for pattern detection
            if len(self.candlestick_history) < 3:
                return None
            
            # Check for fallback activation
            self._check_fallback_activation(market_data)
            
            # Clean up expired patterns
            self._cleanup_expired_patterns()
            
            # Detect patterns
            detected_patterns = self._detect_all_patterns(market_data)
            
            # Update active patterns
            self._update_active_patterns(detected_patterns)
            
            # Generate trading signal from best pattern
            trading_signal = self._generate_trading_signal(market_data)
            
            if trading_signal:
                self._update_performance_metrics(trading_signal)
                return trading_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in candlestick analysis: {e}")
            return None
    
    def _create_candlestick(self, market_data: MarketData) -> Optional[Candlestick]:
        """Create candlestick from market data."""
        try:
            # For real-time data, we need to aggregate tick data into candlesticks
            # For now, we'll simulate OHLC from current price
            price = market_data.price
            
            # Estimate OHLC (in real implementation, this would come from proper aggregation)
            open_price = price * (1 + (hash(str(market_data.timestamp)) % 21 - 10) / 10000)
            high_price = max(price, open_price) * (1 + abs(hash(str(market_data.timestamp)) % 11) / 10000)
            low_price = min(price, open_price) * (1 - abs(hash(str(market_data.timestamp)) % 11) / 10000)
            close_price = price
            
            return Candlestick(
                timestamp=market_data.timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=market_data.volume or 0
            )
            
        except Exception as e:
            self.logger.error(f"Error creating candlestick: {e}")
            return None
    
    def _check_fallback_activation(self, market_data: MarketData) -> None:
        """Check if fallback mode should be activated based on liquidity conditions."""
        try:
            # Get liquidity confidence from market data metadata
            liquidity_confidence = getattr(market_data, 'liquidity_confidence', None)
            
            if liquidity_confidence is not None:
                self.liquidity_confidence_history.append(liquidity_confidence)
                
                # Calculate average liquidity confidence
                if len(self.liquidity_confidence_history) >= 3:
                    avg_liquidity = statistics.mean(list(self.liquidity_confidence_history)[-3:])
                    
                    # Activate fallback if liquidity is consistently low
                    if avg_liquidity < self.parameters['liquidity_fallback_threshold']:
                        if not self.is_fallback_active:
                            self.is_fallback_active = True
                            self.fallback_activation_time = datetime.now()
                            self.performance_metrics['fallback_activations'] += 1
                            self.logger.info(f"Candlestick fallback activated - liquidity confidence: {avg_liquidity:.2f}")
                    
                    # Deactivate fallback if liquidity improves
                    elif avg_liquidity > self.parameters['liquidity_fallback_threshold'] + 0.1:  # Hysteresis
                        if self.is_fallback_active:
                            self.is_fallback_active = False
                            self.fallback_activation_time = None
                            self.logger.info(f"Candlestick fallback deactivated - liquidity confidence: {avg_liquidity:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error checking fallback activation: {e}")
    
    def _detect_all_patterns(self, market_data: MarketData) -> List[CandlestickPattern]:
        """Detect all enabled candlestick patterns."""
        patterns = []
        
        try:
            candlesticks = list(self.candlestick_history)
            market_context = self._build_market_context(market_data)
            
            # Single candlestick patterns
            if self.parameters['enable_single_patterns']:
                single_patterns = self.single_detector.detect_patterns(candlesticks, market_context)
                patterns.extend(single_patterns)
            
            # Two candlestick patterns
            if self.parameters['enable_two_patterns']:
                two_patterns = self.two_detector.detect_patterns(candlesticks, market_context)
                patterns.extend(two_patterns)
            
            # Three candlestick patterns
            if self.parameters['enable_three_patterns']:
                three_patterns = self.three_detector.detect_patterns(candlesticks, market_context)
                patterns.extend(three_patterns)
            
            # Filter by minimum confidence
            min_confidence = self.parameters['min_pattern_confidence']
            
            # Apply fallback confidence boost if active
            if self.is_fallback_active:
                for pattern in patterns:
                    pattern.confidence += self.parameters['fallback_confidence_boost']
                    pattern.confidence = min(pattern.confidence, 0.95)  # Cap at 95%
            
            # Filter patterns
            filtered_patterns = [p for p in patterns if p.confidence >= min_confidence]
            
            return filtered_patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _build_market_context(self, market_data: MarketData) -> Dict[str, Any]:
        """Build market context for pattern detection."""
        try:
            context = {}
            
            # Determine trend
            if len(self.candlestick_history) >= 10:
                recent_closes = [c.close for c in list(self.candlestick_history)[-10:]]
                
                # Simple trend detection
                if recent_closes[-1] > recent_closes[0] * 1.02:  # 2% higher
                    context['trend'] = 'uptrend'
                elif recent_closes[-1] < recent_closes[0] * 0.98:  # 2% lower
                    context['trend'] = 'downtrend'
                else:
                    context['trend'] = 'neutral'
            else:
                context['trend'] = 'neutral'
            
            # Volume analysis
            if len(self.candlestick_history) >= 5:
                recent_volumes = [c.volume for c in list(self.candlestick_history)[-5:]]
                avg_volume = statistics.mean(recent_volumes)
                current_volume = self.candlestick_history[-1].volume
                
                context['volume_above_average'] = current_volume > avg_volume * 1.2
                context['avg_volume'] = avg_volume
            
            # Volatility context
            if len(self.candlestick_history) >= 20:
                recent_closes = [c.close for c in list(self.candlestick_history)[-20:]]
                returns = [(recent_closes[i] - recent_closes[i-1]) / recent_closes[i-1] 
                          for i in range(1, len(recent_closes))]
                volatility = statistics.stdev(returns) if len(returns) > 1 else 0
                context['volatility'] = volatility
                context['high_volatility'] = volatility > 0.02  # 2% daily volatility
            
            # Fallback mode context
            context['fallback_active'] = self.is_fallback_active
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error building market context: {e}")
            return {}
    
    def _update_active_patterns(self, new_patterns: List[CandlestickPattern]) -> None:
        """Update active patterns list."""
        try:
            # Add new patterns
            for pattern in new_patterns:
                if len(self.active_patterns) < self.parameters['max_active_patterns']:
                    self.active_patterns.append(pattern)
                    self.pattern_history.append({
                        'pattern': pattern,
                        'detected_at': pattern.detected_at,
                        'status': 'active',
                        'fallback_mode': self.is_fallback_active
                    })
            
            # Sort by confidence and strength
            sorted_patterns = sorted(self.active_patterns, 
                                   key=lambda x: x.confidence * x.strength, reverse=True)
            
            # Keep only the best patterns
            self.active_patterns.clear()
            self.active_patterns.extend(sorted_patterns[:self.parameters['max_active_patterns']])
            
        except Exception as e:
            self.logger.error(f"Error updating active patterns: {e}")
    
    def _cleanup_expired_patterns(self) -> None:
        """Remove expired patterns."""
        try:
            timeout_delta = timedelta(minutes=self.parameters['pattern_timeout_minutes'])
            current_time = datetime.now()
            
            # Filter out expired patterns
            active_patterns = []
            for pattern in self.active_patterns:
                if current_time - pattern.detected_at < timeout_delta:
                    active_patterns.append(pattern)
            
            self.active_patterns.clear()
            self.active_patterns.extend(active_patterns)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up patterns: {e}")
    
    def _generate_trading_signal(self, market_data: MarketData) -> Optional[TradingSignal]:
        """Generate trading signal from best active pattern."""
        try:
            if not self.active_patterns:
                return None
            
            # Get the best pattern
            best_pattern = self.active_patterns[0]
            
            # Only generate signals for directional patterns
            if best_pattern.signal_direction in ['neutral', 'reversal']:
                return None
            
            # Determine signal action
            if best_pattern.signal_direction == 'bullish':
                action = SignalAction.BUY
            elif best_pattern.signal_direction == 'bearish':
                action = SignalAction.SELL
            else:
                return None
            
            # Calculate position size
            position_size = self._calculate_position_size(best_pattern, market_data)
            
            # Calculate target and stop loss
            target_price, stop_loss = self._calculate_targets(best_pattern, market_data)
            
            # Create trading signal
            trading_signal = TradingSignal(
                symbol=market_data.symbol,
                action=action,
                price=market_data.price,
                quantity=position_size,
                confidence=best_pattern.confidence,
                timestamp=market_data.timestamp,
                strategy_name=self.name,
                metadata={
                    'pattern_type': best_pattern.pattern_type,
                    'pattern_strength': best_pattern.strength,
                    'pattern_range': best_pattern.pattern_range,
                    'volume_profile': best_pattern.volume_profile,
                    'candlesticks': [
                        {
                            'timestamp': c.timestamp,
                            'open': c.open,
                            'high': c.high,
                            'low': c.low,
                            'close': c.close,
                            'volume': c.volume
                        }
                        for c in best_pattern.candlesticks
                    ],
                    'target_price': target_price,
                    'stop_loss': stop_loss,
                    'fallback_mode': self.is_fallback_active,
                    'signal_reasoning': f"{best_pattern.pattern_type} pattern with {best_pattern.confidence:.1%} confidence"
                }
            )
            
            # Remove the pattern after generating signal
            self.active_patterns.popleft()
            
            return trading_signal
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return None
    
    def _calculate_position_size(self, pattern: CandlestickPattern, market_data: MarketData) -> float:
        """Calculate position size based on pattern characteristics."""
        try:
            base_size = self.parameters['position_size_factor']
            
            # Adjust for pattern confidence and strength
            confidence_multiplier = pattern.confidence
            strength_multiplier = pattern.strength
            
            # Adjust for pattern type reliability
            type_multipliers = {
                'bullish_engulfing': 1.2,
                'bearish_engulfing': 1.2,
                'morning_star': 1.3,
                'evening_star': 1.3,
                'hammer': 1.1,
                'shooting_star': 1.1,
                'three_white_soldiers': 1.25,
                'three_black_crows': 1.25,
                'doji': 0.8,
                'spinning_top': 0.7
            }
            
            type_multiplier = type_multipliers.get(pattern.pattern_type, 1.0)
            
            # Fallback mode adjustment
            fallback_multiplier = 1.2 if self.is_fallback_active else 1.0
            
            # Calculate final position size
            position_size = (base_size * confidence_multiplier * strength_multiplier * 
                           type_multiplier * fallback_multiplier)
            
            # Cap position size
            max_size = base_size * 2.5
            return min(position_size, max_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self.parameters['position_size_factor']
    
    def _calculate_targets(self, pattern: CandlestickPattern, market_data: MarketData) -> Tuple[float, float]:
        """Calculate target price and stop loss."""
        try:
            current_price = market_data.price
            pattern_range = pattern.pattern_range
            
            if pattern_range == 0:
                pattern_range = current_price * 0.02  # Default 2% range
            
            # Calculate target and stop based on pattern direction
            if pattern.signal_direction == 'bullish':
                target_price = current_price + (pattern_range * 1.5)
                stop_loss = current_price - (pattern_range * 0.5)
            else:  # bearish
                target_price = current_price - (pattern_range * 1.5)
                stop_loss = current_price + (pattern_range * 0.5)
            
            return target_price, stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating targets: {e}")
            return market_data.price, market_data.price
    
    def _update_performance_metrics(self, signal: TradingSignal) -> None:
        """Update strategy performance metrics."""
        try:
            self.performance_metrics['total_patterns'] += 1
            
            # Update pattern type performance
            pattern_type = signal.metadata.get('pattern_type', 'unknown')
            if pattern_type not in self.performance_metrics['pattern_type_performance']:
                self.performance_metrics['pattern_type_performance'][pattern_type] = {
                    'count': 0, 'avg_confidence': 0.0, 'fallback_count': 0
                }
            
            type_perf = self.performance_metrics['pattern_type_performance'][pattern_type]
            type_perf['count'] += 1
            
            if signal.metadata.get('fallback_mode', False):
                type_perf['fallback_count'] += 1
            
            # Update average confidence
            total_conf = type_perf['avg_confidence'] * (type_perf['count'] - 1) + signal.confidence
            type_perf['avg_confidence'] = total_conf / type_perf['count']
            
            # Update overall metrics
            total_conf = (self.performance_metrics['avg_confidence'] * 
                         (self.performance_metrics['total_patterns'] - 1) + signal.confidence)
            self.performance_metrics['avg_confidence'] = total_conf / self.performance_metrics['total_patterns']
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate strategy parameters."""
        try:
            required_params = [
                'lookback_periods', 'min_pattern_confidence', 'position_size_factor',
                'liquidity_fallback_threshold'
            ]
            
            for param in required_params:
                if param not in parameters:
                    return False
            
            # Validate ranges
            if not (10 <= parameters['lookback_periods'] <= 200):
                return False
            
            if not (0.0 <= parameters['min_pattern_confidence'] <= 1.0):
                return False
            
            if not (0.0 <= parameters['liquidity_fallback_threshold'] <= 1.0):
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
                'candlestick_history_length': len(self.candlestick_history),
                'active_patterns': len(self.active_patterns),
                'pattern_history_length': len(self.pattern_history),
                'is_fallback_active': self.is_fallback_active,
                'fallback_activation_time': self.fallback_activation_time,
                'liquidity_confidence_avg': (
                    statistics.mean(self.liquidity_confidence_history) 
                    if self.liquidity_confidence_history else None
                ),
                'active_pattern_types': [p.pattern_type for p in self.active_patterns]
            }
        }