"""
Momentum trading strategy with multi-timeframe analysis.

This strategy analyzes price momentum across multiple timeframes using
RSI, MACD, ROC, ADX and other momentum indicators to identify trending opportunities.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import statistics

from ..strategies.base_strategy import BaseStrategy
from ..models.trading import TradingSignal, MarketData, SignalAction
from ..utils.technical_analysis import (
    TechnicalAnalyzer, MomentumIndicators, TrendIndicators, 
    MovingAverages, VolatilityIndicators
)


class TimeframeData:
    """Container for timeframe-specific market data and indicators."""
    
    def __init__(self, timeframe: str, max_length: int = 200):
        self.timeframe = timeframe
        self.max_length = max_length
        
        # Price data
        self.opens = deque(maxlen=max_length)
        self.highs = deque(maxlen=max_length)
        self.lows = deque(maxlen=max_length)
        self.closes = deque(maxlen=max_length)
        self.volumes = deque(maxlen=max_length)
        self.timestamps = deque(maxlen=max_length)
        
        # Calculated indicators
        self.rsi = deque(maxlen=max_length)
        self.macd = {'macd': deque(maxlen=max_length), 'signal': deque(maxlen=max_length), 'histogram': deque(maxlen=max_length)}
        self.roc = deque(maxlen=max_length)
        self.adx = {'ADX': deque(maxlen=max_length), '+DI': deque(maxlen=max_length), '-DI': deque(maxlen=max_length)}
        self.sma_20 = deque(maxlen=max_length)
        self.ema_12 = deque(maxlen=max_length)
        self.ema_26 = deque(maxlen=max_length)
        
        # Momentum metrics
        self.momentum_score = deque(maxlen=max_length)
        self.trend_strength = deque(maxlen=max_length)
        self.volatility = deque(maxlen=max_length)
        
        self.last_update = None
    
    def add_data(self, market_data: MarketData) -> None:
        """Add new market data point."""
        # Use price as OHLC for now (in real implementation, aggregate tick data)
        self.opens.append(market_data.price * 0.9995)  # Estimate
        self.highs.append(market_data.high_24h or market_data.price * 1.001)
        self.lows.append(market_data.low_24h or market_data.price * 0.999)
        self.closes.append(market_data.price)
        self.volumes.append(market_data.volume)
        self.timestamps.append(market_data.timestamp)
        
        self.last_update = market_data.timestamp
    
    def calculate_indicators(self) -> None:
        """Calculate all technical indicators for this timeframe."""
        if len(self.closes) < 26:  # Need minimum data for indicators
            return
        
        try:
            closes_list = list(self.closes)
            highs_list = list(self.highs)
            lows_list = list(self.lows)
            volumes_list = list(self.volumes)
            
            # RSI
            rsi_values = MomentumIndicators.rsi(closes_list, 14)
            if rsi_values:
                self.rsi.extend(rsi_values[-len(self.rsi):])
            
            # MACD
            macd_data = MomentumIndicators.macd(closes_list)
            if macd_data['macd']:
                for key in ['macd', 'signal', 'histogram']:
                    if macd_data[key]:
                        self.macd[key].extend(macd_data[key][-len(self.macd[key]):])
            
            # ROC
            roc_values = MomentumIndicators.roc(closes_list, 12)
            if roc_values:
                self.roc.extend(roc_values[-len(self.roc):])
            
            # ADX
            adx_data = TrendIndicators.adx(highs_list, lows_list, closes_list, 14)
            if adx_data['ADX']:
                for key in ['ADX', '+DI', '-DI']:
                    if adx_data[key]:
                        self.adx[key].extend(adx_data[key][-len(self.adx[key]):])
            
            # Moving Averages
            sma_values = MovingAverages.sma(closes_list, 20)
            if sma_values:
                self.sma_20.extend(sma_values[-len(self.sma_20):])
            
            ema_12_values = MovingAverages.ema(closes_list, 12)
            if ema_12_values:
                self.ema_12.extend(ema_12_values[-len(self.ema_12):])
            
            ema_26_values = MovingAverages.ema(closes_list, 26)
            if ema_26_values:
                self.ema_26.extend(ema_26_values[-len(self.ema_26):])
            
            # Calculate composite momentum metrics
            self._calculate_momentum_score()
            self._calculate_trend_strength()
            self._calculate_volatility()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating indicators for {self.timeframe}: {e}")
    
    def _calculate_momentum_score(self) -> None:
        """Calculate composite momentum score."""
        try:
            if not all([self.rsi, self.macd['macd'], self.roc]):
                return
            
            # Get latest values
            rsi_val = self.rsi[-1] if self.rsi and self.rsi[-1] is not None else 50
            macd_val = self.macd['histogram'][-1] if self.macd['histogram'] and self.macd['histogram'][-1] is not None else 0
            roc_val = self.roc[-1] if self.roc and self.roc[-1] is not None else 0
            
            # Normalize indicators to 0-1 scale
            rsi_norm = (rsi_val - 50) / 50  # -1 to 1
            macd_norm = max(-1, min(1, macd_val / 10))  # Clamp to -1 to 1
            roc_norm = max(-1, min(1, roc_val / 20))  # Clamp to -1 to 1
            
            # Weighted momentum score
            momentum = (rsi_norm * 0.4 + macd_norm * 0.4 + roc_norm * 0.2)
            self.momentum_score.append(momentum)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating momentum score: {e}")
    
    def _calculate_trend_strength(self) -> None:
        """Calculate trend strength using ADX and moving averages."""
        try:
            if not self.adx['ADX'] or not self.ema_12 or not self.ema_26:
                return
            
            adx_val = self.adx['ADX'][-1] if self.adx['ADX'][-1] is not None else 0
            ema_12_val = self.ema_12[-1] if self.ema_12[-1] is not None else 0
            ema_26_val = self.ema_26[-1] if self.ema_26[-1] is not None else 0
            
            # ADX strength (0-100 scale)
            adx_strength = min(adx_val / 50, 1.0)  # Normalize to 0-1
            
            # EMA divergence strength
            if ema_26_val != 0:
                ema_divergence = abs(ema_12_val - ema_26_val) / ema_26_val
                ema_strength = min(ema_divergence * 10, 1.0)  # Scale and clamp
            else:
                ema_strength = 0
            
            # Combined trend strength
            trend_strength = (adx_strength * 0.6 + ema_strength * 0.4)
            self.trend_strength.append(trend_strength)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating trend strength: {e}")
    
    def _calculate_volatility(self) -> None:
        """Calculate volatility measure."""
        try:
            if len(self.closes) < 20:
                return
            
            # Calculate 20-period price volatility
            recent_closes = list(self.closes)[-20:]
            returns = [(recent_closes[i] - recent_closes[i-1]) / recent_closes[i-1] 
                      for i in range(1, len(recent_closes))]
            
            if returns:
                volatility = statistics.stdev(returns) * 100  # Convert to percentage
                self.volatility.append(volatility)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating volatility: {e}")
    
    def get_latest_indicators(self) -> Dict[str, Any]:
        """Get latest indicator values."""
        return {
            'rsi': self.rsi[-1] if self.rsi and self.rsi[-1] is not None else None,
            'macd': {
                'macd': self.macd['macd'][-1] if self.macd['macd'] and self.macd['macd'][-1] is not None else None,
                'signal': self.macd['signal'][-1] if self.macd['signal'] and self.macd['signal'][-1] is not None else None,
                'histogram': self.macd['histogram'][-1] if self.macd['histogram'] and self.macd['histogram'][-1] is not None else None
            },
            'roc': self.roc[-1] if self.roc and self.roc[-1] is not None else None,
            'adx': {
                'ADX': self.adx['ADX'][-1] if self.adx['ADX'] and self.adx['ADX'][-1] is not None else None,
                '+DI': self.adx['+DI'][-1] if self.adx['+DI'] and self.adx['+DI'][-1] is not None else None,
                '-DI': self.adx['-DI'][-1] if self.adx['-DI'] and self.adx['-DI'][-1] is not None else None
            },
            'sma_20': self.sma_20[-1] if self.sma_20 and self.sma_20[-1] is not None else None,
            'ema_12': self.ema_12[-1] if self.ema_12 and self.ema_12[-1] is not None else None,
            'ema_26': self.ema_26[-1] if self.ema_26 and self.ema_26[-1] is not None else None,
            'momentum_score': self.momentum_score[-1] if self.momentum_score else None,
            'trend_strength': self.trend_strength[-1] if self.trend_strength else None,
            'volatility': self.volatility[-1] if self.volatility else None,
            'current_price': self.closes[-1] if self.closes else None
        }


class MomentumSignalGenerator:
    """Generates momentum-based trading signals across multiple timeframes."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Signal thresholds
        self.momentum_thresholds = {
            'strong_bullish': 0.6,
            'bullish': 0.3,
            'neutral': 0.1,
            'bearish': -0.3,
            'strong_bearish': -0.6
        }
        
        self.trend_strength_threshold = 0.5
        self.rsi_overbought = 70
        self.rsi_oversold = 30
    
    def generate_momentum_signals(self, timeframes: Dict[str, TimeframeData]) -> List[Dict[str, Any]]:
        """Generate momentum signals from multiple timeframes."""
        signals = []
        
        try:
            # Individual timeframe signals
            for tf_name, tf_data in timeframes.items():
                tf_signals = self._analyze_timeframe_momentum(tf_name, tf_data)
                signals.extend(tf_signals)
            
            # Multi-timeframe confluence signals
            confluence_signals = self._analyze_timeframe_confluence(timeframes)
            signals.extend(confluence_signals)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating momentum signals: {e}")
            return []  
  
    def _analyze_timeframe_momentum(self, timeframe: str, tf_data: TimeframeData) -> List[Dict[str, Any]]:
        """Analyze momentum for a single timeframe."""
        signals = []
        
        try:
            indicators = tf_data.get_latest_indicators()
            
            if not indicators['current_price']:
                return signals
            
            # RSI momentum signals
            rsi_signal = self._analyze_rsi_momentum(indicators, timeframe)
            if rsi_signal:
                signals.append(rsi_signal)
            
            # MACD momentum signals
            macd_signal = self._analyze_macd_momentum(indicators, timeframe, tf_data)
            if macd_signal:
                signals.append(macd_signal)
            
            # ROC momentum signals
            roc_signal = self._analyze_roc_momentum(indicators, timeframe)
            if roc_signal:
                signals.append(roc_signal)
            
            # ADX trend strength signals
            adx_signal = self._analyze_adx_momentum(indicators, timeframe)
            if adx_signal:
                signals.append(adx_signal)
            
            # Composite momentum signals
            composite_signal = self._analyze_composite_momentum(indicators, timeframe)
            if composite_signal:
                signals.append(composite_signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing {timeframe} momentum: {e}")
            return []
    
    def _analyze_rsi_momentum(self, indicators: Dict[str, Any], timeframe: str) -> Optional[Dict[str, Any]]:
        """Analyze RSI for momentum signals."""
        try:
            rsi = indicators.get('rsi')
            if rsi is None:
                return None
            
            # RSI divergence and momentum signals
            if rsi < self.rsi_oversold:
                return {
                    'signal': SignalAction.BUY,
                    'strength': (self.rsi_oversold - rsi) / self.rsi_oversold,
                    'confidence': 0.7,
                    'timeframe': timeframe,
                    'indicator': 'RSI',
                    'reason': f'RSI oversold at {rsi:.1f}',
                    'metadata': {'rsi_value': rsi}
                }
            
            elif rsi > self.rsi_overbought:
                return {
                    'signal': SignalAction.SELL,
                    'strength': (rsi - self.rsi_overbought) / (100 - self.rsi_overbought),
                    'confidence': 0.7,
                    'timeframe': timeframe,
                    'indicator': 'RSI',
                    'reason': f'RSI overbought at {rsi:.1f}',
                    'metadata': {'rsi_value': rsi}
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing RSI momentum: {e}")
            return None
    
    def _analyze_macd_momentum(self, indicators: Dict[str, Any], timeframe: str, 
                              tf_data: TimeframeData) -> Optional[Dict[str, Any]]:
        """Analyze MACD for momentum signals."""
        try:
            macd_data = indicators.get('macd', {})
            macd_line = macd_data.get('macd')
            signal_line = macd_data.get('signal')
            histogram = macd_data.get('histogram')
            
            if not all([macd_line, signal_line, histogram]):
                return None
            
            # Check for MACD crossovers
            if len(tf_data.macd['macd']) >= 2 and len(tf_data.macd['signal']) >= 2:
                prev_macd = tf_data.macd['macd'][-2]
                prev_signal = tf_data.macd['signal'][-2]
                
                if prev_macd is not None and prev_signal is not None:
                    # Bullish crossover
                    if prev_macd <= prev_signal and macd_line > signal_line:
                        return {
                            'signal': SignalAction.BUY,
                            'strength': min(abs(histogram) / 5, 1.0),
                            'confidence': 0.8,
                            'timeframe': timeframe,
                            'indicator': 'MACD',
                            'reason': 'MACD bullish crossover',
                            'metadata': {
                                'macd': macd_line,
                                'signal': signal_line,
                                'histogram': histogram
                            }
                        }
                    
                    # Bearish crossover
                    elif prev_macd >= prev_signal and macd_line < signal_line:
                        return {
                            'signal': SignalAction.SELL,
                            'strength': min(abs(histogram) / 5, 1.0),
                            'confidence': 0.8,
                            'timeframe': timeframe,
                            'indicator': 'MACD',
                            'reason': 'MACD bearish crossover',
                            'metadata': {
                                'macd': macd_line,
                                'signal': signal_line,
                                'histogram': histogram
                            }
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing MACD momentum: {e}")
            return None
    
    def _analyze_roc_momentum(self, indicators: Dict[str, Any], timeframe: str) -> Optional[Dict[str, Any]]:
        """Analyze Rate of Change for momentum signals."""
        try:
            roc = indicators.get('roc')
            if roc is None:
                return None
            
            # Strong momentum signals
            if roc > 5:  # Strong positive momentum
                return {
                    'signal': SignalAction.BUY,
                    'strength': min(roc / 20, 1.0),
                    'confidence': 0.6,
                    'timeframe': timeframe,
                    'indicator': 'ROC',
                    'reason': f'Strong positive momentum (ROC: {roc:.1f}%)',
                    'metadata': {'roc_value': roc}
                }
            
            elif roc < -5:  # Strong negative momentum
                return {
                    'signal': SignalAction.SELL,
                    'strength': min(abs(roc) / 20, 1.0),
                    'confidence': 0.6,
                    'timeframe': timeframe,
                    'indicator': 'ROC',
                    'reason': f'Strong negative momentum (ROC: {roc:.1f}%)',
                    'metadata': {'roc_value': roc}
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing ROC momentum: {e}")
            return None
    
    def _analyze_adx_momentum(self, indicators: Dict[str, Any], timeframe: str) -> Optional[Dict[str, Any]]:
        """Analyze ADX for trend strength signals."""
        try:
            adx_data = indicators.get('adx', {})
            adx = adx_data.get('ADX')
            plus_di = adx_data.get('+DI')
            minus_di = adx_data.get('-DI')
            
            if not all([adx, plus_di, minus_di]):
                return None
            
            # Strong trend with directional bias
            if adx > 25:  # Strong trend
                if plus_di > minus_di * 1.2:  # Strong bullish trend
                    return {
                        'signal': SignalAction.BUY,
                        'strength': min(adx / 50, 1.0),
                        'confidence': 0.75,
                        'timeframe': timeframe,
                        'indicator': 'ADX',
                        'reason': f'Strong bullish trend (ADX: {adx:.1f}, +DI: {plus_di:.1f})',
                        'metadata': {
                            'adx': adx,
                            'plus_di': plus_di,
                            'minus_di': minus_di
                        }
                    }
                
                elif minus_di > plus_di * 1.2:  # Strong bearish trend
                    return {
                        'signal': SignalAction.SELL,
                        'strength': min(adx / 50, 1.0),
                        'confidence': 0.75,
                        'timeframe': timeframe,
                        'indicator': 'ADX',
                        'reason': f'Strong bearish trend (ADX: {adx:.1f}, -DI: {minus_di:.1f})',
                        'metadata': {
                            'adx': adx,
                            'plus_di': plus_di,
                            'minus_di': minus_di
                        }
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing ADX momentum: {e}")
            return None
    
    def _analyze_composite_momentum(self, indicators: Dict[str, Any], timeframe: str) -> Optional[Dict[str, Any]]:
        """Analyze composite momentum score."""
        try:
            momentum_score = indicators.get('momentum_score')
            trend_strength = indicators.get('trend_strength')
            
            if momentum_score is None or trend_strength is None:
                return None
            
            # Strong momentum with good trend strength
            if trend_strength > self.trend_strength_threshold:
                if momentum_score > self.momentum_thresholds['strong_bullish']:
                    return {
                        'signal': SignalAction.BUY,
                        'strength': momentum_score,
                        'confidence': 0.85,
                        'timeframe': timeframe,
                        'indicator': 'COMPOSITE',
                        'reason': f'Strong bullish momentum (Score: {momentum_score:.2f})',
                        'metadata': {
                            'momentum_score': momentum_score,
                            'trend_strength': trend_strength
                        }
                    }
                
                elif momentum_score < self.momentum_thresholds['strong_bearish']:
                    return {
                        'signal': SignalAction.SELL,
                        'strength': abs(momentum_score),
                        'confidence': 0.85,
                        'timeframe': timeframe,
                        'indicator': 'COMPOSITE',
                        'reason': f'Strong bearish momentum (Score: {momentum_score:.2f})',
                        'metadata': {
                            'momentum_score': momentum_score,
                            'trend_strength': trend_strength
                        }
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing composite momentum: {e}")
            return None  
  
    def _analyze_timeframe_confluence(self, timeframes: Dict[str, TimeframeData]) -> List[Dict[str, Any]]:
        """Analyze momentum confluence across multiple timeframes."""
        signals = []
        
        try:
            if len(timeframes) < 2:
                return signals
            
            # Get momentum scores for all timeframes
            tf_momentum = {}
            tf_trends = {}
            
            for tf_name, tf_data in timeframes.items():
                indicators = tf_data.get_latest_indicators()
                tf_momentum[tf_name] = indicators.get('momentum_score')
                tf_trends[tf_name] = indicators.get('trend_strength')
            
            # Filter out None values
            valid_momentum = {k: v for k, v in tf_momentum.items() if v is not None}
            valid_trends = {k: v for k, v in tf_trends.items() if v is not None}
            
            if len(valid_momentum) < 2:
                return signals
            
            # Check for momentum alignment
            bullish_count = sum(1 for score in valid_momentum.values() if score > 0.2)
            bearish_count = sum(1 for score in valid_momentum.values() if score < -0.2)
            total_timeframes = len(valid_momentum)
            
            # Strong confluence signals
            if bullish_count >= total_timeframes * 0.7:  # 70% agreement
                avg_momentum = sum(valid_momentum.values()) / len(valid_momentum)
                avg_trend = sum(valid_trends.values()) / len(valid_trends) if valid_trends else 0.5
                
                signals.append({
                    'signal': SignalAction.BUY,
                    'strength': min(avg_momentum, 1.0),
                    'confidence': 0.9,
                    'timeframe': 'MULTI',
                    'indicator': 'CONFLUENCE',
                    'reason': f'Bullish momentum confluence ({bullish_count}/{total_timeframes} timeframes)',
                    'metadata': {
                        'timeframe_momentum': valid_momentum,
                        'avg_momentum': avg_momentum,
                        'avg_trend_strength': avg_trend,
                        'agreement_ratio': bullish_count / total_timeframes
                    }
                })
            
            elif bearish_count >= total_timeframes * 0.7:  # 70% agreement
                avg_momentum = sum(valid_momentum.values()) / len(valid_momentum)
                avg_trend = sum(valid_trends.values()) / len(valid_trends) if valid_trends else 0.5
                
                signals.append({
                    'signal': SignalAction.SELL,
                    'strength': min(abs(avg_momentum), 1.0),
                    'confidence': 0.9,
                    'timeframe': 'MULTI',
                    'indicator': 'CONFLUENCE',
                    'reason': f'Bearish momentum confluence ({bearish_count}/{total_timeframes} timeframes)',
                    'metadata': {
                        'timeframe_momentum': valid_momentum,
                        'avg_momentum': avg_momentum,
                        'avg_trend_strength': avg_trend,
                        'agreement_ratio': bearish_count / total_timeframes
                    }
                })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe confluence: {e}")
            return []


class MomentumStrategy(BaseStrategy):
    """Multi-timeframe momentum trading strategy."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__("MomentumStrategy", parameters)
        
        # Default parameters
        default_params = {
            'timeframes': ['1m', '5m', '15m'],
            'min_trend_strength': 0.5,
            'momentum_threshold': 0.3,
            'confluence_threshold': 0.7,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'adx_trend_threshold': 25,
            'position_size_factor': 1.0,
            'max_positions': 3,
            'cooldown_minutes': 15
        }
        
        self.parameters.update(default_params)
        if parameters:
            self.parameters.update(parameters)
        
        # Initialize components
        self.signal_generator = MomentumSignalGenerator()
        self.timeframes = {}
        
        # Initialize timeframe data containers
        for tf in self.parameters['timeframes']:
            self.timeframes[tf] = TimeframeData(tf)
        
        # Strategy state
        self.last_signal_time = None
        self.signal_history = deque(maxlen=100)
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'avg_confidence': 0.0,
            'timeframe_performance': {}
        }
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_market(self, market_data: MarketData) -> Optional[TradingSignal]:
        """Analyze market data and generate momentum-based trading signals."""
        try:
            # Update timeframe data
            self._update_timeframes(market_data)
            
            # Calculate indicators for all timeframes
            for tf_data in self.timeframes.values():
                tf_data.calculate_indicators()
            
            # Generate momentum signals
            momentum_signals = self.signal_generator.generate_momentum_signals(self.timeframes)
            
            if not momentum_signals:
                return None
            
            # Process and rank signals
            processed_signal = self._process_momentum_signals(momentum_signals, market_data)
            
            if processed_signal:
                self._update_performance_metrics(processed_signal)
                return processed_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in momentum analysis: {e}")
            return None    

    def _update_timeframes(self, market_data: MarketData) -> None:
        """Update all timeframe data with new market data."""
        try:
            for tf_data in self.timeframes.values():
                tf_data.add_data(market_data)
        except Exception as e:
            self.logger.error(f"Error updating timeframes: {e}")
    
    def _process_momentum_signals(self, signals: List[Dict[str, Any]], 
                                 market_data: MarketData) -> Optional[TradingSignal]:
        """Process and rank momentum signals to generate final trading signal."""
        try:
            if not signals:
                return None
            
            # Check cooldown period
            if self._is_in_cooldown():
                return None
            
            # Filter signals by confidence and strength
            filtered_signals = [
                s for s in signals 
                if s['confidence'] >= 0.6 and s['strength'] >= self.parameters['momentum_threshold']
            ]
            
            if not filtered_signals:
                return None
            
            # Prioritize confluence signals
            confluence_signals = [s for s in filtered_signals if s['indicator'] == 'CONFLUENCE']
            if confluence_signals:
                best_signal = max(confluence_signals, key=lambda x: x['confidence'] * x['strength'])
            else:
                # Rank by confidence * strength
                best_signal = max(filtered_signals, key=lambda x: x['confidence'] * x['strength'])
            
            # Calculate position size based on momentum strength
            position_size = self._calculate_position_size(best_signal, market_data)
            
            # Create trading signal
            trading_signal = TradingSignal(
                symbol=market_data.symbol,
                action=best_signal['signal'],
                price=market_data.price,
                quantity=position_size,
                confidence=best_signal['confidence'],
                timestamp=market_data.timestamp,
                strategy_name=self.name,
                metadata={
                    'momentum_signals': signals,
                    'best_signal': best_signal,
                    'timeframe_data': self._get_timeframe_summary(),
                    'signal_reasoning': best_signal['reason']
                }
            )
            
            self.last_signal_time = market_data.timestamp
            self.signal_history.append({
                'timestamp': market_data.timestamp,
                'signal': best_signal,
                'market_price': market_data.price
            })
            
            return trading_signal
            
        except Exception as e:
            self.logger.error(f"Error processing momentum signals: {e}")
            return None
    
    def _is_in_cooldown(self) -> bool:
        """Check if strategy is in cooldown period."""
        if not self.last_signal_time:
            return False
        
        cooldown_delta = timedelta(minutes=self.parameters['cooldown_minutes'])
        return datetime.now() - self.last_signal_time < cooldown_delta
    
    def _calculate_position_size(self, signal: Dict[str, Any], market_data: MarketData) -> float:
        """Calculate position size based on momentum strength and confidence."""
        try:
            base_size = self.parameters['position_size_factor']
            momentum_multiplier = signal['strength']
            confidence_multiplier = signal['confidence']
            
            # Adjust for timeframe (longer timeframes get higher weight)
            timeframe_multiplier = 1.0
            if signal['timeframe'] == '15m':
                timeframe_multiplier = 1.2
            elif signal['timeframe'] == '5m':
                timeframe_multiplier = 1.1
            elif signal['timeframe'] == 'MULTI':
                timeframe_multiplier = 1.3
            
            position_size = base_size * momentum_multiplier * confidence_multiplier * timeframe_multiplier
            
            # Cap position size
            max_size = base_size * 2.0
            return min(position_size, max_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self.parameters['position_size_factor']
    
    def _get_timeframe_summary(self) -> Dict[str, Any]:
        """Get summary of current timeframe indicators."""
        summary = {}
        
        try:
            for tf_name, tf_data in self.timeframes.items():
                indicators = tf_data.get_latest_indicators()
                summary[tf_name] = {
                    'momentum_score': indicators.get('momentum_score'),
                    'trend_strength': indicators.get('trend_strength'),
                    'rsi': indicators.get('rsi'),
                    'adx': indicators.get('adx', {}).get('ADX'),
                    'current_price': indicators.get('current_price')
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting timeframe summary: {e}")
            return {}
    
    def _update_performance_metrics(self, signal: TradingSignal) -> None:
        """Update strategy performance metrics."""
        try:
            self.performance_metrics['total_signals'] += 1
            
            # Update average confidence
            total_confidence = (self.performance_metrics['avg_confidence'] * 
                              (self.performance_metrics['total_signals'] - 1) + signal.confidence)
            self.performance_metrics['avg_confidence'] = total_confidence / self.performance_metrics['total_signals']
            
            # Update timeframe performance
            timeframe = signal.metadata.get('best_signal', {}).get('timeframe', 'unknown')
            if timeframe not in self.performance_metrics['timeframe_performance']:
                self.performance_metrics['timeframe_performance'][timeframe] = {
                    'signals': 0, 'avg_confidence': 0.0
                }
            
            tf_perf = self.performance_metrics['timeframe_performance'][timeframe]
            tf_perf['signals'] += 1
            tf_total_conf = tf_perf['avg_confidence'] * (tf_perf['signals'] - 1) + signal.confidence
            tf_perf['avg_confidence'] = tf_total_conf / tf_perf['signals']
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate strategy parameters."""
        try:
            required_params = [
                'timeframes', 'min_trend_strength', 'momentum_threshold',
                'confluence_threshold', 'position_size_factor'
            ]
            
            for param in required_params:
                if param not in parameters:
                    return False
            
            # Validate timeframes
            if not isinstance(parameters['timeframes'], list) or len(parameters['timeframes']) == 0:
                return False
            
            # Validate thresholds
            for threshold in ['min_trend_strength', 'momentum_threshold', 'confluence_threshold']:
                if not 0.0 <= parameters[threshold] <= 1.0:
                    return False
            
            # Validate RSI levels
            if not (0 < parameters['rsi_oversold'] < parameters['rsi_overbought'] < 100):
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
                'timeframes': list(self.timeframes.keys()),
                'last_signal_time': self.last_signal_time,
                'signal_history_length': len(self.signal_history),
                'timeframe_data_lengths': {
                    tf: len(data.closes) for tf, data in self.timeframes.items()
                }
            }
        }