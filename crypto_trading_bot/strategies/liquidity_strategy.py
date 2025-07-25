"""
Liquidity analysis trading strategy.

This strategy analyzes order book depth, liquidity patterns, and market
microstructure to identify optimal entry and exit points.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import statistics

from ..strategies.base_strategy import BaseStrategy
from ..models.trading import TradingSignal, MarketData, OrderBook, SignalAction
from ..utils.technical_analysis import MovingAverages


class OrderBookAnalyzer:
    """Analyzes order book data for liquidity patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_depth_metrics(self, order_book: OrderBook, levels: int = 10) -> Dict[str, float]:
        """Calculate order book depth metrics."""
        try:
            if not order_book or not order_book.bids or not order_book.asks:
                return {}
            
            # Get top levels
            top_bids = order_book.bids[:levels]
            top_asks = order_book.asks[:levels]
            
            # Calculate bid/ask volumes
            bid_volume = sum(qty for _, qty in top_bids)
            ask_volume = sum(qty for _, qty in top_asks)
            total_volume = bid_volume + ask_volume
            
            # Calculate weighted average prices
            bid_vwap = sum(price * qty for price, qty in top_bids) / bid_volume if bid_volume > 0 else 0
            ask_vwap = sum(price * qty for price, qty in top_asks) / ask_volume if ask_volume > 0 else 0
            
            # Calculate imbalance
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Calculate spread metrics
            best_bid = order_book.best_bid[0] if order_book.bids else 0
            best_ask = order_book.best_ask[0] if order_book.asks else 0
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0
            spread_pct = (spread / mid_price * 100) if mid_price > 0 else 0
            
            # Calculate depth at different price levels
            depth_1pct = self._calculate_depth_at_distance(order_book, 0.01)
            depth_2pct = self._calculate_depth_at_distance(order_book, 0.02)
            depth_5pct = self._calculate_depth_at_distance(order_book, 0.05)
            
            return {
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': total_volume,
                'imbalance': imbalance,
                'bid_vwap': bid_vwap,
                'ask_vwap': ask_vwap,
                'spread': spread,
                'spread_pct': spread_pct,
                'mid_price': mid_price,
                'depth_1pct': depth_1pct,
                'depth_2pct': depth_2pct,
                'depth_5pct': depth_5pct,
                'levels_count': min(len(order_book.bids), len(order_book.asks))
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating depth metrics: {e}")
            return {}
    
    def _calculate_depth_at_distance(self, order_book: OrderBook, distance_pct: float) -> Dict[str, float]:
        """Calculate liquidity depth at a specific distance from mid price."""
        try:
            mid_price = order_book.mid_price
            if mid_price <= 0:
                return {'bid_depth': 0, 'ask_depth': 0}
            
            # Calculate price thresholds
            bid_threshold = mid_price * (1 - distance_pct)
            ask_threshold = mid_price * (1 + distance_pct)
            
            # Calculate bid depth
            bid_depth = sum(qty for price, qty in order_book.bids if price >= bid_threshold)
            
            # Calculate ask depth
            ask_depth = sum(qty for price, qty in order_book.asks if price <= ask_threshold)
            
            return {'bid_depth': bid_depth, 'ask_depth': ask_depth}
            
        except Exception as e:
            self.logger.error(f"Error calculating depth at distance: {e}")
            return {'bid_depth': 0, 'ask_depth': 0}
    
    def detect_liquidity_walls(self, order_book: OrderBook, 
                              wall_threshold: float = 10.0) -> Dict[str, List[Tuple[float, float]]]:
        """Detect large liquidity walls in the order book."""
        try:
            if not order_book or not order_book.bids or not order_book.asks:
                return {'bid_walls': [], 'ask_walls': []}
            
            # Calculate average order size
            all_orders = order_book.bids + order_book.asks
            avg_size = statistics.mean(qty for _, qty in all_orders) if all_orders else 0
            
            if avg_size == 0:
                return {'bid_walls': [], 'ask_walls': []}
            
            # Find bid walls
            bid_walls = []
            for price, qty in order_book.bids:
                if qty >= avg_size * wall_threshold:
                    bid_walls.append((price, qty))
            
            # Find ask walls
            ask_walls = []
            for price, qty in order_book.asks:
                if qty >= avg_size * wall_threshold:
                    ask_walls.append((price, qty))
            
            return {'bid_walls': bid_walls, 'ask_walls': ask_walls}
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidity walls: {e}")
            return {'bid_walls': [], 'ask_walls': []}
    
    def calculate_market_impact(self, order_book: OrderBook, trade_size: float, 
                               side: str) -> Dict[str, float]:
        """Calculate estimated market impact for a given trade size."""
        try:
            if not order_book:
                return {'impact_price': 0, 'impact_pct': 0, 'slippage': 0}
            
            orders = order_book.asks if side.upper() == 'BUY' else order_book.bids
            if not orders:
                return {'impact_price': 0, 'impact_pct': 0, 'slippage': 0}
            
            remaining_size = trade_size
            total_cost = 0
            last_price = 0
            
            for price, qty in orders:
                if remaining_size <= 0:
                    break
                
                fill_qty = min(remaining_size, qty)
                total_cost += fill_qty * price
                remaining_size -= fill_qty
                last_price = price
            
            if trade_size == 0 or total_cost == 0:
                return {'impact_price': 0, 'impact_pct': 0, 'slippage': 0}
            
            # Calculate average fill price
            avg_fill_price = total_cost / (trade_size - remaining_size)
            
            # Calculate impact
            best_price = orders[0][0]
            impact_pct = abs(avg_fill_price - best_price) / best_price * 100
            slippage = avg_fill_price - best_price if side.upper() == 'BUY' else best_price - avg_fill_price
            
            return {
                'impact_price': avg_fill_price,
                'impact_pct': impact_pct,
                'slippage': slippage,
                'unfilled_size': remaining_size
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating market impact: {e}")
            return {'impact_price': 0, 'impact_pct': 0, 'slippage': 0}


class LiquidityScorer:
    """Scores market liquidity conditions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Scoring thresholds
        self.spread_thresholds = {
            'excellent': 0.05,  # 0.05%
            'good': 0.1,        # 0.1%
            'fair': 0.2,        # 0.2%
            'poor': 0.5         # 0.5%
        }
        
        self.depth_thresholds = {
            'excellent': 1000,
            'good': 500,
            'fair': 200,
            'poor': 50
        }
        
        self.imbalance_thresholds = {
            'balanced': 0.1,    # 10%
            'moderate': 0.3,    # 30%
            'high': 0.6         # 60%
        }
    
    def calculate_liquidity_score(self, depth_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate overall liquidity score."""
        try:
            if not depth_metrics:
                return {'score': 0.0, 'grade': 'poor', 'components': {}}
            
            # Spread score (lower is better)
            spread_score = self._score_spread(depth_metrics.get('spread_pct', 100))
            
            # Depth score (higher is better)
            depth_score = self._score_depth(depth_metrics.get('total_volume', 0))
            
            # Imbalance score (closer to 0 is better)
            imbalance_score = self._score_imbalance(abs(depth_metrics.get('imbalance', 1)))
            
            # Market impact score
            impact_score = self._score_market_impact(depth_metrics)
            
            # Weighted overall score
            weights = {
                'spread': 0.3,
                'depth': 0.3,
                'imbalance': 0.2,
                'impact': 0.2
            }
            
            overall_score = (
                spread_score * weights['spread'] +
                depth_score * weights['depth'] +
                imbalance_score * weights['imbalance'] +
                impact_score * weights['impact']
            )
            
            # Determine grade
            grade = self._score_to_grade(overall_score)
            
            return {
                'score': overall_score,
                'grade': grade,
                'components': {
                    'spread_score': spread_score,
                    'depth_score': depth_score,
                    'imbalance_score': imbalance_score,
                    'impact_score': impact_score
                },
                'metrics': depth_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return {'score': 0.0, 'grade': 'poor', 'components': {}}
    
    def _score_spread(self, spread_pct: float) -> float:
        """Score based on bid-ask spread."""
        if spread_pct <= self.spread_thresholds['excellent']:
            return 1.0
        elif spread_pct <= self.spread_thresholds['good']:
            return 0.8
        elif spread_pct <= self.spread_thresholds['fair']:
            return 0.6
        elif spread_pct <= self.spread_thresholds['poor']:
            return 0.4
        else:
            return 0.2
    
    def _score_depth(self, total_volume: float) -> float:
        """Score based on order book depth."""
        if total_volume >= self.depth_thresholds['excellent']:
            return 1.0
        elif total_volume >= self.depth_thresholds['good']:
            return 0.8
        elif total_volume >= self.depth_thresholds['fair']:
            return 0.6
        elif total_volume >= self.depth_thresholds['poor']:
            return 0.4
        else:
            return 0.2
    
    def _score_imbalance(self, imbalance: float) -> float:
        """Score based on order book imbalance."""
        if imbalance <= self.imbalance_thresholds['balanced']:
            return 1.0
        elif imbalance <= self.imbalance_thresholds['moderate']:
            return 0.7
        elif imbalance <= self.imbalance_thresholds['high']:
            return 0.4
        else:
            return 0.2
    
    def _score_market_impact(self, depth_metrics: Dict[str, float]) -> float:
        """Score based on estimated market impact."""
        # Use depth at different levels as proxy for market impact
        depth_1pct = depth_metrics.get('depth_1pct', {})
        if not depth_1pct:
            return 0.2
        
        min_depth = min(depth_1pct.get('bid_depth', 0), depth_1pct.get('ask_depth', 0))
        
        if min_depth >= 500:
            return 1.0
        elif min_depth >= 200:
            return 0.8
        elif min_depth >= 100:
            return 0.6
        elif min_depth >= 50:
            return 0.4
        else:
            return 0.2
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        else:
            return 'poor'


class LiquiditySignalGenerator:
    """Advanced signal generation based on liquidity patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Signal strength thresholds
        self.signal_thresholds = {
            'strong': 0.8,
            'moderate': 0.6,
            'weak': 0.4
        }
        
        # Pattern detection parameters
        self.pattern_params = {
            'min_pattern_length': 5,
            'breakout_threshold': 0.02,  # 2%
            'volume_confirmation': 1.5,   # 50% above average
            'momentum_lookback': 10
        }
    
    def generate_buying_pressure_signal(self, market_data: MarketData, 
                                       depth_metrics: Dict[str, float],
                                       liquidity_history: List[float]) -> Optional[Dict[str, Any]]:
        """Generate signal based on buying pressure analysis."""
        try:
            if not market_data.orderbook:
                return None
            
            imbalance = depth_metrics.get('imbalance', 0)
            bid_volume = depth_metrics.get('bid_volume', 0)
            ask_volume = depth_metrics.get('ask_volume', 0)
            
            # Strong buying pressure indicators
            buying_pressure_score = 0.0
            reasons = []
            
            # 1. Order book imbalance favoring bids
            if imbalance > 0.2:  # 20% more bids than asks
                buying_pressure_score += 0.3
                reasons.append(f"bid_imbalance_{imbalance:.2f}")
            
            # 2. Large bid walls near current price
            walls = self._detect_nearby_walls(market_data.orderbook, market_data.price)
            if walls['strong_bid_support']:
                buying_pressure_score += 0.25
                reasons.append("strong_bid_walls")
            
            # 3. Increasing bid volume trend
            if len(liquidity_history) >= 5:
                recent_trend = self._calculate_liquidity_momentum(liquidity_history)
                if recent_trend > 0.05:  # Improving liquidity
                    buying_pressure_score += 0.2
                    reasons.append(f"improving_liquidity_{recent_trend:.3f}")
            
            # 4. Tight spreads with high volume
            spread_pct = depth_metrics.get('spread_pct', 100)
            if spread_pct < 0.1 and bid_volume > 500:  # Tight spread + good volume
                buying_pressure_score += 0.15
                reasons.append("tight_spread_high_volume")
            
            # 5. Volume confirmation
            if hasattr(market_data, 'volume') and market_data.volume > 0:
                # This would need historical volume data for proper comparison
                buying_pressure_score += 0.1
                reasons.append("volume_confirmation")
            
            if buying_pressure_score >= 0.6:
                return {
                    'signal': 'BUY',
                    'strength': buying_pressure_score,
                    'confidence': min(buying_pressure_score, 0.95),
                    'reasons': reasons,
                    'target_price': self._calculate_target_price(market_data, 'BUY', depth_metrics),
                    'stop_loss': self._calculate_stop_loss(market_data, 'BUY', depth_metrics),
                    'metadata': {
                        'imbalance': imbalance,
                        'bid_volume': bid_volume,
                        'spread_pct': spread_pct
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating buying pressure signal: {e}")
            return None
    
    def generate_selling_pressure_signal(self, market_data: MarketData,
                                        depth_metrics: Dict[str, float],
                                        liquidity_history: List[float]) -> Optional[Dict[str, Any]]:
        """Generate signal based on selling pressure analysis."""
        try:
            if not market_data.orderbook:
                return None
            
            imbalance = depth_metrics.get('imbalance', 0)
            bid_volume = depth_metrics.get('bid_volume', 0)
            ask_volume = depth_metrics.get('ask_volume', 0)
            
            # Strong selling pressure indicators
            selling_pressure_score = 0.0
            reasons = []
            
            # 1. Order book imbalance favoring asks
            if imbalance < -0.2:  # 20% more asks than bids
                selling_pressure_score += 0.3
                reasons.append(f"ask_imbalance_{abs(imbalance):.2f}")
            
            # 2. Large ask walls near current price
            walls = self._detect_nearby_walls(market_data.orderbook, market_data.price)
            if walls['strong_ask_resistance']:
                selling_pressure_score += 0.25
                reasons.append("strong_ask_walls")
            
            # 3. Deteriorating liquidity trend
            if len(liquidity_history) >= 5:
                recent_trend = self._calculate_liquidity_momentum(liquidity_history)
                if recent_trend < -0.05:  # Deteriorating liquidity
                    selling_pressure_score += 0.2
                    reasons.append(f"deteriorating_liquidity_{recent_trend:.3f}")
            
            # 4. Widening spreads with high ask volume
            spread_pct = depth_metrics.get('spread_pct', 0)
            if spread_pct > 0.15 and ask_volume > bid_volume * 1.5:
                selling_pressure_score += 0.15
                reasons.append("widening_spread_ask_pressure")
            
            # 5. Volume confirmation
            if hasattr(market_data, 'volume') and market_data.volume > 0:
                selling_pressure_score += 0.1
                reasons.append("volume_confirmation")
            
            if selling_pressure_score >= 0.6:
                return {
                    'signal': 'SELL',
                    'strength': selling_pressure_score,
                    'confidence': min(selling_pressure_score, 0.95),
                    'reasons': reasons,
                    'target_price': self._calculate_target_price(market_data, 'SELL', depth_metrics),
                    'stop_loss': self._calculate_stop_loss(market_data, 'SELL', depth_metrics),
                    'metadata': {
                        'imbalance': imbalance,
                        'ask_volume': ask_volume,
                        'spread_pct': spread_pct
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating selling pressure signal: {e}")
            return None
    
    def generate_liquidity_breakout_signal(self, market_data: MarketData,
                                          depth_metrics: Dict[str, float],
                                          price_history: List[float]) -> Optional[Dict[str, Any]]:
        """Generate signal based on liquidity breakout patterns."""
        try:
            if len(price_history) < self.pattern_params['min_pattern_length']:
                return None
            
            current_price = market_data.price
            recent_prices = price_history[-self.pattern_params['min_pattern_length']:]
            
            # Detect support/resistance levels
            support_level = min(recent_prices)
            resistance_level = max(recent_prices)
            price_range = resistance_level - support_level
            
            if price_range == 0:
                return None
            
            # Check for breakout conditions
            breakout_threshold = price_range * self.pattern_params['breakout_threshold']
            
            # Upward breakout
            if current_price > resistance_level + breakout_threshold:
                # Confirm with liquidity conditions
                if self._confirm_breakout_liquidity(market_data, depth_metrics, 'UP'):
                    return {
                        'signal': 'BUY',
                        'strength': 0.8,
                        'confidence': 0.75,
                        'reasons': ['upward_breakout', 'liquidity_confirmation'],
                        'target_price': current_price + (price_range * 0.618),  # Fibonacci extension
                        'stop_loss': resistance_level,
                        'metadata': {
                            'breakout_level': resistance_level,
                            'breakout_strength': (current_price - resistance_level) / price_range,
                            'support_level': support_level
                        }
                    }
            
            # Downward breakout
            elif current_price < support_level - breakout_threshold:
                # Confirm with liquidity conditions
                if self._confirm_breakout_liquidity(market_data, depth_metrics, 'DOWN'):
                    return {
                        'signal': 'SELL',
                        'strength': 0.8,
                        'confidence': 0.75,
                        'reasons': ['downward_breakout', 'liquidity_confirmation'],
                        'target_price': current_price - (price_range * 0.618),
                        'stop_loss': support_level,
                        'metadata': {
                            'breakout_level': support_level,
                            'breakout_strength': (support_level - current_price) / price_range,
                            'resistance_level': resistance_level
                        }
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating breakout signal: {e}")
            return None
    
    def generate_mean_reversion_signal(self, market_data: MarketData,
                                      depth_metrics: Dict[str, float],
                                      liquidity_history: List[float]) -> Optional[Dict[str, Any]]:
        """Generate mean reversion signal based on liquidity extremes."""
        try:
            if len(liquidity_history) < 10:
                return None
            
            current_liquidity = liquidity_history[-1]
            avg_liquidity = sum(liquidity_history) / len(liquidity_history)
            liquidity_std = statistics.stdev(liquidity_history)
            
            if liquidity_std == 0:
                return None
            
            # Z-score of current liquidity
            liquidity_zscore = (current_liquidity - avg_liquidity) / liquidity_std
            
            # Extreme low liquidity (potential reversal opportunity)
            if liquidity_zscore < -1.5 and current_liquidity < 0.4:
                # Look for signs of liquidity improvement
                recent_trend = self._calculate_liquidity_momentum(liquidity_history[-5:])
                
                if recent_trend > 0:  # Liquidity starting to improve
                    imbalance = depth_metrics.get('imbalance', 0)
                    
                    # Determine direction based on imbalance
                    if imbalance > 0.1:  # Slight bid preference
                        return {
                            'signal': 'BUY',
                            'strength': 0.6,
                            'confidence': 0.65,
                            'reasons': ['liquidity_mean_reversion', 'improving_conditions'],
                            'target_price': self._calculate_target_price(market_data, 'BUY', depth_metrics),
                            'stop_loss': self._calculate_stop_loss(market_data, 'BUY', depth_metrics),
                            'metadata': {
                                'liquidity_zscore': liquidity_zscore,
                                'liquidity_trend': recent_trend,
                                'imbalance': imbalance
                            }
                        }
                    elif imbalance < -0.1:  # Slight ask preference
                        return {
                            'signal': 'SELL',
                            'strength': 0.6,
                            'confidence': 0.65,
                            'reasons': ['liquidity_mean_reversion', 'improving_conditions'],
                            'target_price': self._calculate_target_price(market_data, 'SELL', depth_metrics),
                            'stop_loss': self._calculate_stop_loss(market_data, 'SELL', depth_metrics),
                            'metadata': {
                                'liquidity_zscore': liquidity_zscore,
                                'liquidity_trend': recent_trend,
                                'imbalance': imbalance
                            }
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion signal: {e}")
            return None
    
    def _detect_nearby_walls(self, order_book: OrderBook, current_price: float) -> Dict[str, bool]:
        """Detect strong liquidity walls near current price."""
        try:
            # Define "nearby" as within 2% of current price
            price_threshold = current_price * 0.02
            
            # Calculate average order size
            all_orders = order_book.bids + order_book.asks
            if not all_orders:
                return {'strong_bid_support': False, 'strong_ask_resistance': False}
            
            avg_size = statistics.mean(qty for _, qty in all_orders)
            wall_threshold = avg_size * 5  # 5x average size
            
            # Check for strong bid walls (support)
            strong_bid_support = any(
                qty > wall_threshold and abs(price - current_price) <= price_threshold
                for price, qty in order_book.bids
            )
            
            # Check for strong ask walls (resistance)
            strong_ask_resistance = any(
                qty > wall_threshold and abs(price - current_price) <= price_threshold
                for price, qty in order_book.asks
            )
            
            return {
                'strong_bid_support': strong_bid_support,
                'strong_ask_resistance': strong_ask_resistance
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting nearby walls: {e}")
            return {'strong_bid_support': False, 'strong_ask_resistance': False}
    
    def _calculate_liquidity_momentum(self, liquidity_values: List[float]) -> float:
        """Calculate liquidity momentum (rate of change)."""
        if len(liquidity_values) < 2:
            return 0.0
        
        try:
            # Simple momentum calculation
            recent_avg = sum(liquidity_values[-3:]) / min(3, len(liquidity_values))
            older_avg = sum(liquidity_values[:-3]) / max(1, len(liquidity_values) - 3)
            
            if older_avg == 0:
                return 0.0
            
            momentum = (recent_avg - older_avg) / older_avg
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity momentum: {e}")
            return 0.0
    
    def _confirm_breakout_liquidity(self, market_data: MarketData,
                                   depth_metrics: Dict[str, float], direction: str) -> bool:
        """Confirm breakout with liquidity conditions."""
        try:
            liquidity_score = depth_metrics.get('total_volume', 0)
            spread_pct = depth_metrics.get('spread_pct', 100)
            imbalance = depth_metrics.get('imbalance', 0)
            
            # Basic liquidity requirements
            if liquidity_score < 200 or spread_pct > 0.3:
                return False
            
            # Direction-specific confirmation
            if direction == 'UP':
                return imbalance > -0.2  # Not too much selling pressure
            elif direction == 'DOWN':
                return imbalance < 0.2   # Not too much buying pressure
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error confirming breakout liquidity: {e}")
            return False
    
    def _calculate_target_price(self, market_data: MarketData, signal_type: str,
                               depth_metrics: Dict[str, float]) -> Optional[float]:
        """Calculate target price based on liquidity analysis."""
        try:
            current_price = market_data.price
            spread = depth_metrics.get('spread', 0)
            
            # Conservative target based on spread and liquidity
            if signal_type == 'BUY':
                # Target above current price
                target_multiplier = 1.005 + (spread / current_price)  # 0.5% + spread
                return current_price * target_multiplier
            elif signal_type == 'SELL':
                # Target below current price
                target_multiplier = 0.995 - (spread / current_price)  # 0.5% - spread
                return current_price * target_multiplier
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating target price: {e}")
            return None
    
    def _calculate_stop_loss(self, market_data: MarketData, signal_type: str,
                            depth_metrics: Dict[str, float]) -> Optional[float]:
        """Calculate stop loss based on liquidity analysis."""
        try:
            current_price = market_data.price
            spread = depth_metrics.get('spread', 0)
            
            # Conservative stop loss
            if signal_type == 'BUY':
                # Stop loss below current price
                stop_multiplier = 0.99 - (spread / current_price * 2)  # 1% - 2x spread
                return current_price * stop_multiplier
            elif signal_type == 'SELL':
                # Stop loss above current price
                stop_multiplier = 1.01 + (spread / current_price * 2)  # 1% + 2x spread
                return current_price * stop_multiplier
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return None


class LiquiditySignalGenerator:
    """Advanced signal generation based on liquidity patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Signal strength thresholds
        self.signal_thresholds = {
            'strong': 0.8,
            'moderate': 0.6,
            'weak': 0.4
        }
        
        # Pattern detection parameters
        self.pattern_params = {
            'min_imbalance_duration': 3,
            'wall_size_multiplier': 5.0,
            'liquidity_improvement_threshold': 0.1,
            'volume_surge_threshold': 2.0
        }
    
    def generate_buying_pressure_signal(self, market_data: MarketData, 
                                       depth_metrics: Dict[str, float],
                                       history: Dict[str, deque]) -> Optional[Dict[str, Any]]:
        """Generate signal based on buying pressure analysis."""
        try:
            imbalance = depth_metrics.get('imbalance', 0)
            bid_volume = depth_metrics.get('bid_volume', 0)
            ask_volume = depth_metrics.get('ask_volume', 0)
            
            # Strong buying pressure indicators
            buying_pressure_score = 0.0
            reasons = []
            
            # 1. Sustained bid imbalance
            if len(history['imbalance']) >= 3:
                recent_imbalances = list(history['imbalance'])[-3:]
                if all(imb > 0.2 for imb in recent_imbalances):
                    buying_pressure_score += 0.3
                    reasons.append("sustained_bid_imbalance")
            
            # 2. Increasing bid volume
            if bid_volume > ask_volume * 1.5:
                buying_pressure_score += 0.25
                reasons.append("strong_bid_volume")
            
            # 3. Improving liquidity with bid preference
            if len(history['liquidity']) >= 2:
                liquidity_trend = history['liquidity'][-1] - history['liquidity'][-2]
                if liquidity_trend > 0.05 and imbalance > 0.1:
                    buying_pressure_score += 0.25
                    reasons.append("improving_liquidity_with_bid_preference")
            
            # 4. Volume surge with bid imbalance
            if len(history['volume']) >= 3:
                recent_volumes = list(history['volume'])[-3:]
                avg_volume = sum(recent_volumes[:-1]) / len(recent_volumes[:-1])
                if recent_volumes[-1] > avg_volume * 1.5 and imbalance > 0.15:
                    buying_pressure_score += 0.2
                    reasons.append("volume_surge_with_bid_imbalance")
            
            if buying_pressure_score >= 0.6:
                return {
                    'signal': SignalAction.BUY,
                    'strength': buying_pressure_score,
                    'reasons': reasons,
                    'confidence': min(buying_pressure_score, 0.9),
                    'metadata': {
                        'imbalance': imbalance,
                        'bid_volume': bid_volume,
                        'ask_volume': ask_volume,
                        'pressure_score': buying_pressure_score
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating buying pressure signal: {e}")
            return None
    
    def generate_selling_pressure_signal(self, market_data: MarketData,
                                        depth_metrics: Dict[str, float],
                                        history: Dict[str, deque]) -> Optional[Dict[str, Any]]:
        """Generate signal based on selling pressure analysis."""
        try:
            imbalance = depth_metrics.get('imbalance', 0)
            bid_volume = depth_metrics.get('bid_volume', 0)
            ask_volume = depth_metrics.get('ask_volume', 0)
            
            # Strong selling pressure indicators
            selling_pressure_score = 0.0
            reasons = []
            
            # 1. Sustained ask imbalance
            if len(history['imbalance']) >= 3:
                recent_imbalances = list(history['imbalance'])[-3:]
                if all(imb < -0.2 for imb in recent_imbalances):
                    selling_pressure_score += 0.3
                    reasons.append("sustained_ask_imbalance")
            
            # 2. Increasing ask volume
            if ask_volume > bid_volume * 1.5:
                selling_pressure_score += 0.25
                reasons.append("strong_ask_volume")
            
            # 3. Deteriorating liquidity with ask preference
            if len(history['liquidity']) >= 2:
                liquidity_trend = history['liquidity'][-1] - history['liquidity'][-2]
                if liquidity_trend < -0.05 and imbalance < -0.1:
                    selling_pressure_score += 0.25
                    reasons.append("deteriorating_liquidity_with_ask_preference")
            
            # 4. Volume surge with ask imbalance
            if len(history['volume']) >= 3:
                recent_volumes = list(history['volume'])[-3:]
                avg_volume = sum(recent_volumes[:-1]) / len(recent_volumes[:-1])
                if recent_volumes[-1] > avg_volume * 1.5 and imbalance < -0.15:
                    selling_pressure_score += 0.2
                    reasons.append("volume_surge_with_ask_imbalance")
            
            if selling_pressure_score >= 0.6:
                return {
                    'signal': SignalAction.SELL,
                    'strength': selling_pressure_score,
                    'reasons': reasons,
                    'confidence': min(selling_pressure_score, 0.9),
                    'metadata': {
                        'imbalance': imbalance,
                        'bid_volume': bid_volume,
                        'ask_volume': ask_volume,
                        'pressure_score': selling_pressure_score
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating selling pressure signal: {e}")
            return None
    
    def generate_liquidity_breakout_signal(self, market_data: MarketData,
                                          depth_metrics: Dict[str, float],
                                          walls: Dict[str, List],
                                          history: Dict[str, deque]) -> Optional[Dict[str, Any]]:
        """Generate signal based on liquidity wall breakouts."""
        try:
            current_price = market_data.price
            bid_walls = walls.get('bid_walls', [])
            ask_walls = walls.get('ask_walls', [])
            
            # Check for potential breakouts
            breakout_signals = []
            
            # Bullish breakout: Price approaching resistance with strong support
            nearby_resistance = [
                (price, qty) for price, qty in ask_walls 
                if current_price * 1.005 <= price <= current_price * 1.02
            ]
            
            strong_support = [
                (price, qty) for price, qty in bid_walls
                if current_price * 0.98 <= price <= current_price * 0.995
            ]
            
            if nearby_resistance and strong_support:
                # Check if volume is increasing (breakout preparation)
                if len(history['volume']) >= 3:
                    recent_volumes = list(history['volume'])[-3:]
                    volume_trend = (recent_volumes[-1] - recent_volumes[0]) / recent_volumes[0]
                    
                    if volume_trend > 0.2:  # 20% volume increase
                        resistance_strength = sum(qty for _, qty in nearby_resistance)
                        support_strength = sum(qty for _, qty in strong_support)
                        
                        if support_strength > resistance_strength * 1.5:
                            breakout_signals.append({
                                'signal': SignalAction.BUY,
                                'strength': 0.75,
                                'reasons': ['bullish_breakout_setup'],
                                'confidence': 0.8,
                                'metadata': {
                                    'resistance_level': min(price for price, _ in nearby_resistance),
                                    'support_strength': support_strength,
                                    'resistance_strength': resistance_strength,
                                    'volume_trend': volume_trend
                                }
                            })
            
            # Bearish breakdown: Price approaching support with strong resistance
            nearby_support = [
                (price, qty) for price, qty in bid_walls
                if current_price * 0.98 <= price <= current_price * 0.995
            ]
            
            strong_resistance = [
                (price, qty) for price, qty in ask_walls
                if current_price * 1.005 <= price <= current_price * 1.02
            ]
            
            if nearby_support and strong_resistance:
                if len(history['volume']) >= 3:
                    recent_volumes = list(history['volume'])[-3:]
                    volume_trend = (recent_volumes[-1] - recent_volumes[0]) / recent_volumes[0]
                    
                    if volume_trend > 0.2:
                        support_strength = sum(qty for _, qty in nearby_support)
                        resistance_strength = sum(qty for _, qty in strong_resistance)
                        
                        if resistance_strength > support_strength * 1.5:
                            breakout_signals.append({
                                'signal': SignalAction.SELL,
                                'strength': 0.75,
                                'reasons': ['bearish_breakdown_setup'],
                                'confidence': 0.8,
                                'metadata': {
                                    'support_level': max(price for price, _ in nearby_support),
                                    'support_strength': support_strength,
                                    'resistance_strength': resistance_strength,
                                    'volume_trend': volume_trend
                                }
                            })
            
            # Return the strongest signal
            if breakout_signals:
                return max(breakout_signals, key=lambda x: x['strength'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating breakout signal: {e}")
            return None


class LiquidityStrategy(BaseStrategy):
    """Advanced liquidity-based trading strategy."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'min_liquidity_score': 0.6,
            'imbalance_threshold': 0.3,
            'spread_threshold': 0.2,
            'depth_threshold': 200,
            'wall_threshold': 10.0,
            'lookback_periods': 20,
            'confidence_decay': 0.95,
            'volume_ma_period': 10,
            'signal_cooldown_minutes': 5,
            'min_signal_strength': 0.6,
            'enable_breakout_signals': True,
            'enable_mean_reversion': True,
            'enable_pressure_signals': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("liquidity", default_params)
        
        # Components
        self.order_book_analyzer = OrderBookAnalyzer()
        self.liquidity_scorer = LiquidityScorer()
        self.signal_generator = LiquiditySignalGenerator()
        
        # State tracking
        self.liquidity_history = deque(maxlen=self.parameters['lookback_periods'])
        self.imbalance_history = deque(maxlen=self.parameters['lookback_periods'])
        self.volume_history = deque(maxlen=self.parameters['volume_ma_period'])
        self.price_history = deque(maxlen=self.parameters['lookback_periods'])
        
        # Signal tracking
        self.last_signal_time = None
        self.signal_confidence = 0.0
        self.recent_signals = deque(maxlen=10)  # Track recent signals
    
    def _generate_signal(self, market_data: MarketData) -> Optional[TradingSignal]:
        """Generate trading signal based on advanced liquidity analysis."""
        try:
            # Check if we have order book data
            if not market_data.orderbook:
                self.logger.debug("No order book data available")
                return None
            
            # Check signal cooldown
            if self._is_in_cooldown():
                return None
            
            # Analyze order book
            depth_metrics = self.order_book_analyzer.calculate_depth_metrics(
                market_data.orderbook, levels=20
            )
            
            if not depth_metrics:
                return None
            
            # Calculate liquidity score
            liquidity_analysis = self.liquidity_scorer.calculate_liquidity_score(depth_metrics)
            liquidity_score = liquidity_analysis['score']
            
            # Update history
            self.liquidity_history.append(liquidity_score)
            self.imbalance_history.append(depth_metrics.get('imbalance', 0))
            self.volume_history.append(market_data.volume)
            self.price_history.append(market_data.price)
            
            # Check minimum liquidity requirement
            if liquidity_score < self.parameters['min_liquidity_score']:
                self.logger.debug(f"Liquidity score too low: {liquidity_score}")
                return None
            
            # Generate signals using different strategies
            signal_data = None
            
            # 1. Buying/Selling pressure signals
            if self.parameters.get('enable_pressure_signals', True):
                if not signal_data:
                    signal_data = self.signal_generator.generate_buying_pressure_signal(
                        market_data, depth_metrics, list(self.liquidity_history)
                    )
                
                if not signal_data:
                    signal_data = self.signal_generator.generate_selling_pressure_signal(
                        market_data, depth_metrics, list(self.liquidity_history)
                    )
            
            # 2. Breakout signals
            if self.parameters.get('enable_breakout_signals', True) and not signal_data:
                signal_data = self.signal_generator.generate_liquidity_breakout_signal(
                    market_data, depth_metrics, list(self.price_history)
                )
            
            # 3. Mean reversion signals
            if self.parameters.get('enable_mean_reversion', True) and not signal_data:
                signal_data = self.signal_generator.generate_mean_reversion_signal(
                    market_data, depth_metrics, list(self.liquidity_history)
                )
            
            # Convert signal data to TradingSignal
            if signal_data and signal_data['strength'] >= self.parameters['min_signal_strength']:
                signal = self._create_trading_signal(market_data, signal_data)
                
                if signal:
                    # Update strategy state
                    self.confidence = signal_data['confidence']
                    self.last_signal_time = market_data.timestamp
                    self.recent_signals.append({
                        'timestamp': market_data.timestamp,
                        'signal': signal_data['signal'],
                        'strength': signal_data['strength'],
                        'reasons': signal_data['reasons']
                    })
                
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating liquidity signal: {e}")
            return None
    
    def _is_in_cooldown(self) -> bool:
        """Check if strategy is in cooldown period."""
        if not self.last_signal_time:
            return False
        
        cooldown_minutes = self.parameters.get('signal_cooldown_minutes', 5)
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.now() - self.last_signal_time < cooldown_period
    
    def _create_trading_signal(self, market_data: MarketData, 
                              signal_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Create TradingSignal from signal data."""
        try:
            # Map signal string to SignalAction
            action_map = {
                'BUY': SignalAction.BUY,
                'SELL': SignalAction.SELL,
                'HOLD': SignalAction.HOLD
            }
            
            action = action_map.get(signal_data['signal'], SignalAction.HOLD)
            if action == SignalAction.HOLD:
                return None
            
            # Create the signal
            signal = TradingSignal(
                symbol=market_data.symbol,
                action=action,
                confidence=signal_data['confidence'],
                strategy=self.name,
                timestamp=market_data.timestamp,
                target_price=signal_data.get('target_price'),
                stop_loss=signal_data.get('stop_loss'),
                metadata={
                    'liquidity_strategy': {
                        'signal_strength': signal_data['strength'],
                        'reasons': signal_data['reasons'],
                        'signal_metadata': signal_data.get('metadata', {}),
                        'liquidity_score': self.liquidity_history[-1] if self.liquidity_history else 0
                    }
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating trading signal: {e}")
            return None
    
    def _analyze_liquidity_patterns(self, market_data: MarketData, 
                                   depth_metrics: Dict[str, float],
                                   liquidity_analysis: Dict[str, Any],
                                   walls: Dict[str, List]) -> Optional[TradingSignal]:
        """Analyze liquidity patterns for trading opportunities."""
        try:
            imbalance = depth_metrics.get('imbalance', 0)
            spread_pct = depth_metrics.get('spread_pct', 100)
            
            # Pattern 1: Strong imbalance with good liquidity
            if abs(imbalance) > self.parameters['imbalance_threshold']:
                if spread_pct < self.parameters['spread_threshold']:
                    
                    # Bullish: More bids than asks
                    if imbalance > 0:
                        return self._create_signal(
                            market_data, SignalAction.BUY,
                            reason="strong_bid_imbalance",
                            metadata={
                                'imbalance': imbalance,
                                'liquidity_score': liquidity_analysis['score'],
                                'spread_pct': spread_pct
                            }
                        )
                    
                    # Bearish: More asks than bids
                    else:
                        return self._create_signal(
                            market_data, SignalAction.SELL,
                            reason="strong_ask_imbalance",
                            metadata={
                                'imbalance': imbalance,
                                'liquidity_score': liquidity_analysis['score'],
                                'spread_pct': spread_pct
                            }
                        )
            
            # Pattern 2: Liquidity wall breakthrough
            wall_signal = self._analyze_wall_patterns(market_data, walls, depth_metrics)
            if wall_signal:
                return wall_signal
            
            # Pattern 3: Improving liquidity trend
            trend_signal = self._analyze_liquidity_trend(market_data, depth_metrics)
            if trend_signal:
                return trend_signal
            
            # Pattern 4: Volume-liquidity divergence
            divergence_signal = self._analyze_volume_liquidity_divergence(market_data, depth_metrics)
            if divergence_signal:
                return divergence_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity patterns: {e}")
            return None
    
    def _analyze_wall_patterns(self, market_data: MarketData, 
                              walls: Dict[str, List], 
                              depth_metrics: Dict[str, float]) -> Optional[TradingSignal]:
        """Analyze liquidity wall patterns."""
        try:
            bid_walls = walls.get('bid_walls', [])
            ask_walls = walls.get('ask_walls', [])
            current_price = market_data.price
            
            # Strong support wall below current price
            strong_support = any(
                price < current_price * 0.99 and qty > depth_metrics.get('total_volume', 0) * 0.1
                for price, qty in bid_walls
            )
            
            # Strong resistance wall above current price
            strong_resistance = any(
                price > current_price * 1.01 and qty > depth_metrics.get('total_volume', 0) * 0.1
                for price, qty in ask_walls
            )
            
            # Bullish: Strong support, weak resistance
            if strong_support and not strong_resistance:
                return self._create_signal(
                    market_data, SignalAction.BUY,
                    reason="strong_support_wall",
                    metadata={
                        'bid_walls': len(bid_walls),
                        'ask_walls': len(ask_walls),
                        'support_strength': max(qty for _, qty in bid_walls) if bid_walls else 0
                    }
                )
            
            # Bearish: Strong resistance, weak support
            elif strong_resistance and not strong_support:
                return self._create_signal(
                    market_data, SignalAction.SELL,
                    reason="strong_resistance_wall",
                    metadata={
                        'bid_walls': len(bid_walls),
                        'ask_walls': len(ask_walls),
                        'resistance_strength': max(qty for _, qty in ask_walls) if ask_walls else 0
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing wall patterns: {e}")
            return None
    
    def _analyze_liquidity_trend(self, market_data: MarketData, 
                                depth_metrics: Dict[str, float]) -> Optional[TradingSignal]:
        """Analyze liquidity trend patterns."""
        try:
            if len(self.liquidity_history) < 5:
                return None
            
            recent_liquidity = list(self.liquidity_history)[-5:]
            liquidity_trend = self._calculate_trend(recent_liquidity)
            
            current_liquidity = recent_liquidity[-1]
            
            # Improving liquidity with volume confirmation
            if liquidity_trend > 0.02 and current_liquidity > 0.7:
                if len(self.volume_history) >= 3:
                    recent_volume = list(self.volume_history)[-3:]
                    avg_volume = sum(recent_volume) / len(recent_volume)
                    
                    if market_data.volume > avg_volume * 1.2:  # 20% above average
                        return self._create_signal(
                            market_data, SignalAction.BUY,
                            reason="improving_liquidity_trend",
                            metadata={
                                'liquidity_trend': liquidity_trend,
                                'current_liquidity': current_liquidity,
                                'volume_ratio': market_data.volume / avg_volume
                            }
                        )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity trend: {e}")
            return None
    
    def _analyze_volume_liquidity_divergence(self, market_data: MarketData,
                                           depth_metrics: Dict[str, float]) -> Optional[TradingSignal]:
        """Analyze volume-liquidity divergence patterns."""
        try:
            if len(self.volume_history) < 5 or len(self.liquidity_history) < 5:
                return None
            
            # Calculate volume moving average
            volume_ma = MovingAverages.sma(list(self.volume_history), 5)
            if not volume_ma or volume_ma[-1] is None:
                return None
            
            volume_ratio = market_data.volume / volume_ma[-1]
            current_liquidity = self.liquidity_history[-1]
            
            # High volume with improving liquidity (bullish)
            if volume_ratio > 1.5 and current_liquidity > 0.7:
                imbalance = depth_metrics.get('imbalance', 0)
                if imbalance > 0.1:  # Slight bid preference
                    return self._create_signal(
                        market_data, SignalAction.BUY,
                        reason="volume_liquidity_bullish_divergence",
                        metadata={
                            'volume_ratio': volume_ratio,
                            'liquidity_score': current_liquidity,
                            'imbalance': imbalance
                        }
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume-liquidity divergence: {e}")
            return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple trend slope."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _calculate_strategy_confidence(self, liquidity_score: float, 
                                     depth_metrics: Dict[str, float]) -> float:
        """Calculate strategy confidence based on market conditions."""
        try:
            # Base confidence on liquidity score
            base_confidence = liquidity_score
            
            # Adjust for spread quality
            spread_pct = depth_metrics.get('spread_pct', 100)
            spread_factor = max(0.5, 1.0 - spread_pct / 0.5)  # Penalize wide spreads
            
            # Adjust for depth
            total_volume = depth_metrics.get('total_volume', 0)
            depth_factor = min(1.0, total_volume / 1000)  # Normalize to 1000 units
            
            # Adjust for imbalance (moderate imbalance is good for signals)
            imbalance = abs(depth_metrics.get('imbalance', 0))
            imbalance_factor = 1.0 if 0.1 <= imbalance <= 0.5 else 0.8
            
            # Combined confidence
            confidence = base_confidence * spread_factor * depth_factor * imbalance_factor
            
            # Apply decay if we haven't seen good conditions recently
            if hasattr(self, 'signal_confidence'):
                confidence = max(confidence, self.signal_confidence * self.parameters['confidence_decay'])
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy confidence: {e}")
            return 0.5
    
    def _create_signal(self, market_data: MarketData, action: SignalAction,
                      reason: str, metadata: Dict[str, Any]) -> TradingSignal:
        """Create a trading signal with proper metadata."""
        return TradingSignal(
            symbol=market_data.symbol,
            action=action,
            confidence=self.confidence,
            strategy=self.name,
            timestamp=market_data.timestamp,
            metadata={
                'reason': reason,
                'liquidity_analysis': metadata,
                'market_price': market_data.price,
                'spread': market_data.spread if hasattr(market_data, 'spread') else None
            }
        )
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate strategy parameters."""
        try:
            required_params = [
                'min_liquidity_score', 'imbalance_threshold', 'spread_threshold',
                'depth_threshold', 'wall_threshold', 'lookback_periods'
            ]
            
            for param in required_params:
                if param not in parameters:
                    return False
                
                value = parameters[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    return False
            
            # Specific validations
            if not 0.0 <= parameters['min_liquidity_score'] <= 1.0:
                return False
            
            if not 0.0 <= parameters['imbalance_threshold'] <= 1.0:
                return False
            
            if parameters['lookback_periods'] < 5:
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
                'liquidity_history_length': len(self.liquidity_history),
                'imbalance_history_length': len(self.imbalance_history),
                'volume_history_length': len(self.volume_history),
                'last_signal_time': self.last_signal_time,
                'current_confidence': self.signal_confidence
            }
        }
    
    def _calculate_target_price(self, market_data: MarketData, signal_type: str,
                               depth_metrics: Dict[str, float]) -> Optional[float]:
        """Calculate target price based on liquidity analysis."""
        try:
            current_price = market_data.price
            spread = depth_metrics.get('spread', 0)

            # Conservative target based on spread and liquidity
            if signal_type == 'BUY':
                # Target above current price
                target_multiplier = 1.005 + (spread / current_price)  # 0.5% + spread
                return current_price * target_multiplier
            elif signal_type == 'SELL':
                # Target below current price
                target_multiplier = 0.995 - (spread / current_price)  # 0.5% - spread
                return current_price * target_multiplier
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating target price: {e}")
            return None
    
    def _calculate_stop_loss(self, market_data: MarketData, signal_type: str,
                            depth_metrics: Dict[str, float]) -> Optional[float]:
        """Calculate stop loss based on liquidity analysis."""
        try:
            current_price = market_data.price
            spread = depth_metrics.get('spread', 0)
            
            # Conservative stop loss
            if signal_type == 'BUY':
                # Stop loss below current price
                stop_multiplier = 0.99 - (spread / current_price * 2)  # 1% - 2x spread
                return current_price * stop_multiplier
            elif signal_type == 'SELL':
                # Stop loss above current price
                stop_multiplier = 1.01 + (spread / current_price * 2)  # 1% + 2x spread
                return current_price * stop_multiplier
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return None


    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate strategy parameters."""
        try:
            required_params = [
                'min_liquidity_score', 'imbalance_threshold', 'spread_threshold',
                'depth_threshold', 'wall_threshold', 'lookback_periods'
            ]
            
            for param in required_params:
                if param not in parameters:
                    return False
                
                value = parameters[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    return False
            
            # Specific validations
            if not 0.0 <= parameters['min_liquidity_score'] <= 1.0:
                return False
            
            if not 0.0 <= parameters['imbalance_threshold'] <= 1.0:
                return False
            
            if parameters['lookback_periods'] < 5:
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
                'liquidity_history_length': len(self.liquidity_history),
                'imbalance_history_length': len(self.imbalance_history),
                'volume_history_length': len(self.volume_history),
                'price_history_length': len(self.price_history),
                'last_signal_time': self.last_signal_time,
                'current_confidence': self.signal_confidence,
                'recent_signals_count': len(self.recent_signals)
            },
            'recent_signals': list(self.recent_signals)
        }
    
    def get_liquidity_analysis(self, market_data: MarketData) -> Dict[str, Any]:
        """Get current liquidity analysis for debugging/monitoring."""
        try:
            if not market_data.orderbook:
                return {'error': 'No order book data available'}
            
            # Analyze order book
            depth_metrics = self.order_book_analyzer.calculate_depth_metrics(
                market_data.orderbook, levels=20
            )
            
            # Calculate liquidity score
            liquidity_analysis = self.liquidity_scorer.calculate_liquidity_score(depth_metrics)
            
            # Detect walls
            walls = self.order_book_analyzer.detect_liquidity_walls(
                market_data.orderbook, self.parameters['wall_threshold']
            )
            
            # Calculate market impact for different sizes
            impact_small = self.order_book_analyzer.calculate_market_impact(
                market_data.orderbook, 100, 'BUY'
            )
            impact_large = self.order_book_analyzer.calculate_market_impact(
                market_data.orderbook, 1000, 'BUY'
            )
            
            return {
                'timestamp': market_data.timestamp,
                'symbol': market_data.symbol,
                'current_price': market_data.price,
                'depth_metrics': depth_metrics,
                'liquidity_score': liquidity_analysis,
                'liquidity_walls': {
                    'bid_walls_count': len(walls['bid_walls']),
                    'ask_walls_count': len(walls['ask_walls']),
                    'strongest_bid_wall': max(walls['bid_walls'], key=lambda x: x[1]) if walls['bid_walls'] else None,
                    'strongest_ask_wall': max(walls['ask_walls'], key=lambda x: x[1]) if walls['ask_walls'] else None
                },
                'market_impact': {
                    'small_trade': impact_small,
                    'large_trade': impact_large
                },
                'strategy_state': {
                    'current_confidence': self.confidence,
                    'liquidity_trend': self.signal_generator._calculate_liquidity_momentum(
                        list(self.liquidity_history)
                    ) if len(self.liquidity_history) >= 2 else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting liquidity analysis: {e}")
            return {'error': str(e)}