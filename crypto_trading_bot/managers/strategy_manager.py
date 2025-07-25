"""
Strategy Manager for coordinating multiple trading strategies.

This manager handles strategy registration, market data distribution,
signal aggregation, and performance tracking across all strategies.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from ..strategies.base_strategy import BaseStrategy
from ..strategies.liquidity_strategy import LiquidityStrategy
from ..strategies.momentum_strategy import MomentumStrategy
from ..strategies.pattern_strategy import PatternStrategy
from ..strategies.candlestick_strategy import CandlestickStrategy
from ..models.trading import TradingSignal, MarketData, SignalAction
from ..utils.logging_config import setup_logging


class StrategyPerformance:
    """Tracks performance metrics for a strategy."""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.total_signals = 0
        self.successful_signals = 0
        self.failed_signals = 0
        self.total_return = 0.0
        self.win_rate = 0.0
        self.avg_confidence = 0.0
        self.avg_return_per_signal = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
        # Signal history for detailed analysis
        self.signal_history = deque(maxlen=1000)
        self.return_history = deque(maxlen=1000)
        self.confidence_history = deque(maxlen=1000)
        
        # Performance over time
        self.daily_returns = defaultdict(float)
        self.monthly_returns = defaultdict(float)
        
        # Last update timestamp
        self.last_updated = datetime.now()
    
    def update_signal_result(self, signal: TradingSignal, return_pct: float, success: bool):
        """Update performance metrics with signal result."""
        try:
            self.total_signals += 1
            self.signal_history.append({
                'signal': signal,
                'return': return_pct,
                'success': success,
                'timestamp': datetime.now()
            })
            
            if success:
                self.successful_signals += 1
            else:
                self.failed_signals += 1
            
            # Update returns
            self.total_return += return_pct
            self.return_history.append(return_pct)
            self.confidence_history.append(signal.confidence)
            
            # Update daily/monthly tracking
            date_key = signal.timestamp.date()
            month_key = signal.timestamp.strftime('%Y-%m')
            self.daily_returns[date_key] += return_pct
            self.monthly_returns[month_key] += return_pct
            
            # Recalculate metrics
            self._recalculate_metrics()
            self.last_updated = datetime.now()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error updating performance for {self.strategy_name}: {e}")
    
    def _recalculate_metrics(self):
        """Recalculate performance metrics."""
        try:
            if self.total_signals > 0:
                self.win_rate = self.successful_signals / self.total_signals
                self.avg_return_per_signal = self.total_return / self.total_signals
                
                if self.confidence_history:
                    self.avg_confidence = statistics.mean(self.confidence_history)
                
                # Calculate Sharpe ratio (simplified)
                if len(self.return_history) > 1:
                    returns_std = statistics.stdev(self.return_history)
                    if returns_std > 0:
                        self.sharpe_ratio = self.avg_return_per_signal / returns_std
                
                # Calculate max drawdown
                self._calculate_max_drawdown()
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Error recalculating metrics for {self.strategy_name}: {e}")
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown."""
        try:
            if len(self.return_history) < 2:
                return
            
            # Calculate cumulative returns
            cumulative_returns = []
            cumulative = 0.0
            
            for ret in self.return_history:
                cumulative += ret
                cumulative_returns.append(cumulative)
            
            # Find maximum drawdown
            peak = cumulative_returns[0]
            max_dd = 0.0
            
            for value in cumulative_returns[1:]:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak if peak != 0 else 0
                    max_dd = max(max_dd, drawdown)
            
            self.max_drawdown = max_dd
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating max drawdown for {self.strategy_name}: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'strategy_name': self.strategy_name,
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'failed_signals': self.failed_signals,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'avg_return_per_signal': self.avg_return_per_signal,
            'avg_confidence': self.avg_confidence,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'last_updated': self.last_updated,
            'recent_performance': {
                'last_10_signals': list(self.signal_history)[-10:] if self.signal_history else [],
                'last_30_days_return': self._get_recent_return(30),
                'last_7_days_return': self._get_recent_return(7)
            }
        }
    
    def _get_recent_return(self, days: int) -> float:
        """Get return for recent period."""
        try:
            cutoff_date = datetime.now().date() - timedelta(days=days)
            recent_return = 0.0
            
            for date, return_val in self.daily_returns.items():
                if date >= cutoff_date:
                    recent_return += return_val
            
            return recent_return
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating recent return: {e}")
            return 0.0


class SignalAggregator:
    """Aggregates and prioritizes signals from multiple strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Aggregation parameters
        self.confidence_weights = {
            'LiquidityStrategy': 1.2,      # Higher weight for liquidity signals
            'MomentumStrategy': 1.0,       # Standard weight
            'PatternStrategy': 1.1,        # Slightly higher for patterns
            'CandlestickStrategy': 0.9     # Lower weight (fallback strategy)
        }
        
        self.min_signal_confidence = 0.6
        self.max_signals_per_symbol = 3
        self.signal_timeout_minutes = 30
    
    def aggregate_signals(self, signals: List[TradingSignal], 
                         strategy_performances: Dict[str, StrategyPerformance]) -> Optional[TradingSignal]:
        """Aggregate multiple signals into a single trading decision."""
        try:
            if not signals:
                return None
            
            # Filter signals by confidence and timeout
            valid_signals = self._filter_valid_signals(signals)
            if not valid_signals:
                return None
            
            # Group signals by symbol and action
            signal_groups = self._group_signals(valid_signals)
            
            # Find the best signal group
            best_group = self._select_best_signal_group(signal_groups, strategy_performances)
            if not best_group:
                return None
            
            # Create aggregated signal
            aggregated_signal = self._create_aggregated_signal(best_group, strategy_performances)
            
            return aggregated_signal
            
        except Exception as e:
            self.logger.error(f"Error aggregating signals: {e}")
            return None
    
    def _filter_valid_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals by confidence and timeout."""
        try:
            valid_signals = []
            current_time = datetime.now()
            timeout_delta = timedelta(minutes=self.signal_timeout_minutes)
            
            for signal in signals:
                # Check confidence threshold
                if signal.confidence < self.min_signal_confidence:
                    continue
                
                # Check timeout
                if current_time - signal.timestamp > timeout_delta:
                    continue
                
                valid_signals.append(signal)
            
            return valid_signals
            
        except Exception as e:
            self.logger.error(f"Error filtering signals: {e}")
            return []
    
    def _group_signals(self, signals: List[TradingSignal]) -> Dict[Tuple[str, str], List[TradingSignal]]:
        """Group signals by symbol and action."""
        try:
            groups = defaultdict(list)
            
            for signal in signals:
                key = (signal.symbol, signal.action.value)
                groups[key].append(signal)
            
            return dict(groups)
            
        except Exception as e:
            self.logger.error(f"Error grouping signals: {e}")
            return {}
    
    def _select_best_signal_group(self, signal_groups: Dict[Tuple[str, str], List[TradingSignal]], 
                                 strategy_performances: Dict[str, StrategyPerformance]) -> Optional[List[TradingSignal]]:
        """Select the best signal group based on weighted scoring."""
        try:
            if not signal_groups:
                return None
            
            best_group = None
            best_score = 0.0
            
            for group_key, signals in signal_groups.items():
                group_score = self._calculate_group_score(signals, strategy_performances)
                
                if group_score > best_score:
                    best_score = group_score
                    best_group = signals
            
            return best_group
            
        except Exception as e:
            self.logger.error(f"Error selecting best signal group: {e}")
            return None
    
    def _calculate_group_score(self, signals: List[TradingSignal], 
                              strategy_performances: Dict[str, StrategyPerformance]) -> float:
        """Calculate weighted score for a signal group."""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            for signal in signals:
                # Base score from confidence
                base_score = signal.confidence
                
                # Strategy weight
                strategy_weight = self.confidence_weights.get(signal.strategy_name, 1.0)
                
                # Performance adjustment
                performance_multiplier = 1.0
                if signal.strategy_name in strategy_performances:
                    perf = strategy_performances[signal.strategy_name]
                    if perf.total_signals > 10:  # Only adjust if we have enough data
                        # Adjust based on win rate and recent performance
                        win_rate_factor = perf.win_rate
                        recent_return_factor = max(0.5, min(1.5, 1.0 + perf._get_recent_return(7)))
                        performance_multiplier = (win_rate_factor + recent_return_factor) / 2
                
                # Calculate weighted score
                weighted_score = base_score * strategy_weight * performance_multiplier
                total_score += weighted_score
                total_weight += strategy_weight
            
            # Return average weighted score
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating group score: {e}")
            return 0.0
    
    def _create_aggregated_signal(self, signals: List[TradingSignal], 
                                 strategy_performances: Dict[str, StrategyPerformance]) -> TradingSignal:
        """Create aggregated signal from multiple signals."""
        try:
            # Use the highest confidence signal as base
            base_signal = max(signals, key=lambda s: s.confidence)
            
            # Calculate aggregated confidence
            confidences = [s.confidence for s in signals]
            strategy_weights = [self.confidence_weights.get(s.strategy_name, 1.0) for s in signals]
            
            # Weighted average confidence
            weighted_confidence = sum(c * w for c, w in zip(confidences, strategy_weights)) / sum(strategy_weights)
            
            # Boost confidence for signal confluence
            confluence_boost = min(0.15, (len(signals) - 1) * 0.05)  # Up to 15% boost
            final_confidence = min(0.95, weighted_confidence + confluence_boost)
            
            # Calculate aggregated quantity
            total_quantity = sum(s.quantity for s in signals) / len(signals)  # Average quantity
            
            # Create aggregated signal
            aggregated_signal = TradingSignal(
                symbol=base_signal.symbol,
                action=base_signal.action,
                price=base_signal.price,
                quantity=total_quantity,
                confidence=final_confidence,
                timestamp=datetime.now(),
                strategy_name="StrategyManager",
                metadata={
                    'aggregated_from': [
                        {
                            'strategy': s.strategy_name,
                            'confidence': s.confidence,
                            'quantity': s.quantity,
                            'timestamp': s.timestamp,
                            'metadata': s.metadata
                        }
                        for s in signals
                    ],
                    'signal_count': len(signals),
                    'confluence_boost': confluence_boost,
                    'weighted_confidence': weighted_confidence,
                    'base_signal_strategy': base_signal.strategy_name
                }
            )
            
            return aggregated_signal
            
        except Exception as e:
            self.logger.error(f"Error creating aggregated signal: {e}")
            return signals[0]  # Return first signal as fallback


class StrategyManager:
    """Manages multiple trading strategies and coordinates their execution."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = setup_logging(__name__)
        
        # Configuration
        self.config = config or {}
        self.max_concurrent_strategies = self.config.get('max_concurrent_strategies', 4)
        self.signal_aggregation_enabled = self.config.get('signal_aggregation_enabled', True)
        self.performance_tracking_enabled = self.config.get('performance_tracking_enabled', True)
        
        # Strategy registry
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_configs: Dict[str, Dict[str, Any]] = {}
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        
        # Signal management
        self.signal_aggregator = SignalAggregator()
        self.recent_signals = deque(maxlen=1000)
        self.active_signals: Dict[str, TradingSignal] = {}  # symbol -> active signal
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_strategies)
        self.strategy_lock = threading.RLock()
        
        # Performance tracking
        self.manager_performance = {
            'total_signals_generated': 0,
            'total_signals_executed': 0,
            'aggregation_success_rate': 0.0,
            'avg_signal_confidence': 0.0,
            'strategy_utilization': defaultdict(int),
            'last_updated': datetime.now()
        }
        
        # Initialize default strategies
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default trading strategies."""
        try:
            # Liquidity Strategy (Primary)
            liquidity_config = self.config.get('liquidity_strategy', {})
            self.register_strategy('LiquidityStrategy', LiquidityStrategy, liquidity_config)
            
            # Momentum Strategy
            momentum_config = self.config.get('momentum_strategy', {})
            self.register_strategy('MomentumStrategy', MomentumStrategy, momentum_config)
            
            # Pattern Strategy
            pattern_config = self.config.get('pattern_strategy', {})
            self.register_strategy('PatternStrategy', PatternStrategy, pattern_config)
            
            # Candlestick Strategy (Fallback)
            candlestick_config = self.config.get('candlestick_strategy', {})
            self.register_strategy('CandlestickStrategy', CandlestickStrategy, candlestick_config)
            
            self.logger.info(f"Initialized {len(self.strategies)} default strategies")
            
        except Exception as e:
            self.logger.error(f"Error initializing default strategies: {e}")
    
    def register_strategy(self, name: str, strategy_class: type, config: Dict[str, Any] = None):
        """Register a new trading strategy."""
        try:
            with self.strategy_lock:
                if name in self.strategies:
                    self.logger.warning(f"Strategy {name} already registered, replacing...")
                
                # Create strategy instance
                strategy_instance = strategy_class(config)
                
                # Validate strategy
                if not isinstance(strategy_instance, BaseStrategy):
                    raise ValueError(f"Strategy {name} must inherit from BaseStrategy")
                
                # Register strategy
                self.strategies[name] = strategy_instance
                self.strategy_configs[name] = config or {}
                
                # Initialize performance tracking
                if self.performance_tracking_enabled:
                    self.strategy_performances[name] = StrategyPerformance(name)
                
                self.logger.info(f"Registered strategy: {name}")
                
        except Exception as e:
            self.logger.error(f"Error registering strategy {name}: {e}")
            raise
    
    def unregister_strategy(self, name: str):
        """Unregister a trading strategy."""
        try:
            with self.strategy_lock:
                if name not in self.strategies:
                    self.logger.warning(f"Strategy {name} not found for unregistration")
                    return
                
                # Remove strategy
                del self.strategies[name]
                del self.strategy_configs[name]
                
                if name in self.strategy_performances:
                    del self.strategy_performances[name]
                
                self.logger.info(f"Unregistered strategy: {name}")
                
        except Exception as e:
            self.logger.error(f"Error unregistering strategy {name}: {e}")
    
    def analyze_market(self, market_data: MarketData) -> Optional[TradingSignal]:
        """Distribute market data to all strategies and aggregate signals."""
        try:
            # Distribute market data to all strategies concurrently
            signals = self._distribute_market_data(market_data)
            
            # Update manager performance
            self.manager_performance['total_signals_generated'] += len(signals)
            
            if not signals:
                return None
            
            # Log individual signals
            for signal in signals:
                self.recent_signals.append(signal)
                self.manager_performance['strategy_utilization'][signal.strategy_name] += 1
                self.logger.debug(f"Signal from {signal.strategy_name}: {signal.action.value} "
                                f"{signal.symbol} @ {signal.confidence:.2%}")
            
            # Aggregate signals if enabled
            if self.signal_aggregation_enabled and len(signals) > 1:
                aggregated_signal = self.signal_aggregator.aggregate_signals(signals, self.strategy_performances)
                if aggregated_signal:
                    self.manager_performance['total_signals_executed'] += 1
                    self._update_manager_metrics(aggregated_signal)
                    return aggregated_signal
            
            # Return best single signal if no aggregation
            elif signals:
                best_signal = max(signals, key=lambda s: s.confidence)
                self.manager_performance['total_signals_executed'] += 1
                self._update_manager_metrics(best_signal)
                return best_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return None   
 
    def _distribute_market_data(self, market_data: MarketData) -> List[TradingSignal]:
        """Distribute market data to all strategies concurrently."""
    
        try:
            signals = []
            
            with self.strategy_lock:
                if not self.strategies:
                    return signals
                
                # Submit analysis tasks to thread pool
                future_to_strategy = {}
                for name, strategy in self.strategies.items():
                    future = self.executor.submit(self._analyze_with_strategy, strategy, market_data)
                    future_to_strategy[future] = name
                
                # Collect results
                for future in as_completed(future_to_strategy, timeout=5.0):
                    strategy_name = future_to_strategy[future]
                    try:
                        signal = future.result()
                        if signal:
                            signals.append(signal)
                    except Exception as e:
                        self.logger.error(f"Error in strategy {strategy_name}: {e}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error distributing market data: {e}")
            return []
    
    def _analyze_with_strategy(self, strategy: BaseStrategy, market_data: MarketData) -> Optional[TradingSignal]:
        """Analyze market data with a single strategy."""
        try:
            return strategy.analyze_market(market_data)
        except Exception as e:
            self.logger.error(f"Error in strategy {strategy.name}: {e}")
            return None
    
    def _update_manager_metrics(self, signal: TradingSignal):
        """Update manager performance metrics."""
        try:
            # Update average confidence
            total_signals = self.manager_performance['total_signals_executed']
            current_avg = self.manager_performance['avg_signal_confidence']
            
            new_avg = ((current_avg * (total_signals - 1)) + signal.confidence) / total_signals
            self.manager_performance['avg_signal_confidence'] = new_avg
            
            # Update aggregation success rate
            if 'aggregated_from' in signal.metadata:
                aggregated_count = len([s for s in self.recent_signals 
                                      if 'aggregated_from' in s.metadata])
                self.manager_performance['aggregation_success_rate'] = aggregated_count / total_signals
            
            self.manager_performance['last_updated'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating manager metrics: {e}")
    
    def update_strategy_performance(self, signal: TradingSignal, return_pct: float, success: bool):
        """Update performance metrics for strategies."""
        try:
            if not self.performance_tracking_enabled:
                return
            
            # Update performance for aggregated signals
            if 'aggregated_from' in signal.metadata:
                for strategy_info in signal.metadata['aggregated_from']:
                    strategy_name = strategy_info['strategy']
                    if strategy_name in self.strategy_performances:
                        # Create a temporary signal for performance tracking
                        temp_signal = TradingSignal(
                            symbol=signal.symbol,
                            action=signal.action,
                            price=signal.price,
                            quantity=strategy_info['quantity'],
                            confidence=strategy_info['confidence'],
                            timestamp=strategy_info['timestamp'],
                            strategy_name=strategy_name,
                            metadata=strategy_info['metadata']
                        )
                        self.strategy_performances[strategy_name].update_signal_result(
                            temp_signal, return_pct, success
                        )
            else:
                # Update performance for single strategy signal
                strategy_name = signal.strategy_name
                if strategy_name in self.strategy_performances:
                    self.strategy_performances[strategy_name].update_signal_result(
                        signal, return_pct, success
                    )
            
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {e}")
    
    def get_strategy_performance(self, strategy_name: str = None) -> Dict[str, Any]:
        """Get performance metrics for strategies."""
        try:
            if strategy_name:
                if strategy_name in self.strategy_performances:
                    return self.strategy_performances[strategy_name].get_performance_summary()
                else:
                    return {}
            
            # Return all strategy performances
            performances = {}
            for name, perf in self.strategy_performances.items():
                performances[name] = perf.get_performance_summary()
            
            return performances
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {e}")
            return {}
    
    def get_manager_performance(self) -> Dict[str, Any]:
        """Get overall manager performance metrics."""
        try:
            return {
                'manager_metrics': self.manager_performance.copy(),
                'strategy_performances': self.get_strategy_performance(),
                'active_strategies': list(self.strategies.keys()),
                'strategy_configs': self.strategy_configs.copy(),
                'recent_signals_count': len(self.recent_signals),
                'active_signals_count': len(self.active_signals)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting manager performance: {e}")
            return {}
    
    def rebalance_strategy_weights(self):
        """Rebalance strategy weights based on recent performance."""
        try:
            if not self.performance_tracking_enabled:
                return
            
            # Calculate performance-based weights
            total_performance_score = 0.0
            strategy_scores = {}
            
            for name, perf in self.strategy_performances.items():
                if perf.total_signals > 5:  # Need minimum signals for reliable scoring
                    # Combine win rate, return, and Sharpe ratio
                    performance_score = (
                        perf.win_rate * 0.4 +
                        max(0, min(1, perf.avg_return_per_signal * 10)) * 0.3 +
                        max(0, min(1, perf.sharpe_ratio)) * 0.3
                    )
                    
                    # Adjust for recent performance
                    recent_return = perf._get_recent_return(7)
                    if recent_return > 0:
                        performance_score *= 1.1
                    elif recent_return < -0.02:  # Penalize poor recent performance
                        performance_score *= 0.9
                    
                    strategy_scores[name] = performance_score
                    total_performance_score += performance_score
            
            # Update signal aggregator weights
            if total_performance_score > 0:
                for name, score in strategy_scores.items():
                    normalized_weight = (score / total_performance_score) * len(strategy_scores)
                    # Smooth weight changes to avoid volatility
                    current_weight = self.signal_aggregator.confidence_weights.get(name, 1.0)
                    new_weight = (current_weight * 0.7) + (normalized_weight * 0.3)
                    self.signal_aggregator.confidence_weights[name] = max(0.5, min(2.0, new_weight))
                
                self.logger.info(f"Rebalanced strategy weights: {self.signal_aggregator.confidence_weights}")
            
        except Exception as e:
            self.logger.error(f"Error rebalancing strategy weights: {e}")
    
    def handle_conflict_resolution(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Resolve conflicts between opposing signals."""
        try:
            if len(signals) <= 1:
                return signals[0] if signals else None
            
            # Group signals by action
            buy_signals = [s for s in signals if s.action == SignalAction.BUY]
            sell_signals = [s for s in signals if s.action == SignalAction.SELL]
            
            # No conflict if all signals agree
            if not buy_signals or not sell_signals:
                return self.signal_aggregator.aggregate_signals(signals, self.strategy_performances)
            
            # Resolve conflict using weighted confidence
            buy_score = self._calculate_conflict_score(buy_signals)
            sell_score = self._calculate_conflict_score(sell_signals)
            
            # Choose winning side
            if buy_score > sell_score * 1.1:  # 10% bias threshold
                winning_signals = buy_signals
                self.logger.info(f"Conflict resolved in favor of BUY (score: {buy_score:.3f} vs {sell_score:.3f})")
            elif sell_score > buy_score * 1.1:
                winning_signals = sell_signals
                self.logger.info(f"Conflict resolved in favor of SELL (score: {sell_score:.3f} vs {buy_score:.3f})")
            else:
                # Too close to call, return None to avoid trading
                self.logger.info(f"Conflict unresolved, scores too close (BUY: {buy_score:.3f}, SELL: {sell_score:.3f})")
                return None
            
            return self.signal_aggregator.aggregate_signals(winning_signals, self.strategy_performances)
            
        except Exception as e:
            self.logger.error(f"Error in conflict resolution: {e}")
            return None
    
    def _calculate_conflict_score(self, signals: List[TradingSignal]) -> float:
        """Calculate weighted score for conflict resolution."""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            for signal in signals:
                # Base score from confidence
                base_score = signal.confidence
                
                # Strategy weight
                strategy_weight = self.signal_aggregator.confidence_weights.get(signal.strategy_name, 1.0)
                
                # Performance adjustment
                performance_multiplier = 1.0
                if signal.strategy_name in self.strategy_performances:
                    perf = self.strategy_performances[signal.strategy_name]
                    if perf.total_signals > 10:
                        performance_multiplier = perf.win_rate * perf.avg_confidence
                
                # Calculate weighted score
                weighted_score = base_score * strategy_weight * performance_multiplier
                total_score += weighted_score
                total_weight += strategy_weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating conflict score: {e}")
            return 0.0
    
    def optimize_strategy_parameters(self, strategy_name: str, optimization_period_days: int = 30):
        """Optimize strategy parameters based on historical performance."""
        try:
            if strategy_name not in self.strategies:
                self.logger.error(f"Strategy {strategy_name} not found for optimization")
                return
            
            if strategy_name not in self.strategy_performances:
                self.logger.error(f"No performance data for strategy {strategy_name}")
                return
            
            strategy = self.strategies[strategy_name]
            performance = self.strategy_performances[strategy_name]
            
            # Get recent signal history
            cutoff_date = datetime.now() - timedelta(days=optimization_period_days)
            recent_signals = [
                entry for entry in performance.signal_history
                if entry['timestamp'] >= cutoff_date
            ]
            
            if len(recent_signals) < 10:
                self.logger.info(f"Insufficient data for optimizing {strategy_name}")
                return
            
            # Analyze performance patterns
            successful_signals = [s for s in recent_signals if s['success']]
            failed_signals = [s for s in recent_signals if not s['success']]
            
            if not successful_signals:
                self.logger.warning(f"No successful signals for {strategy_name} optimization")
                return
            
            # Extract patterns from successful signals
            success_confidences = [s['signal'].confidence for s in successful_signals]
            failure_confidences = [s['signal'].confidence for s in failed_signals] if failed_signals else []
            
            # Suggest confidence threshold adjustment
            if success_confidences and failure_confidences:
                success_avg = statistics.mean(success_confidences)
                failure_avg = statistics.mean(failure_confidences)
                
                if success_avg > failure_avg + 0.1:  # Significant difference
                    suggested_threshold = (success_avg + failure_avg) / 2
                    self.logger.info(f"Suggested confidence threshold for {strategy_name}: {suggested_threshold:.2f}")
                    
                    # Update strategy parameters if it supports it
                    if hasattr(strategy, 'update_parameters'):
                        strategy.update_parameters({'min_confidence': suggested_threshold})
            
            self.logger.info(f"Optimization completed for {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy {strategy_name}: {e}")
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current status of all strategies."""
        try:
            status = {}
            
            with self.strategy_lock:
                for name, strategy in self.strategies.items():
                    try:
                        strategy_info = strategy.get_strategy_info() if hasattr(strategy, 'get_strategy_info') else {}
                        
                        status[name] = {
                            'name': name,
                            'class': strategy.__class__.__name__,
                            'confidence': strategy.confidence,
                            'active': True,
                            'config': self.strategy_configs.get(name, {}),
                            'info': strategy_info,
                            'performance': (
                                self.strategy_performances[name].get_performance_summary()
                                if name in self.strategy_performances else {}
                            )
                        }
                    except Exception as e:
                        status[name] = {
                            'name': name,
                            'active': False,
                            'error': str(e)
                        }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting strategy status: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown the strategy manager and cleanup resources."""
        try:
            self.logger.info("Shutting down Strategy Manager...")
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            # Clear strategies
            with self.strategy_lock:
                self.strategies.clear()
                self.strategy_configs.clear()
                self.strategy_performances.clear()
            
            self.logger.info("Strategy Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()