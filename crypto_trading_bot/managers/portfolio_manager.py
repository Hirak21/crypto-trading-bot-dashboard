"""
Portfolio Manager for comprehensive portfolio tracking and performance analysis.

This manager handles portfolio position tracking, P&L calculations,
performance metrics, and detailed reporting across all strategies.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from datetime import datetime, timedelta, date
from collections import deque, defaultdict
import statistics
import math
from enum import Enum
import json

from ..models.trading import TradingSignal, MarketData
from ..models.config import TradingConfig
from ..utils.logging_config import setup_logging


class PortfolioPosition:
    """Represents a portfolio position with comprehensive tracking."""
    
    def __init__(self, symbol: str, quantity: float, entry_price: float, 
                 entry_time: datetime, strategy_name: str, trade_id: str):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.strategy_name = strategy_name
        self.trade_id = trade_id
        
        # Current state
        self.current_price = entry_price
        self.market_value = abs(quantity * entry_price)
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        
        # Performance tracking
        self.max_unrealized_pnl = 0.0
        self.min_unrealized_pnl = 0.0
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0
        
        # Historical data
        self.price_history = deque(maxlen=1000)
        self.pnl_history = deque(maxlen=1000)
        self.last_updated = entry_time
        
        # Position flags
        self.is_active = True
        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.realized_pnl: Optional[float] = None
    
    def update_market_price(self, current_price: float, timestamp: datetime = None):
        """Update position with current market price."""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            self.current_price = current_price
            self.market_value = abs(self.quantity * current_price)
            
            # Calculate unrealized P&L
            if self.quantity > 0:  # Long position
                self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            else:  # Short position
                self.unrealized_pnl = (self.entry_price - current_price) * abs(self.quantity)
            
            # Calculate percentage P&L
            position_cost = abs(self.quantity * self.entry_price)
            self.unrealized_pnl_pct = (self.unrealized_pnl / position_cost) * 100 if position_cost > 0 else 0
            
            # Update excursions
            if self.unrealized_pnl > self.max_unrealized_pnl:
                self.max_unrealized_pnl = self.unrealized_pnl
                self.max_favorable_excursion = self.unrealized_pnl_pct
            
            if self.unrealized_pnl < self.min_unrealized_pnl:
                self.min_unrealized_pnl = self.unrealized_pnl
                self.max_adverse_excursion = self.unrealized_pnl_pct
            
            # Store historical data
            self.price_history.append((timestamp, current_price))
            self.pnl_history.append((timestamp, self.unrealized_pnl))
            self.last_updated = timestamp
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error updating position {self.symbol}: {e}")
    
    def close_position(self, exit_price: float, exit_time: datetime = None):
        """Close the position and calculate realized P&L."""
        try:
            if exit_time is None:
                exit_time = datetime.now()
            
            self.exit_price = exit_price
            self.exit_time = exit_time
            self.is_active = False
            
            # Calculate realized P&L
            if self.quantity > 0:  # Long position
                self.realized_pnl = (exit_price - self.entry_price) * self.quantity
            else:  # Short position
                self.realized_pnl = (self.entry_price - exit_price) * abs(self.quantity)
            
            # Final update
            self.update_market_price(exit_price, exit_time)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error closing position {self.symbol}: {e}")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get comprehensive position summary."""
        duration = (self.last_updated - self.entry_time).total_seconds() / 3600  # Hours
        
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'exit_price': self.exit_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'realized_pnl': self.realized_pnl,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion,
            'strategy_name': self.strategy_name,
            'trade_id': self.trade_id,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'duration_hours': duration,
            'is_active': self.is_active,
            'last_updated': self.last_updated
        }


class StrategyPerformance:
    """Tracks performance metrics for individual strategies."""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.reset_time = datetime.now()
        
        # Trade statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.breakeven_trades = 0
        
        # P&L statistics
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        
        # Performance ratios
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.expectancy = 0.0
        self.sharpe_ratio = 0.0
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        self.current_streak = 0
        self.current_streak_type = None  # 'win' or 'loss'
        
        # Historical tracking
        self.trade_history = deque(maxlen=1000)
        self.daily_pnl = defaultdict(float)
        self.monthly_pnl = defaultdict(float)
        self.equity_curve = deque(maxlen=10000)
        
        # Current positions
        self.active_positions = 0
        self.total_exposure = 0.0
        
        self.last_updated = datetime.now()
    
    def add_trade(self, position: PortfolioPosition):
        """Add a completed trade to performance tracking."""
        try:
            if not position.realized_pnl or position.is_active:
                return
            
            self.total_trades += 1
            pnl = position.realized_pnl
            
            # Categorize trade
            if pnl > 0:
                self.winning_trades += 1
                self.gross_profit += pnl
                self.largest_win = max(self.largest_win, pnl)
                self._update_streak('win')
            elif pnl < 0:
                self.losing_trades += 1
                self.gross_loss += abs(pnl)
                self.largest_loss = min(self.largest_loss, pnl)
                self._update_streak('loss')
            else:
                self.breakeven_trades += 1
                self._reset_streak()
            
            # Update totals
            self.total_realized_pnl += pnl
            
            # Update daily/monthly tracking
            trade_date = position.exit_time.date()
            trade_month = position.exit_time.strftime('%Y-%m')
            self.daily_pnl[trade_date] += pnl
            self.monthly_pnl[trade_month] += pnl
            
            # Store trade history
            self.trade_history.append({
                'symbol': position.symbol,
                'pnl': pnl,
                'pnl_pct': position.unrealized_pnl_pct,
                'duration': (position.exit_time - position.entry_time).total_seconds() / 3600,
                'exit_time': position.exit_time,
                'strategy': position.strategy_name
            })
            
            # Update equity curve
            self.equity_curve.append((position.exit_time, self.total_realized_pnl))
            
            # Recalculate metrics
            self._recalculate_metrics()
            self.last_updated = datetime.now()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error adding trade to {self.strategy_name}: {e}")
    
    def update_unrealized_pnl(self, unrealized_pnl: float, active_positions: int, total_exposure: float):
        """Update unrealized P&L and position metrics."""
        try:
            self.total_unrealized_pnl = unrealized_pnl
            self.active_positions = active_positions
            self.total_exposure = total_exposure
            
            # Update current drawdown
            total_equity = self.total_realized_pnl + self.total_unrealized_pnl
            if hasattr(self, 'peak_equity'):
                if total_equity > self.peak_equity:
                    self.peak_equity = total_equity
                    self.current_drawdown = 0.0
                else:
                    self.current_drawdown = (self.peak_equity - total_equity) / self.peak_equity if self.peak_equity > 0 else 0
                    self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            else:
                self.peak_equity = total_equity
                self.current_drawdown = 0.0
            
            self.last_updated = datetime.now()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error updating unrealized P&L for {self.strategy_name}: {e}")
    
    def _update_streak(self, result_type: str):
        """Update winning/losing streak tracking."""
        try:
            if self.current_streak_type == result_type:
                self.current_streak += 1
            else:
                self.current_streak = 1
                self.current_streak_type = result_type
            
            if result_type == 'win':
                self.max_consecutive_wins = max(self.max_consecutive_wins, self.current_streak)
            else:
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.current_streak)
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Error updating streak: {e}")
    
    def _reset_streak(self):
        """Reset streak tracking for breakeven trades."""
        self.current_streak = 0
        self.current_streak_type = None
    
    def _recalculate_metrics(self):
        """Recalculate all performance metrics."""
        try:
            if self.total_trades == 0:
                return
            
            # Basic ratios
            self.win_rate = (self.winning_trades / self.total_trades) * 100
            self.avg_win = self.gross_profit / self.winning_trades if self.winning_trades > 0 else 0
            self.avg_loss = self.gross_loss / self.losing_trades if self.losing_trades > 0 else 0
            
            # Profit factor
            self.profit_factor = self.gross_profit / self.gross_loss if self.gross_loss > 0 else float('inf')
            
            # Expectancy
            win_prob = self.winning_trades / self.total_trades
            loss_prob = self.losing_trades / self.total_trades
            self.expectancy = (win_prob * self.avg_win) - (loss_prob * self.avg_loss)
            
            # Sharpe ratio (simplified)
            if len(self.trade_history) > 1:
                returns = [trade['pnl'] for trade in self.trade_history]
                if statistics.stdev(returns) > 0:
                    self.sharpe_ratio = statistics.mean(returns) / statistics.stdev(returns)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error recalculating metrics for {self.strategy_name}: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'strategy_name': self.strategy_name,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'breakeven_trades': self.breakeven_trades,
            'win_rate': self.win_rate,
            'total_realized_pnl': self.total_realized_pnl,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'expectancy': self.expectancy,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'current_streak': self.current_streak,
            'current_streak_type': self.current_streak_type,
            'active_positions': self.active_positions,
            'total_exposure': self.total_exposure,
            'last_updated': self.last_updated
        }
class PortfolioManager:
    """Comprehensive portfolio management and performance tracking system."""
    
    def __init__(self, config: TradingConfig, initial_capital: float = 10000.0):
        self.logger = setup_logging(__name__)
        self.config = config
        
        # Portfolio state
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        
        # Position tracking
        self.active_positions: Dict[str, PortfolioPosition] = {}
        self.closed_positions: List[PortfolioPosition] = []
        self.position_history = deque(maxlen=10000)
        
        # Strategy performance tracking
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        
        # Portfolio-level metrics
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.total_commission = 0.0
        self.net_pnl = 0.0
        self.total_return = 0.0
        
        # Risk metrics
        self.total_exposure = 0.0
        self.net_exposure = 0.0
        self.gross_exposure = 0.0
        self.leverage = 0.0
        self.portfolio_beta = 0.0
        
        # Drawdown tracking
        self.peak_portfolio_value = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_duration = 0
        self.underwater_periods = deque(maxlen=1000)
        
        # Performance history
        self.daily_returns = defaultdict(float)
        self.monthly_returns = defaultdict(float)
        self.equity_curve = deque(maxlen=10000)
        self.drawdown_curve = deque(maxlen=10000)
        
        # Market data cache for position updates
        self.market_data_cache: Dict[str, MarketData] = {}
        
        # Performance analytics
        self.portfolio_metrics_history = deque(maxlen=1000)
        self.last_portfolio_update = datetime.now()
        
        # Initialize equity curve
        self.equity_curve.append((datetime.now(), initial_capital))
        
        self.logger.info(f"Portfolio Manager initialized with capital: {initial_capital:.2f}")
    
    def add_position(self, symbol: str, quantity: float, entry_price: float, 
                    strategy_name: str, trade_id: str, entry_time: datetime = None) -> bool:
        """Add a new position to the portfolio."""
        try:
            if entry_time is None:
                entry_time = datetime.now()
            
            # Check if we already have a position in this symbol
            if symbol in self.active_positions:
                self.logger.warning(f"Position already exists for {symbol}, updating...")
                existing_pos = self.active_positions[symbol]
                # Combine positions (simplified - in practice might need more complex logic)
                new_quantity = existing_pos.quantity + quantity
                new_avg_price = ((existing_pos.quantity * existing_pos.entry_price) + 
                               (quantity * entry_price)) / new_quantity
                existing_pos.quantity = new_quantity
                existing_pos.entry_price = new_avg_price
                existing_pos.market_value = abs(new_quantity * new_avg_price)
                return True
            
            # Create new position
            position = PortfolioPosition(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                entry_time=entry_time,
                strategy_name=strategy_name,
                trade_id=trade_id
            )
            
            # Add to active positions
            self.active_positions[symbol] = position
            
            # Update available capital
            position_cost = abs(quantity * entry_price)
            self.available_capital -= position_cost
            
            # Initialize strategy performance if needed
            if strategy_name not in self.strategy_performances:
                self.strategy_performances[strategy_name] = StrategyPerformance(strategy_name)
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            self.logger.info(f"Added position: {symbol} {quantity:.6f} @ {entry_price:.6f} ({strategy_name})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding position {symbol}: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, commission: float = 0.0, 
                      exit_time: datetime = None) -> bool:
        """Close an active position."""
        try:
            if symbol not in self.active_positions:
                self.logger.error(f"No active position found for {symbol}")
                return False
            
            if exit_time is None:
                exit_time = datetime.now()
            
            position = self.active_positions[symbol]
            
            # Close the position
            position.close_position(exit_price, exit_time)
            
            # Calculate realized P&L
            realized_pnl = position.realized_pnl
            
            # Update capital
            position_proceeds = abs(position.quantity * exit_price)
            self.available_capital += position_proceeds
            self.current_capital += realized_pnl - commission
            self.total_realized_pnl += realized_pnl
            self.total_commission += commission
            
            # Update strategy performance
            if position.strategy_name in self.strategy_performances:
                self.strategy_performances[position.strategy_name].add_trade(position)
            
            # Move to closed positions
            self.closed_positions.append(position)
            self.position_history.append(position.get_position_summary())
            del self.active_positions[symbol]
            
            # Update daily/monthly returns
            trade_date = exit_time.date()
            trade_month = exit_time.strftime('%Y-%m')
            self.daily_returns[trade_date] += realized_pnl
            self.monthly_returns[trade_month] += realized_pnl
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            self.logger.info(f"Closed position: {symbol} @ {exit_price:.6f}, P&L: {realized_pnl:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    def update_market_data(self, market_data: Dict[str, MarketData]):
        """Update all positions with current market data."""
        try:
            # Update market data cache
            self.market_data_cache.update(market_data)
            
            # Update active positions
            for symbol, position in self.active_positions.items():
                if symbol in market_data:
                    position.update_market_price(market_data[symbol].price)
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def _update_portfolio_metrics(self):
        """Update comprehensive portfolio metrics."""
        try:
            # Calculate unrealized P&L
            self.total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            
            # Calculate exposures
            long_exposure = sum(pos.market_value for pos in self.active_positions.values() if pos.quantity > 0)
            short_exposure = sum(pos.market_value for pos in self.active_positions.values() if pos.quantity < 0)
            
            self.gross_exposure = long_exposure + short_exposure
            self.net_exposure = long_exposure - short_exposure
            self.total_exposure = self.gross_exposure
            
            # Calculate leverage
            portfolio_value = self.current_capital + self.total_unrealized_pnl
            self.leverage = self.gross_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate net P&L and returns
            self.net_pnl = self.total_realized_pnl + self.total_unrealized_pnl - self.total_commission
            self.total_return = (self.net_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0
            
            # Update drawdown
            current_portfolio_value = self.current_capital + self.total_unrealized_pnl
            if current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_portfolio_value
                self.drawdown_duration = 0
            else:
                self.drawdown_duration += 1
            
            self.current_drawdown = ((self.peak_portfolio_value - current_portfolio_value) / 
                                   self.peak_portfolio_value) if self.peak_portfolio_value > 0 else 0
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Update curves
            now = datetime.now()
            self.equity_curve.append((now, current_portfolio_value))
            self.drawdown_curve.append((now, self.current_drawdown))
            
            if self.current_drawdown > 0:
                self.underwater_periods.append((now, self.current_drawdown))
            
            # Update strategy unrealized P&L
            for strategy_name, strategy_perf in self.strategy_performances.items():
                strategy_unrealized = sum(
                    pos.unrealized_pnl for pos in self.active_positions.values() 
                    if pos.strategy_name == strategy_name
                )
                strategy_positions = len([
                    pos for pos in self.active_positions.values() 
                    if pos.strategy_name == strategy_name
                ])
                strategy_exposure = sum(
                    pos.market_value for pos in self.active_positions.values() 
                    if pos.strategy_name == strategy_name
                )
                
                strategy_perf.update_unrealized_pnl(strategy_unrealized, strategy_positions, strategy_exposure)
            
            # Store metrics history
            self.portfolio_metrics_history.append({
                'timestamp': now,
                'portfolio_value': current_portfolio_value,
                'total_pnl': self.net_pnl,
                'unrealized_pnl': self.total_unrealized_pnl,
                'drawdown': self.current_drawdown,
                'leverage': self.leverage,
                'active_positions': len(self.active_positions)
            })
            
            self.last_portfolio_update = now
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio metrics: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        try:
            current_value = self.current_capital + self.total_unrealized_pnl
            
            return {
                'capital': {
                    'initial_capital': self.initial_capital,
                    'current_capital': self.current_capital,
                    'available_capital': self.available_capital,
                    'portfolio_value': current_value
                },
                'pnl': {
                    'total_realized_pnl': self.total_realized_pnl,
                    'total_unrealized_pnl': self.total_unrealized_pnl,
                    'total_commission': self.total_commission,
                    'net_pnl': self.net_pnl,
                    'total_return': self.total_return
                },
                'exposure': {
                    'gross_exposure': self.gross_exposure,
                    'net_exposure': self.net_exposure,
                    'leverage': self.leverage
                },
                'risk': {
                    'current_drawdown': self.current_drawdown,
                    'max_drawdown': self.max_drawdown,
                    'drawdown_duration': self.drawdown_duration,
                    'underwater_periods': len(self.underwater_periods)
                },
                'positions': {
                    'active_positions': len(self.active_positions),
                    'total_trades': len(self.closed_positions),
                    'symbols': list(self.active_positions.keys())
                },
                'last_updated': self.last_portfolio_update
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def get_position_details(self, symbol: str = None) -> Dict[str, Any]:
        """Get detailed position information."""
        try:
            if symbol:
                if symbol in self.active_positions:
                    return self.active_positions[symbol].get_position_summary()
                else:
                    return {}
            
            # Return all active positions
            return {
                symbol: position.get_position_summary()
                for symbol, position in self.active_positions.items()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting position details: {e}")
            return {}
    
    def get_strategy_performance(self, strategy_name: str = None) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        try:
            if strategy_name:
                if strategy_name in self.strategy_performances:
                    return self.strategy_performances[strategy_name].get_performance_summary()
                else:
                    return {}
            
            # Return all strategy performances
            return {
                name: perf.get_performance_summary()
                for name, perf in self.strategy_performances.items()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {e}")
            return {}
    
    def get_trade_history(self, strategy_name: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Get trade history with optional filtering."""
        try:
            trades = []
            
            # Get trades from closed positions
            positions_to_process = self.closed_positions
            if limit:
                positions_to_process = positions_to_process[-limit:]
            
            for position in positions_to_process:
                if strategy_name and position.strategy_name != strategy_name:
                    continue
                
                trade_summary = position.get_position_summary()
                if position.realized_pnl is not None:
                    trade_summary['return_pct'] = (position.realized_pnl / 
                                                 (abs(position.quantity * position.entry_price))) * 100
                    trades.append(trade_summary)
            
            return sorted(trades, key=lambda x: x['exit_time'] or datetime.min, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get advanced performance analytics."""
        try:
            if not self.closed_positions:
                return {'message': 'No completed trades for analysis'}
            
            # Calculate overall statistics
            total_trades = len(self.closed_positions)
            winning_trades = len([p for p in self.closed_positions if p.realized_pnl > 0])
            losing_trades = len([p for p in self.closed_positions if p.realized_pnl < 0])
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # P&L statistics
            all_pnl = [p.realized_pnl for p in self.closed_positions if p.realized_pnl is not None]
            winning_pnl = [pnl for pnl in all_pnl if pnl > 0]
            losing_pnl = [pnl for pnl in all_pnl if pnl < 0]
            
            avg_win = statistics.mean(winning_pnl) if winning_pnl else 0
            avg_loss = statistics.mean(losing_pnl) if losing_pnl else 0
            largest_win = max(winning_pnl) if winning_pnl else 0
            largest_loss = min(losing_pnl) if losing_pnl else 0
            
            # Risk metrics
            profit_factor = abs(sum(winning_pnl) / sum(losing_pnl)) if losing_pnl else float('inf')
            expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
            
            # Sharpe ratio (simplified)
            sharpe_ratio = 0
            if len(all_pnl) > 1 and statistics.stdev(all_pnl) > 0:
                sharpe_ratio = statistics.mean(all_pnl) / statistics.stdev(all_pnl)
            
            # Time-based analysis
            trade_durations = []
            for pos in self.closed_positions:
                if pos.exit_time:
                    duration = (pos.exit_time - pos.entry_time).total_seconds() / 3600  # Hours
                    trade_durations.append(duration)
            
            avg_trade_duration = statistics.mean(trade_durations) if trade_durations else 0
            
            return {
                'overall_statistics': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_pnl': self.total_realized_pnl,
                    'net_pnl': self.net_pnl,
                    'total_return': self.total_return
                },
                'pnl_statistics': {
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'largest_win': largest_win,
                    'largest_loss': largest_loss,
                    'profit_factor': profit_factor,
                    'expectancy': expectancy
                },
                'risk_metrics': {
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': self.max_drawdown,
                    'current_drawdown': self.current_drawdown,
                    'avg_trade_duration_hours': avg_trade_duration
                },
                'monthly_returns': dict(self.monthly_returns),
                'strategy_breakdown': self.get_strategy_performance()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance analytics: {e}")
            return {}
    
    def export_portfolio_data(self, file_path: str = None) -> str:
        """Export portfolio data to JSON file."""
        try:
            if file_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_path = f'portfolio_export_{timestamp}.json'
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'portfolio_summary': self.get_portfolio_summary(),
                'strategy_performance': self.get_strategy_performance(),
                'trade_history': self.get_trade_history(),
                'performance_analytics': self.get_performance_analytics(),
                'active_positions': self.get_position_details(),
                'configuration': {
                    'initial_capital': self.initial_capital,
                    'total_commission': self.total_commission
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Portfolio data exported to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error exporting portfolio data: {e}")
            return ""
    
    def reset_portfolio(self, new_initial_capital: float = None):
        """Reset portfolio to initial state."""
        try:
            if new_initial_capital:
                self.initial_capital = new_initial_capital
            
            # Reset all tracking
            self.current_capital = self.initial_capital
            self.available_capital = self.initial_capital
            self.active_positions.clear()
            self.closed_positions.clear()
            self.position_history.clear()
            self.strategy_performances.clear()
            
            # Reset metrics
            self.total_realized_pnl = 0.0
            self.total_unrealized_pnl = 0.0
            self.total_commission = 0.0
            self.net_pnl = 0.0
            self.total_return = 0.0
            
            # Reset risk metrics
            self.peak_portfolio_value = self.initial_capital
            self.current_drawdown = 0.0
            self.max_drawdown = 0.0
            self.drawdown_duration = 0
            
            # Reset history
            self.daily_returns.clear()
            self.monthly_returns.clear()
            self.equity_curve.clear()
            self.drawdown_curve.clear()
            self.underwater_periods.clear()
            self.portfolio_metrics_history.clear()
            
            # Initialize equity curve
            self.equity_curve.append((datetime.now(), self.initial_capital))
            
            self.logger.info(f"Portfolio reset with capital: {self.initial_capital:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error resetting portfolio: {e}")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics."""
        try:
            # Calculate VaR (simplified)
            if len(self.daily_returns) > 30:
                daily_returns_list = list(self.daily_returns.values())
                var_95 = statistics.quantiles(daily_returns_list, n=20)[1] if len(daily_returns_list) > 20 else 0  # 5th percentile
                var_99 = statistics.quantiles(daily_returns_list, n=100)[1] if len(daily_returns_list) > 100 else 0  # 1st percentile
            else:
                var_95 = var_99 = 0
            
            # Calculate beta (simplified - would need market data for proper calculation)
            portfolio_beta = 1.0  # Placeholder
            
            # Calculate correlation with strategies
            strategy_correlations = {}
            if len(self.strategy_performances) > 1:
                # Simplified correlation calculation
                for strategy_name in self.strategy_performances:
                    strategy_correlations[strategy_name] = 0.5  # Placeholder
            
            return {
                'drawdown_metrics': {
                    'current_drawdown': self.current_drawdown,
                    'max_drawdown': self.max_drawdown,
                    'drawdown_duration': self.drawdown_duration,
                    'underwater_periods': len(self.underwater_periods)
                },
                'exposure_metrics': {
                    'gross_exposure': self.gross_exposure,
                    'net_exposure': self.net_exposure,
                    'leverage': self.leverage,
                    'concentration': self._calculate_concentration()
                },
                'var_metrics': {
                    'var_95': var_95,
                    'var_99': var_99,
                    'portfolio_beta': portfolio_beta
                },
                'strategy_correlations': strategy_correlations,
                'position_metrics': {
                    'active_positions': len(self.active_positions),
                    'avg_position_size': self.gross_exposure / len(self.active_positions) if self.active_positions else 0,
                    'largest_position': max([pos.market_value for pos in self.active_positions.values()], default=0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}")
            return {}
    
    def _calculate_concentration(self) -> float:
        """Calculate portfolio concentration (Herfindahl index)."""
        try:
            if not self.active_positions or self.gross_exposure == 0:
                return 0.0
            
            # Calculate concentration using Herfindahl index
            concentration = sum(
                (pos.market_value / self.gross_exposure) ** 2 
                for pos in self.active_positions.values()
            )
            
            return concentration
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration: {e}")
            return 0.0
    
    def shutdown(self):
        """Shutdown portfolio manager and save final state."""
        try:
            self.logger.info("Shutting down Portfolio Manager...")
            
            # Export final portfolio state
            final_export = self.export_portfolio_data("final_portfolio_state.json")
            
            # Log final summary
            final_summary = self.get_portfolio_summary()
            self.logger.info(f"Final portfolio summary: {final_summary}")
            
            self.logger.info("Portfolio Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during portfolio manager shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()