"""
Backtesting engine for historical strategy testing.

This module provides the core backtesting engine that simulates trading
strategies against historical market data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import pandas as pd
import numpy as np

from ..models.trading import TradingSignal, Position, Trade, MarketData
from ..models.config import BotConfig, RiskConfig
from ..strategies.base_strategy import BaseStrategy


class BacktestStatus(Enum):
    """Backtest execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframe: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d
    commission_rate: float = 0.001  # 0.1% commission
    slippage_rate: float = 0.0005   # 0.05% slippage
    max_positions: int = 5
    enable_shorting: bool = True
    
    def __post_init__(self):
        """Validate backtest configuration."""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        
        if not self.symbols:
            raise ValueError("At least one symbol must be specified")
        
        if self.commission_rate < 0 or self.commission_rate > 0.1:
            raise ValueError("commission_rate must be between 0 and 10%")
        
        if self.slippage_rate < 0 or self.slippage_rate > 0.01:
            raise ValueError("slippage_rate must be between 0 and 1%")


@dataclass
class BacktestResult:
    """Results from a backtest execution."""
    config: BacktestConfig
    status: BacktestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trading statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_return: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Portfolio evolution
    portfolio_values: List[Tuple[datetime, float]] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    positions: List[Position] = field(default_factory=list)
    
    # Strategy-specific metrics
    strategy_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    error_message: Optional[str] = None
    
    def get_duration(self) -> Optional[timedelta]:
        """Get backtest execution duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'config': {
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'initial_balance': self.config.initial_balance,
                'symbols': self.config.symbols,
                'timeframe': self.config.timeframe,
                'commission_rate': self.config.commission_rate,
                'slippage_rate': self.config.slippage_rate
            },
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.get_duration().total_seconds() if self.get_duration() else None,
            'performance': {
                'total_return': self.total_return,
                'annualized_return': self.annualized_return,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor
            },
            'trading_stats': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'avg_trade_return': self.avg_trade_return,
                'avg_winning_trade': self.avg_winning_trade,
                'avg_losing_trade': self.avg_losing_trade,
                'max_consecutive_wins': self.max_consecutive_wins,
                'max_consecutive_losses': self.max_consecutive_losses
            },
            'strategy_metrics': self.strategy_metrics,
            'error_message': self.error_message
        }


class BacktestEngine:
    """
    Core backtesting engine for strategy evaluation.
    
    Simulates trading strategies against historical data with realistic
    market conditions including commissions, slippage, and position limits.
    """
    
    def __init__(self, data_provider=None):
        """Initialize backtest engine."""
        self.data_provider = data_provider
        self.logger = logging.getLogger(__name__)
        
        # Current backtest state
        self._current_backtest: Optional[BacktestResult] = None
        self._is_running = False
        self._should_cancel = False
        
        # Portfolio state during backtest
        self._portfolio_value = 0.0
        self._cash_balance = 0.0
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._portfolio_history: List[Tuple[datetime, float]] = []
        
        # Performance tracking
        self._peak_value = 0.0
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._max_consecutive_wins = 0
        self._max_consecutive_losses = 0
    
    async def run_backtest(self, strategy: BaseStrategy, config: BacktestConfig,
                          bot_config: Optional[BotConfig] = None) -> BacktestResult:
        """
        Run backtest for given strategy and configuration.
        
        Args:
            strategy: Trading strategy to test
            config: Backtest configuration
            bot_config: Bot configuration (optional)
            
        Returns:
            BacktestResult with performance metrics and trade history
        """
        if self._is_running:
            raise RuntimeError("Backtest is already running")
        
        self._is_running = True
        self._should_cancel = False
        
        # Initialize backtest result
        result = BacktestResult(
            config=config,
            status=BacktestStatus.RUNNING,
            start_time=datetime.now()
        )
        self._current_backtest = result
        
        try:
            self.logger.info(f"Starting backtest: {config.start_date} to {config.end_date}")
            
            # Initialize portfolio
            self._initialize_portfolio(config.initial_balance)
            
            # Get historical data
            if not self.data_provider:
                raise ValueError("Data provider not configured")
            
            historical_data = await self.data_provider.get_historical_data(
                symbols=config.symbols,
                start_date=config.start_date,
                end_date=config.end_date,
                timeframe=config.timeframe
            )
            
            if not historical_data:
                raise ValueError("No historical data available for specified period")
            
            # Run simulation
            await self._run_simulation(strategy, config, historical_data, bot_config)
            
            # Calculate final metrics
            self._calculate_performance_metrics(result, config)
            
            result.status = BacktestStatus.COMPLETED
            result.end_time = datetime.now()
            
            self.logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            result.status = BacktestStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
        
        finally:
            self._is_running = False
            self._current_backtest = None
        
        return result
    
    def _initialize_portfolio(self, initial_balance: float) -> None:
        """Initialize portfolio for backtest."""
        self._portfolio_value = initial_balance
        self._cash_balance = initial_balance
        self._positions.clear()
        self._trades.clear()
        self._portfolio_history.clear()
        
        self._peak_value = initial_balance
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._max_consecutive_wins = 0
        self._max_consecutive_losses = 0
    
    async def _run_simulation(self, strategy: BaseStrategy, config: BacktestConfig,
                            historical_data: Dict[str, pd.DataFrame],
                            bot_config: Optional[BotConfig]) -> None:
        """Run the main simulation loop."""
        # Combine all data and sort by timestamp
        all_data_points = []
        
        for symbol, df in historical_data.items():
            for _, row in df.iterrows():
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                all_data_points.append(market_data)
        
        # Sort by timestamp
        all_data_points.sort(key=lambda x: x.timestamp)
        
        # Process each data point
        for i, market_data in enumerate(all_data_points):
            if self._should_cancel:
                break
            
            # Update portfolio value
            self._update_portfolio_value(market_data)
            
            # Record portfolio history
            self._portfolio_history.append((market_data.timestamp, self._portfolio_value))
            
            # Generate trading signal
            try:
                signal = await strategy.analyze_market_data(market_data)
                
                if signal and signal.action in ['BUY', 'SELL']:
                    # Process trading signal
                    await self._process_signal(signal, market_data, config)
                
            except Exception as e:
                self.logger.warning(f"Strategy error at {market_data.timestamp}: {e}")
                continue
            
            # Update positions (check for stop losses, take profits)
            self._update_positions(market_data, config)
            
            # Progress logging
            if i % 1000 == 0:
                progress = (i / len(all_data_points)) * 100
                self.logger.debug(f"Backtest progress: {progress:.1f}%")
    
    def _update_portfolio_value(self, market_data: MarketData) -> None:
        """Update portfolio value based on current market data."""
        total_position_value = 0.0
        
        # Calculate value of all positions
        for position in self._positions.values():
            if position.symbol == market_data.symbol:
                if position.side == 'LONG':
                    position_value = position.size * market_data.close
                else:  # SHORT
                    position_value = position.size * (2 * position.entry_price - market_data.close)
                
                total_position_value += position_value
                
                # Update position current price and unrealized PnL
                position.current_price = market_data.close
                if position.side == 'LONG':
                    position.unrealized_pnl = (market_data.close - position.entry_price) * position.size
                else:
                    position.unrealized_pnl = (position.entry_price - market_data.close) * position.size
        
        self._portfolio_value = self._cash_balance + total_position_value
        
        # Update peak value for drawdown calculation
        if self._portfolio_value > self._peak_value:
            self._peak_value = self._portfolio_value
    
    async def _process_signal(self, signal: TradingSignal, market_data: MarketData,
                            config: BacktestConfig) -> None:
        """Process a trading signal and execute trades."""
        try:
            # Check if we can open new positions
            if len(self._positions) >= config.max_positions and signal.symbol not in self._positions:
                return
            
            # Calculate position size (simplified)
            position_size = self._calculate_position_size(signal, config)
            if position_size <= 0:
                return
            
            # Apply slippage
            execution_price = self._apply_slippage(market_data.close, signal.action, config.slippage_rate)
            
            # Calculate commission
            commission = position_size * execution_price * config.commission_rate
            
            if signal.action == 'BUY':
                await self._execute_buy_order(signal, execution_price, position_size, commission, market_data.timestamp)
            elif signal.action == 'SELL':
                await self._execute_sell_order(signal, execution_price, position_size, commission, market_data.timestamp)
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    def _calculate_position_size(self, signal: TradingSignal, config: BacktestConfig) -> float:
        """Calculate position size based on available cash and risk management."""
        # Simple position sizing: use 2% of portfolio per trade
        max_position_value = self._portfolio_value * 0.02
        
        if signal.target_price and signal.target_price > 0:
            position_size = max_position_value / signal.target_price
        else:
            # Use current market price estimate
            position_size = max_position_value / 100  # Placeholder
        
        # Ensure we have enough cash
        required_cash = position_size * (signal.target_price or 100)  # Placeholder price
        if required_cash > self._cash_balance:
            position_size = self._cash_balance / (signal.target_price or 100)
        
        return max(0, position_size)
    
    def _apply_slippage(self, price: float, action: str, slippage_rate: float) -> float:
        """Apply slippage to execution price."""
        if action == 'BUY':
            return price * (1 + slippage_rate)
        else:  # SELL
            return price * (1 - slippage_rate)
    
    async def _execute_buy_order(self, signal: TradingSignal, price: float, size: float,
                               commission: float, timestamp: datetime) -> None:
        """Execute a buy order."""
        total_cost = (price * size) + commission
        
        if total_cost > self._cash_balance:
            return  # Insufficient funds
        
        # Update cash balance
        self._cash_balance -= total_cost
        
        # Create or update position
        if signal.symbol in self._positions:
            # Add to existing position
            existing_pos = self._positions[signal.symbol]
            total_size = existing_pos.size + size
            avg_price = ((existing_pos.entry_price * existing_pos.size) + (price * size)) / total_size
            
            existing_pos.size = total_size
            existing_pos.entry_price = avg_price
        else:
            # Create new position
            position = Position(
                symbol=signal.symbol,
                side='LONG',
                size=size,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.target_price,
                timestamp=timestamp
            )
            self._positions[signal.symbol] = position
        
        # Record trade
        trade = Trade(
            id=f"trade_{len(self._trades) + 1}",
            symbol=signal.symbol,
            side='BUY',
            size=size,
            price=price,
            commission=commission,
            timestamp=timestamp,
            strategy=signal.strategy
        )
        self._trades.append(trade)
    
    async def _execute_sell_order(self, signal: TradingSignal, price: float, size: float,
                                commission: float, timestamp: datetime) -> None:
        """Execute a sell order."""
        if signal.symbol not in self._positions:
            return  # No position to sell
        
        position = self._positions[signal.symbol]
        sell_size = min(size, position.size)
        
        # Calculate proceeds
        proceeds = (price * sell_size) - commission
        self._cash_balance += proceeds
        
        # Calculate realized PnL
        if position.side == 'LONG':
            pnl = (price - position.entry_price) * sell_size
        else:  # SHORT
            pnl = (position.entry_price - price) * sell_size
        
        # Update position
        position.size -= sell_size
        if position.size <= 0:
            del self._positions[signal.symbol]
        
        # Record trade
        trade = Trade(
            id=f"trade_{len(self._trades) + 1}",
            symbol=signal.symbol,
            side='SELL',
            size=sell_size,
            price=price,
            commission=commission,
            timestamp=timestamp,
            strategy=signal.strategy,
            pnl=pnl
        )
        self._trades.append(trade)
        
        # Update consecutive win/loss tracking
        if pnl > 0:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
            self._max_consecutive_wins = max(self._max_consecutive_wins, self._consecutive_wins)
        elif pnl < 0:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
            self._max_consecutive_losses = max(self._max_consecutive_losses, self._consecutive_losses)
    
    def _update_positions(self, market_data: MarketData, config: BacktestConfig) -> None:
        """Update positions and check for stop losses/take profits."""
        positions_to_close = []
        
        for symbol, position in self._positions.items():
            if symbol != market_data.symbol:
                continue
            
            current_price = market_data.close
            
            # Check stop loss
            if position.stop_loss:
                if (position.side == 'LONG' and current_price <= position.stop_loss) or \
                   (position.side == 'SHORT' and current_price >= position.stop_loss):
                    positions_to_close.append((symbol, current_price, 'STOP_LOSS'))
            
            # Check take profit
            if position.take_profit:
                if (position.side == 'LONG' and current_price >= position.take_profit) or \
                   (position.side == 'SHORT' and current_price <= position.take_profit):
                    positions_to_close.append((symbol, current_price, 'TAKE_PROFIT'))
        
        # Close positions that hit stop loss or take profit
        for symbol, price, reason in positions_to_close:
            asyncio.create_task(self._close_position(symbol, price, market_data.timestamp, reason))
    
    async def _close_position(self, symbol: str, price: float, timestamp: datetime, reason: str) -> None:
        """Close a position at market price."""
        if symbol not in self._positions:
            return
        
        position = self._positions[symbol]
        
        # Create sell signal to close position
        signal = TradingSignal(
            symbol=symbol,
            action='SELL',
            confidence=1.0,
            strategy=f"system_{reason.lower()}",
            timestamp=timestamp,
            target_price=price
        )
        
        # Execute sell order
        commission = position.size * price * 0.001  # Use default commission
        await self._execute_sell_order(signal, price, position.size, commission, timestamp)
    
    def _calculate_performance_metrics(self, result: BacktestResult, config: BacktestConfig) -> None:
        """Calculate performance metrics for backtest result."""
        if not self._portfolio_history:
            return
        
        initial_value = config.initial_balance
        final_value = self._portfolio_value
        
        # Basic returns
        result.total_return = (final_value - initial_value) / initial_value
        
        # Annualized return
        days = (config.end_date - config.start_date).days
        if days > 0:
            result.annualized_return = ((final_value / initial_value) ** (365.25 / days)) - 1
        
        # Maximum drawdown
        result.max_drawdown = (self._peak_value - min(v for _, v in self._portfolio_history)) / self._peak_value
        
        # Trading statistics
        result.total_trades = len([t for t in self._trades if t.side == 'SELL'])
        result.trades = self._trades.copy()
        result.positions = list(self._positions.values())
        result.portfolio_values = self._portfolio_history.copy()
        
        # Win/loss statistics
        profitable_trades = [t for t in self._trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self._trades if t.pnl and t.pnl < 0]
        
        result.winning_trades = len(profitable_trades)
        result.losing_trades = len(losing_trades)
        
        if result.total_trades > 0:
            result.win_rate = result.winning_trades / result.total_trades
        
        if profitable_trades:
            result.avg_winning_trade = sum(t.pnl for t in profitable_trades) / len(profitable_trades)
        
        if losing_trades:
            result.avg_losing_trade = sum(t.pnl for t in losing_trades) / len(losing_trades)
        
        # Profit factor
        total_profit = sum(t.pnl for t in profitable_trades) if profitable_trades else 0
        total_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        
        if total_loss > 0:
            result.profit_factor = total_profit / total_loss
        
        # Average trade return
        if result.total_trades > 0:
            total_pnl = sum(t.pnl for t in self._trades if t.pnl)
            result.avg_trade_return = total_pnl / result.total_trades
        
        # Consecutive wins/losses
        result.max_consecutive_wins = self._max_consecutive_wins
        result.max_consecutive_losses = self._max_consecutive_losses
        
        # Risk-adjusted returns (simplified)
        if self._portfolio_history:
            returns = []
            for i in range(1, len(self._portfolio_history)):
                prev_value = self._portfolio_history[i-1][1]
                curr_value = self._portfolio_history[i][1]
                returns.append((curr_value - prev_value) / prev_value)
            
            if returns:
                avg_return = np.mean(returns)
                return_std = np.std(returns)
                
                if return_std > 0:
                    result.sharpe_ratio = (avg_return * 252) / (return_std * np.sqrt(252))  # Annualized
                
                # Sortino ratio (downside deviation)
                negative_returns = [r for r in returns if r < 0]
                if negative_returns:
                    downside_std = np.std(negative_returns)
                    if downside_std > 0:
                        result.sortino_ratio = (avg_return * 252) / (downside_std * np.sqrt(252))
    
    def cancel_backtest(self) -> bool:
        """Cancel running backtest."""
        if self._is_running:
            self._should_cancel = True
            return True
        return False
    
    def get_current_status(self) -> Optional[Dict[str, Any]]:
        """Get current backtest status."""
        if self._current_backtest:
            return {
                'status': self._current_backtest.status.value,
                'start_time': self._current_backtest.start_time.isoformat(),
                'portfolio_value': self._portfolio_value,
                'total_trades': len(self._trades),
                'cash_balance': self._cash_balance,
                'open_positions': len(self._positions)
            }
        return None