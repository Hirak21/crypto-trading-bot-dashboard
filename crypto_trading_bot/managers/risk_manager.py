"""
Risk Manager for comprehensive trading risk management.

This manager handles position sizing, risk validation, portfolio exposure monitoring,
drawdown controls, and emergency stop functionality.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics
import math
from enum import Enum

from ..models.trading import TradingSignal, MarketData, SignalAction
from ..models.config import RiskConfig
from ..utils.logging_config import setup_logging


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAlert(NamedTuple):
    """Risk alert structure."""
    timestamp: datetime
    level: RiskLevel
    category: str
    message: str
    data: Dict[str, Any]


class PositionRisk:
    """Tracks risk metrics for individual positions."""
    
    def __init__(self, symbol: str, entry_price: float, quantity: float, 
                 stop_loss: float = None, take_profit: float = None):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = datetime.now()
        
        # Risk metrics
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0
        self.position_value = abs(quantity * entry_price)
        
        # Risk calculations
        self.risk_amount = self._calculate_risk_amount()
        self.risk_reward_ratio = self._calculate_risk_reward_ratio()
        
        # Tracking
        self.price_history = deque(maxlen=1000)
        self.pnl_history = deque(maxlen=1000)
        self.last_updated = datetime.now()
    
    def update_price(self, current_price: float):
        """Update position with current market price."""
        try:
            self.current_price = current_price
            self.price_history.append((datetime.now(), current_price))
            
            # Calculate P&L
            if self.quantity > 0:  # Long position
                self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            else:  # Short position
                self.unrealized_pnl = (self.entry_price - current_price) * abs(self.quantity)
            
            self.unrealized_pnl_pct = self.unrealized_pnl / self.position_value if self.position_value > 0 else 0
            self.pnl_history.append((datetime.now(), self.unrealized_pnl_pct))
            
            # Update excursions
            if self.unrealized_pnl_pct > self.max_favorable_excursion:
                self.max_favorable_excursion = self.unrealized_pnl_pct
            
            if self.unrealized_pnl_pct < self.max_adverse_excursion:
                self.max_adverse_excursion = self.unrealized_pnl_pct
            
            self.last_updated = datetime.now()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error updating position price for {self.symbol}: {e}")
    
    def _calculate_risk_amount(self) -> float:
        """Calculate the risk amount for this position."""
        try:
            if not self.stop_loss:
                return self.position_value * 0.02  # Default 2% risk
            
            if self.quantity > 0:  # Long position
                risk_per_share = self.entry_price - self.stop_loss
            else:  # Short position
                risk_per_share = self.stop_loss - self.entry_price
            
            return abs(risk_per_share * self.quantity)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating risk amount: {e}")
            return 0.0
    
    def _calculate_risk_reward_ratio(self) -> float:
        """Calculate risk-reward ratio."""
        try:
            if not self.stop_loss or not self.take_profit:
                return 0.0
            
            if self.quantity > 0:  # Long position
                risk = self.entry_price - self.stop_loss
                reward = self.take_profit - self.entry_price
            else:  # Short position
                risk = self.stop_loss - self.entry_price
                reward = self.entry_price - self.take_profit
            
            return abs(reward / risk) if risk != 0 else 0.0
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating risk-reward ratio: {e}")
            return 0.0
    
    def should_stop_out(self) -> bool:
        """Check if position should be stopped out."""
        try:
            if not self.stop_loss:
                return False
            
            if self.quantity > 0:  # Long position
                return self.current_price <= self.stop_loss
            else:  # Short position
                return self.current_price >= self.stop_loss
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Error checking stop loss for {self.symbol}: {e}")
            return False
    
    def should_take_profit(self) -> bool:
        """Check if position should take profit."""
        try:
            if not self.take_profit:
                return False
            
            if self.quantity > 0:  # Long position
                return self.current_price >= self.take_profit
            else:  # Short position
                return self.current_price <= self.take_profit
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Error checking take profit for {self.symbol}: {e}")
            return False
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics for this position."""
        return {
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'quantity': self.quantity,
            'position_value': self.position_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'risk_amount': self.risk_amount,
            'risk_reward_ratio': self.risk_reward_ratio,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'should_stop_out': self.should_stop_out(),
            'should_take_profit': self.should_take_profit(),
            'entry_time': self.entry_time,
            'last_updated': self.last_updated
        }


class PortfolioRisk:
    """Tracks portfolio-level risk metrics."""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, PositionRisk] = {}
        
        # Portfolio metrics
        self.total_exposure = 0.0
        self.net_exposure = 0.0
        self.gross_exposure = 0.0
        self.leverage = 0.0
        self.beta = 0.0
        
        # P&L tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.total_return = 0.0
        
        # Drawdown tracking
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_duration = 0
        self.underwater_periods = deque(maxlen=1000)
        
        # Risk metrics history
        self.capital_history = deque(maxlen=10000)
        self.drawdown_history = deque(maxlen=10000)
        self.exposure_history = deque(maxlen=1000)
        
        # Daily tracking
        self.daily_pnl = defaultdict(float)
        self.daily_returns = deque(maxlen=365)
        
        self.last_updated = datetime.now()
    
    def add_position(self, position: PositionRisk):
        """Add a new position to the portfolio."""
        try:
            self.positions[position.symbol] = position
            self._recalculate_metrics()
        except Exception as e:
            logging.getLogger(__name__).error(f"Error adding position {position.symbol}: {e}")
    
    def remove_position(self, symbol: str, realized_pnl: float = 0.0):
        """Remove a position from the portfolio."""
        try:
            if symbol in self.positions:
                del self.positions[symbol]
                self.realized_pnl += realized_pnl
                self.current_capital += realized_pnl
                
                # Track daily P&L
                today = datetime.now().date()
                self.daily_pnl[today] += realized_pnl
                
                self._recalculate_metrics()
        except Exception as e:
            logging.getLogger(__name__).error(f"Error removing position {symbol}: {e}")
    
    def update_positions(self, market_prices: Dict[str, float]):
        """Update all positions with current market prices."""
        try:
            for symbol, position in self.positions.items():
                if symbol in market_prices:
                    position.update_price(market_prices[symbol])
            
            self._recalculate_metrics()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error updating positions: {e}")
    
    def _recalculate_metrics(self):
        """Recalculate portfolio-level metrics."""
        try:
            # Calculate exposures
            long_exposure = sum(pos.position_value for pos in self.positions.values() if pos.quantity > 0)
            short_exposure = sum(pos.position_value for pos in self.positions.values() if pos.quantity < 0)
            
            self.gross_exposure = long_exposure + short_exposure
            self.net_exposure = long_exposure - short_exposure
            self.total_exposure = self.gross_exposure
            self.leverage = self.gross_exposure / self.current_capital if self.current_capital > 0 else 0
            
            # Calculate unrealized P&L
            self.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            self.total_pnl = self.realized_pnl + self.unrealized_pnl
            self.total_return = self.total_pnl / self.initial_capital if self.initial_capital > 0 else 0
            
            # Update capital
            portfolio_value = self.current_capital + self.unrealized_pnl
            
            # Update drawdown
            if portfolio_value > self.peak_capital:
                self.peak_capital = portfolio_value
                self.drawdown_duration = 0
            else:
                self.drawdown_duration += 1
            
            self.current_drawdown = (self.peak_capital - portfolio_value) / self.peak_capital if self.peak_capital > 0 else 0
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Update history
            now = datetime.now()
            self.capital_history.append((now, portfolio_value))
            self.drawdown_history.append((now, self.current_drawdown))
            self.exposure_history.append((now, self.total_exposure))
            
            if self.current_drawdown > 0:
                self.underwater_periods.append((now, self.current_drawdown))
            
            self.last_updated = now
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error recalculating portfolio metrics: {e}")
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio risk metrics."""
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'portfolio_value': self.current_capital + self.unrealized_pnl,
            'total_positions': len(self.positions),
            'gross_exposure': self.gross_exposure,
            'net_exposure': self.net_exposure,
            'leverage': self.leverage,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'total_return': self.total_return,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'drawdown_duration': self.drawdown_duration,
            'underwater_periods': len(self.underwater_periods),
            'last_updated': self.last_updated
        }


class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(self, config: RiskConfig):
        self.logger = setup_logging(__name__)
        self.config = config
        
        # Portfolio tracking
        self.portfolio = PortfolioRisk(config.initial_capital)
        
        # Risk limits
        self.max_position_size = config.max_position_size
        self.max_portfolio_risk = config.max_portfolio_risk
        self.max_daily_loss = config.max_daily_loss
        self.max_drawdown_limit = config.max_drawdown_limit
        self.max_leverage = config.max_leverage
        self.max_correlation_exposure = config.max_correlation_exposure
        
        # Position sizing parameters
        self.default_risk_per_trade = config.default_risk_per_trade
        self.position_sizing_method = config.position_sizing_method
        self.volatility_lookback = config.volatility_lookback
        
        # Emergency controls
        self.emergency_stop_active = False
        self.emergency_stop_reason = None
        self.emergency_stop_time = None
        
        # Risk monitoring
        self.risk_alerts = deque(maxlen=1000)
        self.risk_violations = deque(maxlen=100)
        self.daily_loss_tracking = defaultdict(float)
        
        # Market data for risk calculations
        self.market_data_cache: Dict[str, MarketData] = {}
        self.volatility_cache: Dict[str, float] = {}
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        
        # Performance tracking
        self.risk_metrics_history = deque(maxlen=10000)
        self.last_risk_check = datetime.now()
        
        self.logger.info("Risk Manager initialized with comprehensive controls")
    
    def validate_trade(self, signal: TradingSignal, market_data: MarketData) -> Tuple[bool, str, float]:
        """Validate a trading signal against risk parameters."""
        try:
            # Check emergency stop
            if self.emergency_stop_active:
                return False, f"Emergency stop active: {self.emergency_stop_reason}", 0.0
            
            # Update market data cache
            self.market_data_cache[signal.symbol] = market_data
            
            # Calculate position size
            position_size = self.calculate_position_size(signal, market_data)
            if position_size == 0:
                return False, "Position size calculated as zero", 0.0
            
            # Validate position size limits
            position_value = abs(position_size * market_data.price)
            if position_value > self.max_position_size:
                return False, f"Position size {position_value:.2f} exceeds limit {self.max_position_size:.2f}", 0.0
            
            # Check portfolio risk limits
            if not self._check_portfolio_risk_limits(signal, position_size, market_data):
                return False, "Portfolio risk limits exceeded", 0.0
            
            # Check daily loss limits
            if not self._check_daily_loss_limits():
                return False, "Daily loss limit exceeded", 0.0
            
            # Check drawdown limits
            if not self._check_drawdown_limits():
                return False, "Drawdown limit exceeded", 0.0
            
            # Check leverage limits
            if not self._check_leverage_limits(position_size, market_data.price):
                return False, "Leverage limit exceeded", 0.0
            
            # Check correlation limits
            if not self._check_correlation_limits(signal.symbol, position_size):
                return False, "Correlation exposure limit exceeded", 0.0
            
            # All checks passed
            return True, "Trade validated", position_size
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {str(e)}", 0.0
    
    def calculate_position_size(self, signal: TradingSignal, market_data: MarketData) -> float:
        """Calculate optimal position size based on risk parameters."""
        try:
            if self.position_sizing_method == "fixed_risk":
                return self._calculate_fixed_risk_size(signal, market_data)
            elif self.position_sizing_method == "volatility_adjusted":
                return self._calculate_volatility_adjusted_size(signal, market_data)
            elif self.position_sizing_method == "kelly_criterion":
                return self._calculate_kelly_size(signal, market_data)
            else:
                return self._calculate_fixed_risk_size(signal, market_data)
                
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _calculate_fixed_risk_size(self, signal: TradingSignal, market_data: MarketData) -> float:
        """Calculate position size based on fixed risk percentage."""
        try:
            # Risk amount based on portfolio
            risk_amount = self.portfolio.current_capital * self.default_risk_per_trade
            
            # Estimate stop loss if not provided
            stop_loss = self._estimate_stop_loss(signal, market_data)
            if not stop_loss:
                return 0.0
            
            # Calculate risk per share
            if signal.action == SignalAction.BUY:
                risk_per_share = market_data.price - stop_loss
            else:
                risk_per_share = stop_loss - market_data.price
            
            if risk_per_share <= 0:
                return 0.0
            
            # Calculate position size
            position_size = risk_amount / risk_per_share
            
            # Apply confidence adjustment
            confidence_adjustment = signal.confidence
            adjusted_size = position_size * confidence_adjustment
            
            return adjusted_size
            
        except Exception as e:
            self.logger.error(f"Error calculating fixed risk size: {e}")
            return 0.0
    
    def _calculate_volatility_adjusted_size(self, signal: TradingSignal, market_data: MarketData) -> float:
        """Calculate position size adjusted for volatility."""
        try:
            # Get base size from fixed risk
            base_size = self._calculate_fixed_risk_size(signal, market_data)
            
            # Get volatility
            volatility = self._get_volatility(signal.symbol, market_data)
            if volatility == 0:
                return base_size
            
            # Adjust for volatility (inverse relationship)
            avg_volatility = 0.02  # 2% average daily volatility
            volatility_adjustment = avg_volatility / volatility
            
            # Cap adjustment to reasonable range
            volatility_adjustment = max(0.5, min(2.0, volatility_adjustment))
            
            return base_size * volatility_adjustment
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility adjusted size: {e}")
            return 0.0
    
    def _calculate_kelly_size(self, signal: TradingSignal, market_data: MarketData) -> float:
        """Calculate position size using Kelly Criterion."""
        try:
            # Simplified Kelly calculation
            # Would need historical win rate and average win/loss for full implementation
            
            # Use signal confidence as win probability estimate
            win_probability = signal.confidence
            loss_probability = 1 - win_probability
            
            # Estimate win/loss ratio from stop loss and take profit
            stop_loss = self._estimate_stop_loss(signal, market_data)
            take_profit = self._estimate_take_profit(signal, market_data)
            
            if not stop_loss or not take_profit:
                return self._calculate_fixed_risk_size(signal, market_data)
            
            if signal.action == SignalAction.BUY:
                avg_win = take_profit - market_data.price
                avg_loss = market_data.price - stop_loss
            else:
                avg_win = market_data.price - take_profit
                avg_loss = stop_loss - market_data.price
            
            if avg_loss <= 0:
                return 0.0
            
            win_loss_ratio = avg_win / avg_loss
            
            # Kelly fraction
            kelly_fraction = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
            
            # Cap Kelly fraction to reasonable range
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Max 25% of capital
            
            # Calculate position size
            risk_amount = self.portfolio.current_capital * kelly_fraction
            position_size = risk_amount / avg_loss if avg_loss > 0 else 0
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly size: {e}")
            return self._calculate_fixed_risk_size(signal, market_data)    
    
    def _estimate_stop_loss(self, signal: TradingSignal, market_data: MarketData) -> Optional[float]:
        """Estimate stop loss level if not provided."""
        try:
            # Check if stop loss is in signal metadata
            if 'stop_loss' in signal.metadata and signal.metadata['stop_loss']:
                return signal.metadata['stop_loss']
            
            # Use volatility-based stop loss
            volatility = self._get_volatility(signal.symbol, market_data)
            if volatility == 0:
                volatility = 0.02  # Default 2% volatility
            
            # 2x volatility stop loss
            stop_distance = market_data.price * volatility * 2
            
            if signal.action == SignalAction.BUY:
                return market_data.price - stop_distance
            else:
                return market_data.price + stop_distance
                
        except Exception as e:
            self.logger.error(f"Error estimating stop loss: {e}")
            return None
    
def _estimate_take_profit(self, signal: TradingSignal, market_data: MarketData) -> Optional[float]:
        """Estimate take profit level if not provided."""
        try:
            # Check if take profit is in signal metadata
            if 'target_price' in signal.metadata and signal.metadata['target_price']:
                return signal.metadata['target_price']
            
            # Use volatility-based take profit (3x stop distance)
            stop_loss = self._estimate_stop_loss(signal, market_data)
            if not stop_loss:
                return None
            
            if signal.action == SignalAction.BUY:
                stop_distance = market_data.price - stop_loss
                return market_data.price + (stop_distance * 3)
            else:
                stop_distance = stop_loss - market_data.price
                return market_data.price - (stop_distance * 3)
                
        except Exception as e:
            self.logger.error(f"Error estimating take profit: {e}")
            return None
    
def _get_volatility(self, symbol: str, market_data: MarketData) -> float:
        """Get volatility for a symbol."""
        try:
            # Use cached volatility if available
            if symbol in self.volatility_cache:
                return self.volatility_cache[symbol]
            
            # Calculate simple volatility estimate
            # In production, this would use historical price data
            if hasattr(market_data, 'volatility') and market_data.volatility:
                volatility = market_data.volatility
            else:
                # Estimate from 24h high/low
                if market_data.high_24h and market_data.low_24h:
                    daily_range = (market_data.high_24h - market_data.low_24h) / market_data.price
                    volatility = daily_range / 4  # Rough approximation
                else:
                    volatility = 0.02  # Default 2%
            
            self.volatility_cache[symbol] = volatility
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error getting volatility for {symbol}: {e}")
            return 0.02
    
def _check_portfolio_risk_limits(self, signal: TradingSignal, position_size: float, 
                                   market_data: MarketData) -> bool:
        """Check if trade would exceed portfolio risk limits."""
        try:
            # Calculate new position risk
            position_value = abs(position_size * market_data.price)
            stop_loss = self._estimate_stop_loss(signal, market_data)
            
            if stop_loss:
                if signal.action == SignalAction.BUY:
                    position_risk = (market_data.price - stop_loss) * position_size
                else:
                    position_risk = (stop_loss - market_data.price) * abs(position_size)
            else:
                position_risk = position_value * 0.02  # Default 2% risk
            
            # Check total portfolio risk
            current_portfolio_risk = sum(pos.risk_amount for pos in self.portfolio.positions.values())
            total_risk = current_portfolio_risk + position_risk
            max_risk = self.portfolio.current_capital * self.max_portfolio_risk
            
            if total_risk > max_risk:
                self._create_risk_alert(
                    RiskLevel.HIGH,
                    "portfolio_risk",
                    f"Portfolio risk {total_risk:.2f} would exceed limit {max_risk:.2f}",
                    {'current_risk': current_portfolio_risk, 'new_risk': position_risk, 'limit': max_risk}
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio risk limits: {e}")
            return False
    
def _check_daily_loss_limits(self) -> bool:
        """Check daily loss limits."""
        try:
            today = datetime.now().date()
            daily_loss = self.daily_loss_tracking[today]
            
            # Add unrealized losses for today's positions
            for position in self.portfolio.positions.values():
                if position.entry_time.date() == today and position.unrealized_pnl < 0:
                    daily_loss += abs(position.unrealized_pnl)
            
            max_daily_loss = self.portfolio.current_capital * self.max_daily_loss
            
            if daily_loss > max_daily_loss:
                self._create_risk_alert(
                    RiskLevel.CRITICAL,
                    "daily_loss",
                    f"Daily loss {daily_loss:.2f} exceeds limit {max_daily_loss:.2f}",
                    {'daily_loss': daily_loss, 'limit': max_daily_loss}
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking daily loss limits: {e}")
            return True
    
def _check_drawdown_limits(self) -> bool:
        """Check drawdown limits."""
        try:
            if self.portfolio.current_drawdown > self.max_drawdown_limit:
                self._create_risk_alert(
                    RiskLevel.CRITICAL,
                    "drawdown",
                    f"Drawdown {self.portfolio.current_drawdown:.2%} exceeds limit {self.max_drawdown_limit:.2%}",
                    {'current_drawdown': self.portfolio.current_drawdown, 'limit': self.max_drawdown_limit}
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking drawdown limits: {e}")
            return True
    
def _check_leverage_limits(self, position_size: float, price: float) -> bool:
        """Check leverage limits."""
        try:
            new_position_value = abs(position_size * price)
            new_gross_exposure = self.portfolio.gross_exposure + new_position_value
            new_leverage = new_gross_exposure / self.portfolio.current_capital
            
            if new_leverage > self.max_leverage:
                self._create_risk_alert(
                    RiskLevel.HIGH,
                    "leverage",
                    f"Leverage {new_leverage:.2f} would exceed limit {self.max_leverage:.2f}",
                    {'current_leverage': self.portfolio.leverage, 'new_leverage': new_leverage, 'limit': self.max_leverage}
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking leverage limits: {e}")
            return True
    
def _check_correlation_limits(self, symbol: str, position_size: float) -> bool:
        """Check correlation exposure limits."""
        try:
            # Simplified correlation check
            # In production, this would use actual correlation matrix
            
            # Check if we already have exposure to this symbol
            if symbol in self.portfolio.positions:
                existing_position = self.portfolio.positions[symbol]
                # Check if new position is in same direction
                if (existing_position.quantity > 0 and position_size > 0) or \
                   (existing_position.quantity < 0 and position_size < 0):
                    
                    total_exposure = abs(existing_position.position_value) + abs(position_size * self.market_data_cache[symbol].price)
                    max_single_exposure = self.portfolio.current_capital * self.max_correlation_exposure
                    
                    if total_exposure > max_single_exposure:
                        self._create_risk_alert(
                            RiskLevel.MEDIUM,
                            "correlation",
                            f"Single symbol exposure {total_exposure:.2f} would exceed limit {max_single_exposure:.2f}",
                            {'symbol': symbol, 'exposure': total_exposure, 'limit': max_single_exposure}
                        )
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking correlation limits: {e}")
            return True
    
def _create_risk_alert(self, level: RiskLevel, category: str, message: str, data: Dict[str, Any]):
        """Create a risk alert."""
        try:
            alert = RiskAlert(
                timestamp=datetime.now(),
                level=level,
                category=category,
                message=message,
                data=data
            )
            
            self.risk_alerts.append(alert)
            
            # Log based on severity
            if level == RiskLevel.CRITICAL:
                self.logger.critical(f"CRITICAL RISK ALERT [{category}]: {message}")
            elif level == RiskLevel.HIGH:
                self.logger.error(f"HIGH RISK ALERT [{category}]: {message}")
            elif level == RiskLevel.MEDIUM:
                self.logger.warning(f"MEDIUM RISK ALERT [{category}]: {message}")
            else:
                self.logger.info(f"LOW RISK ALERT [{category}]: {message}")
            
            # Check if emergency stop should be triggered
            if level == RiskLevel.CRITICAL:
                self._check_emergency_stop_trigger(alert)
                
        except Exception as e:
            self.logger.error(f"Error creating risk alert: {e}")
    
def _check_emergency_stop_trigger(self, alert: RiskAlert):
        """Check if emergency stop should be triggered."""
        try:
            # Trigger emergency stop for critical alerts
            if alert.level == RiskLevel.CRITICAL and not self.emergency_stop_active:
                self.activate_emergency_stop(f"Critical risk alert: {alert.message}")
                
        except Exception as e:
            self.logger.error(f"Error checking emergency stop trigger: {e}")
    
def activate_emergency_stop(self, reason: str):
        """Activate emergency stop."""
        try:
            self.emergency_stop_active = True
            self.emergency_stop_reason = reason
            self.emergency_stop_time = datetime.now()
            
            self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
            
            # Create critical alert
            self._create_risk_alert(
                RiskLevel.CRITICAL,
                "emergency_stop",
                f"Emergency stop activated: {reason}",
                {'activation_time': self.emergency_stop_time}
            )
            
        except Exception as e:
            self.logger.error(f"Error activating emergency stop: {e}")
    
    
def deactivate_emergency_stop(self, reason: str = "Manual override"):
        """Deactivate emergency stop."""
        try:
            if self.emergency_stop_active:
                self.emergency_stop_active = False
                deactivation_time = datetime.now()
                duration = deactivation_time - self.emergency_stop_time if self.emergency_stop_time else timedelta(0)
                
                self.logger.warning(f"Emergency stop deactivated: {reason} (Duration: {duration})")
                
                # Create alert
                self._create_risk_alert(
                    RiskLevel.MEDIUM,
                    "emergency_stop",
                    f"Emergency stop deactivated: {reason}",
                    {
                        'deactivation_time': deactivation_time,
                        'duration': duration.total_seconds(),
                        'original_reason': self.emergency_stop_reason
                    }
                )
                
                self.emergency_stop_reason = None
                self.emergency_stop_time = None
                
        except Exception as e:
            self.logger.error(f"Error deactivating emergency stop: {e}")
    
def add_position(self, signal: TradingSignal, position_size: float, market_data: MarketData):
        """Add a new position to risk tracking."""
        try:
            stop_loss = self._estimate_stop_loss(signal, market_data)
            take_profit = self._estimate_take_profit(signal, market_data)
            
            position = PositionRisk(
                symbol=signal.symbol,
                entry_price=market_data.price,
                quantity=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            self.portfolio.add_position(position)
            
            self.logger.info(f"Added position: {signal.symbol} {position_size:.4f} @ {market_data.price:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
    
def remove_position(self, symbol: str, exit_price: float, realized_pnl: float):
        """Remove a position from risk tracking."""
        try:
            self.portfolio.remove_position(symbol, realized_pnl)
            
            # Track daily loss
            if realized_pnl < 0:
                today = datetime.now().date()
                self.daily_loss_tracking[today] += abs(realized_pnl)
            
            self.logger.info(f"Removed position: {symbol} @ {exit_price:.4f}, P&L: {realized_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error removing position: {e}")
    
def update_market_data(self, market_data: Dict[str, MarketData]):
        """Update market data for all positions."""
        try:
            # Update market data cache
            self.market_data_cache.update(market_data)
            
            # Extract prices for portfolio update
            market_prices = {symbol: data.price for symbol, data in market_data.items()}
            
            # Update portfolio positions
            self.portfolio.update_positions(market_prices)
            
            # Check for stop losses and take profits
            self._check_position_exits()
            
            # Update risk metrics
            self._update_risk_metrics()
            
            self.last_risk_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
def _check_position_exits(self):
        """Check positions for stop loss or take profit triggers."""
        try:
            positions_to_exit = []
            
            for symbol, position in self.portfolio.positions.items():
                if position.should_stop_out():
                    positions_to_exit.append((symbol, 'stop_loss', position.stop_loss))
                elif position.should_take_profit():
                    positions_to_exit.append((symbol, 'take_profit', position.take_profit))
            
            # Log exit recommendations
            for symbol, exit_type, exit_price in positions_to_exit:
                self.logger.warning(f"Position {symbol} should exit ({exit_type}) at {exit_price:.4f}")
                
                # Create alert
                self._create_risk_alert(
                    RiskLevel.MEDIUM,
                    "position_exit",
                    f"Position {symbol} triggered {exit_type} at {exit_price:.4f}",
                    {'symbol': symbol, 'exit_type': exit_type, 'exit_price': exit_price}
                )
            
        except Exception as e:
            self.logger.error(f"Error checking position exits: {e}")
    
def _update_risk_metrics(self):
        """Update comprehensive risk metrics."""
        try:
            # Calculate portfolio-level metrics
            portfolio_metrics = self.portfolio.get_portfolio_metrics()
            
            # Add risk-specific metrics
            risk_metrics = {
                'timestamp': datetime.now(),
                'portfolio_metrics': portfolio_metrics,
                'risk_limits': {
                    'max_position_size': self.max_position_size,
                    'max_portfolio_risk': self.max_portfolio_risk,
                    'max_daily_loss': self.max_daily_loss,
                    'max_drawdown_limit': self.max_drawdown_limit,
                    'max_leverage': self.max_leverage
                },
                'current_utilization': {
                    'leverage_utilization': portfolio_metrics['leverage'] / self.max_leverage if self.max_leverage > 0 else 0,
                    'drawdown_utilization': portfolio_metrics['current_drawdown'] / self.max_drawdown_limit if self.max_drawdown_limit > 0 else 0,
                    'daily_loss_utilization': self._get_daily_loss_utilization()
                },
                'emergency_stop': {
                    'active': self.emergency_stop_active,
                    'reason': self.emergency_stop_reason,
                    'time': self.emergency_stop_time
                },
                'recent_alerts': len([a for a in self.risk_alerts if datetime.now() - a.timestamp < timedelta(hours=1)])
            }
            
            self.risk_metrics_history.append(risk_metrics)
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")
    
def _get_daily_loss_utilization(self) -> float:
        """Get daily loss utilization percentage."""
        try:
            today = datetime.now().date()
            daily_loss = self.daily_loss_tracking[today]
            max_daily_loss = self.portfolio.current_capital * self.max_daily_loss
            
            return daily_loss / max_daily_loss if max_daily_loss > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating daily loss utilization: {e}")
            return 0.0
    
def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        try:
            portfolio_metrics = self.portfolio.get_portfolio_metrics()
            
            # Recent alerts by level
            recent_alerts = [a for a in self.risk_alerts if datetime.now() - a.timestamp < timedelta(hours=24)]
            alerts_by_level = defaultdict(int)
            for alert in recent_alerts:
                alerts_by_level[alert.level.value] += 1
            
            return {
                'portfolio': portfolio_metrics,
                'risk_limits': {
                    'max_position_size': self.max_position_size,
                    'max_portfolio_risk': self.max_portfolio_risk,
                    'max_daily_loss': self.max_daily_loss,
                    'max_drawdown_limit': self.max_drawdown_limit,
                    'max_leverage': self.max_leverage,
                    'max_correlation_exposure': self.max_correlation_exposure
                },
                'current_utilization': {
                    'leverage': f"{(portfolio_metrics['leverage'] / self.max_leverage * 100):.1f}%" if self.max_leverage > 0 else "N/A",
                    'drawdown': f"{(portfolio_metrics['current_drawdown'] / self.max_drawdown_limit * 100):.1f}%" if self.max_drawdown_limit > 0 else "N/A",
                    'daily_loss': f"{(self._get_daily_loss_utilization() * 100):.1f}%"
                },
                'emergency_stop': {
                    'active': self.emergency_stop_active,
                    'reason': self.emergency_stop_reason,
                    'duration': (datetime.now() - self.emergency_stop_time).total_seconds() if self.emergency_stop_time else 0
                },
                'alerts_24h': dict(alerts_by_level),
                'total_alerts': len(self.risk_alerts),
                'positions': {
                    symbol: pos.get_risk_metrics() 
                    for symbol, pos in self.portfolio.positions.items()
                },
                'last_updated': self.last_risk_check
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {}
    
def get_position_recommendations(self) -> List[Dict[str, Any]]:
        """Get position management recommendations."""
        try:
            recommendations = []
            
            for symbol, position in self.portfolio.positions.items():
                # Check for stop loss triggers
                if position.should_stop_out():
                    recommendations.append({
                        'symbol': symbol,
                        'action': 'close',
                        'reason': 'stop_loss_triggered',
                        'urgency': 'high',
                        'current_price': position.current_price,
                        'trigger_price': position.stop_loss,
                        'unrealized_pnl': position.unrealized_pnl
                    })
                
                # Check for take profit triggers
                elif position.should_take_profit():
                    recommendations.append({
                        'symbol': symbol,
                        'action': 'close',
                        'reason': 'take_profit_triggered',
                        'urgency': 'medium',
                        'current_price': position.current_price,
                        'trigger_price': position.take_profit,
                        'unrealized_pnl': position.unrealized_pnl
                    })
                
                # Check for large adverse moves
                elif position.unrealized_pnl_pct < -0.05:  # 5% loss
                    recommendations.append({
                        'symbol': symbol,
                        'action': 'review',
                        'reason': 'large_adverse_move',
                        'urgency': 'medium',
                        'current_price': position.current_price,
                        'unrealized_pnl_pct': position.unrealized_pnl_pct,
                        'max_adverse_excursion': position.max_adverse_excursion
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting position recommendations: {e}")
            return []
    
def shutdown(self):
        """Shutdown risk manager and cleanup resources."""
        try:
            self.logger.info("Shutting down Risk Manager...")
            
            # Log final risk summary
            final_summary = self.get_risk_summary()
            self.logger.info(f"Final risk summary: {final_summary}")
            
            # Clear caches
            self.market_data_cache.clear()
            self.volatility_cache.clear()
            self.correlation_matrix.clear()
            
            self.logger.info("Risk Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during risk manager shutdown: {e}")
    
def __enter__(self):
        """Context manager entry."""
        return self
    
def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()