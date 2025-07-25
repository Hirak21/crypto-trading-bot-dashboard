"""
Performance analysis tools for backtesting results.

This module provides comprehensive performance analysis and metrics
calculation for trading strategy backtests.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd
from scipy import stats
import json

from .backtest_engine import BacktestResult
from ..models.trading import Trade, Position


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a trading strategy."""
    
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    compound_annual_growth_rate: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # Days
    value_at_risk_95: float = 0.0
    conditional_value_at_risk_95: float = 0.0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Trading statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trade analysis
    avg_trade_return: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    largest_winning_trade: float = 0.0
    largest_losing_trade: float = 0.0
    
    # Consistency metrics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_duration: float = 0.0  # Hours
    
    # Monthly/yearly breakdown
    monthly_returns: List[float] = field(default_factory=list)
    yearly_returns: List[float] = field(default_factory=list)
    best_month: float = 0.0
    worst_month: float = 0.0
    positive_months: int = 0
    negative_months: int = 0
    
    # Benchmark comparison
    benchmark_correlation: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    tracking_error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'returns': {
                'total_return': self.total_return,
                'annualized_return': self.annualized_return,
                'compound_annual_growth_rate': self.compound_annual_growth_rate
            },
            'risk': {
                'volatility': self.volatility,
                'max_drawdown': self.max_drawdown,
                'max_drawdown_duration': self.max_drawdown_duration,
                'value_at_risk_95': self.value_at_risk_95,
                'conditional_value_at_risk_95': self.conditional_value_at_risk_95
            },
            'risk_adjusted': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
                'information_ratio': self.information_ratio
            },
            'trading': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor
            },
            'trade_analysis': {
                'avg_trade_return': self.avg_trade_return,
                'avg_winning_trade': self.avg_winning_trade,
                'avg_losing_trade': self.avg_losing_trade,
                'largest_winning_trade': self.largest_winning_trade,
                'largest_losing_trade': self.largest_losing_trade
            },
            'consistency': {
                'max_consecutive_wins': self.max_consecutive_wins,
                'max_consecutive_losses': self.max_consecutive_losses,
                'avg_trade_duration': self.avg_trade_duration
            },
            'monthly_analysis': {
                'monthly_returns': self.monthly_returns,
                'yearly_returns': self.yearly_returns,
                'best_month': self.best_month,
                'worst_month': self.worst_month,
                'positive_months': self.positive_months,
                'negative_months': self.negative_months
            },
            'benchmark': {
                'correlation': self.benchmark_correlation,
                'alpha': self.alpha,
                'beta': self.beta,
                'tracking_error': self.tracking_error
            }
        }


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for trading strategies.
    
    Calculates detailed performance metrics, risk measures, and
    provides analysis tools for backtesting results.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize performance analyzer."""
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def analyze_backtest_result(self, backtest_result: BacktestResult,
                              benchmark_returns: Optional[List[float]] = None) -> PerformanceMetrics:
        """
        Analyze backtest result and calculate comprehensive performance metrics.
        
        Args:
            backtest_result: Backtest result to analyze
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            PerformanceMetrics with detailed analysis
        """
        metrics = PerformanceMetrics()
        
        try:
            # Basic validation
            if not backtest_result.portfolio_values:
                self.logger.warning("No portfolio values found in backtest result")
                return metrics
            
            # Extract data
            portfolio_values = backtest_result.portfolio_values
            trades = backtest_result.trades
            
            # Calculate return metrics
            self._calculate_return_metrics(metrics, portfolio_values, backtest_result.config)
            
            # Calculate risk metrics
            self._calculate_risk_metrics(metrics, portfolio_values)
            
            # Calculate risk-adjusted returns
            self._calculate_risk_adjusted_returns(metrics, portfolio_values)
            
            # Analyze trades
            self._analyze_trades(metrics, trades)
            
            # Calculate monthly/yearly breakdown
            self._calculate_periodic_returns(metrics, portfolio_values)
            
            # Benchmark comparison if provided
            if benchmark_returns:
                self._calculate_benchmark_metrics(metrics, portfolio_values, benchmark_returns)
            
            self.logger.info("Performance analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in performance analysis: {e}")
        
        return metrics
    
    def _calculate_return_metrics(self, metrics: PerformanceMetrics, 
                                portfolio_values: List[Tuple[datetime, float]],
                                backtest_config) -> None:
        """Calculate return-based metrics."""
        if len(portfolio_values) < 2:
            return
        
        initial_value = portfolio_values[0][1]
        final_value = portfolio_values[-1][1]
        
        # Total return
        metrics.total_return = (final_value - initial_value) / initial_value
        
        # Annualized return
        start_date = portfolio_values[0][0]
        end_date = portfolio_values[-1][0]
        days = (end_date - start_date).days
        
        if days > 0:
            years = days / 365.25
            metrics.annualized_return = ((final_value / initial_value) ** (1 / years)) - 1
            metrics.compound_annual_growth_rate = metrics.annualized_return
    
    def _calculate_risk_metrics(self, metrics: PerformanceMetrics,
                              portfolio_values: List[Tuple[datetime, float]]) -> None:
        """Calculate risk-based metrics."""
        if len(portfolio_values) < 2:
            return
        
        # Convert to returns series
        values = [pv[1] for pv in portfolio_values]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        if not returns:
            return
        
        # Volatility (annualized)
        daily_vol = np.std(returns)
        metrics.volatility = daily_vol * np.sqrt(252)  # Assuming daily data
        
        # Maximum drawdown
        peak = values[0]
        max_dd = 0.0
        dd_duration = 0
        max_dd_duration = 0
        current_dd_start = None
        
        for i, value in enumerate(values):
            if value > peak:
                peak = value
                if current_dd_start is not None:
                    # End of drawdown period
                    max_dd_duration = max(max_dd_duration, dd_duration)
                    current_dd_start = None
                    dd_duration = 0
            else:
                if current_dd_start is None:
                    current_dd_start = i
                dd_duration += 1
                
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        # Handle case where drawdown continues to the end
        if current_dd_start is not None:
            max_dd_duration = max(max_dd_duration, dd_duration)
        
        metrics.max_drawdown = max_dd
        metrics.max_drawdown_duration = max_dd_duration
        
        # Value at Risk (VaR) and Conditional VaR
        if len(returns) > 20:  # Need sufficient data
            sorted_returns = sorted(returns)
            var_index = int(len(sorted_returns) * 0.05)  # 5% VaR
            metrics.value_at_risk_95 = sorted_returns[var_index]
            
            # Conditional VaR (Expected Shortfall)
            tail_returns = sorted_returns[:var_index]
            if tail_returns:
                metrics.conditional_value_at_risk_95 = np.mean(tail_returns)
    
    def _calculate_risk_adjusted_returns(self, metrics: PerformanceMetrics,
                                       portfolio_values: List[Tuple[datetime, float]]) -> None:
        """Calculate risk-adjusted return metrics."""
        if len(portfolio_values) < 2:
            return
        
        # Convert to returns series
        values = [pv[1] for pv in portfolio_values]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        if not returns:
            return
        
        avg_return = np.mean(returns)
        return_std = np.std(returns)
        
        # Sharpe ratio
        if return_std > 0:
            daily_rf_rate = self.risk_free_rate / 252
            excess_return = avg_return - daily_rf_rate
            metrics.sharpe_ratio = (excess_return * 252) / (return_std * np.sqrt(252))
        
        # Sortino ratio (using downside deviation)
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                daily_rf_rate = self.risk_free_rate / 252
                excess_return = avg_return - daily_rf_rate
                metrics.sortino_ratio = (excess_return * 252) / (downside_std * np.sqrt(252))
        
        # Calmar ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
    
    def _analyze_trades(self, metrics: PerformanceMetrics, trades: List[Trade]) -> None:
        """Analyze individual trades."""
        if not trades:
            return
        
        # Filter completed trades (with PnL)
        completed_trades = [t for t in trades if t.pnl is not None]
        
        if not completed_trades:
            return
        
        metrics.total_trades = len(completed_trades)
        
        # Separate winning and losing trades
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        
        # Win rate
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        # Trade returns
        trade_pnls = [t.pnl for t in completed_trades]
        metrics.avg_trade_return = np.mean(trade_pnls)
        
        if winning_trades:
            winning_pnls = [t.pnl for t in winning_trades]
            metrics.avg_winning_trade = np.mean(winning_pnls)
            metrics.largest_winning_trade = max(winning_pnls)
        
        if losing_trades:
            losing_pnls = [t.pnl for t in losing_trades]
            metrics.avg_losing_trade = np.mean(losing_pnls)
            metrics.largest_losing_trade = min(losing_pnls)
        
        # Profit factor
        total_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        total_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        
        if total_loss > 0:
            metrics.profit_factor = total_profit / total_loss
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_wins = 0
        max_losses = 0
        
        for trade in completed_trades:
            if trade.pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_wins = max(max_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_losses = max(max_losses, consecutive_losses)
        
        metrics.max_consecutive_wins = max_wins
        metrics.max_consecutive_losses = max_losses
        
        # Average trade duration (simplified - would need entry/exit timestamps)
        # For now, we'll estimate based on trade frequency
        if len(completed_trades) > 1:
            first_trade_time = completed_trades[0].timestamp
            last_trade_time = completed_trades[-1].timestamp
            total_duration = (last_trade_time - first_trade_time).total_seconds() / 3600  # Hours
            metrics.avg_trade_duration = total_duration / len(completed_trades)
    
    def _calculate_periodic_returns(self, metrics: PerformanceMetrics,
                                  portfolio_values: List[Tuple[datetime, float]]) -> None:
        """Calculate monthly and yearly returns."""
        if len(portfolio_values) < 2:
            return
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(portfolio_values, columns=['timestamp', 'value'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate daily returns
        df['returns'] = df['value'].pct_change()
        
        # Monthly returns
        monthly_returns = df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        metrics.monthly_returns = monthly_returns.dropna().tolist()
        
        if metrics.monthly_returns:
            metrics.best_month = max(metrics.monthly_returns)
            metrics.worst_month = min(metrics.monthly_returns)
            metrics.positive_months = sum(1 for r in metrics.monthly_returns if r > 0)
            metrics.negative_months = sum(1 for r in metrics.monthly_returns if r < 0)
        
        # Yearly returns
        yearly_returns = df['returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
        metrics.yearly_returns = yearly_returns.dropna().tolist()
    
    def _calculate_benchmark_metrics(self, metrics: PerformanceMetrics,
                                   portfolio_values: List[Tuple[datetime, float]],
                                   benchmark_returns: List[float]) -> None:
        """Calculate benchmark comparison metrics."""
        if len(portfolio_values) < 2 or not benchmark_returns:
            return
        
        # Convert portfolio to returns
        values = [pv[1] for pv in portfolio_values]
        portfolio_returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        # Align lengths
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]
        
        if min_length < 2:
            return
        
        # Correlation
        correlation, _ = stats.pearsonr(portfolio_returns, benchmark_returns)
        metrics.benchmark_correlation = correlation
        
        # Beta (portfolio sensitivity to benchmark)
        portfolio_var = np.var(portfolio_returns)
        benchmark_var = np.var(benchmark_returns)
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        
        if benchmark_var > 0:
            metrics.beta = covariance / benchmark_var
        
        # Alpha (excess return over benchmark)
        avg_portfolio_return = np.mean(portfolio_returns)
        avg_benchmark_return = np.mean(benchmark_returns)
        
        if metrics.beta is not None:
            expected_return = self.risk_free_rate / 252 + metrics.beta * (avg_benchmark_return - self.risk_free_rate / 252)
            metrics.alpha = (avg_portfolio_return - expected_return) * 252  # Annualized
        
        # Tracking error
        excess_returns = [portfolio_returns[i] - benchmark_returns[i] for i in range(min_length)]
        metrics.tracking_error = np.std(excess_returns) * np.sqrt(252)  # Annualized
        
        # Information ratio
        if metrics.tracking_error > 0:
            metrics.information_ratio = (np.mean(excess_returns) * 252) / metrics.tracking_error
    
    def generate_performance_report(self, metrics: PerformanceMetrics,
                                  strategy_name: str = "Strategy") -> str:
        """Generate a formatted performance report."""
        report = f"""
PERFORMANCE REPORT - {strategy_name}
{'=' * 50}

RETURN METRICS
--------------
Total Return:              {metrics.total_return:.2%}
Annualized Return:         {metrics.annualized_return:.2%}
CAGR:                     {metrics.compound_annual_growth_rate:.2%}

RISK METRICS
------------
Volatility (Annualized):   {metrics.volatility:.2%}
Maximum Drawdown:          {metrics.max_drawdown:.2%}
Max Drawdown Duration:     {metrics.max_drawdown_duration} periods
Value at Risk (95%):       {metrics.value_at_risk_95:.2%}
Conditional VaR (95%):     {metrics.conditional_value_at_risk_95:.2%}

RISK-ADJUSTED RETURNS
--------------------
Sharpe Ratio:              {metrics.sharpe_ratio:.3f}
Sortino Ratio:             {metrics.sortino_ratio:.3f}
Calmar Ratio:              {metrics.calmar_ratio:.3f}
Information Ratio:         {metrics.information_ratio:.3f}

TRADING STATISTICS
-----------------
Total Trades:              {metrics.total_trades}
Winning Trades:            {metrics.winning_trades}
Losing Trades:             {metrics.losing_trades}
Win Rate:                  {metrics.win_rate:.2%}
Profit Factor:             {metrics.profit_factor:.3f}

TRADE ANALYSIS
--------------
Average Trade Return:      {metrics.avg_trade_return:.2f}
Average Winning Trade:     {metrics.avg_winning_trade:.2f}
Average Losing Trade:      {metrics.avg_losing_trade:.2f}
Largest Winning Trade:     {metrics.largest_winning_trade:.2f}
Largest Losing Trade:      {metrics.largest_losing_trade:.2f}

CONSISTENCY METRICS
------------------
Max Consecutive Wins:      {metrics.max_consecutive_wins}
Max Consecutive Losses:    {metrics.max_consecutive_losses}
Avg Trade Duration:        {metrics.avg_trade_duration:.1f} hours

MONTHLY ANALYSIS
---------------
Best Month:                {metrics.best_month:.2%}
Worst Month:               {metrics.worst_month:.2%}
Positive Months:           {metrics.positive_months}
Negative Months:           {metrics.negative_months}
"""
        
        if metrics.benchmark_correlation is not None:
            report += f"""
BENCHMARK COMPARISON
-------------------
Correlation:               {metrics.benchmark_correlation:.3f}
Alpha:                     {metrics.alpha:.2%}
Beta:                      {metrics.beta:.3f}
Tracking Error:            {metrics.tracking_error:.2%}
"""
        
        return report
    
    def save_performance_report(self, metrics: PerformanceMetrics, file_path: str,
                              strategy_name: str = "Strategy") -> bool:
        """Save performance report to file."""
        try:
            report_data = {
                'strategy_name': strategy_name,
                'generated_at': datetime.now().isoformat(),
                'metrics': metrics.to_dict(),
                'formatted_report': self.generate_performance_report(metrics, strategy_name)
            }
            
            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Performance report saved to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save performance report: {e}")
            return False
    
    def compare_strategies(self, metrics_list: List[Tuple[str, PerformanceMetrics]]) -> Dict[str, Any]:
        """Compare multiple strategies and rank them."""
        if not metrics_list:
            return {}
        
        comparison = {
            'strategies': [],
            'rankings': {},
            'summary': {}
        }
        
        # Extract metrics for comparison
        for name, metrics in metrics_list:
            strategy_data = {
                'name': name,
                'total_return': metrics.total_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'total_trades': metrics.total_trades
            }
            comparison['strategies'].append(strategy_data)
        
        # Rank strategies by different metrics
        strategies = comparison['strategies']
        
        comparison['rankings'] = {
            'by_total_return': sorted(strategies, key=lambda x: x['total_return'], reverse=True),
            'by_sharpe_ratio': sorted(strategies, key=lambda x: x['sharpe_ratio'], reverse=True),
            'by_min_drawdown': sorted(strategies, key=lambda x: x['max_drawdown']),
            'by_win_rate': sorted(strategies, key=lambda x: x['win_rate'], reverse=True),
            'by_profit_factor': sorted(strategies, key=lambda x: x['profit_factor'], reverse=True)
        }
        
        # Summary statistics
        returns = [s['total_return'] for s in strategies]
        sharpe_ratios = [s['sharpe_ratio'] for s in strategies]
        drawdowns = [s['max_drawdown'] for s in strategies]
        
        comparison['summary'] = {
            'best_return': max(returns),
            'worst_return': min(returns),
            'avg_return': np.mean(returns),
            'best_sharpe': max(sharpe_ratios),
            'worst_sharpe': min(sharpe_ratios),
            'avg_sharpe': np.mean(sharpe_ratios),
            'min_drawdown': min(drawdowns),
            'max_drawdown': max(drawdowns),
            'avg_drawdown': np.mean(drawdowns)
        }
        
        return comparison