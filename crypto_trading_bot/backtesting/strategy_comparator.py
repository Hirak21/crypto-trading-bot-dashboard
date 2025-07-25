"""
Strategy comparison tools for backtesting framework.

This module provides tools to compare multiple trading strategies
and analyze their relative performance.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import numpy as np
import pandas as pd
import json

from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from .data_provider import HistoricalDataProvider
from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from ..strategies.base_strategy import BaseStrategy
from ..models.config import BotConfig


class ComparisonMetric(Enum):
    """Metrics for strategy comparison."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    VOLATILITY = "volatility"


@dataclass
class StrategyConfig:
    """Configuration for a strategy in comparison."""
    name: str
    strategy_class: Type[BaseStrategy]
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: Optional[str] = None


@dataclass
class ComparisonResult:
    """Results from strategy comparison."""
    comparison_config: Dict[str, Any]
    backtest_config: BacktestConfig
    strategies: List[StrategyConfig]
    
    # Individual results
    strategy_results: Dict[str, BacktestResult] = field(default_factory=dict)
    strategy_metrics: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    
    # Comparison analysis
    rankings: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    risk_return_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical analysis
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    
    # Summary
    best_strategy: Optional[str] = None
    best_metric_value: Optional[float] = None
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Execution info
    execution_time: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'comparison_config': self.comparison_config,
            'backtest_config': {
                'start_date': self.backtest_config.start_date.isoformat(),
                'end_date': self.backtest_config.end_date.isoformat(),
                'initial_balance': self.backtest_config.initial_balance,
                'symbols': self.backtest_config.symbols,
                'timeframe': self.backtest_config.timeframe
            },
            'strategies': [
                {
                    'name': s.name,
                    'parameters': s.parameters,
                    'enabled': s.enabled,
                    'description': s.description
                }
                for s in self.strategies
            ],
            'strategy_results': {
                name: result.to_dict() for name, result in self.strategy_results.items()
            },
            'strategy_metrics': {
                name: metrics.to_dict() for name, metrics in self.strategy_metrics.items()
            },
            'rankings': self.rankings,
            'correlation_matrix': self.correlation_matrix.to_dict() if self.correlation_matrix is not None else None,
            'risk_return_analysis': self.risk_return_analysis,
            'statistical_tests': self.statistical_tests,
            'best_strategy': self.best_strategy,
            'best_metric_value': self.best_metric_value,
            'summary_statistics': self.summary_statistics,
            'execution_time': self.execution_time,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


class StrategyComparator:
    """
    Strategy comparison framework.
    
    Provides tools to compare multiple trading strategies across
    various performance metrics and statistical measures.
    """
    
    def __init__(self, backtest_engine: BacktestEngine, data_provider: HistoricalDataProvider,
                 performance_analyzer: PerformanceAnalyzer):
        """Initialize strategy comparator."""
        self.backtest_engine = backtest_engine
        self.data_provider = data_provider
        self.performance_analyzer = performance_analyzer
        self.logger = logging.getLogger(__name__)
        
        # Comparison state
        self._is_running = False
        self._current_comparison: Optional[ComparisonResult] = None
    
    async def compare_strategies(self, strategies: List[StrategyConfig], 
                               backtest_config: BacktestConfig,
                               comparison_metric: ComparisonMetric = ComparisonMetric.SHARPE_RATIO,
                               bot_config: Optional[BotConfig] = None,
                               run_parallel: bool = True) -> ComparisonResult:
        """
        Compare multiple strategies using backtesting.
        
        Args:
            strategies: List of strategy configurations to compare
            backtest_config: Backtest configuration
            comparison_metric: Primary metric for ranking strategies
            bot_config: Bot configuration (optional)
            run_parallel: Whether to run backtests in parallel
            
        Returns:
            ComparisonResult with detailed analysis
        """
        if self._is_running:
            raise RuntimeError("Strategy comparison is already running")
        
        self._is_running = True
        start_time = datetime.now()
        
        # Initialize result
        result = ComparisonResult(
            comparison_config={
                'primary_metric': comparison_metric.value,
                'run_parallel': run_parallel,
                'total_strategies': len(strategies)
            },
            backtest_config=backtest_config,
            strategies=strategies,
            start_time=start_time
        )
        self._current_comparison = result
        
        try:
            self.logger.info(f"Starting strategy comparison: {len(strategies)} strategies")
            
            # Filter enabled strategies
            enabled_strategies = [s for s in strategies if s.enabled]
            
            if not enabled_strategies:
                raise ValueError("No enabled strategies found")
            
            # Run backtests
            if run_parallel:
                await self._run_parallel_backtests(enabled_strategies, backtest_config, result, bot_config)
            else:
                await self._run_sequential_backtests(enabled_strategies, backtest_config, result, bot_config)
            
            # Analyze results
            self._analyze_strategy_performance(result)
            self._calculate_rankings(result, comparison_metric)
            self._calculate_correlations(result)
            self._perform_statistical_analysis(result)
            self._generate_summary_statistics(result)
            
            result.end_time = datetime.now()
            result.execution_time = (result.end_time - start_time).total_seconds()
            
            self.logger.info(f"Strategy comparison completed in {result.execution_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Strategy comparison failed: {e}")
            raise
        
        finally:
            self._is_running = False
            self._current_comparison = None
        
        return result
    
    async def _run_parallel_backtests(self, strategies: List[StrategyConfig],
                                    backtest_config: BacktestConfig,
                                    result: ComparisonResult,
                                    bot_config: Optional[BotConfig]) -> None:
        """Run backtests in parallel."""
        tasks = []
        
        for strategy_config in strategies:
            task = asyncio.create_task(
                self._run_single_backtest(strategy_config, backtest_config, bot_config)
            )
            tasks.append((strategy_config.name, task))
        
        # Wait for all tasks to complete
        for strategy_name, task in tasks:
            try:
                backtest_result = await task
                result.strategy_results[strategy_name] = backtest_result
                self.logger.info(f"Completed backtest for {strategy_name}")
            except Exception as e:
                self.logger.error(f"Backtest failed for {strategy_name}: {e}")
                continue
    
    async def _run_sequential_backtests(self, strategies: List[StrategyConfig],
                                      backtest_config: BacktestConfig,
                                      result: ComparisonResult,
                                      bot_config: Optional[BotConfig]) -> None:
        """Run backtests sequentially."""
        for i, strategy_config in enumerate(strategies):
            try:
                self.logger.info(f"Running backtest {i+1}/{len(strategies)}: {strategy_config.name}")
                
                backtest_result = await self._run_single_backtest(strategy_config, backtest_config, bot_config)
                result.strategy_results[strategy_config.name] = backtest_result
                
                self.logger.info(f"Completed backtest for {strategy_config.name}")
                
            except Exception as e:
                self.logger.error(f"Backtest failed for {strategy_config.name}: {e}")
                continue
    
    async def _run_single_backtest(self, strategy_config: StrategyConfig,
                                 backtest_config: BacktestConfig,
                                 bot_config: Optional[BotConfig]) -> BacktestResult:
        """Run backtest for a single strategy."""
        # Create strategy instance
        strategy = strategy_config.strategy_class(**strategy_config.parameters)
        
        # Run backtest
        return await self.backtest_engine.run_backtest(strategy, backtest_config, bot_config)
    
    def _analyze_strategy_performance(self, result: ComparisonResult) -> None:
        """Analyze performance metrics for all strategies."""
        for strategy_name, backtest_result in result.strategy_results.items():
            try:
                metrics = self.performance_analyzer.analyze_backtest_result(backtest_result)
                result.strategy_metrics[strategy_name] = metrics
            except Exception as e:
                self.logger.error(f"Performance analysis failed for {strategy_name}: {e}")
                continue
    
    def _calculate_rankings(self, result: ComparisonResult, primary_metric: ComparisonMetric) -> None:
        """Calculate strategy rankings for various metrics."""
        if not result.strategy_metrics:
            return
        
        # Define metrics to rank
        metrics_to_rank = [
            ComparisonMetric.TOTAL_RETURN,
            ComparisonMetric.SHARPE_RATIO,
            ComparisonMetric.SORTINO_RATIO,
            ComparisonMetric.MAX_DRAWDOWN,
            ComparisonMetric.WIN_RATE,
            ComparisonMetric.PROFIT_FACTOR,
            ComparisonMetric.CALMAR_RATIO,
            ComparisonMetric.VOLATILITY
        ]
        
        for metric in metrics_to_rank:
            strategy_values = []
            
            for strategy_name, metrics in result.strategy_metrics.items():
                value = self._get_metric_value(metrics, metric)
                if value is not None:
                    strategy_values.append((strategy_name, value))
            
            # Sort strategies (higher is better for most metrics, except drawdown and volatility)
            reverse_sort = metric not in [ComparisonMetric.MAX_DRAWDOWN, ComparisonMetric.VOLATILITY]
            strategy_values.sort(key=lambda x: x[1], reverse=reverse_sort)
            
            result.rankings[metric.value] = strategy_values
        
        # Set best strategy based on primary metric
        if primary_metric.value in result.rankings and result.rankings[primary_metric.value]:
            best_strategy, best_value = result.rankings[primary_metric.value][0]
            result.best_strategy = best_strategy
            result.best_metric_value = best_value
    
    def _get_metric_value(self, metrics: PerformanceMetrics, metric: ComparisonMetric) -> Optional[float]:
        """Get metric value from performance metrics."""
        metric_mapping = {
            ComparisonMetric.TOTAL_RETURN: metrics.total_return,
            ComparisonMetric.SHARPE_RATIO: metrics.sharpe_ratio,
            ComparisonMetric.SORTINO_RATIO: metrics.sortino_ratio,
            ComparisonMetric.MAX_DRAWDOWN: metrics.max_drawdown,
            ComparisonMetric.WIN_RATE: metrics.win_rate,
            ComparisonMetric.PROFIT_FACTOR: metrics.profit_factor,
            ComparisonMetric.CALMAR_RATIO: metrics.calmar_ratio,
            ComparisonMetric.VOLATILITY: metrics.volatility
        }
        
        return metric_mapping.get(metric)
    
    def _calculate_correlations(self, result: ComparisonResult) -> None:
        """Calculate correlation matrix between strategy returns."""
        if len(result.strategy_results) < 2:
            return
        
        try:
            # Extract portfolio value series for each strategy
            strategy_returns = {}
            
            for strategy_name, backtest_result in result.strategy_results.items():
                if backtest_result.portfolio_values:
                    values = [pv[1] for pv in backtest_result.portfolio_values]
                    if len(values) > 1:
                        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
                        strategy_returns[strategy_name] = returns
            
            if len(strategy_returns) < 2:
                return
            
            # Align return series (use shortest length)
            min_length = min(len(returns) for returns in strategy_returns.values())
            aligned_returns = {name: returns[:min_length] for name, returns in strategy_returns.items()}
            
            # Create correlation matrix
            strategy_names = list(aligned_returns.keys())
            correlation_data = []
            
            for name1 in strategy_names:
                row = []
                for name2 in strategy_names:
                    if name1 == name2:
                        correlation = 1.0
                    else:
                        returns1 = aligned_returns[name1]
                        returns2 = aligned_returns[name2]
                        correlation = np.corrcoef(returns1, returns2)[0, 1]
                        if np.isnan(correlation):
                            correlation = 0.0
                    row.append(correlation)
                correlation_data.append(row)
            
            result.correlation_matrix = pd.DataFrame(
                correlation_data,
                index=strategy_names,
                columns=strategy_names
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {e}")
    
    def _perform_statistical_analysis(self, result: ComparisonResult) -> None:
        """Perform statistical tests on strategy performance."""
        if len(result.strategy_metrics) < 2:
            return
        
        try:
            # Extract returns for statistical tests
            strategy_returns = {}
            
            for strategy_name, backtest_result in result.strategy_results.items():
                if backtest_result.portfolio_values:
                    values = [pv[1] for pv in backtest_result.portfolio_values]
                    if len(values) > 1:
                        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
                        strategy_returns[strategy_name] = returns
            
            if len(strategy_returns) < 2:
                return
            
            # Perform pairwise t-tests (simplified)
            from scipy import stats
            
            strategy_names = list(strategy_returns.keys())
            t_test_results = {}
            
            for i, name1 in enumerate(strategy_names):
                for j, name2 in enumerate(strategy_names[i+1:], i+1):
                    returns1 = strategy_returns[name1]
                    returns2 = strategy_returns[name2]
                    
                    # Align lengths
                    min_length = min(len(returns1), len(returns2))
                    returns1 = returns1[:min_length]
                    returns2 = returns2[:min_length]
                    
                    if min_length > 10:  # Need sufficient data
                        t_stat, p_value = stats.ttest_ind(returns1, returns2)
                        t_test_results[f"{name1}_vs_{name2}"] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
            
            result.statistical_tests['t_tests'] = t_test_results
            
            # Calculate risk-return analysis
            risk_return_data = []
            for strategy_name, metrics in result.strategy_metrics.items():
                risk_return_data.append({
                    'strategy': strategy_name,
                    'return': metrics.annualized_return,
                    'risk': metrics.volatility,
                    'sharpe': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown
                })
            
            result.risk_return_analysis = {
                'data': risk_return_data,
                'efficient_frontier': self._calculate_efficient_frontier(risk_return_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {e}")
    
    def _calculate_efficient_frontier(self, risk_return_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate efficient frontier analysis (simplified)."""
        if len(risk_return_data) < 2:
            return {}
        
        try:
            # Find strategies with best risk-adjusted returns
            sorted_by_sharpe = sorted(risk_return_data, key=lambda x: x['sharpe'], reverse=True)
            sorted_by_return = sorted(risk_return_data, key=lambda x: x['return'], reverse=True)
            sorted_by_risk = sorted(risk_return_data, key=lambda x: x['risk'])
            
            return {
                'best_sharpe': sorted_by_sharpe[0]['strategy'] if sorted_by_sharpe else None,
                'highest_return': sorted_by_return[0]['strategy'] if sorted_by_return else None,
                'lowest_risk': sorted_by_risk[0]['strategy'] if sorted_by_risk else None,
                'risk_return_ratio': {
                    'best': max(d['return'] / d['risk'] if d['risk'] > 0 else 0 for d in risk_return_data),
                    'worst': min(d['return'] / d['risk'] if d['risk'] > 0 else 0 for d in risk_return_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating efficient frontier: {e}")
            return {}
    
    def _generate_summary_statistics(self, result: ComparisonResult) -> None:
        """Generate summary statistics for the comparison."""
        if not result.strategy_metrics:
            return
        
        # Collect all metric values
        returns = [m.total_return for m in result.strategy_metrics.values()]
        sharpe_ratios = [m.sharpe_ratio for m in result.strategy_metrics.values()]
        drawdowns = [m.max_drawdown for m in result.strategy_metrics.values()]
        win_rates = [m.win_rate for m in result.strategy_metrics.values()]
        
        result.summary_statistics = {
            'total_strategies_tested': len(result.strategy_metrics),
            'successful_backtests': len([r for r in result.strategy_results.values() 
                                       if r.status.value == "completed"]),
            'returns': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': min(returns),
                'max': max(returns),
                'median': np.median(returns)
            },
            'sharpe_ratios': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'min': min(sharpe_ratios),
                'max': max(sharpe_ratios),
                'median': np.median(sharpe_ratios)
            },
            'max_drawdowns': {
                'mean': np.mean(drawdowns),
                'std': np.std(drawdowns),
                'min': min(drawdowns),
                'max': max(drawdowns),
                'median': np.median(drawdowns)
            },
            'win_rates': {
                'mean': np.mean(win_rates),
                'std': np.std(win_rates),
                'min': min(win_rates),
                'max': max(win_rates),
                'median': np.median(win_rates)
            }
        }
    
    def generate_comparison_report(self, result: ComparisonResult) -> str:
        """Generate a formatted comparison report."""
        report = f"""
STRATEGY COMPARISON REPORT
{'=' * 50}

COMPARISON OVERVIEW
------------------
Total Strategies:          {result.comparison_config['total_strategies']}
Successful Backtests:       {result.summary_statistics.get('successful_backtests', 0)}
Primary Metric:             {result.comparison_config['primary_metric']}
Best Strategy:              {result.best_strategy or 'N/A'}
Best Metric Value:          {result.best_metric_value:.4f if result.best_metric_value else 'N/A'}
Execution Time:             {result.execution_time:.2f} seconds

BACKTEST CONFIGURATION
---------------------
Period:                     {result.backtest_config.start_date.strftime('%Y-%m-%d')} to {result.backtest_config.end_date.strftime('%Y-%m-%d')}
Initial Balance:            ${result.backtest_config.initial_balance:,.2f}
Symbols:                    {', '.join(result.backtest_config.symbols)}
Timeframe:                  {result.backtest_config.timeframe}

PERFORMANCE SUMMARY
------------------
"""
        
        if result.summary_statistics:
            stats = result.summary_statistics
            report += f"""
Returns:
  Mean:                     {stats['returns']['mean']:.2%}
  Std Dev:                  {stats['returns']['std']:.2%}
  Range:                    {stats['returns']['min']:.2%} to {stats['returns']['max']:.2%}

Sharpe Ratios:
  Mean:                     {stats['sharpe_ratios']['mean']:.3f}
  Std Dev:                  {stats['sharpe_ratios']['std']:.3f}
  Range:                    {stats['sharpe_ratios']['min']:.3f} to {stats['sharpe_ratios']['max']:.3f}

Max Drawdowns:
  Mean:                     {stats['max_drawdowns']['mean']:.2%}
  Std Dev:                  {stats['max_drawdowns']['std']:.2%}
  Range:                    {stats['max_drawdowns']['min']:.2%} to {stats['max_drawdowns']['max']:.2%}
"""
        
        # Add rankings
        if result.rankings:
            report += "\nSTRATEGY RANKINGS\n" + "-" * 17 + "\n"
            
            for metric, rankings in result.rankings.items():
                if rankings:
                    report += f"\nBy {metric.replace('_', ' ').title()}:\n"
                    for i, (strategy, value) in enumerate(rankings[:5], 1):  # Top 5
                        report += f"  {i}. {strategy}: {value:.4f}\n"
        
        # Add correlation analysis
        if result.correlation_matrix is not None:
            report += "\nCORRELATION ANALYSIS\n" + "-" * 19 + "\n"
            report += "Strategy correlation matrix (top correlations):\n"
            
            # Find highest correlations (excluding self-correlations)
            correlations = []
            for i in range(len(result.correlation_matrix)):
                for j in range(i+1, len(result.correlation_matrix)):
                    strategy1 = result.correlation_matrix.index[i]
                    strategy2 = result.correlation_matrix.columns[j]
                    corr_value = result.correlation_matrix.iloc[i, j]
                    correlations.append((strategy1, strategy2, corr_value))
            
            correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            for strategy1, strategy2, corr in correlations[:5]:
                report += f"  {strategy1} vs {strategy2}: {corr:.3f}\n"
        
        return report
    
    def save_comparison_result(self, result: ComparisonResult, file_path: str) -> bool:
        """Save comparison result to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Comparison result saved to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save comparison result: {e}")
            return False
    
    def get_comparison_status(self) -> Optional[Dict[str, Any]]:
        """Get current comparison status."""
        if self._current_comparison:
            completed_backtests = len(self._current_comparison.strategy_results)
            total_strategies = len(self._current_comparison.strategies)
            
            return {
                'is_running': self._is_running,
                'completed_backtests': completed_backtests,
                'total_strategies': total_strategies,
                'progress': (completed_backtests / total_strategies) * 100 if total_strategies > 0 else 0,
                'current_best': self._current_comparison.best_strategy,
                'current_best_value': self._current_comparison.best_metric_value
            }
        
        return None