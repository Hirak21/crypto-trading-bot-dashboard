"""
Parameter optimization framework for trading strategies.

This module provides optimization algorithms to find optimal parameters
for trading strategies based on historical performance.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from scipy.optimize import minimize, differential_evolution
import json

from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from .data_provider import HistoricalDataProvider
from ..strategies.base_strategy import BaseStrategy
from ..models.config import BotConfig


class OptimizationMethod(Enum):
    """Available optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"


class OptimizationObjective(Enum):
    """Optimization objectives."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    CUSTOM = "custom"


@dataclass
class ParameterRange:
    """Parameter range for optimization."""
    name: str
    min_value: Union[int, float]
    max_value: Union[int, float]
    step: Optional[Union[int, float]] = None
    values: Optional[List[Any]] = None  # For discrete values
    parameter_type: str = "float"  # "int", "float", "bool", "choice"
    
    def __post_init__(self):
        """Validate parameter range."""
        if self.parameter_type in ["int", "float"]:
            if self.min_value >= self.max_value:
                raise ValueError(f"min_value must be less than max_value for {self.name}")
        
        if self.parameter_type == "choice" and not self.values:
            raise ValueError(f"values must be provided for choice parameter {self.name}")
    
    def generate_values(self, n_samples: Optional[int] = None) -> List[Any]:
        """Generate parameter values for optimization."""
        if self.parameter_type == "choice":
            return self.values
        
        if self.parameter_type == "bool":
            return [True, False]
        
        if self.step is not None:
            # Grid search with step
            if self.parameter_type == "int":
                return list(range(int(self.min_value), int(self.max_value) + 1, int(self.step)))
            else:
                values = []
                current = self.min_value
                while current <= self.max_value:
                    values.append(current)
                    current += self.step
                return values
        
        if n_samples:
            # Random sampling
            if self.parameter_type == "int":
                return np.random.randint(self.min_value, self.max_value + 1, n_samples).tolist()
            else:
                return np.random.uniform(self.min_value, self.max_value, n_samples).tolist()
        
        # Default grid
        if self.parameter_type == "int":
            return list(range(int(self.min_value), int(self.max_value) + 1))
        else:
            return np.linspace(self.min_value, self.max_value, 10).tolist()


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    parameter_ranges: List[ParameterRange] = field(default_factory=list)
    
    # Method-specific settings
    max_iterations: int = 100
    population_size: int = 50  # For genetic algorithm
    n_random_samples: int = 100  # For random search
    convergence_threshold: float = 1e-6
    
    # Validation settings
    validation_split: float = 0.3  # Portion of data for validation
    cross_validation_folds: int = 3
    
    # Performance settings
    max_workers: int = 4
    timeout_seconds: int = 3600  # 1 hour timeout
    
    # Custom objective function
    custom_objective: Optional[Callable[[BacktestResult], float]] = None
    
    def __post_init__(self):
        """Validate optimization configuration."""
        if not self.parameter_ranges:
            raise ValueError("At least one parameter range must be specified")
        
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        
        if self.cross_validation_folds < 2:
            raise ValueError("cross_validation_folds must be at least 2")
        
        if self.objective == OptimizationObjective.CUSTOM and not self.custom_objective:
            raise ValueError("custom_objective function must be provided for CUSTOM objective")


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    config: OptimizationConfig
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Validation results
    validation_score: Optional[float] = None
    cross_validation_scores: List[float] = field(default_factory=list)
    
    # Performance metrics
    total_evaluations: int = 0
    execution_time: float = 0.0
    convergence_iteration: Optional[int] = None
    
    # Best backtest result
    best_backtest_result: Optional[BacktestResult] = None
    
    # All results for analysis
    all_results: List[Tuple[Dict[str, Any], float, BacktestResult]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'best_parameters': self.best_parameters,
            'best_score': self.best_score,
            'validation_score': self.validation_score,
            'cross_validation_scores': self.cross_validation_scores,
            'total_evaluations': self.total_evaluations,
            'execution_time': self.execution_time,
            'convergence_iteration': self.convergence_iteration,
            'optimization_history': self.optimization_history,
            'best_backtest_summary': self.best_backtest_result.to_dict() if self.best_backtest_result else None
        }


class ParameterOptimizer:
    """
    Parameter optimizer for trading strategies.
    
    Provides various optimization algorithms to find optimal parameters
    for trading strategies based on historical performance.
    """
    
    def __init__(self, backtest_engine: BacktestEngine, data_provider: HistoricalDataProvider):
        """Initialize parameter optimizer."""
        self.backtest_engine = backtest_engine
        self.data_provider = data_provider
        self.logger = logging.getLogger(__name__)
        
        # Optimization state
        self._is_running = False
        self._should_cancel = False
        self._current_optimization: Optional[OptimizationResult] = None
    
    async def optimize_parameters(self, strategy_class: type, backtest_config: BacktestConfig,
                                optimization_config: OptimizationConfig,
                                bot_config: Optional[BotConfig] = None) -> OptimizationResult:
        """
        Optimize strategy parameters using specified method.
        
        Args:
            strategy_class: Strategy class to optimize
            backtest_config: Backtest configuration
            optimization_config: Optimization configuration
            bot_config: Bot configuration (optional)
            
        Returns:
            OptimizationResult with best parameters and performance metrics
        """
        if self._is_running:
            raise RuntimeError("Optimization is already running")
        
        self._is_running = True
        self._should_cancel = False
        
        start_time = datetime.now()
        
        # Initialize result
        result = OptimizationResult(
            config=optimization_config,
            best_parameters={},
            best_score=float('-inf')
        )
        self._current_optimization = result
        
        try:
            self.logger.info(f"Starting parameter optimization using {optimization_config.method.value}")
            
            # Split data for validation if required
            train_config, validation_config = self._split_data_for_validation(
                backtest_config, optimization_config.validation_split
            )
            
            # Run optimization based on method
            if optimization_config.method == OptimizationMethod.GRID_SEARCH:
                await self._grid_search_optimization(strategy_class, train_config, optimization_config, result, bot_config)
            elif optimization_config.method == OptimizationMethod.RANDOM_SEARCH:
                await self._random_search_optimization(strategy_class, train_config, optimization_config, result, bot_config)
            elif optimization_config.method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
                await self._differential_evolution_optimization(strategy_class, train_config, optimization_config, result, bot_config)
            else:
                raise ValueError(f"Optimization method not implemented: {optimization_config.method}")
            
            # Validate best parameters on validation set
            if validation_config and result.best_parameters:
                validation_score = await self._validate_parameters(
                    strategy_class, result.best_parameters, validation_config, optimization_config, bot_config
                )
                result.validation_score = validation_score
            
            # Cross-validation
            if optimization_config.cross_validation_folds > 1 and result.best_parameters:
                cv_scores = await self._cross_validate_parameters(
                    strategy_class, result.best_parameters, backtest_config, optimization_config, bot_config
                )
                result.cross_validation_scores = cv_scores
            
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Optimization completed. Best score: {result.best_score:.4f}")
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
        
        finally:
            self._is_running = False
            self._current_optimization = None
        
        return result
    
    def _split_data_for_validation(self, backtest_config: BacktestConfig, 
                                 validation_split: float) -> Tuple[BacktestConfig, BacktestConfig]:
        """Split backtest period into training and validation sets."""
        total_days = (backtest_config.end_date - backtest_config.start_date).days
        train_days = int(total_days * (1 - validation_split))
        
        train_end_date = backtest_config.start_date + timedelta(days=train_days)
        
        train_config = BacktestConfig(
            start_date=backtest_config.start_date,
            end_date=train_end_date,
            initial_balance=backtest_config.initial_balance,
            symbols=backtest_config.symbols,
            timeframe=backtest_config.timeframe,
            commission_rate=backtest_config.commission_rate,
            slippage_rate=backtest_config.slippage_rate,
            max_positions=backtest_config.max_positions,
            enable_shorting=backtest_config.enable_shorting
        )
        
        validation_config = BacktestConfig(
            start_date=train_end_date,
            end_date=backtest_config.end_date,
            initial_balance=backtest_config.initial_balance,
            symbols=backtest_config.symbols,
            timeframe=backtest_config.timeframe,
            commission_rate=backtest_config.commission_rate,
            slippage_rate=backtest_config.slippage_rate,
            max_positions=backtest_config.max_positions,
            enable_shorting=backtest_config.enable_shorting
        )
        
        return train_config, validation_config
    
    async def _grid_search_optimization(self, strategy_class: type, backtest_config: BacktestConfig,
                                      optimization_config: OptimizationConfig, result: OptimizationResult,
                                      bot_config: Optional[BotConfig]) -> None:
        """Perform grid search optimization."""
        # Generate all parameter combinations
        parameter_combinations = self._generate_parameter_combinations(optimization_config.parameter_ranges)
        
        self.logger.info(f"Grid search: evaluating {len(parameter_combinations)} parameter combinations")
        
        # Evaluate all combinations
        await self._evaluate_parameter_combinations(
            strategy_class, parameter_combinations, backtest_config, optimization_config, result, bot_config
        )
    
    async def _random_search_optimization(self, strategy_class: type, backtest_config: BacktestConfig,
                                        optimization_config: OptimizationConfig, result: OptimizationResult,
                                        bot_config: Optional[BotConfig]) -> None:
        """Perform random search optimization."""
        # Generate random parameter combinations
        parameter_combinations = self._generate_random_parameter_combinations(
            optimization_config.parameter_ranges, optimization_config.n_random_samples
        )
        
        self.logger.info(f"Random search: evaluating {len(parameter_combinations)} random parameter combinations")
        
        # Evaluate combinations
        await self._evaluate_parameter_combinations(
            strategy_class, parameter_combinations, backtest_config, optimization_config, result, bot_config
        )
    
    async def _differential_evolution_optimization(self, strategy_class: type, backtest_config: BacktestConfig,
                                                 optimization_config: OptimizationConfig, result: OptimizationResult,
                                                 bot_config: Optional[BotConfig]) -> None:
        """Perform differential evolution optimization."""
        # Prepare bounds for scipy optimization
        bounds = []
        param_names = []
        
        for param_range in optimization_config.parameter_ranges:
            if param_range.parameter_type in ["int", "float"]:
                bounds.append((param_range.min_value, param_range.max_value))
                param_names.append(param_range.name)
            else:
                # For discrete parameters, we'll handle them separately
                self.logger.warning(f"Differential evolution doesn't support discrete parameter: {param_range.name}")
        
        if not bounds:
            raise ValueError("No continuous parameters found for differential evolution")
        
        # Objective function for scipy
        def objective_function(x):
            if self._should_cancel:
                return float('inf')
            
            # Convert array to parameter dictionary
            parameters = {param_names[i]: x[i] for i in range(len(param_names))}
            
            # Handle integer parameters
            for param_range in optimization_config.parameter_ranges:
                if param_range.name in parameters and param_range.parameter_type == "int":
                    parameters[param_range.name] = int(round(parameters[param_range.name]))
            
            # Run backtest
            try:
                strategy = strategy_class(**parameters)
                backtest_result = asyncio.run(self.backtest_engine.run_backtest(strategy, backtest_config, bot_config))
                
                score = self._calculate_objective_score(backtest_result, optimization_config)
                
                # Track result
                result.all_results.append((parameters.copy(), score, backtest_result))
                result.total_evaluations += 1
                
                # Update best result
                if score > result.best_score:
                    result.best_score = score
                    result.best_parameters = parameters.copy()
                    result.best_backtest_result = backtest_result
                
                # Add to history
                result.optimization_history.append({
                    'iteration': result.total_evaluations,
                    'parameters': parameters.copy(),
                    'score': score
                })
                
                return -score  # Minimize negative score (maximize score)
                
            except Exception as e:
                self.logger.error(f"Error evaluating parameters {parameters}: {e}")
                return float('inf')
        
        # Run differential evolution
        try:
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=optimization_config.max_iterations,
                popsize=optimization_config.population_size,
                seed=42
            )
            
            if de_result.success:
                result.convergence_iteration = de_result.nit
                self.logger.info(f"Differential evolution converged after {de_result.nit} iterations")
            
        except Exception as e:
            self.logger.error(f"Differential evolution failed: {e}")
    
    def _generate_parameter_combinations(self, parameter_ranges: List[ParameterRange]) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations for grid search."""
        param_values = {}
        
        for param_range in parameter_ranges:
            param_values[param_range.name] = param_range.generate_values()
        
        # Generate all combinations
        param_names = list(param_values.keys())
        value_combinations = itertools.product(*[param_values[name] for name in param_names])
        
        combinations = []
        for values in value_combinations:
            combination = {param_names[i]: values[i] for i in range(len(param_names))}
            combinations.append(combination)
        
        return combinations
    
    def _generate_random_parameter_combinations(self, parameter_ranges: List[ParameterRange], 
                                              n_samples: int) -> List[Dict[str, Any]]:
        """Generate random parameter combinations."""
        combinations = []
        
        for _ in range(n_samples):
            combination = {}
            
            for param_range in parameter_ranges:
                if param_range.parameter_type == "choice":
                    combination[param_range.name] = np.random.choice(param_range.values)
                elif param_range.parameter_type == "bool":
                    combination[param_range.name] = np.random.choice([True, False])
                elif param_range.parameter_type == "int":
                    combination[param_range.name] = np.random.randint(param_range.min_value, param_range.max_value + 1)
                else:  # float
                    combination[param_range.name] = np.random.uniform(param_range.min_value, param_range.max_value)
            
            combinations.append(combination)
        
        return combinations
    
    async def _evaluate_parameter_combinations(self, strategy_class: type, parameter_combinations: List[Dict[str, Any]],
                                             backtest_config: BacktestConfig, optimization_config: OptimizationConfig,
                                             result: OptimizationResult, bot_config: Optional[BotConfig]) -> None:
        """Evaluate parameter combinations using parallel processing."""
        # Use ThreadPoolExecutor for parallel backtesting
        with ThreadPoolExecutor(max_workers=optimization_config.max_workers) as executor:
            # Submit all tasks
            future_to_params = {}
            
            for i, parameters in enumerate(parameter_combinations):
                if self._should_cancel:
                    break
                
                future = executor.submit(
                    self._evaluate_single_combination,
                    strategy_class, parameters, backtest_config, optimization_config, bot_config
                )
                future_to_params[future] = (i, parameters)
            
            # Process completed tasks
            for future in as_completed(future_to_params, timeout=optimization_config.timeout_seconds):
                if self._should_cancel:
                    break
                
                try:
                    i, parameters = future_to_params[future]
                    score, backtest_result = future.result()
                    
                    # Track result
                    result.all_results.append((parameters.copy(), score, backtest_result))
                    result.total_evaluations += 1
                    
                    # Update best result
                    if score > result.best_score:
                        result.best_score = score
                        result.best_parameters = parameters.copy()
                        result.best_backtest_result = backtest_result
                    
                    # Add to history
                    result.optimization_history.append({
                        'iteration': i + 1,
                        'parameters': parameters.copy(),
                        'score': score
                    })
                    
                    # Progress logging
                    if result.total_evaluations % 10 == 0:
                        progress = (result.total_evaluations / len(parameter_combinations)) * 100
                        self.logger.info(f"Optimization progress: {progress:.1f}% (Best score: {result.best_score:.4f})")
                
                except Exception as e:
                    self.logger.error(f"Error evaluating parameter combination: {e}")
                    continue
    
    def _evaluate_single_combination(self, strategy_class: type, parameters: Dict[str, Any],
                                   backtest_config: BacktestConfig, optimization_config: OptimizationConfig,
                                   bot_config: Optional[BotConfig]) -> Tuple[float, BacktestResult]:
        """Evaluate a single parameter combination."""
        try:
            # Create strategy instance with parameters
            strategy = strategy_class(**parameters)
            
            # Run backtest
            backtest_result = asyncio.run(self.backtest_engine.run_backtest(strategy, backtest_config, bot_config))
            
            # Calculate objective score
            score = self._calculate_objective_score(backtest_result, optimization_config)
            
            return score, backtest_result
            
        except Exception as e:
            self.logger.error(f"Error in single combination evaluation: {e}")
            return float('-inf'), None
    
    def _calculate_objective_score(self, backtest_result: BacktestResult, 
                                 optimization_config: OptimizationConfig) -> float:
        """Calculate objective score from backtest result."""
        if not backtest_result or backtest_result.status.value != "completed":
            return float('-inf')
        
        objective = optimization_config.objective
        
        if objective == OptimizationObjective.TOTAL_RETURN:
            return backtest_result.total_return
        elif objective == OptimizationObjective.SHARPE_RATIO:
            return backtest_result.sharpe_ratio
        elif objective == OptimizationObjective.SORTINO_RATIO:
            return backtest_result.sortino_ratio
        elif objective == OptimizationObjective.PROFIT_FACTOR:
            return backtest_result.profit_factor
        elif objective == OptimizationObjective.MAX_DRAWDOWN:
            return -backtest_result.max_drawdown  # Minimize drawdown
        elif objective == OptimizationObjective.WIN_RATE:
            return backtest_result.win_rate
        elif objective == OptimizationObjective.CUSTOM:
            return optimization_config.custom_objective(backtest_result)
        else:
            raise ValueError(f"Unknown optimization objective: {objective}")
    
    async def _validate_parameters(self, strategy_class: type, parameters: Dict[str, Any],
                                 validation_config: BacktestConfig, optimization_config: OptimizationConfig,
                                 bot_config: Optional[BotConfig]) -> float:
        """Validate parameters on validation set."""
        try:
            strategy = strategy_class(**parameters)
            backtest_result = await self.backtest_engine.run_backtest(strategy, validation_config, bot_config)
            return self._calculate_objective_score(backtest_result, optimization_config)
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return float('-inf')
    
    async def _cross_validate_parameters(self, strategy_class: type, parameters: Dict[str, Any],
                                       backtest_config: BacktestConfig, optimization_config: OptimizationConfig,
                                       bot_config: Optional[BotConfig]) -> List[float]:
        """Perform cross-validation on parameters."""
        scores = []
        
        # Split data into folds
        total_days = (backtest_config.end_date - backtest_config.start_date).days
        fold_days = total_days // optimization_config.cross_validation_folds
        
        for fold in range(optimization_config.cross_validation_folds):
            start_date = backtest_config.start_date + timedelta(days=fold * fold_days)
            end_date = start_date + timedelta(days=fold_days)
            
            if end_date > backtest_config.end_date:
                end_date = backtest_config.end_date
            
            fold_config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_balance=backtest_config.initial_balance,
                symbols=backtest_config.symbols,
                timeframe=backtest_config.timeframe,
                commission_rate=backtest_config.commission_rate,
                slippage_rate=backtest_config.slippage_rate,
                max_positions=backtest_config.max_positions,
                enable_shorting=backtest_config.enable_shorting
            )
            
            try:
                strategy = strategy_class(**parameters)
                backtest_result = await self.backtest_engine.run_backtest(strategy, fold_config, bot_config)
                score = self._calculate_objective_score(backtest_result, optimization_config)
                scores.append(score)
            except Exception as e:
                self.logger.error(f"Cross-validation fold {fold} error: {e}")
                scores.append(float('-inf'))
        
        return scores
    
    def cancel_optimization(self) -> bool:
        """Cancel running optimization."""
        if self._is_running:
            self._should_cancel = True
            return True
        return False
    
    def get_optimization_status(self) -> Optional[Dict[str, Any]]:
        """Get current optimization status."""
        if self._current_optimization:
            return {
                'is_running': self._is_running,
                'total_evaluations': self._current_optimization.total_evaluations,
                'best_score': self._current_optimization.best_score,
                'best_parameters': self._current_optimization.best_parameters
            }
        return None
    
    def save_optimization_result(self, result: OptimizationResult, file_path: str) -> bool:
        """Save optimization result to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Optimization result saved to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save optimization result: {e}")
            return False
    
    def load_optimization_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load optimization result from file."""
        try:
            with open(file_path, 'r') as f:
                result_data = json.load(f)
            
            self.logger.info(f"Optimization result loaded from: {file_path}")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Failed to load optimization result: {e}")
            return None