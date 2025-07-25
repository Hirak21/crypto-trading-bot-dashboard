"""
Main application entry point for the crypto trading bot.

This module orchestrates all components and manages the bot lifecycle.
"""

import asyncio
import signal
import sys
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import traceback

from .utils.config import ConfigManager
from .utils.logging_config import setup_logging, get_logger
from .utils.error_recovery import ErrorRecoveryManager, with_error_recovery
from .models.config import BotConfig, RiskConfig
from .managers.market_manager import MarketManager
from .managers.strategy_manager import StrategyManager
from .managers.risk_manager import RiskManager
from .managers.trade_manager import TradeManager
from .managers.portfolio_manager import PortfolioManager
from .managers.notification_manager import NotificationManager
from .api.binance_client import BinanceRestClient


class ApplicationState:
    """Application state management."""
    
    def __init__(self):
        self.startup_time = datetime.now()
        self.last_heartbeat = datetime.now()
        self.error_count = 0
        self.restart_count = 0
        self.is_healthy = True
        self.component_status = {}
        self.performance_metrics = {
            'uptime_seconds': 0,
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'last_trade_time': None
        }
    
    def update_heartbeat(self):
        """Update application heartbeat."""
        self.last_heartbeat = datetime.now()
        self.performance_metrics['uptime_seconds'] = (
            self.last_heartbeat - self.startup_time
        ).total_seconds()
    
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
        if self.error_count > 10:  # Too many errors
            self.is_healthy = False
    
    def record_restart(self):
        """Record an application restart."""
        self.restart_count += 1
        self.error_count = 0  # Reset error count on restart
        self.is_healthy = True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current application status."""
        return {
            'startup_time': self.startup_time,
            'last_heartbeat': self.last_heartbeat,
            'uptime_seconds': self.performance_metrics['uptime_seconds'],
            'error_count': self.error_count,
            'restart_count': self.restart_count,
            'is_healthy': self.is_healthy,
            'component_status': self.component_status.copy(),
            'performance_metrics': self.performance_metrics.copy()
        }


class TradingBotApplication:
    """Main trading bot application that coordinates all managers."""
    
    def __init__(self):
        # Configuration management
        self.config_manager = ConfigManager()
        self.config_dict = self.config_manager.load_config()
        self.config = BotConfig.from_dict(self.config_dict)
        
        # Set up logging
        setup_logging(self.config.logging_config.to_dict())
        self.logger = get_logger(__name__)
        
        # Application state
        self.state = ApplicationState()
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.restart_requested = False
        
        # Error recovery system
        self.error_recovery_manager = ErrorRecoveryManager({
            'state_dir': 'state',
            'max_recovery_attempts': 3,
            'recovery_cooldown_minutes': 5,
            'critical_error_threshold': 5
        })
        
        # Component managers
        self.market_manager: Optional[MarketManager] = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.trade_manager: Optional[TradeManager] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.notification_manager: Optional[NotificationManager] = None
        self.binance_client: Optional[BinanceRestClient] = None
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.performance_tracker_task: Optional[asyncio.Task] = None
        
        self.logger.info("Trading bot application initialized")
    
    async def start(self) -> None:
        """Start the trading bot with full component coordination."""
        try:
            self.logger.info("Starting crypto trading bot...")
            
            # Validate configuration
            if not await self._validate_startup_config():
                self.logger.error("Invalid configuration, cannot start bot")
                return
            
            # Initialize all components
            if not await self._initialize_components():
                self.logger.error("Component initialization failed, cannot start bot")
                return
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start background monitoring tasks
            await self._start_background_tasks()
            
            self.is_running = True
            self.state.record_restart()
            self.logger.info("Trading bot started successfully")
            
            # Main application loop
            await self._run_main_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start trading bot: {e}")
            self.logger.error(traceback.format_exc())
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the trading bot."""
        try:
            self.logger.info("Shutting down trading bot...")
            self.is_running = False
            
            # Signal shutdown to all components
            self.shutdown_event.set()
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Cleanup components in reverse order of initialization
            await self._cleanup_components()
            
            self.logger.info("Trading bot shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.logger.error(traceback.format_exc())
    
    async def restart(self) -> None:
        """Restart the trading bot."""
        self.logger.info("Restarting trading bot...")
        self.restart_requested = True
        await self.shutdown()
    
    async def _validate_startup_config(self) -> bool:
        """Validate configuration required for startup."""
        try:
            # Check if API credentials are available
            try:
                api_key, api_secret = self.config_manager.get_api_credentials()
                if not api_key or not api_secret:
                    self.logger.error("API credentials not found")
                    return False
            except Exception as e:
                self.logger.error(f"Failed to load API credentials: {e}")
                return False
            
            # Validate required config sections
            if not self.config.symbols:
                self.logger.error("No trading symbols configured")
                return False
            
            # Validate at least one strategy is enabled
            enabled_strategies = self.config.get_enabled_strategies()
            if not enabled_strategies:
                self.logger.error("No trading strategies enabled")
                return False
            
            self.logger.info(f"Configuration validated: {len(self.config.symbols)} symbols, "
                           f"{len(enabled_strategies)} strategies enabled")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    async def _initialize_components(self) -> bool:
        """Initialize all bot components in proper order."""
        try:
            self.logger.info("Initializing bot components...")
            
            # 1. Initialize Binance REST client
            api_key, api_secret = self.config_manager.get_api_credentials()
            self.binance_client = BinanceRestClient(api_key, api_secret, self.config.testnet)
            
            if not await self.binance_client.connect():
                self.logger.error("Failed to connect to Binance API")
                return False
            
            self.state.component_status['binance_client'] = 'connected'
            self.logger.info("Binance REST client initialized")
            
            # 2. Initialize Portfolio Manager
            initial_capital = 10000.0  # Default, should be configurable
            self.portfolio_manager = PortfolioManager(
                config=self.config_dict,
                initial_capital=initial_capital
            )
            self.state.component_status['portfolio_manager'] = 'initialized'
            self.logger.info("Portfolio Manager initialized")
            
            # 3. Initialize Risk Manager
            self.risk_manager = RiskManager(self.config.risk_config)
            self.state.component_status['risk_manager'] = 'initialized'
            self.logger.info("Risk Manager initialized")
            
            # 4. Initialize Trade Manager
            self.trade_manager = TradeManager(
                config=self.config_dict,
                binance_client=self.binance_client
            )
            self.state.component_status['trade_manager'] = 'initialized'
            self.logger.info("Trade Manager initialized")
            
            # 5. Initialize Strategy Manager
            self.strategy_manager = StrategyManager(self.config_dict)
            self.state.component_status['strategy_manager'] = 'initialized'
            self.logger.info("Strategy Manager initialized")
            
            # 6. Initialize Market Manager
            self.market_manager = MarketManager(self.config_dict)
            if not await self.market_manager.start():
                self.logger.error("Failed to start Market Manager")
                return False
            
            self.state.component_status['market_manager'] = 'running'
            self.logger.info("Market Manager started")
            
            # 7. Initialize Notification Manager
            self.notification_manager = NotificationManager(self.config.notification_config)
            await self.notification_manager.start()
            self.state.component_status['notification_manager'] = 'running'
            self.logger.info("Notification Manager started")
            
            # 8. Set up component interconnections
            await self._setup_component_connections()
            
            # 9. Register error recovery callbacks
            await self._setup_error_recovery()
            
            # 10. Attempt to recover from previous state if available
            await self._attempt_state_recovery()
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def _setup_component_connections(self) -> None:
        """Set up connections between components."""
        try:
            # Market Manager -> Strategy Manager (market data flow)
            self.market_manager.add_data_handler(self._handle_market_data)
            
            # Market Manager -> Portfolio Manager (price updates)
            self.market_manager.add_data_handler(self._handle_portfolio_price_update)
            
            # Set up error handlers
            self.market_manager.add_error_handler(self._handle_market_error)
            
            self.logger.info("Component connections established")
            
        except Exception as e:
            self.logger.error(f"Failed to setup component connections: {e}")
            raise
    
    async def _handle_market_data(self, market_data) -> None:
        """Handle incoming market data and coordinate trading decisions."""
        try:
            # Update application heartbeat
            self.state.update_heartbeat()
            
            # 1. Analyze market data with strategies
            signal = self.strategy_manager.analyze_market(market_data)
            
            if not signal:
                return  # No trading signal generated
            
            self.logger.debug(f"Trading signal generated: {signal.action.value} {signal.symbol} "
                            f"confidence: {signal.confidence:.2%}")
            
            # 2. Validate signal with risk manager
            is_valid, reason, position_size = self.risk_manager.validate_trade(signal, market_data)
            
            if not is_valid:
                self.logger.info(f"Trade rejected by risk manager: {reason}")
                return
            
            # 3. Execute trade through trade manager
            success, message, order_id = self.trade_manager.execute_trade(
                signal, position_size, market_data
            )
            
            if success:
                self.logger.info(f"Trade executed successfully: {order_id}")
                self.state.performance_metrics['total_trades'] += 1
                self.state.performance_metrics['successful_trades'] += 1
                self.state.performance_metrics['last_trade_time'] = datetime.now()
                
                # Send notification
                if self.notification_manager:
                    await self.notification_manager.send_trade_notification(
                        f"Trade executed: {signal.action.value} {signal.symbol} "
                        f"size: {position_size:.6f} @ {market_data.price:.6f}"
                    )
            else:
                self.logger.error(f"Trade execution failed: {message}")
                self.state.performance_metrics['failed_trades'] += 1
                self.state.record_error()
                
        except Exception as e:
            self.logger.error(f"Error handling market data: {e}")
            self.state.record_error()
    
    async def _handle_portfolio_price_update(self, market_data) -> None:
        """Handle portfolio price updates."""
        try:
            if self.portfolio_manager:
                # Update portfolio with current market prices
                market_prices = {market_data.symbol: market_data}
                self.portfolio_manager.update_market_data(market_prices)
                
        except Exception as e:
            self.logger.error(f"Error updating portfolio prices: {e}")
    
    async def _handle_market_error(self, error: Exception) -> None:
        """Handle market data errors."""
        self.logger.error(f"Market data error: {error}")
        self.state.record_error()
        
        # Send error notification
        if self.notification_manager:
            await self.notification_manager.send_alert(
                f"Market data error: {str(error)}", "ERROR"
            )
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        try:
            # Heartbeat task
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Health monitoring task
            self.health_monitor_task = asyncio.create_task(self._enhanced_health_monitor_loop())
            
            # Performance tracking task
            self.performance_tracker_task = asyncio.create_task(self._performance_tracker_loop())
            
            self.logger.info("Background tasks started")
            
        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}")
            raise
    
    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        try:
            tasks = [
                self.heartbeat_task,
                self.health_monitor_task,
                self.performance_tracker_task
            ]
            
            for task in tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.logger.info("Background tasks stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping background tasks: {e}")
    
    async def _cleanup_components(self) -> None:
        """Cleanup all bot components in reverse order."""
        try:
            self.logger.info("Cleaning up bot components...")
            
            # Stop notification manager
            if self.notification_manager:
                await self.notification_manager.stop()
                self.state.component_status['notification_manager'] = 'stopped'
            
            # Stop market manager
            if self.market_manager:
                await self.market_manager.stop()
                self.state.component_status['market_manager'] = 'stopped'
            
            # Close any open positions (if configured to do so)
            if self.trade_manager and self.config_dict.get('close_positions_on_shutdown', False):
                # Close all positions
                for symbol in list(self.trade_manager.positions.keys()):
                    await self.trade_manager.close_position(symbol, "Application shutdown")
            
            # Disconnect Binance client
            if self.binance_client:
                await self.binance_client.disconnect()
                self.state.component_status['binance_client'] = 'disconnected'
            
            self.logger.info("Component cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during component cleanup: {e}")
    
    async def _run_main_loop(self) -> None:
        """Main application loop with health monitoring."""
        self.logger.info("Starting main application loop")
        
        try:
            while self.is_running and not self.shutdown_event.is_set():
                # Check if restart was requested
                if self.restart_requested:
                    break
                
                # Main loop just waits - actual work is done in event handlers
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            self.logger.info("Main loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.state.record_error()
            await self.shutdown()
    
    async def _heartbeat_loop(self) -> None:
        """Background heartbeat monitoring."""
        try:
            while self.is_running and not self.shutdown_event.is_set():
                self.state.update_heartbeat()
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Heartbeat loop error: {e}")
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring."""
        try:
            while self.is_running and not self.shutdown_event.is_set():
                await asyncio.sleep(60)  # Check every minute
                
                # Check component health
                unhealthy_components = []
                
                # Check market manager
                if self.market_manager and not self.market_manager.is_running:
                    unhealthy_components.append('market_manager')
                
                # Check if we're receiving market data
                if self.market_manager:
                    stats = self.market_manager.get_stats()
                    last_update = stats['manager_stats'].get('last_update_time')
                    if last_update:
                        time_since_update = datetime.now() - last_update
                        if time_since_update > timedelta(minutes=5):
                            unhealthy_components.append('market_data_stale')
                
                # Report unhealthy components
                if unhealthy_components:
                    self.logger.warning(f"Unhealthy components detected: {unhealthy_components}")
                    self.state.is_healthy = False
                    
                    if self.notification_manager:
                        await self.notification_manager.send_alert(
                            f"Health check failed: {', '.join(unhealthy_components)}", "WARNING"
                        )
                else:
                    self.state.is_healthy = True
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Health monitor loop error: {e}")
    
    async def _performance_tracker_loop(self) -> None:
        """Background performance tracking."""
        try:
            while self.is_running and not self.shutdown_event.is_set():
                await asyncio.sleep(300)  # Update every 5 minutes
                
                # Update performance metrics
                if self.portfolio_manager:
                    portfolio_summary = self.portfolio_manager.get_portfolio_summary()
                    self.state.performance_metrics['total_pnl'] = portfolio_summary['pnl']['net_pnl']
                
                # Log performance summary
                self.logger.info(f"Performance update - "
                               f"Trades: {self.state.performance_metrics['total_trades']}, "
                               f"P&L: {self.state.performance_metrics['total_pnl']:.4f}, "
                               f"Uptime: {self.state.performance_metrics['uptime_seconds']:.0f}s")
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Performance tracker loop error: {e}")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _setup_error_recovery(self) -> None:
        """Set up error recovery callbacks for all components."""
        try:
            # Register recovery callbacks for each component
            self.error_recovery_manager.register_component_recovery(
                'market_manager', self._recover_market_manager
            )
            
            self.error_recovery_manager.register_component_recovery(
                'trade_manager', self._recover_trade_manager
            )
            
            self.error_recovery_manager.register_component_recovery(
                'binance_client', self._recover_binance_client
            )
            
            self.error_recovery_manager.register_component_recovery(
                'notification_manager', self._recover_notification_manager
            )
            
            self.logger.info("Error recovery callbacks registered")
            
        except Exception as e:
            self.logger.error(f"Failed to setup error recovery: {e}")
            raise
    
    async def _attempt_state_recovery(self) -> None:
        """Attempt to recover from previous application state."""
        try:
            saved_state = self.error_recovery_manager.load_application_state()
            
            if saved_state:
                self.logger.info(f"Found saved state from {saved_state.timestamp}")
                
                # Check if state is recent enough to be useful (within last hour)
                if datetime.now() - saved_state.timestamp < timedelta(hours=1):
                    self.logger.info("Attempting state recovery...")
                    
                    # Recover portfolio positions if available
                    if saved_state.active_positions and self.portfolio_manager:
                        for symbol, position_data in saved_state.active_positions.items():
                            try:
                                self.portfolio_manager.add_position(
                                    symbol=symbol,
                                    quantity=position_data.get('quantity', 0),
                                    entry_price=position_data.get('entry_price', 0),
                                    strategy_name=position_data.get('strategy_name', 'recovered'),
                                    trade_id=position_data.get('trade_id', f'recovered_{symbol}'),
                                    entry_time=datetime.fromisoformat(position_data.get('entry_time', datetime.now().isoformat()))
                                )
                                self.logger.info(f"Recovered position: {symbol}")
                            except Exception as e:
                                self.logger.warning(f"Failed to recover position {symbol}: {e}")
                    
                    # Update performance metrics
                    if saved_state.performance_metrics:
                        self.state.performance_metrics.update(saved_state.performance_metrics)
                    
                    self.logger.info("State recovery completed")
                else:
                    self.logger.info("Saved state too old, starting fresh")
            else:
                self.logger.info("No saved state found, starting fresh")
                
        except Exception as e:
            self.logger.error(f"Error during state recovery: {e}")
            # Continue startup even if recovery fails
    
    async def _save_current_state(self) -> None:
        """Save current application state for recovery."""
        try:
            # Collect component states
            component_states = {}
            
            if self.market_manager:
                component_states['market_manager'] = {
                    'is_running': self.market_manager.is_running,
                    'subscribed_symbols': list(self.market_manager.subscribed_symbols)
                }
            
            if self.trade_manager:
                component_states['trade_manager'] = {
                    'active_positions': len(self.trade_manager.positions),
                    'pending_orders': len(self.trade_manager.pending_orders)
                }
            
            # Collect active positions
            active_positions = {}
            if self.portfolio_manager:
                for symbol, position in self.portfolio_manager.active_positions.items():
                    active_positions[symbol] = {
                        'quantity': position.quantity,
                        'entry_price': position.entry_price,
                        'entry_time': position.entry_time.isoformat(),
                        'strategy_name': position.strategy_name,
                        'trade_id': position.trade_id
                    }
            
            # Collect portfolio metrics
            portfolio_metrics = {}
            if self.portfolio_manager:
                portfolio_summary = self.portfolio_manager.get_portfolio_summary()
                portfolio_metrics = portfolio_summary
            
            # Save state
            success = self.error_recovery_manager.save_application_state(
                component_states=component_states,
                active_positions=active_positions,
                portfolio_metrics=portfolio_metrics,
                configuration=self.config_dict,
                performance_metrics=self.state.performance_metrics
            )
            
            if success:
                self.logger.debug("Application state saved successfully")
            else:
                self.logger.warning("Failed to save application state")
                
        except Exception as e:
            self.logger.error(f"Error saving application state: {e}")
    
    # Component recovery methods
    async def _recover_market_manager(self) -> None:
        """Recover market manager component."""
        try:
            self.logger.info("Recovering Market Manager...")
            
            if self.market_manager:
                await self.market_manager.stop()
            
            # Reinitialize market manager
            self.market_manager = MarketManager(self.config_dict)
            
            if await self.market_manager.start():
                # Re-establish connections
                self.market_manager.add_data_handler(self._handle_market_data)
                self.market_manager.add_data_handler(self._handle_portfolio_price_update)
                self.market_manager.add_error_handler(self._handle_market_error)
                
                self.state.component_status['market_manager'] = 'recovered'
                self.logger.info("Market Manager recovered successfully")
            else:
                raise Exception("Failed to restart Market Manager")
                
        except Exception as e:
            self.logger.error(f"Market Manager recovery failed: {e}")
            raise
    
    async def _recover_trade_manager(self) -> None:
        """Recover trade manager component."""
        try:
            self.logger.info("Recovering Trade Manager...")
            
            # Reinitialize trade manager
            self.trade_manager = TradeManager(
                config=self.config_dict,
                binance_client=self.binance_client
            )
            
            self.state.component_status['trade_manager'] = 'recovered'
            self.logger.info("Trade Manager recovered successfully")
            
        except Exception as e:
            self.logger.error(f"Trade Manager recovery failed: {e}")
            raise
    
    async def _recover_binance_client(self) -> None:
        """Recover Binance client connection."""
        try:
            self.logger.info("Recovering Binance client...")
            
            if self.binance_client:
                await self.binance_client.disconnect()
            
            # Reinitialize Binance client
            api_key, api_secret = self.config_manager.get_api_credentials()
            self.binance_client = BinanceRestClient(api_key, api_secret, self.config.testnet)
            
            if await self.binance_client.connect():
                self.state.component_status['binance_client'] = 'recovered'
                self.logger.info("Binance client recovered successfully")
            else:
                raise Exception("Failed to reconnect Binance client")
                
        except Exception as e:
            self.logger.error(f"Binance client recovery failed: {e}")
            raise
    
    async def _recover_notification_manager(self) -> None:
        """Recover notification manager component."""
        try:
            self.logger.info("Recovering Notification Manager...")
            
            if self.notification_manager:
                await self.notification_manager.stop()
            
            # Reinitialize notification manager
            self.notification_manager = NotificationManager(self.config.notification_config)
            await self.notification_manager.start()
            
            self.state.component_status['notification_manager'] = 'recovered'
            self.logger.info("Notification Manager recovered successfully")
            
        except Exception as e:
            self.logger.error(f"Notification Manager recovery failed: {e}")
            raise
    
    async def _enhanced_health_monitor_loop(self) -> None:
        """Enhanced health monitoring with error recovery integration."""
        try:
            while self.is_running and not self.shutdown_event.is_set():
                await asyncio.sleep(60)  # Check every minute
                
                # Perform comprehensive health check
                health_checks = {
                    'market_manager': lambda: self.market_manager and self.market_manager.is_running,
                    'binance_client': lambda: self.binance_client and hasattr(self.binance_client, 'is_connected') and self.binance_client.is_connected,
                    'portfolio_manager': lambda: self.portfolio_manager is not None,
                    'trade_manager': lambda: self.trade_manager is not None,
                    'notification_manager': lambda: self.notification_manager is not None
                }
                
                health_results = await self.error_recovery_manager.perform_health_check(health_checks)
                
                # Handle unhealthy components
                unhealthy_components = [comp for comp, healthy in health_results.items() if not healthy]
                
                if unhealthy_components:
                    self.logger.warning(f"Unhealthy components detected: {unhealthy_components}")
                    self.state.is_healthy = False
                    
                    # Trigger error recovery for unhealthy components
                    for component in unhealthy_components:
                        await self.error_recovery_manager.handle_error(
                            Exception(f"Health check failed for {component}"),
                            component,
                            {'health_check': True}
                        )
                    
                    if self.notification_manager:
                        await self.notification_manager.send_alert(
                            f"Health check failed: {', '.join(unhealthy_components)}", "WARNING"
                        )
                else:
                    self.state.is_healthy = True
                
                # Save current state periodically
                await self._save_current_state()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Enhanced health monitor loop error: {e}")
            await self.error_recovery_manager.handle_error(e, 'health_monitor', {})
    
    def get_application_status(self) -> Dict[str, Any]:
        """Get comprehensive application status including error recovery metrics."""
        status = self.state.get_status()
        
        # Add component-specific status
        if self.portfolio_manager:
            status['portfolio'] = self.portfolio_manager.get_portfolio_summary()
        
        if self.strategy_manager:
            status['strategies'] = self.strategy_manager.get_manager_performance()
        
        if self.market_manager:
            status['market_data'] = self.market_manager.get_stats()
        
        # Add error recovery statistics
        status['error_recovery'] = self.error_recovery_manager.get_error_statistics()
        
        # Add component health status
        status['component_health'] = {}
        for component in ['market_manager', 'trade_manager', 'binance_client', 'notification_manager']:
            status['component_health'][component] = self.error_recovery_manager.get_component_health(component)
        
        return status


async def main():
    """Main entry point."""
    app = TradingBotApplication()
    try:
        await app.start()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        await app.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)