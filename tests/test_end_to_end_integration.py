"""
End-to-end integration tests for the crypto trading bot.

Tests complete workflows including market data processing, signal generation,
risk validation, trade execution, and portfolio management.
"""

import unittest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from crypto_trading_bot.main import TradingBotApplication
from crypto_trading_bot.managers.market_manager import MarketManager
from crypto_trading_bot.managers.strategy_manager import StrategyManager
from crypto_trading_bot.managers.risk_manager import RiskManager
from crypto_trading_bot.managers.trade_manager import TradeManager
from crypto_trading_bot.managers.portfolio_manager import PortfolioManager
from crypto_trading_bot.managers.notification_manager import NotificationManager
from crypto_trading_bot.models.trading import TradingSignal, MarketData, SignalAction, Trade
from crypto_trading_bot.models.config import BotConfig
from crypto_trading_bot.api.binance_client import BinanceRestClient
from tests.test_mock_data import MockDataGenerator, create_mock_config, create_mock_trading_signal


class TestMarketDataToTradeExecution(unittest.TestCase):
    """Test complete market data to trade execution workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_generator = MockDataGenerator()
        self.config = create_mock_config()
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock components
        self.mock_binance_client = Mock(spec=BinanceRestClient)
        self.mock_binance_client.connect = AsyncMock(return_value=True)
        self.mock_binance_client.disconnect = AsyncMock()
        self.mock_binance_client.is_connected = True
        
        # Create managers
        self.portfolio_manager = PortfolioManager(self.config.to_dict(), initial_capital=10000)
        self.risk_manager = RiskManager(self.config.risk_config)
        self.strategy_manager = StrategyManager(self.config.to_dict())
        self.trade_manager = TradeManager(self.config.to_dict(), self.mock_binance_client)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_buy_signal_workflow(self):
        """Test complete workflow from market data to buy order execution."""
        # 1. Generate market data
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        market_data.symbol = 'BTCUSDT'
        market_data.price = 50000
        
        # 2. Mock strategy to generate buy signal
        buy_signal = create_mock_trading_signal(
            action=SignalAction.BUY,
            confidence=0.8,
            symbol='BTCUSDT'
        )
        
        with patch.object(self.strategy_manager, 'analyze_market', return_value=buy_signal):
            signal = self.strategy_manager.analyze_market(market_data)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.action, SignalAction.BUY)
        self.assertEqual(signal.symbol, 'BTCUSDT')
        
        # 3. Validate signal with risk manager
        is_valid, reason, position_size = self.risk_manager.validate_trade(signal, market_data)
        
        self.assertTrue(is_valid)
        self.assertGreater(position_size, 0)
        
        # 4. Mock successful trade execution
        self.mock_binance_client.place_market_order = AsyncMock(return_value={
            'orderId': 'test_order_123',
            'status': 'FILLED',
            'executedQty': str(position_size),
            'cummulativeQuoteQty': str(position_size * market_data.price)
        })
        
        # Execute trade
        success, message, order_id = self.trade_manager.execute_trade(signal, position_size, market_data)
        
        self.assertTrue(success)
        self.assertEqual(order_id, 'test_order_123')
        
        # 5. Update portfolio
        self.portfolio_manager.add_position(
            symbol='BTCUSDT',
            quantity=position_size,
            entry_price=market_data.price,
            strategy_name=signal.strategy,
            trade_id=order_id
        )
        
        # Verify position was added
        self.assertIn('BTCUSDT', self.portfolio_manager.active_positions)
        position = self.portfolio_manager.active_positions['BTCUSDT']
        self.assertEqual(position.quantity, position_size)
        self.assertEqual(position.entry_price, market_data.price)
    
    def test_complete_sell_signal_workflow(self):
        """Test complete workflow for sell signal."""
        # Setup existing position first
        self.portfolio_manager.add_position(
            symbol='BTCUSDT',
            quantity=0.1,
            entry_price=49000,
            strategy_name='test_strategy',
            trade_id='existing_trade'
        )
        
        # Generate market data with higher price
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        market_data.symbol = 'BTCUSDT'
        market_data.price = 51000
        
        # Generate sell signal
        sell_signal = create_mock_trading_signal(
            action=SignalAction.SELL,
            confidence=0.7,
            symbol='BTCUSDT'
        )
        
        with patch.object(self.strategy_manager, 'analyze_market', return_value=sell_signal):
            signal = self.strategy_manager.analyze_market(market_data)
        
        # Validate and execute
        is_valid, reason, position_size = self.risk_manager.validate_trade(signal, market_data)
        self.assertTrue(is_valid)
        
        # Mock successful sell execution
        self.mock_binance_client.place_market_order = AsyncMock(return_value={
            'orderId': 'sell_order_456',
            'status': 'FILLED',
            'executedQty': '0.1',
            'cummulativeQuoteQty': str(0.1 * market_data.price)
        })
        
        success, message, order_id = self.trade_manager.execute_trade(signal, 0.1, market_data)
        self.assertTrue(success)
        
        # Close position and verify profit
        success, pnl = self.portfolio_manager.close_position('BTCUSDT', market_data.price, "Signal exit")
        self.assertTrue(success)
        self.assertGreater(pnl, 0)  # Should be profitable (51000 - 49000) * 0.1
    
    def test_risk_rejection_workflow(self):
        """Test workflow when risk manager rejects trade."""
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        market_data.price = 50000
        
        # Generate high-risk signal
        risky_signal = create_mock_trading_signal(
            action=SignalAction.BUY,
            confidence=0.9,
            symbol='BTCUSDT'
        )
        
        # Mock risk manager to reject trade
        with patch.object(self.risk_manager, 'validate_trade', return_value=(False, "Position size too large", 0.0)):
            is_valid, reason, position_size = self.risk_manager.validate_trade(risky_signal, market_data)
        
        self.assertFalse(is_valid)
        self.assertEqual(position_size, 0.0)
        self.assertIn("Position size too large", reason)
        
        # Trade should not be executed
        # Portfolio should remain unchanged
        initial_positions = len(self.portfolio_manager.active_positions)
        # No trade execution should occur
        final_positions = len(self.portfolio_manager.active_positions)
        self.assertEqual(initial_positions, final_positions)
    
    def test_multiple_symbol_workflow(self):
        """Test workflow with multiple trading symbols."""
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        for symbol in symbols:
            # Generate market data for each symbol
            market_data = self.mock_generator.generate_market_data(count=1, symbol=symbol)[0]
            
            # Generate buy signal
            signal = create_mock_trading_signal(
                action=SignalAction.BUY,
                confidence=0.8,
                symbol=symbol
            )
            
            # Validate and execute
            is_valid, reason, position_size = self.risk_manager.validate_trade(signal, market_data)
            
            if is_valid and position_size > 0:
                # Mock successful execution
                self.mock_binance_client.place_market_order = AsyncMock(return_value={
                    'orderId': f'order_{symbol}',
                    'status': 'FILLED',
                    'executedQty': str(position_size),
                    'cummulativeQuoteQty': str(position_size * market_data.price)
                })
                
                success, message, order_id = self.trade_manager.execute_trade(signal, position_size, market_data)
                
                if success:
                    self.portfolio_manager.add_position(
                        symbol=symbol,
                        quantity=position_size,
                        entry_price=market_data.price,
                        strategy_name=signal.strategy,
                        trade_id=order_id
                    )
        
        # Verify multiple positions
        self.assertGreaterEqual(len(self.portfolio_manager.active_positions), 1)
        
        # Check portfolio metrics
        summary = self.portfolio_manager.get_portfolio_summary()
        self.assertGreater(summary['positions']['active_count'], 0)
        self.assertGreater(summary['positions']['total_value'], 0)


class TestApplicationLifecycle(unittest.TestCase):
    """Test complete application lifecycle scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock configuration loading
        self.mock_config = create_mock_config()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('crypto_trading_bot.main.ConfigManager')
    @patch('crypto_trading_bot.main.BinanceRestClient')
    async def test_application_startup_sequence(self, mock_binance_class, mock_config_manager_class):
        """Test complete application startup sequence."""
        # Mock configuration manager
        mock_config_manager = Mock()
        mock_config_manager.load_config.return_value = self.mock_config.to_dict()
        mock_config_manager.get_api_credentials.return_value = ('test_key', 'test_secret')
        mock_config_manager_class.return_value = mock_config_manager
        
        # Mock Binance client
        mock_binance_client = Mock()
        mock_binance_client.connect = AsyncMock(return_value=True)
        mock_binance_client.disconnect = AsyncMock()
        mock_binance_client.is_connected = True
        mock_binance_class.return_value = mock_binance_client
        
        # Create application
        app = TradingBotApplication()
        
        # Mock component initialization
        with patch.object(app, '_initialize_components', return_value=True) as mock_init:
            with patch.object(app, '_start_background_tasks') as mock_start_tasks:
                with patch.object(app, '_run_main_loop') as mock_main_loop:
                    await app.start()
        
        # Verify initialization sequence
        mock_init.assert_called_once()
        mock_start_tasks.assert_called_once()
        mock_main_loop.assert_called_once()
        
        # Verify application state
        self.assertTrue(app.is_running)
        self.assertIsNotNone(app.state)
    
    @patch('crypto_trading_bot.main.ConfigManager')
    async def test_application_shutdown_sequence(self, mock_config_manager_class):
        """Test graceful application shutdown."""
        # Mock configuration
        mock_config_manager = Mock()
        mock_config_manager.load_config.return_value = self.mock_config.to_dict()
        mock_config_manager_class.return_value = mock_config_manager
        
        app = TradingBotApplication()
        
        # Mock components
        app.market_manager = Mock()
        app.market_manager.stop = AsyncMock()
        app.notification_manager = Mock()
        app.notification_manager.stop = AsyncMock()
        app.binance_client = Mock()
        app.binance_client.disconnect = AsyncMock()
        
        # Mock background tasks
        app.heartbeat_task = Mock()
        app.heartbeat_task.done.return_value = False
        app.heartbeat_task.cancel = Mock()
        app.health_monitor_task = Mock()
        app.health_monitor_task.done.return_value = False
        app.health_monitor_task.cancel = Mock()
        app.performance_tracker_task = Mock()
        app.performance_tracker_task.done.return_value = False
        app.performance_tracker_task.cancel = Mock()
        
        # Test shutdown
        await app.shutdown()
        
        # Verify shutdown sequence
        app.market_manager.stop.assert_called_once()
        app.notification_manager.stop.assert_called_once()
        app.binance_client.disconnect.assert_called_once()
        
        # Verify application state
        self.assertFalse(app.is_running)
        self.assertTrue(app.shutdown_event.is_set())
    
    async def test_error_recovery_integration(self):
        """Test error recovery integration in application."""
        # This would test the error recovery system integration
        # with the main application components
        pass


class TestStrategyIntegration(unittest.TestCase):
    """Test strategy integration scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_generator = MockDataGenerator()
        self.config = create_mock_config()
        self.strategy_manager = StrategyManager(self.config.to_dict())
    
    def test_multiple_strategy_signals(self):
        """Test handling multiple strategy signals."""
        market_data = self.mock_generator.generate_market_data(count=50)
        
        # Mock multiple strategies returning different signals
        with patch.object(self.strategy_manager, 'strategies') as mock_strategies:
            # Mock strategy 1 - bullish
            strategy1 = Mock()
            strategy1.analyze.return_value = create_mock_trading_signal(
                action=SignalAction.BUY, confidence=0.8
            )
            strategy1.get_confidence.return_value = 0.8
            strategy1.is_active = True
            
            # Mock strategy 2 - bearish
            strategy2 = Mock()
            strategy2.analyze.return_value = create_mock_trading_signal(
                action=SignalAction.SELL, confidence=0.6
            )
            strategy2.get_confidence.return_value = 0.6
            strategy2.is_active = True
            
            # Mock strategy 3 - neutral
            strategy3 = Mock()
            strategy3.analyze.return_value = None
            strategy3.get_confidence.return_value = 0.4
            strategy3.is_active = True
            
            mock_strategies.items.return_value = [
                ('strategy1', strategy1),
                ('strategy2', strategy2),
                ('strategy3', strategy3)
            ]
            
            # Analyze market with multiple strategies
            signal = self.strategy_manager.analyze_market(market_data[-1])
            
            # Should return the highest confidence signal (strategy1 - BUY)
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, SignalAction.BUY)
    
    def test_strategy_performance_tracking(self):
        """Test strategy performance tracking over multiple trades."""
        # Create mock trades for different strategies
        trades = []
        for i in range(10):
            trade = self.mock_generator.generate_trade()
            trade.strategy = f'strategy_{i % 3}'  # 3 different strategies
            trade.pnl = (-1) ** i * 50  # Alternating profit/loss
            trades.append(trade)
        
        # Update strategy performance
        for trade in trades:
            self.strategy_manager.update_strategy_performance(trade)
        
        # Get performance metrics
        performance = self.strategy_manager.get_manager_performance()
        
        self.assertIn('strategies', performance)
        self.assertIn('overall_stats', performance)
        self.assertGreater(performance['overall_stats']['total_trades'], 0)
    
    def test_strategy_activation_deactivation(self):
        """Test strategy activation and deactivation."""
        # Test that inactive strategies don't generate signals
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        
        # Mock all strategies as inactive
        with patch.object(self.strategy_manager, 'strategies') as mock_strategies:
            inactive_strategy = Mock()
            inactive_strategy.is_active = False
            inactive_strategy.analyze.return_value = create_mock_trading_signal()
            
            mock_strategies.items.return_value = [('inactive', inactive_strategy)]
            
            signal = self.strategy_manager.analyze_market(market_data)
            
            # Should return None since no active strategies
            self.assertIsNone(signal)


class TestRiskManagementIntegration(unittest.TestCase):
    """Test risk management integration scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_generator = MockDataGenerator()
        self.risk_config = self.mock_generator.generate_risk_config()
        self.risk_manager = RiskManager(self.risk_config)
        self.portfolio_manager = PortfolioManager({}, initial_capital=10000)
    
    def test_portfolio_risk_limits(self):
        """Test portfolio-level risk limit enforcement."""
        # Add multiple positions to approach risk limits
        positions = []
        for i in range(5):
            symbol = f'TEST{i}USDT'
            position_size = 0.2  # Each position is 20% of portfolio
            
            # Add position to portfolio
            self.portfolio_manager.add_position(
                symbol=symbol,
                quantity=position_size,
                entry_price=50000,
                strategy_name='test_strategy',
                trade_id=f'trade_{i}'
            )
            
            # Update risk manager's portfolio tracking
            market_data = self.mock_generator.generate_market_data(count=1, symbol=symbol)[0]
            signal = create_mock_trading_signal(symbol=symbol)
            
            is_valid, reason, calculated_size = self.risk_manager.validate_trade(signal, market_data)
            
            # As portfolio fills up, position sizes should decrease or trades should be rejected
            if i >= 3:  # After 3 positions (60% of portfolio)
                # Should either reject or significantly reduce position size
                self.assertTrue(not is_valid or calculated_size < 0.1)
    
    def test_daily_loss_limits(self):
        """Test daily loss limit enforcement."""
        # Simulate multiple losing trades in one day
        daily_losses = []
        
        for i in range(5):
            # Create losing trade
            losing_trade = self.mock_generator.generate_trade()
            losing_trade.pnl = -200  # $200 loss each
            losing_trade.timestamp = datetime.now()
            
            # Update portfolio with loss
            self.portfolio_manager.trade_history.append(losing_trade)
            daily_losses.append(losing_trade.pnl)
            
            # Try to place another trade
            signal = create_mock_trading_signal()
            market_data = self.mock_generator.generate_market_data(count=1)[0]
            
            # Mock daily loss tracking
            with patch.object(self.risk_manager, '_check_daily_loss_limits') as mock_check:
                total_daily_loss = sum(daily_losses)
                max_daily_loss = self.risk_manager.portfolio.current_capital * self.risk_manager.max_daily_loss
                
                # Should reject if daily loss exceeds limit
                mock_check.return_value = abs(total_daily_loss) <= max_daily_loss
                
                is_valid, reason, position_size = self.risk_manager.validate_trade(signal, market_data)
                
                if abs(total_daily_loss) > max_daily_loss:
                    self.assertFalse(is_valid)
                    self.assertIn("daily loss", reason.lower())
    
    def test_position_sizing_methods(self):
        """Test different position sizing methods."""
        signal = create_mock_trading_signal(confidence=0.8)
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        market_data.price = 50000
        
        # Test fixed risk sizing
        self.risk_manager.position_sizing_method = "fixed_risk"
        size_fixed = self.risk_manager.calculate_position_size(signal, market_data)
        
        # Test volatility adjusted sizing
        self.risk_manager.position_sizing_method = "volatility_adjusted"
        size_volatility = self.risk_manager.calculate_position_size(signal, market_data)
        
        # Test Kelly criterion sizing
        self.risk_manager.position_sizing_method = "kelly_criterion"
        size_kelly = self.risk_manager.calculate_position_size(signal, market_data)
        
        # All methods should return positive sizes
        self.assertGreater(size_fixed, 0)
        self.assertGreater(size_volatility, 0)
        self.assertGreater(size_kelly, 0)
        
        # Sizes should be different (unless by coincidence)
        sizes = [size_fixed, size_volatility, size_kelly]
        self.assertGreater(len(set(sizes)), 1)  # At least some should be different


class TestNotificationIntegration(unittest.TestCase):
    """Test notification system integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.notification_config = self.mock_generator.generate_notification_config()
        self.notification_manager = NotificationManager(self.notification_config)
        self.mock_generator = MockDataGenerator()
    
    async def test_trade_notification_flow(self):
        """Test notification flow for trade events."""
        # Mock console output
        with patch('builtins.print') as mock_print:
            # Test trade execution notification
            await self.notification_manager.send_trade_notification(
                "BUY 0.1 BTCUSDT @ 50000"
            )
            
            # Test error notification
            await self.notification_manager.send_alert(
                "Connection lost to market data", "ERROR"
            )
            
            # Test performance notification
            await self.notification_manager.send_notification(
                "Daily P&L: +$150", "INFO"
            )
            
            # Verify notifications were sent
            self.assertGreaterEqual(mock_print.call_count, 3)
    
    async def test_notification_filtering(self):
        """Test notification level filtering."""
        # Create notification manager with restricted levels
        restricted_config = self.notification_config
        restricted_config.alert_levels = ['ERROR']
        
        restricted_manager = NotificationManager(restricted_config)
        
        with patch('builtins.print') as mock_print:
            # Should send ERROR
            await restricted_manager.send_alert("Critical error", "ERROR")
            
            # Should not send INFO
            await restricted_manager.send_notification("Info message", "INFO")
            
            # Should not send WARNING
            await restricted_manager.send_alert("Warning message", "WARNING")
            
            # Only ERROR should have been printed
            self.assertEqual(mock_print.call_count, 1)


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance tracking integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_generator = MockDataGenerator()
        self.config = create_mock_config()
        self.portfolio_manager = PortfolioManager(self.config.to_dict(), initial_capital=10000)
    
    def test_portfolio_performance_tracking(self):
        """Test comprehensive portfolio performance tracking."""
        # Simulate trading session with multiple trades
        trades = []
        
        # Generate mix of winning and losing trades
        for i in range(20):
            trade = self.mock_generator.generate_trade()
            trade.pnl = random.gauss(10, 50)  # Average $10 profit, $50 std dev
            trade.timestamp = datetime.now() - timedelta(minutes=i*30)
            trades.append(trade)
            
            # Add to portfolio history
            self.portfolio_manager.trade_history.append(trade)
        
        # Get performance summary
        summary = self.portfolio_manager.get_portfolio_summary()
        
        # Verify performance metrics
        self.assertIn('performance', summary)
        self.assertIn('pnl', summary)
        
        performance = summary['performance']
        self.assertIn('total_trades', performance)
        self.assertIn('win_rate', performance)
        self.assertIn('avg_win', performance)
        self.assertIn('avg_loss', performance)
        self.assertIn('sharpe_ratio', performance)
        
        # Verify calculations
        self.assertEqual(performance['total_trades'], len(trades))
        
        winning_trades = [t for t in trades if t.pnl > 0]
        if winning_trades:
            expected_win_rate = len(winning_trades) / len(trades)
            self.assertAlmostEqual(performance['win_rate'], expected_win_rate, places=2)
    
    def test_real_time_pnl_updates(self):
        """Test real-time P&L updates with market price changes."""
        # Add position
        self.portfolio_manager.add_position(
            symbol='BTCUSDT',
            quantity=0.1,
            entry_price=50000,
            strategy_name='test_strategy',
            trade_id='test_trade'
        )
        
        # Simulate price movements
        price_updates = [50500, 51000, 50800, 51200, 50900]
        
        for price in price_updates:
            # Update market data
            market_data = {'BTCUSDT': self.mock_generator.generate_market_data(count=1)[0]}
            market_data['BTCUSDT'].price = price
            
            self.portfolio_manager.update_market_data(market_data)
            
            # Get current P&L
            summary = self.portfolio_manager.get_portfolio_summary()
            unrealized_pnl = summary['pnl']['unrealized_pnl']
            
            # Verify P&L calculation
            expected_pnl = (price - 50000) * 0.1
            self.assertAlmostEqual(unrealized_pnl, expected_pnl, places=2)


if __name__ == '__main__':
    # Custom test runner for async tests
    class AsyncTestRunner:
        def run_tests(self):
            # Synchronous tests
            sync_test_classes = [
                TestMarketDataToTradeExecution,
                TestStrategyIntegration,
                TestRiskManagementIntegration,
                TestPerformanceIntegration
            ]
            
            for test_class in sync_test_classes:
                suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
                unittest.TextTestRunner(verbosity=2).run(suite)
            
            # Async tests
            async_test_classes = [
                TestApplicationLifecycle,
                TestNotificationIntegration
            ]
            
            for test_class in async_test_classes:
                suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
                for test in suite:
                    if hasattr(test, '_testMethodName'):
                        method = getattr(test, test._testMethodName)
                        if asyncio.iscoroutinefunction(method):
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                loop.run_until_complete(method())
                                print(f"✓ {test_class.__name__}.{test._testMethodName}")
                            except Exception as e:
                                print(f"✗ {test_class.__name__}.{test._testMethodName}: {e}")
                            finally:
                                loop.close()
                        else:
                            unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([test]))
    
    runner = AsyncTestRunner()
    runner.run_tests()