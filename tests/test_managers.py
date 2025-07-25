"""
Unit tests for manager components.

Tests all manager classes including risk manager, portfolio manager,
trade manager, strategy manager, and notification manager.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

from crypto_trading_bot.managers.risk_manager import RiskManager, PositionRisk, PortfolioRisk, RiskLevel
from crypto_trading_bot.managers.portfolio_manager import PortfolioManager
from crypto_trading_bot.managers.trade_manager import TradeManager
from crypto_trading_bot.managers.strategy_manager import StrategyManager
from crypto_trading_bot.managers.notification_manager import NotificationManager
from crypto_trading_bot.models.trading import TradingSignal, MarketData, SignalAction, Trade, Position
from crypto_trading_bot.models.config import RiskConfig, NotificationConfig
from tests.test_mock_data import MockDataGenerator, create_mock_config, create_mock_trading_signal


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_generator = MockDataGenerator()
        self.risk_config = self.mock_generator.generate_risk_config()
        self.risk_manager = RiskManager(self.risk_config)
        
    def test_initialization(self):
        """Test risk manager initialization."""
        self.assertIsInstance(self.risk_manager.portfolio, PortfolioRisk)
        self.assertEqual(self.risk_manager.max_position_size, self.risk_config.max_position_size)
        self.assertFalse(self.risk_manager.emergency_stop_active)
        
    def test_validate_trade_success(self):
        """Test successful trade validation."""
        signal = create_mock_trading_signal(action=SignalAction.BUY, confidence=0.8)
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        
        # Mock position size calculation
        with patch.object(self.risk_manager, 'calculate_position_size', return_value=0.1):
            with patch.object(self.risk_manager, '_check_portfolio_risk_limits', return_value=True):
                with patch.object(self.risk_manager, '_check_daily_loss_limits', return_value=True):
                    with patch.object(self.risk_manager, '_check_drawdown_limits', return_value=True):
                        with patch.object(self.risk_manager, '_check_leverage_limits', return_value=True):
                            with patch.object(self.risk_manager, '_check_correlation_limits', return_value=True):
                                is_valid, reason, position_size = self.risk_manager.validate_trade(signal, market_data)
        
        self.assertTrue(is_valid)
        self.assertEqual(reason, "Trade validated")
        self.assertEqual(position_size, 0.1)
    
    def test_validate_trade_emergency_stop(self):
        """Test trade validation with emergency stop active."""
        self.risk_manager.emergency_stop_active = True
        self.risk_manager.emergency_stop_reason = "Test emergency"
        
        signal = create_mock_trading_signal()
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        
        is_valid, reason, position_size = self.risk_manager.validate_trade(signal, market_data)
        
        self.assertFalse(is_valid)
        self.assertIn("Emergency stop active", reason)
        self.assertEqual(position_size, 0.0)
    
    def test_validate_trade_zero_position_size(self):
        """Test trade validation with zero position size."""
        signal = create_mock_trading_signal()
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        
        with patch.object(self.risk_manager, 'calculate_position_size', return_value=0.0):
            is_valid, reason, position_size = self.risk_manager.validate_trade(signal, market_data)
        
        self.assertFalse(is_valid)
        self.assertEqual(reason, "Position size calculated as zero")
        self.assertEqual(position_size, 0.0)
    
    def test_calculate_fixed_risk_size(self):
        """Test fixed risk position size calculation."""
        signal = create_mock_trading_signal(action=SignalAction.BUY)
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        market_data.price = 50000
        
        # Mock stop loss estimation
        with patch.object(self.risk_manager, '_estimate_stop_loss', return_value=49000):
            position_size = self.risk_manager._calculate_fixed_risk_size(signal, market_data)
        
        self.assertGreater(position_size, 0)
        self.assertIsInstance(position_size, float)
    
    def test_calculate_volatility_adjusted_size(self):
        """Test volatility adjusted position size calculation."""
        signal = create_mock_trading_signal()
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        
        with patch.object(self.risk_manager, '_calculate_fixed_risk_size', return_value=0.1):
            with patch.object(self.risk_manager, '_get_volatility', return_value=0.02):
                position_size = self.risk_manager._calculate_volatility_adjusted_size(signal, market_data)
        
        self.assertGreater(position_size, 0)
        self.assertIsInstance(position_size, float)
    
    def test_emergency_stop_activation(self):
        """Test emergency stop activation."""
        reason = "Test emergency stop"
        self.risk_manager.activate_emergency_stop(reason)
        
        self.assertTrue(self.risk_manager.emergency_stop_active)
        self.assertEqual(self.risk_manager.emergency_stop_reason, reason)
        self.assertIsInstance(self.risk_manager.emergency_stop_time, datetime)
    
    def test_emergency_stop_deactivation(self):
        """Test emergency stop deactivation."""
        self.risk_manager.activate_emergency_stop("Test")
        self.risk_manager.deactivate_emergency_stop()
        
        self.assertFalse(self.risk_manager.emergency_stop_active)
        self.assertIsNone(self.risk_manager.emergency_stop_reason)


class TestPositionRisk(unittest.TestCase):
    """Test cases for PositionRisk class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.position = PositionRisk(
            symbol="BTCUSDT",
            entry_price=50000,
            quantity=0.1,
            stop_loss=49000,
            take_profit=52000
        )
    
    def test_initialization(self):
        """Test position risk initialization."""
        self.assertEqual(self.position.symbol, "BTCUSDT")
        self.assertEqual(self.position.entry_price, 50000)
        self.assertEqual(self.position.quantity, 0.1)
        self.assertEqual(self.position.stop_loss, 49000)
        self.assertEqual(self.position.take_profit, 52000)
        self.assertGreater(self.position.risk_amount, 0)
        self.assertGreater(self.position.risk_reward_ratio, 0)
    
    def test_price_update(self):
        """Test position price update."""
        new_price = 51000
        self.position.update_price(new_price)
        
        self.assertEqual(self.position.current_price, new_price)
        self.assertGreater(self.position.unrealized_pnl, 0)  # Profit for long position
        self.assertGreater(self.position.unrealized_pnl_pct, 0)
    
    def test_stop_loss_check(self):
        """Test stop loss condition check."""
        # Price above stop loss - should not stop out
        self.position.update_price(49500)
        self.assertFalse(self.position.should_stop_out())
        
        # Price at stop loss - should stop out
        self.position.update_price(49000)
        self.assertTrue(self.position.should_stop_out())
        
        # Price below stop loss - should stop out
        self.position.update_price(48500)
        self.assertTrue(self.position.should_stop_out())
    
    def test_take_profit_check(self):
        """Test take profit condition check."""
        # Price below take profit - should not take profit
        self.position.update_price(51500)
        self.assertFalse(self.position.should_take_profit())
        
        # Price at take profit - should take profit
        self.position.update_price(52000)
        self.assertTrue(self.position.should_take_profit())
        
        # Price above take profit - should take profit
        self.position.update_price(52500)
        self.assertTrue(self.position.should_take_profit())
    
    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        metrics = self.position.get_risk_metrics()
        
        required_keys = [
            'symbol', 'entry_price', 'current_price', 'quantity',
            'position_value', 'unrealized_pnl', 'risk_amount',
            'risk_reward_ratio', 'should_stop_out', 'should_take_profit'
        ]
        
        for key in required_keys:
            self.assertIn(key, metrics)


class TestPortfolioRisk(unittest.TestCase):
    """Test cases for PortfolioRisk class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.portfolio = PortfolioRisk(initial_capital=10000)
    
    def test_initialization(self):
        """Test portfolio risk initialization."""
        self.assertEqual(self.portfolio.initial_capital, 10000)
        self.assertEqual(self.portfolio.current_capital, 10000)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(self.portfolio.total_exposure, 0)
    
    def test_add_position(self):
        """Test adding position to portfolio."""
        position = PositionRisk("BTCUSDT", 50000, 0.1)
        self.portfolio.add_position(position)
        
        self.assertIn("BTCUSDT", self.portfolio.positions)
        self.assertGreater(self.portfolio.total_exposure, 0)
    
    def test_remove_position(self):
        """Test removing position from portfolio."""
        position = PositionRisk("BTCUSDT", 50000, 0.1)
        self.portfolio.add_position(position)
        
        realized_pnl = 100
        self.portfolio.remove_position("BTCUSDT", realized_pnl)
        
        self.assertNotIn("BTCUSDT", self.portfolio.positions)
        self.assertEqual(self.portfolio.realized_pnl, realized_pnl)
        self.assertEqual(self.portfolio.current_capital, 10000 + realized_pnl)
    
    def test_update_positions(self):
        """Test updating positions with market prices."""
        position = PositionRisk("BTCUSDT", 50000, 0.1)
        self.portfolio.add_position(position)
        
        market_prices = {"BTCUSDT": 51000}
        self.portfolio.update_positions(market_prices)
        
        self.assertEqual(position.current_price, 51000)
        self.assertGreater(self.portfolio.unrealized_pnl, 0)
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        # Add profitable position
        position = PositionRisk("BTCUSDT", 50000, 0.1)
        self.portfolio.add_position(position)
        
        # Update with profit
        self.portfolio.update_positions({"BTCUSDT": 52000})
        initial_peak = self.portfolio.peak_capital
        
        # Update with loss
        self.portfolio.update_positions({"BTCUSDT": 48000})
        
        self.assertGreater(self.portfolio.current_drawdown, 0)
        self.assertGreater(self.portfolio.drawdown_duration, 0)
    
    def test_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        metrics = self.portfolio.get_portfolio_metrics()
        
        required_keys = [
            'initial_capital', 'current_capital', 'portfolio_value',
            'total_positions', 'gross_exposure', 'net_exposure',
            'leverage', 'realized_pnl', 'unrealized_pnl', 'total_pnl',
            'current_drawdown', 'max_drawdown'
        ]
        
        for key in required_keys:
            self.assertIn(key, metrics)


class TestPortfolioManager(unittest.TestCase):
    """Test cases for PortfolioManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = create_mock_config().to_dict()
        self.portfolio_manager = PortfolioManager(self.config, initial_capital=10000)
        self.mock_generator = MockDataGenerator()
    
    def test_initialization(self):
        """Test portfolio manager initialization."""
        self.assertEqual(self.portfolio_manager.initial_capital, 10000)
        self.assertEqual(len(self.portfolio_manager.active_positions), 0)
        self.assertEqual(len(self.portfolio_manager.trade_history), 0)
    
    def test_add_position(self):
        """Test adding position to portfolio."""
        success = self.portfolio_manager.add_position(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000,
            strategy_name="test_strategy",
            trade_id="test_trade_1"
        )
        
        self.assertTrue(success)
        self.assertIn("BTCUSDT", self.portfolio_manager.active_positions)
        
        position = self.portfolio_manager.active_positions["BTCUSDT"]
        self.assertEqual(position.quantity, 0.1)
        self.assertEqual(position.entry_price, 50000)
    
    def test_close_position(self):
        """Test closing position."""
        # Add position first
        self.portfolio_manager.add_position(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000,
            strategy_name="test_strategy",
            trade_id="test_trade_1"
        )
        
        # Close position
        success, pnl = self.portfolio_manager.close_position(
            symbol="BTCUSDT",
            exit_price=51000,
            reason="Test close"
        )
        
        self.assertTrue(success)
        self.assertGreater(pnl, 0)  # Should be profitable
        self.assertNotIn("BTCUSDT", self.portfolio_manager.active_positions)
        self.assertEqual(len(self.portfolio_manager.trade_history), 1)
    
    def test_update_market_data(self):
        """Test updating market data."""
        # Add position
        self.portfolio_manager.add_position(
            symbol="BTCUSDT",
            quantity=0.1,
            entry_price=50000,
            strategy_name="test_strategy",
            trade_id="test_trade_1"
        )
        
        # Update market data
        market_data = {"BTCUSDT": self.mock_generator.generate_market_data(count=1)[0]}
        market_data["BTCUSDT"].price = 51000
        
        self.portfolio_manager.update_market_data(market_data)
        
        position = self.portfolio_manager.active_positions["BTCUSDT"]
        self.assertEqual(position.current_price, 51000)
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        summary = self.portfolio_manager.get_portfolio_summary()
        
        required_keys = ['positions', 'pnl', 'performance', 'risk_metrics']
        for key in required_keys:
            self.assertIn(key, summary)
        
        self.assertIsInstance(summary['positions']['active_count'], int)
        self.assertIsInstance(summary['pnl']['net_pnl'], (int, float))


class TestTradeManager(unittest.TestCase):
    """Test cases for TradeManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = create_mock_config().to_dict()
        self.mock_binance_client = Mock()
        self.trade_manager = TradeManager(self.config, self.mock_binance_client)
        self.mock_generator = MockDataGenerator()
    
    def test_initialization(self):
        """Test trade manager initialization."""
        self.assertEqual(self.trade_manager.binance_client, self.mock_binance_client)
        self.assertEqual(len(self.trade_manager.positions), 0)
        self.assertEqual(len(self.trade_manager.pending_orders), 0)
    
    @patch('crypto_trading_bot.managers.trade_manager.TradeManager._place_market_order')
    def test_execute_trade_success(self, mock_place_order):
        """Test successful trade execution."""
        # Mock successful order placement
        mock_place_order.return_value = (True, "Order placed", "order_123")
        
        signal = create_mock_trading_signal(action=SignalAction.BUY)
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        position_size = 0.1
        
        success, message, order_id = self.trade_manager.execute_trade(signal, position_size, market_data)
        
        self.assertTrue(success)
        self.assertEqual(message, "Order placed")
        self.assertEqual(order_id, "order_123")
        mock_place_order.assert_called_once()
    
    @patch('crypto_trading_bot.managers.trade_manager.TradeManager._place_market_order')
    def test_execute_trade_failure(self, mock_place_order):
        """Test failed trade execution."""
        # Mock failed order placement
        mock_place_order.return_value = (False, "Order failed", None)
        
        signal = create_mock_trading_signal()
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        position_size = 0.1
        
        success, message, order_id = self.trade_manager.execute_trade(signal, position_size, market_data)
        
        self.assertFalse(success)
        self.assertEqual(message, "Order failed")
        self.assertIsNone(order_id)
    
    def test_position_tracking(self):
        """Test position tracking functionality."""
        # Simulate adding a position
        position_data = {
            'symbol': 'BTCUSDT',
            'quantity': 0.1,
            'entry_price': 50000,
            'side': 'BUY',
            'timestamp': datetime.now()
        }
        
        self.trade_manager.positions['BTCUSDT'] = position_data
        
        self.assertIn('BTCUSDT', self.trade_manager.positions)
        self.assertEqual(self.trade_manager.positions['BTCUSDT']['quantity'], 0.1)
    
    def test_order_status_tracking(self):
        """Test order status tracking."""
        order_data = {
            'order_id': 'order_123',
            'symbol': 'BTCUSDT',
            'status': 'PENDING',
            'timestamp': datetime.now()
        }
        
        self.trade_manager.pending_orders['order_123'] = order_data
        
        self.assertIn('order_123', self.trade_manager.pending_orders)
        self.assertEqual(self.trade_manager.pending_orders['order_123']['status'], 'PENDING')


class TestStrategyManager(unittest.TestCase):
    """Test cases for StrategyManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = create_mock_config().to_dict()
        self.strategy_manager = StrategyManager(self.config)
        self.mock_generator = MockDataGenerator()
    
    def test_initialization(self):
        """Test strategy manager initialization."""
        self.assertIsInstance(self.strategy_manager.strategies, dict)
        self.assertGreater(len(self.strategy_manager.strategies), 0)
    
    def test_analyze_market(self):
        """Test market analysis with strategies."""
        market_data = self.mock_generator.generate_market_data(count=1)[0]
        
        # Mock strategy analysis
        with patch.object(self.strategy_manager, '_get_best_signal') as mock_get_best:
            mock_signal = create_mock_trading_signal()
            mock_get_best.return_value = mock_signal
            
            signal = self.strategy_manager.analyze_market(market_data)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal, mock_signal)
    
    def test_strategy_performance_tracking(self):
        """Test strategy performance tracking."""
        trade = self.mock_generator.generate_trade()
        trade.strategy = "test_strategy"
        
        # Mock strategy exists
        mock_strategy = Mock()
        mock_strategy.update_performance = Mock()
        self.strategy_manager.strategies["test_strategy"] = mock_strategy
        
        self.strategy_manager.update_strategy_performance(trade)
        
        mock_strategy.update_performance.assert_called_once_with(trade)
    
    def test_get_manager_performance(self):
        """Test getting manager performance metrics."""
        performance = self.strategy_manager.get_manager_performance()
        
        self.assertIsInstance(performance, dict)
        self.assertIn('strategies', performance)
        self.assertIn('overall_stats', performance)


class TestNotificationManager(unittest.TestCase):
    """Test cases for NotificationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.notification_config = NotificationConfig(
            email_enabled=False,
            webhook_enabled=False,
            console_enabled=True,
            email_settings={},
            webhook_settings={},
            alert_levels=['ERROR', 'WARNING', 'INFO']
        )
        self.notification_manager = NotificationManager(self.notification_config)
    
    def test_initialization(self):
        """Test notification manager initialization."""
        self.assertEqual(self.notification_manager.config, self.notification_config)
        self.assertFalse(self.notification_manager.email_enabled)
        self.assertFalse(self.notification_manager.webhook_enabled)
        self.assertTrue(self.notification_manager.console_enabled)
    
    @patch('builtins.print')
    async def test_send_console_notification(self, mock_print):
        """Test sending console notification."""
        await self.notification_manager.send_notification("Test message", "INFO")
        mock_print.assert_called()
    
    async def test_send_trade_notification(self):
        """Test sending trade notification."""
        with patch.object(self.notification_manager, 'send_notification') as mock_send:
            await self.notification_manager.send_trade_notification("Trade executed")
            mock_send.assert_called_once_with("Trade executed", "INFO")
    
    async def test_send_alert(self):
        """Test sending alert notification."""
        with patch.object(self.notification_manager, 'send_notification') as mock_send:
            await self.notification_manager.send_alert("Error occurred", "ERROR")
            mock_send.assert_called_once_with("Error occurred", "ERROR")
    
    async def test_start_stop(self):
        """Test starting and stopping notification manager."""
        await self.notification_manager.start()
        self.assertTrue(self.notification_manager.is_running)
        
        await self.notification_manager.stop()
        self.assertFalse(self.notification_manager.is_running)
    
    def test_alert_level_filtering(self):
        """Test alert level filtering."""
        # Test with restricted alert levels
        restricted_config = NotificationConfig(
            console_enabled=True,
            alert_levels=['ERROR']
        )
        restricted_manager = NotificationManager(restricted_config)
        
        # Should filter out non-ERROR alerts
        self.assertTrue(restricted_manager._should_send_alert("ERROR"))
        self.assertFalse(restricted_manager._should_send_alert("INFO"))
        self.assertFalse(restricted_manager._should_send_alert("WARNING"))


if __name__ == '__main__':
    # Run async tests
    def run_async_test(test_func):
        """Helper to run async test functions."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_func())
        finally:
            loop.close()
    
    unittest.main()