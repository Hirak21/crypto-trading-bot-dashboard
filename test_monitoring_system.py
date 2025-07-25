"""
Test script for the monitoring and notification system.

This script demonstrates the monitoring system capabilities including:
- System health monitoring
- Performance tracking
- Alert generation
- Connection monitoring
"""

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Dict, Any

from crypto_trading_bot.managers.monitoring_system import MonitoringSystem
from crypto_trading_bot.models.config import NotificationConfig
from crypto_trading_bot.models.trading import Trade, Position


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def simulate_api_connection() -> bool:
    """Simulate API connection check"""
    # Randomly simulate connection issues
    return random.random() > 0.1  # 90% success rate


async def simulate_websocket_connection() -> bool:
    """Simulate WebSocket connection check"""
    # Randomly simulate connection issues
    return random.random() > 0.05  # 95% success rate


async def test_monitoring_system():
    """Test the monitoring system with simulated data"""
    
    # Create notification configuration
    notification_config = NotificationConfig(
        enabled=True,
        console={'enabled': True, 'min_level': 'info'},
        email=None,  # Disable email for testing
        webhook=None,  # Disable webhook for testing
        trade_notifications=True,
        error_notifications=True,
        performance_notifications=True,
        system_notifications=True
    )
    
    # Initialize monitoring system
    monitoring = MonitoringSystem(notification_config)
    
    try:
        # Start monitoring
        logger.info("Starting monitoring system...")
        await monitoring.start()
        
        # Register connection checks
        monitoring.register_connection_check("binance_api", simulate_api_connection)
        monitoring.register_connection_check("websocket", simulate_websocket_connection)
        
        logger.info("Monitoring system started. Running simulation...")
        
        # Simulate trading activity
        for i in range(20):
            await asyncio.sleep(2)  # Wait 2 seconds between activities
            
            # Simulate API requests
            is_error = random.random() < 0.1  # 10% error rate
            response_time = random.uniform(50, 200)  # 50-200ms response time
            monitoring.record_api_request("binance_api", is_error, response_time)
            
            # Simulate signal generation
            strategy = random.choice(["liquidity", "momentum", "chart_patterns"])
            signal_time = random.uniform(10, 50)  # 10-50ms signal generation
            monitoring.record_signal_generation(strategy, signal_time)
            
            # Simulate market data
            symbol = random.choice(["BTCUSDT", "ETHUSDT"])
            price = random.uniform(30000, 50000)
            volume = random.uniform(100, 1000)
            monitoring.record_market_data(symbol, price, volume)
            
            # Simulate technical indicators
            indicators = {
                'rsi': random.uniform(20, 80),
                'bb_upper': price * 1.02,
                'bb_lower': price * 0.98,
                'price': price
            }
            monitoring.record_technical_indicators(symbol, indicators)
            
            # Occasionally simulate trades
            if random.random() < 0.3:  # 30% chance of trade
                trade = Trade(
                    id=f"trade_{i}",
                    symbol=symbol,
                    side="BUY" if random.random() > 0.5 else "SELL",
                    size=random.uniform(0.1, 1.0),
                    price=price,
                    commission=price * 0.001,  # 0.1% commission
                    timestamp=datetime.now(),
                    strategy=strategy,
                    pnl=random.uniform(-100, 200)  # Random P&L
                )
                monitoring.record_trade(trade)
            
            # Occasionally simulate position changes
            if random.random() < 0.2:  # 20% chance of position change
                position = Position(
                    symbol=symbol,
                    side="LONG" if random.random() > 0.5 else "SHORT",
                    size=random.uniform(0.5, 2.0),
                    entry_price=price,
                    current_price=price * random.uniform(0.98, 1.02),
                    unrealized_pnl=random.uniform(-50, 100),
                    timestamp=datetime.now()
                )
                
                if random.random() > 0.5:
                    monitoring.record_position_opened(position)
                else:
                    pnl = random.uniform(-100, 200)
                    monitoring.record_position_closed(position, pnl)
            
            # Occasionally simulate errors
            if random.random() < 0.05:  # 5% chance of error
                error_types = [
                    ("API rate limit exceeded", "RATE_LIMIT"),
                    ("Connection timeout", "TIMEOUT"),
                    ("Invalid order parameters", "INVALID_ORDER"),
                    ("Insufficient balance", "INSUFFICIENT_BALANCE")
                ]
                error_msg, error_code = random.choice(error_types)
                monitoring.alert_api_error(error_msg, error_code)
            
            # Print progress
            if (i + 1) % 5 == 0:
                logger.info(f"Completed {i + 1}/20 simulation steps")
        
        # Get final monitoring stats
        logger.info("Getting final monitoring statistics...")
        stats = monitoring.get_monitoring_stats()
        
        print("\n" + "="*60)
        print("FINAL MONITORING STATISTICS")
        print("="*60)
        print(f"Uptime: {stats['uptime_seconds']:.1f} seconds")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Active Positions: {stats['active_positions']}")
        print(f"Error Counts: {stats['error_counts']}")
        print(f"System Status: {stats['system_status']['status']}")
        print(f"CPU Usage: {stats['system_status']['cpu_usage_percent']:.1f}%")
        print(f"Memory Usage: {stats['system_status']['memory_usage_mb']:.1f} MB")
        print(f"Registered Connections: {stats['registered_connections']}")
        
        # Run health check
        logger.info("Running comprehensive health check...")
        health_check = await monitoring.run_health_check()
        
        print("\n" + "="*60)
        print("HEALTH CHECK RESULTS")
        print("="*60)
        print(f"Overall Status: {health_check['system_status']['status']}")
        print(f"Health Metrics Count: {len(health_check['health_metrics'])}")
        
        # Display health metrics
        for metric in health_check['health_metrics']:
            status_symbol = {
                'healthy': '✓',
                'warning': '⚠',
                'critical': '✗',
                'unknown': '?'
            }.get(metric['status'], '?')
            
            print(f"  {status_symbol} {metric['name']}: {metric['value']:.1f}{metric['unit']}")
        
        # Test system health
        is_healthy = monitoring.is_healthy()
        uptime = monitoring.get_uptime()
        error_rate = monitoring.get_error_rate()
        
        print(f"\nSystem Health: {'✓ HEALTHY' if is_healthy else '✗ UNHEALTHY'}")
        print(f"Uptime: {uptime}")
        print(f"Error Rate: {error_rate:.2%}")
        
        # Wait a bit to see any final alerts
        logger.info("Waiting for final alerts...")
        await asyncio.sleep(5)
        
    except Exception as e:
        logger.error(f"Error during monitoring test: {e}")
        raise
    
    finally:
        # Stop monitoring
        logger.info("Stopping monitoring system...")
        await monitoring.stop()
        logger.info("Monitoring system stopped.")


async def test_alert_scenarios():
    """Test specific alert scenarios"""
    
    logger.info("Testing specific alert scenarios...")
    
    notification_config = NotificationConfig(
        enabled=True,
        console={'enabled': True, 'min_level': 'info'},
        market_events={
            'price_change_threshold': 0.02,  # 2% for testing
            'volume_spike_threshold': 1.5,   # 1.5x for testing
        },
        technical_indicators={
            'rsi_overbought': 75,  # Lower threshold for testing
            'rsi_oversold': 25,    # Higher threshold for testing
        }
    )
    
    monitoring = MonitoringSystem(notification_config)
    
    try:
        await monitoring.start()
        
        # Test market event alerts
        logger.info("Testing market event alerts...")
        
        # Simulate normal price
        monitoring.record_market_data("BTCUSDT", 40000, 100)
        await asyncio.sleep(1)
        
        # Simulate significant price change
        monitoring.record_market_data("BTCUSDT", 41000, 100)  # 2.5% increase
        await asyncio.sleep(1)
        
        # Simulate volume spike
        monitoring.record_market_data("BTCUSDT", 41000, 200)  # 2x volume
        await asyncio.sleep(1)
        
        # Test technical indicator alerts
        logger.info("Testing technical indicator alerts...")
        
        # RSI overbought
        monitoring.record_technical_indicators("BTCUSDT", {
            'rsi': 78,  # Above 75 threshold
            'price': 41000,
            'bb_upper': 41500,
            'bb_lower': 40500
        })
        await asyncio.sleep(1)
        
        # RSI oversold
        monitoring.record_technical_indicators("ETHUSDT", {
            'rsi': 22,  # Below 25 threshold
            'price': 2500,
            'bb_upper': 2550,
            'bb_lower': 2450
        })
        await asyncio.sleep(1)
        
        # Test risk alerts
        logger.info("Testing risk alerts...")
        monitoring.alert_risk_limit_reached("daily_loss_limit", 0.06, 0.05)
        await asyncio.sleep(1)
        
        # Test system alerts
        logger.info("Testing system alerts...")
        monitoring.alert_system_error("Memory usage critical", "risk_manager")
        await asyncio.sleep(1)
        
        monitoring.alert_connection_lost("binance_api")
        await asyncio.sleep(2)
        monitoring.alert_connection_restored("binance_api")
        
        logger.info("Alert scenario testing completed")
        await asyncio.sleep(3)  # Wait for alerts to process
        
    finally:
        await monitoring.stop()


async def main():
    """Main test function"""
    print("="*60)
    print("CRYPTO TRADING BOT - MONITORING SYSTEM TEST")
    print("="*60)
    
    try:
        # Test main monitoring system
        await test_monitoring_system()
        
        print("\n" + "="*60)
        print("TESTING SPECIFIC ALERT SCENARIOS")
        print("="*60)
        
        # Test specific alert scenarios
        await test_alert_scenarios()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())