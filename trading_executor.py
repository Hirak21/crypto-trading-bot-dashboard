#!/usr/bin/env python3
"""
Trading Executor - Executes trades based on scanner signals
Reads signals from scalping_results.json or strategy_results.json
Uses original bot logic for execution
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging

class TradingExecutor:
    """Executes trades based on scanner signals"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.testnet = testnet
        self.active_positions = {}
        self.executed_signals = set()  # Track executed signals to avoid duplicates
        
        # Risk management settings
        self.max_position_size = 0.02  # 2% of portfolio per trade
        self.max_daily_trades = 10
        self.max_concurrent_positions = 5
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Track daily stats
        self.daily_stats = {
            'trades_executed': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now()
        }
    
    async def initialize(self):
        """Initialize the trading executor"""
        try:
            self.logger.info("🚀 Initializing Trading Executor...")
            
            # Test API connection
            account_info = self.client.futures_account()
            balance = float(account_info['totalWalletBalance'])
            
            self.logger.info(f"✅ Connected to Binance {'Testnet' if self.testnet else 'Live'}")
            self.logger.info(f"💰 Account Balance: ${balance:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def load_scanner_signals(self, signal_file: str) -> List[Dict]:
        """Load signals from scanner result files"""
        try:
            with open(signal_file, 'r') as f:
                data = json.load(f)
            
            signals = data.get('selected_trades', [])
            scan_time = data.get('timestamp', '')
            
            self.logger.info(f"📊 Loaded {len(signals)} signals from {signal_file}")
            self.logger.info(f"🕐 Scan time: {scan_time}")
            
            return signals
            
        except FileNotFoundError:
            self.logger.warning(f"⚠️ Signal file {signal_file} not found")
            return []
        except Exception as e:
            self.logger.error(f"❌ Error loading signals: {e}")
            return []
    
    def validate_signal(self, signal: Dict) -> bool:
        """Validate signal before execution"""
        try:
            # Check required fields
            required_fields = ['symbol', 'action', 'confidence', 'current_price', 'target_price', 'stop_loss']
            for field in required_fields:
                if field not in signal:
                    self.logger.warning(f"⚠️ Signal missing field: {field}")
                    return False
            
            # Check confidence threshold
            if signal['confidence'] < 75:
                self.logger.info(f"⚠️ Signal confidence too low: {signal['confidence']:.1f}%")
                return False
            
            # Check if already executed
            signal_id = f"{signal['symbol']}_{signal['action']}_{signal.get('timestamp', '')}"
            if signal_id in self.executed_signals:
                self.logger.info(f"⚠️ Signal already executed: {signal['symbol']}")
                return False
            
            # Check daily limits
            if self.daily_stats['trades_executed'] >= self.max_daily_trades:
                self.logger.warning("⚠️ Daily trade limit reached")
                return False
            
            # Check concurrent positions
            if len(self.active_positions) >= self.max_concurrent_positions:
                self.logger.warning("⚠️ Maximum concurrent positions reached")
                return False
            
            # Check daily loss limit
            if self.daily_stats['total_pnl'] <= -self.max_daily_trades:
                self.logger.warning("⚠️ Daily loss limit reached")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating signal: {e}")
            return False
    
    def calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            account_info = self.client.futures_account()
            balance = float(account_info['totalWalletBalance'])
            
            # Calculate risk per trade
            current_price = signal['current_price']
            stop_loss = signal['stop_loss']
            
            # Calculate risk percentage
            if signal['action'] == 'BUY':
                risk_per_unit = abs(current_price - stop_loss) / current_price
            else:  # SELL
                risk_per_unit = abs(stop_loss - current_price) / current_price
            
            # Position size based on risk
            risk_amount = balance * self.max_position_size
            position_value = risk_amount / risk_per_unit
            position_size = position_value / current_price
            
            # Apply minimum and maximum limits
            min_size = 0.001  # Minimum position size
            max_value = balance * 0.1  # Maximum 10% of balance per trade
            max_size = max_value / current_price
            
            position_size = max(min_size, min(position_size, max_size))
            
            self.logger.info(f"💰 Position size calculated: {position_size:.6f} (${position_size * current_price:.2f})")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 0.001  # Default minimum size
    
    async def execute_trade(self, signal: Dict) -> bool:
        """Execute a trade based on signal"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            current_price = signal['current_price']
            target_price = signal['target_price']
            stop_loss = signal['stop_loss']
            confidence = signal['confidence']
            strategy = signal.get('strategy', 'unknown')
            
            self.logger.info(f"🎯 Executing {action} {symbol} (Strategy: {strategy})")
            self.logger.info(f"   Confidence: {confidence:.1f}%")
            self.logger.info(f"   Price: ${current_price:.6f}")
            self.logger.info(f"   Target: ${target_price:.6f}")
            self.logger.info(f"   Stop Loss: ${stop_loss:.6f}")
            
            # Calculate position size
            position_size = self.calculate_position_size(signal)
            
            if position_size <= 0:
                self.logger.error("❌ Invalid position size")
                return False
            
            # Execute market order
            side = 'BUY' if action == 'BUY' else 'SELL'
            
            if self.testnet:
                # Simulate order for testnet
                order_result = {
                    'orderId': f"TEST_{int(time.time())}",
                    'symbol': symbol,
                    'side': side,
                    'origQty': str(position_size),
                    'price': str(current_price),
                    'status': 'FILLED'
                }
                self.logger.info("📝 Testnet: Simulated order execution")
            else:
                # Real order execution
                order_result = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=position_size
                )
            
            if order_result:
                order_id = order_result['orderId']
                
                # Record position
                position_data = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': position_size,
                    'entry_price': current_price,
                    'target_price': target_price,
                    'stop_loss': stop_loss,
                    'strategy': strategy,
                    'confidence': confidence,
                    'entry_time': datetime.now(),
                    'status': 'ACTIVE'
                }
                
                self.active_positions[symbol] = position_data
                
                # Mark signal as executed
                signal_id = f"{symbol}_{action}_{signal.get('timestamp', '')}"
                self.executed_signals.add(signal_id)
                
                # Update daily stats
                self.daily_stats['trades_executed'] += 1
                
                # Log trade execution
                self.log_trade_execution(position_data)
                
                self.logger.info(f"✅ Trade executed successfully: {order_id}")
                return True
            else:
                self.logger.error("❌ Order execution failed")
                return False
                
        except BinanceAPIException as e:
            self.logger.error(f"❌ Binance API error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Trade execution error: {e}")
            return False
    
    def log_trade_execution(self, position_data: Dict):
        """Log trade execution to file"""
        try:
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'order_id': position_data['order_id'],
                'symbol': position_data['symbol'],
                'side': position_data['side'],
                'quantity': position_data['quantity'],
                'entry_price': position_data['entry_price'],
                'target_price': position_data['target_price'],
                'stop_loss': position_data['stop_loss'],
                'strategy': position_data['strategy'],
                'confidence': position_data['confidence']
            }
            
            # Append to trade log file
            with open('trade_executions.json', 'a') as f:
                f.write(json.dumps(trade_log) + '\n')
                
        except Exception as e:
            self.logger.error(f"❌ Error logging trade: {e}")
    
    async def monitor_positions(self):
        """Monitor active positions for stop loss and take profit"""
        if not self.active_positions:
            return
        
        self.logger.info(f"👁️ Monitoring {len(self.active_positions)} active positions...")
        
        try:
            # Get current prices
            tickers = self.client.futures_ticker()
            price_dict = {t['symbol']: float(t['price']) for t in tickers}
            
            positions_to_close = []
            
            for symbol, position in self.active_positions.items():
                if symbol not in price_dict:
                    continue
                
                current_price = price_dict[symbol]
                entry_price = position['entry_price']
                target_price = position['target_price']
                stop_loss = position['stop_loss']
                side = position['side']
                
                # Calculate P&L
                if side == 'BUY':
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    # Check stop loss and take profit
                    if current_price <= stop_loss:
                        positions_to_close.append((symbol, 'STOP_LOSS', pnl_pct))
                    elif current_price >= target_price:
                        positions_to_close.append((symbol, 'TAKE_PROFIT', pnl_pct))
                else:  # SELL
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    # Check stop loss and take profit
                    if current_price >= stop_loss:
                        positions_to_close.append((symbol, 'STOP_LOSS', pnl_pct))
                    elif current_price <= target_price:
                        positions_to_close.append((symbol, 'TAKE_PROFIT', pnl_pct))
                
                # Log position status
                self.logger.info(f"📊 {symbol}: ${current_price:.6f} | P&L: {pnl_pct:.2f}%")
            
            # Close positions that hit targets
            for symbol, reason, pnl_pct in positions_to_close:
                await self.close_position(symbol, reason, pnl_pct)
                
        except Exception as e:
            self.logger.error(f"❌ Error monitoring positions: {e}")
    
    async def close_position(self, symbol: str, reason: str, pnl_pct: float):
        """Close a position"""
        try:
            if symbol not in self.active_positions:
                return
            
            position = self.active_positions[symbol]
            side = 'SELL' if position['side'] == 'BUY' else 'BUY'  # Opposite side to close
            quantity = position['quantity']
            
            self.logger.info(f"🔄 Closing {symbol} position - Reason: {reason} | P&L: {pnl_pct:.2f}%")
            
            if self.testnet:
                # Simulate close for testnet
                self.logger.info("📝 Testnet: Simulated position close")
                close_success = True
            else:
                # Real position close
                close_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
                close_success = bool(close_order)
            
            if close_success:
                # Update daily stats
                position_value = position['quantity'] * position['entry_price']
                pnl_amount = position_value * (pnl_pct / 100)
                self.daily_stats['total_pnl'] += pnl_amount
                
                # Log position close
                close_log = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'reason': reason,
                    'pnl_pct': pnl_pct,
                    'pnl_amount': pnl_amount,
                    'strategy': position['strategy'],
                    'hold_time': (datetime.now() - position['entry_time']).total_seconds() / 60  # minutes
                }
                
                with open('position_closes.json', 'a') as f:
                    f.write(json.dumps(close_log) + '\n')
                
                # Remove from active positions
                del self.active_positions[symbol]
                
                self.logger.info(f"✅ Position closed: {symbol} | P&L: ${pnl_amount:.2f}")
            else:
                self.logger.error(f"❌ Failed to close position: {symbol}")
                
        except Exception as e:
            self.logger.error(f"❌ Error closing position {symbol}: {e}")
    
    async def execute_signals_from_file(self, signal_file: str):
        """Execute all valid signals from a file"""
        signals = self.load_scanner_signals(signal_file)
        
        if not signals:
            self.logger.info("📊 No signals to execute")
            return
        
        executed_count = 0
        
        for signal in signals:
            if self.validate_signal(signal):
                success = await self.execute_trade(signal)
                if success:
                    executed_count += 1
                    # Wait between executions
                    await asyncio.sleep(2)
            else:
                self.logger.info(f"⚠️ Signal validation failed: {signal['symbol']}")
        
        self.logger.info(f"✅ Executed {executed_count}/{len(signals)} signals")
    
    async def run_trading_session(self, scalping_mode: bool = False):
        """Run a complete trading session"""
        self.logger.info("🚀 Starting Trading Session...")
        
        # Choose signal file based on mode
        if scalping_mode:
            signal_file = 'scalping_results.json'
            self.logger.info("⚡ Mode: Scalping (15-minute signals)")
        else:
            signal_file = 'strategy_results.json'
            self.logger.info("🎯 Mode: Strategy (Original bot strategies)")
        
        # Execute signals
        await self.execute_signals_from_file(signal_file)
        
        # Monitor positions
        self.logger.info("👁️ Starting position monitoring...")
        
        # Monitor loop
        monitor_count = 0
        while self.active_positions and monitor_count < 120:  # Max 2 hours monitoring
            await self.monitor_positions()
            await asyncio.sleep(30)  # Check every 30 seconds
            monitor_count += 1
        
        # Print session summary
        self.print_session_summary()
    
    def print_session_summary(self):
        """Print trading session summary"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 TRADING SESSION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Trades Executed: {self.daily_stats['trades_executed']}")
        self.logger.info(f"Active Positions: {len(self.active_positions)}")
        self.logger.info(f"Total P&L: ${self.daily_stats['total_pnl']:.2f}")
        self.logger.info(f"Session Duration: {datetime.now() - self.daily_stats['start_time']}")
        
        if self.active_positions:
            self.logger.info("\n🔄 Active Positions:")
            for symbol, position in self.active_positions.items():
                self.logger.info(f"  {symbol}: {position['side']} | Strategy: {position['strategy']}")

async def main():
    """Main trading executor function"""
    try:
        # Load credentials
        with open('config/credentials.json', 'r') as f:
            creds = json.load(f)
        api_key = creds['binance']['api_key']
        api_secret = creds['binance']['api_secret']
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        return
    
    # Initialize executor
    executor = TradingExecutor(api_key, api_secret, testnet=True)
    
    if await executor.initialize():
        print("\n🎯 Trading Executor Ready!")
        print("Options:")
        print("1. Execute Scalping Signals (15-minute)")
        print("2. Execute Strategy Signals (Original bot)")
        print("3. Monitor existing positions only")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == '1':
            await executor.run_trading_session(scalping_mode=True)
        elif choice == '2':
            await executor.run_trading_session(scalping_mode=False)
        elif choice == '3':
            await executor.monitor_positions()
        else:
            print("Invalid choice")
    else:
        print("❌ Failed to initialize trading executor")

if __name__ == "__main__":
    asyncio.run(main())