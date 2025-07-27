#!/usr/bin/env python3
"""
Mobile-Optimized Scalping Scanner
Optimized for Termux/Android with reduced resource usage
"""

import asyncio
import json
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from binance.client import Client
from binance.exceptions import BinanceAPIException
import numpy as np
from datetime import datetime, timedelta

@dataclass
class MobileSignal:
    symbol: str
    action: str
    confidence: float
    strategy: str
    current_price: float
    target_price: float
    stop_loss: float
    volume_24h: float
    price_change: float
    score: float = 0.0

class MobileScanner:
    """Mobile-optimized scanner with reduced resource usage"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.client = Client(api_key, api_secret, testnet=testnet)
        
        # Mobile-optimized settings
        self.batch_size = 50  # Smaller batches for mobile
        self.max_trades = 3   # Fewer concurrent trades
        self.scan_delay = 2   # Longer delays to save battery
        self.max_symbols = 200  # Limit symbols for mobile
        
        # Battery saving mode
        self.battery_save_mode = self.detect_battery_save_mode()
        
        if self.battery_save_mode:
            print("🔋 Battery save mode enabled")
            self.batch_size = 30
            self.scan_delay = 5
            self.max_symbols = 100
    
    def detect_battery_save_mode(self) -> bool:
        """Detect if device is in battery save mode"""
        try:
            # Check if termux-battery-status is available
            import subprocess
            result = subprocess.run(['termux-battery-status'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                battery_info = json.loads(result.stdout)
                return battery_info.get('percentage', 100) < 30
        except:
            pass
        return False
    
    def send_mobile_notification(self, title: str, message: str):
        """Send notification using Termux API"""
        try:
            import subprocess
            subprocess.run([
                'termux-notification',
                '--title', title,
                '--content', message
            ], timeout=5)
        except:
            pass  # Fail silently if termux-api not available
    
    async def get_top_symbols(self) -> List[str]:
        """Get top trading symbols optimized for mobile"""
        try:
            # Get 24hr ticker stats
            tickers = self.client.futures_ticker()
            
            # Filter USDT perpetual futures
            usdt_tickers = [
                t for t in tickers 
                if t['symbol'].endswith('USDT') and float(t['volume']) > 1000000
            ]
            
            # Sort by volume and take top symbols
            usdt_tickers.sort(key=lambda x: float(x['volume']), reverse=True)
            symbols = [t['symbol'] for t in usdt_tickers[:self.max_symbols]]
            
            print(f"📱 Mobile mode: Scanning top {len(symbols)} symbols")
            return symbols
            
        except Exception as e:
            print(f"❌ Error fetching symbols: {e}")
            # Fallback to popular symbols
            return [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
                'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT',
                'LTCUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'FILUSDT'
            ]
    
    def analyze_mobile_signal(self, symbol: str, klines: List, ticker: Dict, timeframe: str) -> Optional[MobileSignal]:
        """Lightweight signal analysis for mobile"""
        try:
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            
            if len(closes) < 10:
                return None
            
            current_price = closes[-1]
            volume_24h = float(ticker['volume'])
            price_change = float(ticker['priceChangePercent'])
            
            # Simple RSI calculation
            def simple_rsi(prices, period=14):
                if len(prices) < period + 1:
                    return 50
                deltas = np.diff(prices)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[:period])
                avg_loss = np.mean(losses[:period])
                if avg_loss == 0:
                    return 100
                rs = avg_gain / avg_loss
                return 100 - (100 / (1 + rs))
            
            rsi = simple_rsi(closes)
            
            # Simple moving averages
            sma_5 = np.mean(closes[-5:])
            sma_10 = np.mean(closes[-10:])
            
            # Volume spike detection
            avg_volume = np.mean(volumes[-5:])
            volume_spike = volumes[-1] > avg_volume * 1.5
            
            # Mobile-optimized strategies
            signals = []
            
            # 1. RSI Strategy (simplified)
            if rsi < 30 and price_change > -5:
                signals.append({
                    'strategy': f'mobile_rsi_long_{timeframe}',
                    'action': 'BUY',
                    'confidence': min(85, (30 - rsi) * 3),
                    'target': current_price * 1.02,
                    'stop_loss': current_price * 0.99
                })
            elif rsi > 70 and price_change < 5:
                signals.append({
                    'strategy': f'mobile_rsi_short_{timeframe}',
                    'action': 'SELL',
                    'confidence': min(85, (rsi - 70) * 3),
                    'target': current_price * 0.98,
                    'stop_loss': current_price * 1.01
                })
            
            # 2. Moving Average Crossover
            if sma_5 > sma_10 and volume_spike:
                signals.append({
                    'strategy': f'mobile_ma_cross_long_{timeframe}',
                    'action': 'BUY',
                    'confidence': 75,
                    'target': current_price * 1.015,
                    'stop_loss': current_price * 0.995
                })
            elif sma_5 < sma_10 and volume_spike:
                signals.append({
                    'strategy': f'mobile_ma_cross_short_{timeframe}',
                    'action': 'SELL',
                    'confidence': 75,
                    'target': current_price * 0.985,
                    'stop_loss': current_price * 1.005
                })
            
            # 3. Volume Momentum
            if volume_spike and abs(price_change) > 1:
                action = 'BUY' if price_change > 0 else 'SELL'
                signals.append({
                    'strategy': f'mobile_volume_momentum_{timeframe}',
                    'action': action,
                    'confidence': min(80, abs(price_change) * 8),
                    'target': current_price * (1.01 if action == 'BUY' else 0.99),
                    'stop_loss': current_price * (0.995 if action == 'BUY' else 1.005)
                })
            
            if not signals:
                return None
            
            # Select best signal
            best_signal = max(signals, key=lambda x: x['confidence'])
            
            # Calculate mobile score (simplified)
            score = (
                best_signal['confidence'] * 0.6 +
                min(volume_24h / 1000000, 30) * 0.4
            )
            
            return MobileSignal(
                symbol=symbol,
                action=best_signal['action'],
                confidence=best_signal['confidence'],
                strategy=best_signal['strategy'],
                current_price=current_price,
                target_price=best_signal['target'],
                stop_loss=best_signal['stop_loss'],
                volume_24h=volume_24h,
                price_change=price_change,
                score=score
            )
            
        except Exception as e:
            print(f"⚠️ Error analyzing {symbol}: {e}")
            return None
    
    async def mobile_scan_batch(self, symbols: List[str], timeframe: str) -> List[MobileSignal]:
        """Mobile-optimized batch scanning"""
        print(f"📱 Scanning {len(symbols)} symbols ({timeframe})")
        signals = []
        
        try:
            # Get ticker data
            tickers = self.client.futures_ticker()
            ticker_dict = {t['symbol']: t for t in tickers if t['symbol'] in symbols}
            
            # Interval mapping
            interval_map = {
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '30m': Client.KLINE_INTERVAL_30MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE
            }
            
            for symbol in symbols:
                try:
                    if symbol not in ticker_dict:
                        continue
                    
                    # Get kline data (reduced limit for mobile)
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=interval_map[timeframe],
                        limit=20  # Reduced from 50 for mobile
                    )
                    
                    signal = self.analyze_mobile_signal(symbol, klines, ticker_dict[symbol], timeframe)
                    if signal and signal.confidence > 70:
                        signals.append(signal)
                    
                    # Mobile-friendly delay
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    continue
            
            print(f"✅ Found {len(signals)} signals")
            return signals
            
        except Exception as e:
            print(f"❌ Batch scan failed: {e}")
            return []
    
    async def mobile_adaptive_scan(self) -> List[MobileSignal]:
        """Mobile-optimized adaptive scanning"""
        print("📱 MOBILE ADAPTIVE SCANNER")
        print("=" * 40)
        
        symbols = await self.get_top_symbols()
        
        # Split into smaller batches for mobile
        batches = [symbols[i:i + self.batch_size] for i in range(0, len(symbols), self.batch_size)]
        
        # Scan timeframes
        for timeframe in ['15m', '30m', '5m']:
            print(f"\n🔍 Scanning {timeframe} timeframe...")
            all_signals = []
            
            for i, batch in enumerate(batches, 1):
                print(f"📊 Batch {i}/{len(batches)}")
                batch_signals = await self.mobile_scan_batch(batch, timeframe)
                all_signals.extend(batch_signals)
                
                # Battery-friendly delay between batches
                if i < len(batches):
                    await asyncio.sleep(self.scan_delay)
            
            # Filter and select best signals
            if all_signals:
                best_signals = self.select_mobile_signals(all_signals)
                if best_signals:
                    print(f"✅ Found {len(best_signals)} signals on {timeframe}!")
                    
                    # Send mobile notification
                    self.send_mobile_notification(
                        "Trading Bot Alert",
                        f"Found {len(best_signals)} signals on {timeframe}"
                    )
                    
                    return best_signals
            
            print(f"⚠️ No signals on {timeframe}")
            
            # Don't wait if this is the last timeframe
            if timeframe != '5m':
                print("⏳ Waiting 30 seconds...")
                await asyncio.sleep(30)
        
        # If no signals found, start continuous 5m scanning
        return await self.continuous_mobile_scan()
    
    async def continuous_mobile_scan(self) -> List[MobileSignal]:
        """Continuous mobile scanning with battery optimization"""
        print("\n📱 CONTINUOUS MOBILE SCANNING")
        print("=" * 40)
        print("🔋 Battery-optimized mode active")
        
        scan_count = 0
        symbols = await self.get_top_symbols()
        batches = [symbols[i:i + self.batch_size] for i in range(0, len(symbols), self.batch_size)]
        
        while True:
            try:
                scan_count += 1
                print(f"\n📱 Mobile Scan #{scan_count} - {datetime.now().strftime('%H:%M:%S')}")
                
                all_signals = []
                for i, batch in enumerate(batches, 1):
                    batch_signals = await self.mobile_scan_batch(batch, '5m')
                    all_signals.extend(batch_signals)
                    
                    # Show progress
                    print(f"📊 Progress: {i}/{len(batches)} batches")
                    
                    if i < len(batches):
                        await asyncio.sleep(1)
                
                if all_signals:
                    best_signals = self.select_mobile_signals(all_signals)
                    if best_signals:
                        print(f"🎉 Found {len(best_signals)} signals!")
                        
                        # Send notification
                        self.send_mobile_notification(
                            "Trading Signals Found!",
                            f"{len(best_signals)} high-quality signals detected"
                        )
                        
                        return best_signals
                
                print("⚠️ No signals found")
                
                # Battery-optimized wait time
                wait_time = 300 if not self.battery_save_mode else 600  # 5 or 10 minutes
                print(f"⏳ Next scan in {wait_time//60} minutes...")
                
                # Countdown with battery check
                for remaining in range(wait_time, 0, -60):
                    mins = remaining // 60
                    print(f"⏳ {mins}m remaining...", end='\r')
                    await asyncio.sleep(60)
                    
                    # Check battery status periodically
                    if remaining % 300 == 0:  # Every 5 minutes
                        self.battery_save_mode = self.detect_battery_save_mode()
                
                print()  # New line after countdown
                
            except KeyboardInterrupt:
                print(f"\n🛑 Mobile scanning stopped")
                print(f"📊 Total scans: {scan_count}")
                return []
            except Exception as e:
                print(f"❌ Mobile scan error: {e}")
                await asyncio.sleep(60)
    
    def select_mobile_signals(self, signals: List[MobileSignal]) -> List[MobileSignal]:
        """Select best signals for mobile trading"""
        if not signals:
            return []
        
        # Sort by score
        sorted_signals = sorted(signals, key=lambda x: x.score, reverse=True)
        
        selected = []
        for signal in sorted_signals:
            # Mobile-friendly filters
            if (signal.confidence < 70 or 
                abs(signal.price_change) > 10 or
                signal.volume_24h < 500000):
                continue
            
            selected.append(signal)
            
            if len(selected) >= self.max_trades:
                break
        
        return selected
    
    def print_mobile_results(self, signals: List[MobileSignal]):
        """Print results optimized for mobile screen"""
        print(f"\n📱 MOBILE SCAN RESULTS")
        print("=" * 30)
        
        if not signals:
            print("❌ No signals found")
            return
        
        for i, signal in enumerate(signals, 1):
            profit_pct = ((signal.target_price - signal.current_price) / signal.current_price) * 100
            
            print(f"\n🎯 SIGNAL #{i}")
            print(f"Symbol: {signal.symbol}")
            print(f"Action: {signal.action}")
            print(f"Confidence: {signal.confidence:.1f}%")
            print(f"Price: ${signal.current_price:.4f}")
            print(f"Target: ${signal.target_price:.4f}")
            print(f"Profit: {profit_pct:.2f}%")
            print(f"Strategy: {signal.strategy}")
            print("-" * 25)

async def main():
    """Main mobile scanner function"""
    try:
        with open('config/credentials.json', 'r') as f:
            creds = json.load(f)
        api_key = creds['binance']['api_key']
        api_secret = creds['binance']['api_secret']
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        return
    
    scanner = MobileScanner(api_key, api_secret, testnet=True)
    
    print("📱 MOBILE CRYPTO TRADING BOT")
    print("=" * 30)
    print("🔋 Optimized for Android/Termux")
    print("📊 Reduced resource usage")
    print("🔔 Mobile notifications enabled")
    print("=" * 30)
    
    # Run adaptive scan
    best_signals = await scanner.mobile_adaptive_scan()
    
    if best_signals:
        scanner.print_mobile_results(best_signals)
        
        # Save results
        results = {
            'scan_type': 'mobile_adaptive',
            'timestamp': datetime.now().isoformat(),
            'device': 'mobile',
            'signals': [
                {
                    'symbol': s.symbol,
                    'action': s.action,
                    'confidence': s.confidence,
                    'strategy': s.strategy,
                    'current_price': s.current_price,
                    'target_price': s.target_price,
                    'stop_loss': s.stop_loss,
                    'score': s.score
                }
                for s in best_signals
            ]
        }
        
        with open('mobile_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to mobile_results.json")
        
        # Final notification
        scanner.send_mobile_notification(
            "Scan Complete!",
            f"Found {len(best_signals)} trading opportunities"
        )
    else:
        print("❌ No trading opportunities found")

if __name__ == "__main__":
    asyncio.run(main())