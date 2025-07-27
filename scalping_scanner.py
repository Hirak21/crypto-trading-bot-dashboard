#!/usr/bin/env python3
"""
15-Minute Scalping Scanner
Fast trades on 15-minute timeframe for quick profits
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@dataclass
class ScalpingSignal:
    symbol: str
    action: str  # BUY/SELL
    confidence: float
    strategy: str
    current_price: float
    target_price: float
    stop_loss: float
    volume_24h: float
    price_change_15m: float
    rsi_15m: Optional[float] = None
    ema_cross: bool = False
    volume_spike: bool = False
    score: float = 0.0

class ScalpingScanner:
    """15-minute scalping scanner for quick trades"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.batch_size = 100
        self.max_trades = 5
        
    async def get_futures_symbols(self) -> List[str]:
        """Get all USDT perpetual futures symbols"""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbols = []
            
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                if (symbol.endswith('USDT') and 
                    symbol_info['status'] == 'TRADING' and
                    symbol_info['contractType'] == 'PERPETUAL'):
                    symbols.append(symbol)
            
            print(f"✅ Found {len(symbols)} USDT perpetual futures symbols")
            return symbols[:1000]
            
        except Exception as e:
            print(f"❌ Error fetching symbols: {e}")
            return []

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices) if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema

    def analyze_scalping_opportunity(self, symbol: str, klines: List, ticker: Dict, timeframe: str = '15m') -> Optional[ScalpingSignal]:
        """Analyze symbol for scalping opportunities on specified timeframe"""
        try:
            # Extract price data
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            
            if len(closes) < 20:
                return None
                
            current_price = closes[-1]
            volume_24h = float(ticker['volume'])
            
            # Calculate price change based on timeframe
            if timeframe == '15m':
                price_ago = closes[-2] if len(closes) >= 2 else current_price
                price_change = ((current_price - price_ago) / price_ago) * 100
                timeframe_label = '15m'
            else:  # 30m
                price_ago = closes[-3] if len(closes) >= 3 else current_price  # 30m ago (2 periods)
                price_change = ((current_price - price_ago) / price_ago) * 100
                timeframe_label = '30m'
            
            # Calculate indicators
            rsi = self.calculate_rsi(closes, 14)
            ema_9 = self.calculate_ema(closes, 9)
            ema_21 = self.calculate_ema(closes, 21)
            
            # Volume analysis
            avg_volume = np.mean(volumes[-10:])
            volume_spike = volumes[-1] > (avg_volume * 1.3)
            
            # EMA crossover
            ema_cross_bullish = ema_9 > ema_21 and closes[-2] <= ema_21
            ema_cross_bearish = ema_9 < ema_21 and closes[-2] >= ema_21
            
            # Adjust thresholds based on timeframe
            if timeframe == '15m':
                rsi_oversold, rsi_overbought = 25, 75
                min_price_change = 0.5
                momentum_threshold = 1.5
                target_multiplier = 1.015  # 1.5% target
                stop_multiplier = 0.995    # 0.5% stop
            else:  # 30m - more relaxed thresholds
                rsi_oversold, rsi_overbought = 30, 70
                min_price_change = 0.3
                momentum_threshold = 1.0
                target_multiplier = 1.02   # 2% target
                stop_multiplier = 0.99     # 1% stop
            
            # Scalping strategies
            signals = []
            
            # 1. RSI Scalping (Quick reversals)
            if rsi < rsi_oversold and price_change < -0.5:  # Oversold bounce
                signals.append({
                    'strategy': f'rsi_scalp_long_{timeframe}',
                    'action': 'BUY',
                    'confidence': min(95, (rsi_oversold - rsi) * 4),
                    'target': current_price * target_multiplier,
                    'stop_loss': current_price * stop_multiplier
                })
            elif rsi > rsi_overbought and price_change > 0.5:  # Overbought reversal
                signals.append({
                    'strategy': f'rsi_scalp_short_{timeframe}',
                    'action': 'SELL',
                    'confidence': min(95, (rsi - rsi_overbought) * 4),
                    'target': current_price * (2 - target_multiplier),  # Inverse for short
                    'stop_loss': current_price * (2 - stop_multiplier)   # Inverse for short
                })
            
            # 2. EMA Crossover Scalping
            if ema_cross_bullish and volume_spike:
                signals.append({
                    'strategy': f'ema_cross_long_{timeframe}',
                    'action': 'BUY',
                    'confidence': 80,
                    'target': current_price * (target_multiplier + 0.005),  # Slightly higher target
                    'stop_loss': current_price * (stop_multiplier - 0.005)  # Slightly tighter stop
                })
            elif ema_cross_bearish and volume_spike:
                signals.append({
                    'strategy': f'ema_cross_short_{timeframe}',
                    'action': 'SELL',
                    'confidence': 80,
                    'target': current_price * (2 - target_multiplier - 0.005),
                    'stop_loss': current_price * (2 - stop_multiplier + 0.005)
                })
            
            # 3. Volume Breakout Scalping
            if volume_spike and abs(price_change) > min_price_change:
                action = 'BUY' if price_change > 0 else 'SELL'
                signals.append({
                    'strategy': f'volume_breakout_scalp_{timeframe}',
                    'action': action,
                    'confidence': min(85, abs(price_change) * 10),
                    'target': current_price * (target_multiplier if action == 'BUY' else (2 - target_multiplier)),
                    'stop_loss': current_price * (stop_multiplier if action == 'BUY' else (2 - stop_multiplier))
                })
            
            # 4. Quick Momentum Scalping
            if abs(price_change) > momentum_threshold and volume_spike:
                action = 'BUY' if price_change > 0 else 'SELL'
                quick_target = 1.008 if timeframe == '15m' else 1.012  # Adjust for timeframe
                quick_stop = 0.997 if timeframe == '15m' else 0.995
                
                signals.append({
                    'strategy': f'momentum_scalp_{timeframe}',
                    'action': action,
                    'confidence': min(90, abs(price_change) * 5),
                    'target': current_price * (quick_target if action == 'BUY' else (2 - quick_target)),
                    'stop_loss': current_price * (quick_stop if action == 'BUY' else (2 - quick_stop))
                })
            
            # Select best signal
            if not signals:
                return None
                
            best_signal = max(signals, key=lambda x: x['confidence'])
            
            # Calculate scalping score (prioritize quick, high-confidence trades)
            score = (
                best_signal['confidence'] * 0.5 +
                min(abs(price_change) * 10, 50) * 0.3 +
                (50 if volume_spike else 0) * 0.2
            )
            
            return ScalpingSignal(
                symbol=symbol,
                action=best_signal['action'],
                confidence=best_signal['confidence'],
                strategy=best_signal['strategy'],
                current_price=current_price,
                target_price=best_signal['target'],
                stop_loss=best_signal['stop_loss'],
                volume_24h=volume_24h,
                price_change_15m=price_change,  # Now represents current timeframe change
                rsi_15m=rsi,  # Now represents current timeframe RSI
                ema_cross=(ema_cross_bullish or ema_cross_bearish),
                volume_spike=volume_spike,
                score=score
            )
            
        except Exception as e:
            print(f"⚠️ Error analyzing {symbol}: {e}")
            return None

    async def scan_batch(self, symbols: List[str], batch_num: int, timeframe: str = '15m') -> List[ScalpingSignal]:
        """Scan a batch of symbols for scalping opportunities on specified timeframe"""
        interval_map = {
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE
        }
        
        print(f"🔍 Scanning batch {batch_num}: {len(symbols)} symbols ({timeframe} timeframe)")
        signals = []
        
        try:
            # Get 24hr ticker data
            tickers = self.client.futures_ticker()
            ticker_dict = {t['symbol']: t for t in tickers if t['symbol'] in symbols}
            
            for symbol in symbols:
                try:
                    if symbol not in ticker_dict:
                        continue
                        
                    # Get kline data for specified timeframe
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=interval_map[timeframe],
                        limit=50
                    )
                    
                    signal = self.analyze_scalping_opportunity(symbol, klines, ticker_dict[symbol], timeframe)
                    if signal and signal.confidence > 70:  # High confidence for scalping
                        signals.append(signal)
                        
                    time.sleep(0.05)  # Faster rate for scalping
                    
                except BinanceAPIException as e:
                    if e.code == -1121:
                        continue
                    print(f"⚠️ API error for {symbol}: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ Error processing {symbol}: {e}")
                    continue
            
            print(f"✅ Batch {batch_num} complete: {len(signals)} scalping signals found")
            return signals
            
        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {e}")
            return []

    async def scan_all_symbols_with_timeframe(self, timeframe: str = '15m') -> List[ScalpingSignal]:
        """Scan all symbols for scalping opportunities on specified timeframe"""
        print(f"🚀 Starting {timeframe} scalping scan...")
        
        all_symbols = await self.get_futures_symbols()
        if not all_symbols:
            return []
        
        all_signals = []
        total_batches = (len(all_symbols) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(all_symbols), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch_symbols = all_symbols[i:i + self.batch_size]
            
            print(f"\n📊 Processing batch {batch_num}/{total_batches}")
            batch_signals = await self.scan_batch(batch_symbols, batch_num, timeframe)
            all_signals.extend(batch_signals)
            
            print(f"📈 Progress: {batch_num}/{total_batches} batches, {len(all_signals)} total signals")
            
            if batch_num < total_batches:
                print("⏳ Waiting 1 second before next batch...")
                time.sleep(1)  # Faster for scalping
        
        return all_signals

    async def adaptive_scan(self) -> List[ScalpingSignal]:
        """Adaptive scanning: 15m -> 30m -> continuous 5m if no signals found"""
        print("🎯 Starting Adaptive Scalping Scanner...")
        print("=" * 80)
        
        # Round 1: 15-minute timeframe
        print("🔍 ROUND 1: Scanning 15-minute timeframe...")
        signals_15m = await self.scan_all_symbols_with_timeframe('15m')
        print(f"📊 Raw signals found: {len(signals_15m)}")
        
        if signals_15m:
            print("🔍 Filtering signals for best trades...")
            best_trades_15m = self.select_best_scalping_trades(signals_15m)
            print(f"🎯 Best trades after filtering: {len(best_trades_15m)}")
            
            if best_trades_15m:
                print(f"✅ Found {len(best_trades_15m)} valid signals on 15-minute timeframe!")
                return best_trades_15m
            else:
                print("⚠️ Signals found but filtered out by quality checks")
                print("📋 Signal details:")
                for i, signal in enumerate(signals_15m, 1):
                    print(f"   {i}. {signal.symbol}: {signal.confidence:.1f}% confidence, "
                          f"Volume: ${signal.volume_24h:,.0f}, Change: {signal.price_change_15m:.2f}%")
        
        print("⚠️ No valid signals found on 15-minute timeframe")
        print("⏳ Waiting 30 seconds before trying 30-minute timeframe...")
        await asyncio.sleep(30)
        
        # Round 2: 30-minute timeframe
        print("\n🔍 ROUND 2: Scanning 30-minute timeframe...")
        signals_30m = await self.scan_all_symbols_with_timeframe('30m')
        print(f"📊 Raw signals found: {len(signals_30m)}")
        
        if signals_30m:
            print("🔍 Filtering signals for best trades...")
            best_trades_30m = self.select_best_scalping_trades(signals_30m)
            print(f"🎯 Best trades after filtering: {len(best_trades_30m)}")
            
            if best_trades_30m:
                print(f"✅ Found {len(best_trades_30m)} valid signals on 30-minute timeframe!")
                return best_trades_30m
            else:
                print("⚠️ Signals found but filtered out by quality checks")
        
        print("⚠️ No valid signals found on 30-minute timeframe")
        print("🔄 Starting continuous 5-minute scanning (24/7)...")
        
        # Round 3: Continuous 5-minute scanning
        return await self.continuous_5m_scan()

    async def continuous_5m_scan(self) -> List[ScalpingSignal]:
        """Continuous 5-minute scanning until signals are found"""
        print("\n" + "=" * 80)
        print("🔄 CONTINUOUS 5-MINUTE SCANNING MODE (24/7)")
        print("=" * 80)
        print("⚡ Scanning every 5 minutes until valid signals are found...")
        print("🛑 Press Ctrl+C to stop")
        
        scan_count = 0
        
        while True:
            try:
                scan_count += 1
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"\n🔍 SCAN #{scan_count} - {current_time}")
                print("-" * 60)
                
                # Scan with 5-minute timeframe
                start_time = time.time()
                signals_5m = await self.scan_all_symbols_with_timeframe('5m')
                best_trades_5m = self.select_best_scalping_trades(signals_5m)
                scan_duration = time.time() - start_time
                
                print(f"⏱️ Scan completed in {scan_duration:.1f} seconds")
                print(f"📊 Total signals: {len(signals_5m)}")
                print(f"🎯 Best trades: {len(best_trades_5m)}")
                
                if best_trades_5m:
                    print(f"🎉 SUCCESS! Found {len(best_trades_5m)} valid signals!")
                    print("🔄 Stopping continuous scan...")
                    return best_trades_5m
                
                print("⚠️ No valid signals found in this scan")
                print("⏳ Waiting 5 minutes before next scan...")
                
                # Wait 5 minutes before next scan
                for remaining in range(300, 0, -30):  # Countdown every 30 seconds
                    mins, secs = divmod(remaining, 60)
                    print(f"⏳ Next scan in {mins:02d}:{secs:02d}...", end='\r')
                    await asyncio.sleep(30)
                
                print()  # New line after countdown
                
            except KeyboardInterrupt:
                print("\n🛑 Continuous scanning stopped by user")
                print(f"📊 Total scans performed: {scan_count}")
                return []
            except Exception as e:
                print(f"❌ Error in continuous scan #{scan_count}: {e}")
                print("⏳ Waiting 1 minute before retry...")
                await asyncio.sleep(60)

    def select_best_scalping_trades(self, signals: List[ScalpingSignal]) -> List[ScalpingSignal]:
        """Select best scalping trades (quick, high-confidence)"""
        if not signals:
            return []
        
        print(f"🔍 Filtering {len(signals)} signals...")
        
        # Sort by score (prioritize quick opportunities)
        sorted_signals = sorted(signals, key=lambda x: x.score, reverse=True)
        
        selected = []
        used_strategies = {}
        filtered_reasons = []
        
        for signal in sorted_signals:
            # Debug: Check each filter condition
            reasons = []
            
            if signal.confidence < 70:  # Lowered from 75 to 70
                reasons.append(f"Low confidence: {signal.confidence:.1f}%")
            
            if abs(signal.price_change_15m) > 8:  # Increased from 5 to 8
                reasons.append(f"Extreme price change: {signal.price_change_15m:.2f}%")
            
            if signal.volume_24h < 100000:  # Lowered from 500000 to 100000
                reasons.append(f"Low volume: ${signal.volume_24h:,.0f}")
            
            # If any filter fails, record reason and skip
            if reasons:
                filtered_reasons.append(f"{signal.symbol}: {', '.join(reasons)}")
                continue
            
            # Limit per strategy for diversification
            strategy_count = used_strategies.get(signal.strategy, 0)
            if strategy_count >= 2:
                filtered_reasons.append(f"{signal.symbol}: Strategy limit reached ({signal.strategy})")
                continue
            
            selected.append(signal)
            used_strategies[signal.strategy] = strategy_count + 1
            print(f"✅ Selected: {signal.symbol} ({signal.confidence:.1f}% confidence)")
            
            if len(selected) >= self.max_trades:
                break
        
        # Show filtering details if signals were filtered out
        if filtered_reasons and len(selected) == 0:
            print("⚠️ All signals filtered out. Reasons:")
            for reason in filtered_reasons[:5]:  # Show first 5 reasons
                print(f"   - {reason}")
        
        print(f"🎯 Final selection: {len(selected)}/{len(signals)} signals passed filters")
        return selected

    def print_scalping_results(self, signals: List[ScalpingSignal]):
        """Print scalping scan results"""
        print(f"\n{'='*80}")
        print(f"⚡ SCALPING SCAN RESULTS: {len(signals)} QUICK OPPORTUNITIES FOUND")
        print(f"{'='*80}")
        
        if not signals:
            print("❌ No scalping opportunities found")
            return
        
        for i, signal in enumerate(signals, 1):
            profit_pct = ((signal.target_price - signal.current_price) / signal.current_price) * 100
            risk_pct = ((signal.current_price - signal.stop_loss) / signal.current_price) * 100
            
            print(f"\n⚡ SCALPING SIGNAL #{i}")
            print(f"Symbol: {signal.symbol}")
            print(f"Action: {signal.action} ({'🟢 LONG' if signal.action == 'BUY' else '🔴 SHORT'})")
            print(f"Strategy: {signal.strategy}")
            print(f"Confidence: {signal.confidence:.1f}%")
            print(f"Current Price: ${signal.current_price:.6f}")
            print(f"Target: ${signal.target_price:.6f} ({profit_pct:.2f}% profit)")
            print(f"Stop Loss: ${signal.stop_loss:.6f} ({risk_pct:.2f}% risk)")
            print(f"15m Change: {signal.price_change_15m:.2f}%")
            print(f"RSI (15m): {signal.rsi_15m:.1f}")
            print(f"Volume Spike: {'✅' if signal.volume_spike else '❌'}")
            print(f"EMA Cross: {'✅' if signal.ema_cross else '❌'}")
            print(f"Score: {signal.score:.1f}")
            print("-" * 50)

async def main():
    """Main adaptive scalping scanner function"""
    try:
        with open('config/credentials.json', 'r') as f:
            creds = json.load(f)
        api_key = creds['binance']['api_key']
        api_secret = creds['binance']['api_secret']
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        return
    
    scanner = ScalpingScanner(api_key, api_secret, testnet=True)
    
    print("🎯 ADAPTIVE SCALPING SCANNER")
    print("=" * 80)
    print("📋 Scanning Strategy:")
    print("   1️⃣ First: 15-minute timeframe")
    print("   2️⃣ If no signals: 30-minute timeframe") 
    print("   3️⃣ If still no signals: Continuous 5-minute scanning (24/7)")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run adaptive scanning
    best_trades = await scanner.adaptive_scan()
    
    total_time = time.time() - start_time
    
    if best_trades:
        print(f"\n⏱️ Total scanning time: {total_time:.1f} seconds")
        print(f"🎉 Final result: {len(best_trades)} high-quality scalping signals found!")
        
        scanner.print_scalping_results(best_trades)
        
        # Determine scan type based on strategy names
        scan_type = 'adaptive_scalping'
        if best_trades[0].strategy.endswith('_15m'):
            scan_type = '15_minute_scalping'
        elif best_trades[0].strategy.endswith('_30m'):
            scan_type = '30_minute_scalping'
        elif best_trades[0].strategy.endswith('_5m'):
            scan_type = '5_minute_continuous_scalping'
        
        # Save results
        results = {
            'scan_type': scan_type,
            'timestamp': datetime.now().isoformat(),
            'total_scan_duration': total_time,
            'selected_trades': [
                {
                    'symbol': s.symbol,
                    'action': s.action,
                    'confidence': s.confidence,
                    'strategy': s.strategy,
                    'current_price': s.current_price,
                    'target_price': s.target_price,
                    'stop_loss': s.stop_loss,
                    'score': s.score,
                    'price_change': s.price_change_15m,  # Now represents timeframe change
                    'rsi': s.rsi_15m,  # Now represents timeframe RSI
                    'volume_spike': bool(s.volume_spike),  # Convert to JSON-serializable bool
                    'ema_cross': bool(s.ema_cross)  # Convert to JSON-serializable bool
                }
                for s in best_trades
            ]
        }
        
        with open('scalping_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to scalping_results.json")
        
    else:
        print(f"\n⏱️ Total scanning time: {total_time:.1f} seconds")
        print("❌ No scalping signals found after all scanning attempts")
        print("💡 Try running again later when market conditions change")

if __name__ == "__main__":
    asyncio.run(main())