#!/usr/bin/env python3
"""
Strategy Scanner - Uses Your Original Bot Strategies
Implements: Liquidity, Momentum, Pattern, Candlestick strategies
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
class StrategySignal:
    symbol: str
    action: str  # BUY/SELL
    confidence: float
    strategy: str
    current_price: float
    target_price: float
    stop_loss: float
    volume_24h: float
    price_change_24h: float
    # Strategy-specific data
    liquidity_score: Optional[float] = None
    momentum_score: Optional[float] = None
    pattern_detected: Optional[str] = None
    candlestick_pattern: Optional[str] = None
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    score: float = 0.0

class StrategyScanner:
    """Scanner using your original bot strategies"""
    
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

    def calculate_technical_indicators(self, klines: List) -> Dict[str, Any]:
        """Calculate technical indicators for strategy analysis"""
        try:
            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            
            if len(closes) < 50:
                return {}
            
            # RSI
            rsi = self.calculate_rsi(closes, 14)
            
            # MACD
            macd_line, signal_line = self.calculate_macd(closes)
            macd_signal = 'bullish' if macd_line > signal_line else 'bearish'
            
            # Moving Averages
            sma_20 = np.mean(closes[-20:])
            sma_50 = np.mean(closes[-50:])
            ema_12 = self.calculate_ema(closes, 12)
            ema_26 = self.calculate_ema(closes, 26)
            
            # Bollinger Bands
            bb_upper, bb_lower = self.calculate_bollinger_bands(closes, 20, 2)
            
            # Volume analysis
            avg_volume = np.mean(volumes[-20:])
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
            
            return {
                'rsi': rsi,
                'macd_signal': macd_signal,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'volume_ratio': volume_ratio,
                'current_price': closes[-1],
                'price_data': {
                    'closes': closes,
                    'highs': highs,
                    'lows': lows,
                    'volumes': volumes
                }
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating indicators: {e}")
            return {}

    def analyze_liquidity_strategy(self, symbol: str, indicators: Dict, ticker: Dict) -> Optional[Dict]:
        """Analyze using liquidity strategy logic"""
        try:
            current_price = indicators['current_price']
            volume_ratio = indicators['volume_ratio']
            
            # Liquidity analysis (simplified version of your original strategy)
            liquidity_score = 0
            
            # High volume indicates good liquidity
            if volume_ratio > 1.5:
                liquidity_score += 30
            elif volume_ratio > 1.2:
                liquidity_score += 15
            
            # Price stability indicates liquidity
            price_volatility = np.std(indicators['price_data']['closes'][-10:]) / current_price
            if price_volatility < 0.02:  # Low volatility
                liquidity_score += 20
            
            # Spread analysis (using price action as proxy)
            recent_highs = indicators['price_data']['highs'][-5:]
            recent_lows = indicators['price_data']['lows'][-5:]
            avg_spread = np.mean([(h - l) / l for h, l in zip(recent_highs, recent_lows)])
            
            if avg_spread < 0.01:  # Tight spread
                liquidity_score += 25
            
            # Volume-price relationship
            volume_24h = float(ticker['volume'])
            if volume_24h > 1000000:  # High 24h volume
                liquidity_score += 25
            
            if liquidity_score >= 50:  # Minimum threshold
                # Determine action based on liquidity patterns
                if (volume_ratio > 1.3 and 
                    current_price > indicators['sma_20']):
                    return {
                        'action': 'BUY',
                        'confidence': min(liquidity_score, 90),
                        'target': current_price * 1.03,
                        'stop_loss': current_price * 0.98,
                        'liquidity_score': liquidity_score
                    }
                elif (volume_ratio > 1.3 and 
                      current_price < indicators['sma_20']):
                    return {
                        'action': 'SELL',
                        'confidence': min(liquidity_score, 90),
                        'target': current_price * 0.97,
                        'stop_loss': current_price * 1.02,
                        'liquidity_score': liquidity_score
                    }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Liquidity analysis error: {e}")
            return None

    def analyze_momentum_strategy(self, symbol: str, indicators: Dict, ticker: Dict) -> Optional[Dict]:
        """Analyze using momentum strategy logic"""
        try:
            current_price = indicators['current_price']
            rsi = indicators['rsi']
            macd_signal = indicators['macd_signal']
            
            momentum_score = 0
            
            # RSI momentum
            if 30 < rsi < 70:  # Not extreme
                momentum_score += 20
            
            # MACD momentum
            if macd_signal == 'bullish':
                momentum_score += 25
            
            # Moving average momentum
            if (current_price > indicators['sma_20'] > indicators['sma_50']):
                momentum_score += 30  # Uptrend
            elif (current_price < indicators['sma_20'] < indicators['sma_50']):
                momentum_score += 30  # Downtrend
            
            # Price momentum
            price_change_24h = float(ticker['priceChangePercent'])
            if 2 < abs(price_change_24h) < 8:  # Good momentum, not extreme
                momentum_score += 25
            
            if momentum_score >= 60:  # Minimum threshold
                if (macd_signal == 'bullish' and 
                    current_price > indicators['sma_20'] and
                    price_change_24h > 0):
                    return {
                        'action': 'BUY',
                        'confidence': min(momentum_score, 95),
                        'target': current_price * 1.04,
                        'stop_loss': current_price * 0.97,
                        'momentum_score': momentum_score
                    }
                elif (macd_signal == 'bearish' and 
                      current_price < indicators['sma_20'] and
                      price_change_24h < 0):
                    return {
                        'action': 'SELL',
                        'confidence': min(momentum_score, 95),
                        'target': current_price * 0.96,
                        'stop_loss': current_price * 1.03,
                        'momentum_score': momentum_score
                    }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Momentum analysis error: {e}")
            return None

    def analyze_pattern_strategy(self, symbol: str, indicators: Dict, ticker: Dict) -> Optional[Dict]:
        """Analyze using pattern recognition strategy"""
        try:
            closes = indicators['price_data']['closes']
            highs = indicators['price_data']['highs']
            lows = indicators['price_data']['lows']
            current_price = indicators['current_price']
            
            # Simplified pattern detection
            pattern_score = 0
            detected_pattern = None
            
            # Support/Resistance levels
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            resistance = np.max(recent_highs)
            support = np.min(recent_lows)
            
            # Breakout patterns
            if current_price > resistance * 1.005:  # Breakout above resistance
                pattern_score += 40
                detected_pattern = 'resistance_breakout'
            elif current_price < support * 0.995:  # Breakdown below support
                pattern_score += 40
                detected_pattern = 'support_breakdown'
            
            # Triangle patterns (simplified)
            if len(closes) >= 30:
                recent_closes = closes[-30:]
                trend_slope = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0]
                
                if abs(trend_slope) < current_price * 0.0001:  # Sideways movement
                    volatility = np.std(recent_closes) / current_price
                    if volatility < 0.02:  # Low volatility triangle
                        pattern_score += 30
                        detected_pattern = 'triangle_consolidation'
            
            # Bollinger Band patterns
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            
            if current_price >= bb_upper:  # At upper band
                pattern_score += 25
                detected_pattern = 'bb_upper_touch'
            elif current_price <= bb_lower:  # At lower band
                pattern_score += 25
                detected_pattern = 'bb_lower_touch'
            
            if pattern_score >= 35 and detected_pattern:
                if detected_pattern in ['resistance_breakout', 'bb_lower_touch']:
                    return {
                        'action': 'BUY',
                        'confidence': min(pattern_score + 20, 90),
                        'target': current_price * 1.035,
                        'stop_loss': current_price * 0.975,
                        'pattern_detected': detected_pattern
                    }
                elif detected_pattern in ['support_breakdown', 'bb_upper_touch']:
                    return {
                        'action': 'SELL',
                        'confidence': min(pattern_score + 20, 90),
                        'target': current_price * 0.965,
                        'stop_loss': current_price * 1.025,
                        'pattern_detected': detected_pattern
                    }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Pattern analysis error: {e}")
            return None

    def analyze_candlestick_strategy(self, symbol: str, indicators: Dict, ticker: Dict) -> Optional[Dict]:
        """Analyze using candlestick pattern strategy"""
        try:
            closes = indicators['price_data']['closes']
            highs = indicators['price_data']['highs']
            lows = indicators['price_data']['lows']
            
            if len(closes) < 5:
                return None
            
            # Get last few candlesticks
            last_candles = []
            for i in range(-3, 0):  # Last 3 candles
                open_price = closes[i-1] if i > -len(closes) else closes[i]
                last_candles.append({
                    'open': open_price,
                    'high': highs[i],
                    'low': lows[i],
                    'close': closes[i],
                    'body_size': abs(closes[i] - open_price),
                    'is_bullish': closes[i] > open_price
                })
            
            current_price = closes[-1]
            candlestick_score = 0
            detected_pattern = None
            
            # Doji pattern
            if last_candles[-1]['body_size'] < current_price * 0.001:
                candlestick_score += 30
                detected_pattern = 'doji'
            
            # Hammer/Hanging Man
            last_candle = last_candles[-1]
            lower_shadow = last_candle['low'] - min(last_candle['open'], last_candle['close'])
            upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            
            if (lower_shadow > last_candle['body_size'] * 2 and 
                upper_shadow < last_candle['body_size'] * 0.5):
                candlestick_score += 35
                detected_pattern = 'hammer' if last_candle['is_bullish'] else 'hanging_man'
            
            # Engulfing patterns
            if len(last_candles) >= 2:
                prev_candle = last_candles[-2]
                curr_candle = last_candles[-1]
                
                # Bullish engulfing
                if (not prev_candle['is_bullish'] and curr_candle['is_bullish'] and
                    curr_candle['close'] > prev_candle['open'] and
                    curr_candle['open'] < prev_candle['close']):
                    candlestick_score += 40
                    detected_pattern = 'bullish_engulfing'
                
                # Bearish engulfing
                elif (prev_candle['is_bullish'] and not curr_candle['is_bullish'] and
                      curr_candle['close'] < prev_candle['open'] and
                      curr_candle['open'] > prev_candle['close']):
                    candlestick_score += 40
                    detected_pattern = 'bearish_engulfing'
            
            if candlestick_score >= 30 and detected_pattern:
                if detected_pattern in ['hammer', 'bullish_engulfing']:
                    return {
                        'action': 'BUY',
                        'confidence': min(candlestick_score + 25, 90),
                        'target': current_price * 1.025,
                        'stop_loss': current_price * 0.985,
                        'candlestick_pattern': detected_pattern
                    }
                elif detected_pattern in ['hanging_man', 'bearish_engulfing']:
                    return {
                        'action': 'SELL',
                        'confidence': min(candlestick_score + 25, 90),
                        'target': current_price * 0.975,
                        'stop_loss': current_price * 1.015,
                        'candlestick_pattern': detected_pattern
                    }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Candlestick analysis error: {e}")
            return None

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
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
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: List[float]) -> tuple:
        """Calculate MACD"""
        if len(prices) < 26:
            return 0, 0
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26
        # Simplified signal line
        signal_line = macd_line * 0.9  # Approximation
        return macd_line, signal_line

    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return np.mean(prices) if prices else 0
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema

    def calculate_bollinger_bands(self, prices: List[float], period: int, std_dev: float) -> tuple:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return 0, 0
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower

    async def analyze_symbol_with_strategies(self, symbol: str, klines: List, ticker: Dict) -> Optional[StrategySignal]:
        """Analyze symbol using all original bot strategies"""
        try:
            indicators = self.calculate_technical_indicators(klines)
            if not indicators:
                return None
            
            current_price = indicators['current_price']
            volume_24h = float(ticker['volume'])
            price_change_24h = float(ticker['priceChangePercent'])
            
            # Run all strategies
            strategies = [
                ('liquidity', self.analyze_liquidity_strategy),
                ('momentum', self.analyze_momentum_strategy),
                ('pattern', self.analyze_pattern_strategy),
                ('candlestick', self.analyze_candlestick_strategy)
            ]
            
            best_signal = None
            best_confidence = 0
            
            for strategy_name, strategy_func in strategies:
                try:
                    result = strategy_func(symbol, indicators, ticker)
                    if result and result['confidence'] > best_confidence:
                        best_confidence = result['confidence']
                        best_signal = {
                            'strategy': strategy_name,
                            'action': result['action'],
                            'confidence': result['confidence'],
                            'target': result['target'],
                            'stop_loss': result['stop_loss'],
                            'extra_data': {k: v for k, v in result.items() 
                                         if k not in ['action', 'confidence', 'target', 'stop_loss']}
                        }
                except Exception as e:
                    print(f"⚠️ Strategy {strategy_name} error for {symbol}: {e}")
                    continue
            
            if best_signal and best_signal['confidence'] >= 70:
                # Calculate composite score
                score = (
                    best_signal['confidence'] * 0.6 +
                    min(volume_24h / 1000000, 50) * 0.2 +
                    min(abs(price_change_24h) * 5, 30) * 0.2
                )
                
                return StrategySignal(
                    symbol=symbol,
                    action=best_signal['action'],
                    confidence=best_signal['confidence'],
                    strategy=best_signal['strategy'],
                    current_price=current_price,
                    target_price=best_signal['target'],
                    stop_loss=best_signal['stop_loss'],
                    volume_24h=volume_24h,
                    price_change_24h=price_change_24h,
                    liquidity_score=best_signal['extra_data'].get('liquidity_score'),
                    momentum_score=best_signal['extra_data'].get('momentum_score'),
                    pattern_detected=best_signal['extra_data'].get('pattern_detected'),
                    candlestick_pattern=best_signal['extra_data'].get('candlestick_pattern'),
                    rsi=indicators['rsi'],
                    macd_signal=indicators['macd_signal'],
                    score=score
                )
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error analyzing {symbol}: {e}")
            return None

    async def scan_batch(self, symbols: List[str], batch_num: int) -> List[StrategySignal]:
        """Scan batch using original bot strategies"""
        print(f"🔍 Scanning batch {batch_num}: {len(symbols)} symbols (Original Strategies)")
        signals = []
        
        try:
            tickers = self.client.futures_ticker()
            ticker_dict = {t['symbol']: t for t in tickers if t['symbol'] in symbols}
            
            for symbol in symbols:
                try:
                    if symbol not in ticker_dict:
                        continue
                        
                    # Get 1-hour klines for strategy analysis
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=Client.KLINE_INTERVAL_1HOUR,
                        limit=100
                    )
                    
                    signal = await self.analyze_symbol_with_strategies(symbol, klines, ticker_dict[symbol])
                    if signal:
                        signals.append(signal)
                        
                    time.sleep(0.1)
                    
                except BinanceAPIException as e:
                    if e.code == -1121:
                        continue
                    print(f"⚠️ API error for {symbol}: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ Error processing {symbol}: {e}")
                    continue
            
            print(f"✅ Batch {batch_num} complete: {len(signals)} strategy signals found")
            return signals
            
        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {e}")
            return []

    async def scan_all_symbols(self) -> List[StrategySignal]:
        """Scan all symbols using original strategies"""
        print("🚀 Starting strategy scan (Original Bot Strategies)...")
        
        all_symbols = await self.get_futures_symbols()
        if not all_symbols:
            return []
        
        all_signals = []
        total_batches = (len(all_symbols) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(all_symbols), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch_symbols = all_symbols[i:i + self.batch_size]
            
            print(f"\n📊 Processing batch {batch_num}/{total_batches}")
            batch_signals = await self.scan_batch(batch_symbols, batch_num)
            all_signals.extend(batch_signals)
            
            print(f"📈 Progress: {batch_num}/{total_batches} batches, {len(all_signals)} total signals")
            
            if batch_num < total_batches:
                print("⏳ Waiting 2 seconds before next batch...")
                time.sleep(2)
        
        return all_signals

    def select_best_strategy_trades(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """Select best trades from strategy analysis"""
        if not signals:
            return []
        
        sorted_signals = sorted(signals, key=lambda x: x.score, reverse=True)
        
        selected = []
        strategy_counts = {}
        
        for signal in sorted_signals:
            # Strategy diversification
            count = strategy_counts.get(signal.strategy, 0)
            if count >= 2:  # Max 2 per strategy
                continue
            
            # Quality filters
            if (signal.confidence < 75 or 
                abs(signal.price_change_24h) > 12 or
                signal.volume_24h < 200000):
                continue
            
            selected.append(signal)
            strategy_counts[signal.strategy] = count + 1
            
            if len(selected) >= self.max_trades:
                break
        
        return selected

    def print_strategy_results(self, signals: List[StrategySignal]):
        """Print strategy scan results"""
        print(f"\n{'='*80}")
        print(f"🎯 STRATEGY SCAN RESULTS: {len(signals)} SIGNALS FROM ORIGINAL BOT STRATEGIES")
        print(f"{'='*80}")
        
        if not signals:
            print("❌ No strategy signals found")
            return
        
        for i, signal in enumerate(signals, 1):
            profit_pct = ((signal.target_price - signal.current_price) / signal.current_price) * 100
            risk_pct = ((signal.current_price - signal.stop_loss) / signal.current_price) * 100
            
            print(f"\n🎯 STRATEGY SIGNAL #{i}")
            print(f"Symbol: {signal.symbol}")
            print(f"Action: {signal.action} ({'🟢 LONG' if signal.action == 'BUY' else '🔴 SHORT'})")
            print(f"Strategy: {signal.strategy.upper()}")
            print(f"Confidence: {signal.confidence:.1f}%")
            print(f"Current Price: ${signal.current_price:.6f}")
            print(f"Target: ${signal.target_price:.6f} ({profit_pct:.2f}% profit)")
            print(f"Stop Loss: ${signal.stop_loss:.6f} ({risk_pct:.2f}% risk)")
            print(f"24h Change: {signal.price_change_24h:.2f}%")
            print(f"24h Volume: ${signal.volume_24h:,.0f}")
            print(f"RSI: {signal.rsi:.1f}")
            print(f"MACD: {signal.macd_signal}")
            
            # Strategy-specific details
            if signal.liquidity_score:
                print(f"Liquidity Score: {signal.liquidity_score:.1f}")
            if signal.momentum_score:
                print(f"Momentum Score: {signal.momentum_score:.1f}")
            if signal.pattern_detected:
                print(f"Pattern: {signal.pattern_detected}")
            if signal.candlestick_pattern:
                print(f"Candlestick: {signal.candlestick_pattern}")
            
            print(f"Overall Score: {signal.score:.1f}")
            print("-" * 50)

async def main():
    """Main strategy scanner function"""
    try:
        with open('config/credentials.json', 'r') as f:
            creds = json.load(f)
        api_key = creds['binance']['api_key']
        api_secret = creds['binance']['api_secret']
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        return
    
    scanner = StrategyScanner(api_key, api_secret, testnet=True)
    
    start_time = time.time()
    all_signals = await scanner.scan_all_symbols()
    best_trades = scanner.select_best_strategy_trades(all_signals)
    
    scan_time = time.time() - start_time
    print(f"\n⏱️ Strategy scan completed in {scan_time:.1f} seconds")
    print(f"📊 Total signals found: {len(all_signals)}")
    print(f"🎯 Best strategy trades: {len(best_trades)}")
    
    scanner.print_strategy_results(best_trades)
    
    # Save results
    if best_trades:
        results = {
            'scan_type': 'original_bot_strategies',
            'timestamp': datetime.now().isoformat(),
            'scan_duration': scan_time,
            'total_signals': len(all_signals),
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
                    'liquidity_score': s.liquidity_score,
                    'momentum_score': s.momentum_score,
                    'pattern_detected': s.pattern_detected,
                    'candlestick_pattern': s.candlestick_pattern,
                    'rsi': s.rsi,
                    'macd_signal': s.macd_signal
                }
                for s in best_trades
            ]
        }
        
        with open('strategy_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Strategy results saved to strategy_results.json")

if __name__ == "__main__":
    asyncio.run(main())