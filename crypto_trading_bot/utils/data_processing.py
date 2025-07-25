"""
Market data preprocessing and normalization utilities.

This module provides functions for cleaning, filtering, and normalizing
market data for analysis and strategy consumption.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import deque
import statistics
import math

from ..models.trading import MarketData, OrderBook


class OutlierDetector:
    """Detects and filters outliers in market data."""
    
    def __init__(self, window_size: int = 50, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.logger = logging.getLogger(__name__)
    
    def is_price_outlier(self, symbol: str, price: float) -> bool:
        """Check if price is an outlier."""
        try:
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.window_size)
            
            history = self.price_history[symbol]
            
            # Need at least 10 data points for meaningful statistics
            if len(history) < 10:
                history.append(price)
                return False
            
            # Calculate z-score
            mean_price = statistics.mean(history)
            std_price = statistics.stdev(history)
            
            if std_price == 0:
                history.append(price)
                return False
            
            z_score = abs((price - mean_price) / std_price)
            is_outlier = z_score > self.z_threshold
            
            if not is_outlier:
                history.append(price)
            else:
                self.logger.warning(f"Price outlier detected for {symbol}: {price} (z-score: {z_score:.2f})")
            
            return is_outlier
            
        except Exception as e:
            self.logger.error(f"Error detecting price outlier: {e}")
            return False
    
    def is_volume_outlier(self, symbol: str, volume: float) -> bool:
        """Check if volume is an outlier."""
        try:
            if symbol not in self.volume_history:
                self.volume_history[symbol] = deque(maxlen=self.window_size)
            
            history = self.volume_history[symbol]
            
            if len(history) < 10:
                history.append(volume)
                return False
            
            # Use log transformation for volume to handle skewness
            log_history = [math.log(v + 1) for v in history if v > 0]
            if not log_history:
                history.append(volume)
                return False
            
            mean_log_volume = statistics.mean(log_history)
            std_log_volume = statistics.stdev(log_history)
            
            if std_log_volume == 0:
                history.append(volume)
                return False
            
            log_volume = math.log(volume + 1)
            z_score = abs((log_volume - mean_log_volume) / std_log_volume)
            is_outlier = z_score > self.z_threshold
            
            if not is_outlier:
                history.append(volume)
            else:
                self.logger.warning(f"Volume outlier detected for {symbol}: {volume} (z-score: {z_score:.2f})")
            
            return is_outlier
            
        except Exception as e:
            self.logger.error(f"Error detecting volume outlier: {e}")
            return False


class DataSmoother:
    """Smooths market data using various techniques."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def exponential_moving_average(self, values: List[float], alpha: float = 0.1) -> List[float]:
        """Calculate exponential moving average."""
        if not values:
            return []
        
        try:
            ema = [values[0]]  # First value is the seed
            
            for i in range(1, len(values)):
                ema_value = alpha * values[i] + (1 - alpha) * ema[i-1]
                ema.append(ema_value)
            
            return ema
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return values
    
    def simple_moving_average(self, values: List[float], window: int = 5) -> List[float]:
        """Calculate simple moving average."""
        if not values or window <= 0:
            return values
        
        try:
            sma = []
            
            for i in range(len(values)):
                start_idx = max(0, i - window + 1)
                window_values = values[start_idx:i+1]
                sma.append(sum(window_values) / len(window_values))
            
            return sma
            
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return values
    
    def median_filter(self, values: List[float], window: int = 3) -> List[float]:
        """Apply median filter to remove spikes."""
        if not values or window <= 0:
            return values
        
        try:
            filtered = []
            
            for i in range(len(values)):
                start_idx = max(0, i - window // 2)
                end_idx = min(len(values), i + window // 2 + 1)
                window_values = values[start_idx:end_idx]
                filtered.append(statistics.median(window_values))
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error applying median filter: {e}")
            return values


class TimestampSynchronizer:
    """Synchronizes timestamps across different data sources."""
    
    def __init__(self, tolerance_ms: int = 1000):
        self.tolerance_ms = tolerance_ms
        self.logger = logging.getLogger(__name__)
    
    def synchronize_data(self, data_list: List[MarketData]) -> List[MarketData]:
        """Synchronize timestamps in market data list."""
        if not data_list:
            return []
        
        try:
            # Sort by timestamp
            sorted_data = sorted(data_list, key=lambda x: x.timestamp)
            
            # Group data by time windows
            synchronized = []
            current_group = [sorted_data[0]]
            base_time = sorted_data[0].timestamp
            
            for data in sorted_data[1:]:
                time_diff = abs((data.timestamp - base_time).total_seconds() * 1000)
                
                if time_diff <= self.tolerance_ms:
                    current_group.append(data)
                else:
                    # Process current group
                    if len(current_group) > 1:
                        synced_data = self._merge_group(current_group)
                        synchronized.append(synced_data)
                    else:
                        synchronized.extend(current_group)
                    
                    # Start new group
                    current_group = [data]
                    base_time = data.timestamp
            
            # Process final group
            if len(current_group) > 1:
                synced_data = self._merge_group(current_group)
                synchronized.append(synced_data)
            else:
                synchronized.extend(current_group)
            
            return synchronized
            
        except Exception as e:
            self.logger.error(f"Error synchronizing timestamps: {e}")
            return data_list
    
    def _merge_group(self, group: List[MarketData]) -> MarketData:
        """Merge a group of market data with similar timestamps."""
        if len(group) == 1:
            return group[0]
        
        # Use the most recent timestamp
        latest_data = max(group, key=lambda x: x.timestamp)
        
        # Average prices and volumes
        avg_price = sum(d.price for d in group) / len(group)
        avg_volume = sum(d.volume for d in group) / len(group)
        avg_bid = sum(d.bid for d in group) / len(group)
        avg_ask = sum(d.ask for d in group) / len(group)
        
        # Create merged data
        merged = MarketData(
            symbol=latest_data.symbol,
            timestamp=latest_data.timestamp,
            price=avg_price,
            volume=avg_volume,
            bid=avg_bid,
            ask=avg_ask,
            orderbook=latest_data.orderbook,  # Use latest order book
            high_24h=latest_data.high_24h,
            low_24h=latest_data.low_24h,
            change_24h=latest_data.change_24h,
            volume_24h=latest_data.volume_24h
        )
        
        return merged


class DataNormalizer:
    """Normalizes market data for analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.price_ranges: Dict[str, Tuple[float, float]] = {}
        self.volume_ranges: Dict[str, Tuple[float, float]] = {}
    
    def normalize_prices(self, symbol: str, prices: List[float], 
                        method: str = 'minmax') -> List[float]:
        """Normalize price data."""
        if not prices:
            return []
        
        try:
            if method == 'minmax':
                return self._minmax_normalize(prices)
            elif method == 'zscore':
                return self._zscore_normalize(prices)
            elif method == 'percentage':
                return self._percentage_normalize(prices)
            else:
                self.logger.warning(f"Unknown normalization method: {method}")
                return prices
                
        except Exception as e:
            self.logger.error(f"Error normalizing prices: {e}")
            return prices
    
    def _minmax_normalize(self, values: List[float]) -> List[float]:
        """Min-max normalization (0-1 range)."""
        if not values:
            return []
        
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            return [0.5] * len(values)  # All values are the same
        
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    def _zscore_normalize(self, values: List[float]) -> List[float]:
        """Z-score normalization (mean=0, std=1)."""
        if len(values) < 2:
            return [0.0] * len(values)
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if std_val == 0:
            return [0.0] * len(values)
        
        return [(v - mean_val) / std_val for v in values]
    
    def _percentage_normalize(self, values: List[float]) -> List[float]:
        """Percentage change normalization."""
        if not values:
            return []
        
        if len(values) == 1:
            return [0.0]
        
        base_value = values[0]
        if base_value == 0:
            return [0.0] * len(values)
        
        return [(v - base_value) / base_value * 100 for v in values]
    
    def denormalize_prices(self, symbol: str, normalized_prices: List[float], 
                          method: str = 'minmax') -> List[float]:
        """Denormalize price data back to original scale."""
        if not normalized_prices or symbol not in self.price_ranges:
            return normalized_prices
        
        try:
            min_val, max_val = self.price_ranges[symbol]
            
            if method == 'minmax':
                return [v * (max_val - min_val) + min_val for v in normalized_prices]
            else:
                # For other methods, we'd need to store additional parameters
                return normalized_prices
                
        except Exception as e:
            self.logger.error(f"Error denormalizing prices: {e}")
            return normalized_prices


class DataConverter:
    """Converts between different data formats and structures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def market_data_to_ohlcv(self, data_list: List[MarketData], 
                            interval_minutes: int = 1) -> List[Dict[str, Any]]:
        """Convert market data to OHLCV format."""
        if not data_list:
            return []
        
        try:
            # Group data by time intervals
            interval_delta = timedelta(minutes=interval_minutes)
            grouped_data = {}
            
            for data in data_list:
                # Round timestamp to interval
                interval_start = self._round_timestamp(data.timestamp, interval_delta)
                
                if interval_start not in grouped_data:
                    grouped_data[interval_start] = []
                
                grouped_data[interval_start].append(data)
            
            # Convert each group to OHLCV
            ohlcv_data = []
            
            for timestamp, group in sorted(grouped_data.items()):
                prices = [d.price for d in group]
                volumes = [d.volume for d in group]
                
                ohlcv = {
                    'timestamp': timestamp,
                    'open': prices[0],
                    'high': max(prices),
                    'low': min(prices),
                    'close': prices[-1],
                    'volume': sum(volumes),
                    'count': len(group)
                }
                
                ohlcv_data.append(ohlcv)
            
            return ohlcv_data
            
        except Exception as e:
            self.logger.error(f"Error converting to OHLCV: {e}")
            return []
    
    def _round_timestamp(self, timestamp: datetime, interval: timedelta) -> datetime:
        """Round timestamp to nearest interval."""
        total_seconds = interval.total_seconds()
        rounded_seconds = (timestamp.timestamp() // total_seconds) * total_seconds
        return datetime.fromtimestamp(rounded_seconds)
    
    def order_book_to_levels(self, order_book: OrderBook, 
                           levels: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """Convert order book to price levels."""
        try:
            return {
                'bids': order_book.bids[:levels],
                'asks': order_book.asks[:levels],
                'timestamp': order_book.timestamp,
                'symbol': order_book.symbol
            }
            
        except Exception as e:
            self.logger.error(f"Error converting order book: {e}")
            return {'bids': [], 'asks': [], 'timestamp': datetime.now(), 'symbol': ''}


class MarketDataProcessor:
    """Main processor that combines all preprocessing functions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.outlier_detector = OutlierDetector(
            window_size=self.config.get('outlier_window', 50),
            z_threshold=self.config.get('outlier_threshold', 3.0)
        )
        
        self.smoother = DataSmoother()
        self.synchronizer = TimestampSynchronizer(
            tolerance_ms=self.config.get('sync_tolerance_ms', 1000)
        )
        self.normalizer = DataNormalizer()
        self.converter = DataConverter()
        
        self.logger = logging.getLogger(__name__)
    
    def process_market_data(self, data: MarketData, 
                           enable_outlier_filter: bool = True,
                           enable_smoothing: bool = False) -> Optional[MarketData]:
        """Process a single market data point."""
        try:
            # Outlier detection
            if enable_outlier_filter:
                if (self.outlier_detector.is_price_outlier(data.symbol, data.price) or
                    self.outlier_detector.is_volume_outlier(data.symbol, data.volume)):
                    self.logger.debug(f"Filtered outlier for {data.symbol}")
                    return None
            
            # Additional processing can be added here
            # For now, return the original data if it passes outlier detection
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return None
    
    def process_batch(self, data_list: List[MarketData],
                     synchronize: bool = True,
                     smooth_prices: bool = False,
                     normalize: bool = False) -> List[MarketData]:
        """Process a batch of market data."""
        try:
            if not data_list:
                return []
            
            processed_data = data_list.copy()
            
            # Synchronize timestamps
            if synchronize:
                processed_data = self.synchronizer.synchronize_data(processed_data)
            
            # Apply smoothing
            if smooth_prices:
                processed_data = self._apply_smoothing(processed_data)
            
            # Apply normalization
            if normalize:
                processed_data = self._apply_normalization(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            return data_list
    
    def _apply_smoothing(self, data_list: List[MarketData]) -> List[MarketData]:
        """Apply smoothing to price data."""
        if not data_list:
            return []
        
        try:
            # Group by symbol
            symbol_groups = {}
            for data in data_list:
                if data.symbol not in symbol_groups:
                    symbol_groups[data.symbol] = []
                symbol_groups[data.symbol].append(data)
            
            smoothed_data = []
            
            for symbol, group in symbol_groups.items():
                prices = [d.price for d in group]
                smoothed_prices = self.smoother.exponential_moving_average(prices, alpha=0.1)
                
                # Create new data with smoothed prices
                for i, data in enumerate(group):
                    smoothed = MarketData(
                        symbol=data.symbol,
                        timestamp=data.timestamp,
                        price=smoothed_prices[i],
                        volume=data.volume,
                        bid=data.bid,
                        ask=data.ask,
                        orderbook=data.orderbook,
                        high_24h=data.high_24h,
                        low_24h=data.low_24h,
                        change_24h=data.change_24h,
                        volume_24h=data.volume_24h
                    )
                    smoothed_data.append(smoothed)
            
            # Sort by timestamp
            return sorted(smoothed_data, key=lambda x: x.timestamp)
            
        except Exception as e:
            self.logger.error(f"Error applying smoothing: {e}")
            return data_list
    
    def _apply_normalization(self, data_list: List[MarketData]) -> List[MarketData]:
        """Apply normalization to market data."""
        # For now, return original data
        # Normalization is typically applied to extracted features, not raw market data
        return data_list
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'outlier_detector_stats': {
                'price_history_symbols': len(self.outlier_detector.price_history),
                'volume_history_symbols': len(self.outlier_detector.volume_history)
            },
            'config': self.config
        }