"""
Historical data provider for backtesting.

This module provides access to historical market data from various sources
for backtesting purposes.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import csv


class DataSource(Enum):
    """Available data sources."""
    BINANCE = "binance"
    CSV_FILE = "csv_file"
    JSON_FILE = "json_file"
    MOCK = "mock"


@dataclass
class DataRequest:
    """Request for historical data."""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    timeframe: str = "1h"
    source: DataSource = DataSource.BINANCE
    source_config: Optional[Dict[str, Any]] = None


class HistoricalDataProvider:
    """
    Provider for historical market data.
    
    Supports multiple data sources including Binance API, CSV files,
    and mock data generation for testing.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize data provider."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Data cache
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_enabled = True
    
    async def get_historical_data(self, symbols: List[str], start_date: datetime,
                                end_date: datetime, timeframe: str = "1h",
                                source: DataSource = DataSource.MOCK) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for specified symbols and time range.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            source: Data source to use
            
        Returns:
            Dictionary mapping symbols to DataFrames with OHLCV data
        """
        request = DataRequest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            source=source
        )
        
        self.logger.info(f"Fetching historical data: {symbols} from {start_date} to {end_date}")
        
        data = {}
        
        for symbol in symbols:
            try:
                # Check cache first
                cache_key = self._get_cache_key(symbol, start_date, end_date, timeframe, source)
                
                if self._cache_enabled and cache_key in self._cache:
                    self.logger.debug(f"Using cached data for {symbol}")
                    data[symbol] = self._cache[cache_key]
                    continue
                
                # Fetch data from source
                if source == DataSource.BINANCE:
                    symbol_data = await self._fetch_binance_data(symbol, start_date, end_date, timeframe)
                elif source == DataSource.CSV_FILE:
                    symbol_data = await self._load_csv_data(symbol, start_date, end_date, timeframe)
                elif source == DataSource.JSON_FILE:
                    symbol_data = await self._load_json_data(symbol, start_date, end_date, timeframe)
                elif source == DataSource.MOCK:
                    symbol_data = await self._generate_mock_data(symbol, start_date, end_date, timeframe)
                else:
                    raise ValueError(f"Unsupported data source: {source}")
                
                if symbol_data is not None and not symbol_data.empty:
                    # Cache the data
                    if self._cache_enabled:
                        self._cache[cache_key] = symbol_data
                    
                    data[symbol] = symbol_data
                else:
                    self.logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        return data
    
    def _get_cache_key(self, symbol: str, start_date: datetime, end_date: datetime,
                      timeframe: str, source: DataSource) -> str:
        """Generate cache key for data request."""
        return f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{timeframe}_{source.value}"
    
    async def _fetch_binance_data(self, symbol: str, start_date: datetime,
                                end_date: datetime, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data from Binance API."""
        try:
            # This would integrate with the actual Binance API client
            # For now, we'll simulate the API call
            self.logger.info(f"Fetching Binance data for {symbol}")
            
            # Simulate API delay
            await asyncio.sleep(0.1)
            
            # In a real implementation, this would call the Binance API
            # For now, generate mock data
            return await self._generate_mock_data(symbol, start_date, end_date, timeframe)
            
        except Exception as e:
            self.logger.error(f"Binance API error for {symbol}: {e}")
            return None
    
    async def _load_csv_data(self, symbol: str, start_date: datetime,
                           end_date: datetime, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from CSV file."""
        try:
            csv_file = self.cache_dir / f"{symbol}_{timeframe}.csv"
            
            if not csv_file.exists():
                self.logger.warning(f"CSV file not found: {csv_file}")
                return None
            
            # Load CSV data
            df = pd.read_csv(csv_file)
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV file missing required columns: {required_columns}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date range
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            df = df[mask]
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(df)} records from CSV for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV data for {symbol}: {e}")
            return None
    
    async def _load_json_data(self, symbol: str, start_date: datetime,
                            end_date: datetime, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from JSON file."""
        try:
            json_file = self.cache_dir / f"{symbol}_{timeframe}.json"
            
            if not json_file.exists():
                self.logger.warning(f"JSON file not found: {json_file}")
                return None
            
            # Load JSON data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date range
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            df = df[mask]
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(df)} records from JSON for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading JSON data for {symbol}: {e}")
            return None
    
    async def _generate_mock_data(self, symbol: str, start_date: datetime,
                                end_date: datetime, timeframe: str) -> pd.DataFrame:
        """Generate mock market data for testing."""
        # Parse timeframe
        timeframe_minutes = self._parse_timeframe(timeframe)
        
        # Generate timestamps
        current_time = start_date
        timestamps = []
        
        while current_time <= end_date:
            timestamps.append(current_time)
            current_time += timedelta(minutes=timeframe_minutes)
        
        # Generate realistic price data using random walk
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        
        # Starting price based on symbol
        if 'BTC' in symbol:
            base_price = 45000
        elif 'ETH' in symbol:
            base_price = 3000
        elif 'ADA' in symbol:
            base_price = 1.2
        else:
            base_price = 100
        
        # Generate price series using geometric Brownian motion
        n_periods = len(timestamps)
        returns = np.random.normal(0.0001, 0.02, n_periods)  # Small positive drift with volatility
        
        prices = [base_price]
        for i in range(1, n_periods):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        # Generate OHLCV data
        data = []
        
        for i, timestamp in enumerate(timestamps):
            if i == 0:
                open_price = prices[i]
            else:
                open_price = data[-1]['close']  # Previous close becomes current open
            
            close_price = prices[i]
            
            # Generate high and low based on volatility
            volatility = abs(returns[i]) * 2
            high_price = max(open_price, close_price) * (1 + volatility)
            low_price = min(open_price, close_price) * (1 - volatility)
            
            # Generate volume (higher volume on larger price moves)
            base_volume = 1000000
            volume_multiplier = 1 + abs(returns[i]) * 10
            volume = base_volume * volume_multiplier * np.random.uniform(0.5, 1.5)
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 8),
                'high': round(high_price, 8),
                'low': round(low_price, 8),
                'close': round(close_price, 8),
                'volume': round(volume, 2)
            })
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated {len(df)} mock data points for {symbol}")
        
        return df
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to minutes."""
        timeframe = timeframe.lower()
        
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 24 * 60
        else:
            raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    def save_data_to_cache(self, symbol: str, data: pd.DataFrame, timeframe: str,
                          format: str = "csv") -> bool:
        """Save data to cache file."""
        try:
            if format.lower() == "csv":
                cache_file = self.cache_dir / f"{symbol}_{timeframe}.csv"
                data.to_csv(cache_file, index=False)
            elif format.lower() == "json":
                cache_file = self.cache_dir / f"{symbol}_{timeframe}.json"
                # Convert timestamps to strings for JSON serialization
                data_copy = data.copy()
                data_copy['timestamp'] = data_copy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                data_copy.to_json(cache_file, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved {len(data)} records to cache: {cache_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save data to cache: {e}")
            return False
    
    def load_data_from_cache(self, symbol: str, timeframe: str,
                           format: str = "csv") -> Optional[pd.DataFrame]:
        """Load data from cache file."""
        try:
            if format.lower() == "csv":
                cache_file = self.cache_dir / f"{symbol}_{timeframe}.csv"
                if cache_file.exists():
                    df = pd.read_csv(cache_file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
            elif format.lower() == "json":
                cache_file = self.cache_dir / f"{symbol}_{timeframe}.json"
                if cache_file.exists():
                    df = pd.read_json(cache_file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load data from cache: {e}")
            return None
    
    def clear_cache(self, symbol: Optional[str] = None) -> bool:
        """Clear data cache."""
        try:
            if symbol:
                # Clear cache for specific symbol
                pattern = f"{symbol}_*"
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()
                
                # Clear memory cache
                keys_to_remove = [k for k in self._cache.keys() if k.startswith(symbol)]
                for key in keys_to_remove:
                    del self._cache[key]
                
                self.logger.info(f"Cleared cache for {symbol}")
            else:
                # Clear all cache
                for cache_file in self.cache_dir.glob("*"):
                    if cache_file.is_file():
                        cache_file.unlink()
                
                self._cache.clear()
                self.logger.info("Cleared all cache")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        cache_info = {
            'memory_cache_size': len(self._cache),
            'cache_directory': str(self.cache_dir),
            'cached_files': []
        }
        
        try:
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file():
                    cache_info['cached_files'].append({
                        'filename': cache_file.name,
                        'size_bytes': cache_file.stat().st_size,
                        'modified': datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat()
                    })
        except Exception as e:
            self.logger.error(f"Error getting cache info: {e}")
        
        return cache_info
    
    def validate_data(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Validate historical data quality."""
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            # Check required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing columns: {missing_columns}")
                return validation_result
            
            # Check for missing data
            missing_data = data.isnull().sum()
            if missing_data.any():
                validation_result['warnings'].append(f"Missing data points: {missing_data.to_dict()}")
            
            # Check price consistency (high >= low, etc.)
            invalid_prices = data[(data['high'] < data['low']) | 
                                (data['high'] < data['open']) | 
                                (data['high'] < data['close']) |
                                (data['low'] > data['open']) | 
                                (data['low'] > data['close'])]
            
            if not invalid_prices.empty:
                validation_result['errors'].append(f"Invalid price relationships in {len(invalid_prices)} records")
                validation_result['valid'] = False
            
            # Check for negative prices or volumes
            negative_values = data[(data['open'] <= 0) | (data['high'] <= 0) | 
                                 (data['low'] <= 0) | (data['close'] <= 0) | 
                                 (data['volume'] < 0)]
            
            if not negative_values.empty:
                validation_result['errors'].append(f"Negative or zero values in {len(negative_values)} records")
                validation_result['valid'] = False
            
            # Check for duplicate timestamps
            duplicates = data[data.duplicated(subset=['timestamp'])]
            if not duplicates.empty:
                validation_result['warnings'].append(f"Duplicate timestamps: {len(duplicates)} records")
            
            # Calculate statistics
            validation_result['statistics'] = {
                'total_records': len(data),
                'date_range': {
                    'start': data['timestamp'].min().isoformat() if not data.empty else None,
                    'end': data['timestamp'].max().isoformat() if not data.empty else None
                },
                'price_range': {
                    'min': float(data['low'].min()) if not data.empty else None,
                    'max': float(data['high'].max()) if not data.empty else None
                },
                'avg_volume': float(data['volume'].mean()) if not data.empty else None,
                'missing_data_percentage': (missing_data.sum() / len(data)) * 100 if not data.empty else 0
            }
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result