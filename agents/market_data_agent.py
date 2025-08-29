"""
Market Data Agent

This agent is responsible for:
- Fetching real-time market data
- Managing historical data
- Providing data feeds to other agents
- Handling multiple data sources with failover
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger

from .base_agent import BaseAgent
from core.data_sources import DataSourceManager, DataSourceError


class MarketDataAgent(BaseAgent):
    """
    Market Data Agent for fetching and managing market data.
    
    Features:
    - Real-time price feeds
    - Historical data management
    - Multiple data source support
    - Data quality validation
    - Caching for performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "MarketDataAgent")
        
        # Configuration
        self.symbols = config.get("symbols", ["AAPL", "GOOGL", "MSFT"])
        self.real_time_interval = config.get("real_time_interval", 60)
        self.historical_days = config.get("historical_lookback_days", 252)
        
        # Initialize DataSourceManager with full market data config
        market_data_config = config.get("market_data", config)
        self.data_source_manager = DataSourceManager(market_data_config)
        
        # Legacy cache (still used for additional caching layer)
        self.price_cache = {}
        self.historical_cache = {}
        self.last_update = {}
        
        # Data quality metrics
        self.data_quality_scores = {}
        
        self.logger.info(f"Market Data Agent initialized for symbols: {self.symbols}")
        self.logger.info(f"Data sources: {market_data_config.get('sources', ['yfinance'])}")
    
    async def _execute_logic(self, input_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, float]]:
        """
        Execute market data fetching logic.
        
        Args:
            input_data: Contains symbols, data_type, and other parameters
            
        Returns:
            Tuple of (market_data, metrics)
        """
        symbols = input_data.get("symbols", self.symbols)
        data_type = input_data.get("data_type", "real_time")  # "real_time", "historical", or "both"
        
        start_time = time.time()
        
        # Fetch data based on type
        if data_type == "real_time":
            data = await self._fetch_real_time_data(symbols)
        elif data_type == "historical":
            data = await self._fetch_historical_data(symbols)
        else:  # both
            real_time_data = await self._fetch_real_time_data(symbols)
            historical_data = await self._fetch_historical_data(symbols)
            data = {
                "real_time": real_time_data,
                "historical": historical_data
            }
        
        # Calculate metrics
        fetch_time = time.time() - start_time
        data_quality = self._calculate_data_quality(data, symbols)
        
        metrics = {
            "fetch_time_seconds": fetch_time,
            "symbols_processed": len(symbols),
            "data_quality_score": data_quality,
            "cache_hit_rate": self._calculate_cache_hit_rate(symbols),
            "data_points_fetched": self._count_data_points(data)
        }
        
        # Update cache
        self._update_cache(data, symbols, data_type)
        
        return data, metrics
    
    async def _fetch_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch real-time market data for given symbols with multi-source failover."""
        real_time_data = {}
        
        for symbol in symbols:
            try:
                # Check local cache first
                if self._is_cache_valid(symbol, "real_time"):
                    real_time_data[symbol] = self.price_cache[symbol]
                    continue
                
                # Use DataSourceManager with automatic failover
                data = await self.data_source_manager.fetch_real_time_data(symbol)
                
                if data:
                    real_time_data[symbol] = data
                    # Update local cache
                    self.price_cache[symbol] = data
                    self.last_update[symbol] = time.time()
                else:
                    self.logger.warning(f"No real-time data received for {symbol}")
                    
            except DataSourceError as e:
                self.logger.error(f"All data sources failed for {symbol}: {e}")
                # Try to use cached data if available
                if symbol in self.price_cache:
                    real_time_data[symbol] = self.price_cache[symbol]
                    self.logger.info(f"Using cached data for {symbol}")
            except Exception as e:
                self.logger.error(f"Unexpected error fetching real-time data for {symbol}: {e}")
                # Try to use cached data if available
                if symbol in self.price_cache:
                    real_time_data[symbol] = self.price_cache[symbol]
        
        return real_time_data
    
    async def _fetch_historical_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch historical market data for given symbols with multi-source failover."""
        historical_data = {}
        
        # Convert historical_days to period string
        period = f"{self.historical_days}d" if self.historical_days <= 365 else f"{self.historical_days // 365}y"
        
        for symbol in symbols:
            try:
                # Check local cache first
                if self._is_cache_valid(symbol, "historical"):
                    historical_data[symbol] = self.historical_cache[symbol]
                    continue
                
                # Use DataSourceManager with automatic failover
                hist = await self.data_source_manager.fetch_historical_data(symbol, period=period)
                
                if hist is not None and not hist.empty:
                    # Add technical indicators
                    hist = self._add_basic_indicators(hist)
                    historical_data[symbol] = hist
                    # Update local cache
                    self.historical_cache[symbol] = hist
                    self.last_update[f"{symbol}_historical"] = time.time()
                else:
                    self.logger.warning(f"No historical data received for {symbol}")
                    
            except DataSourceError as e:
                self.logger.error(f"All data sources failed for historical {symbol}: {e}")
                # Try to use cached data if available
                if symbol in self.historical_cache:
                    historical_data[symbol] = self.historical_cache[symbol]
                    self.logger.info(f"Using cached historical data for {symbol}")
            except Exception as e:
                self.logger.error(f"Unexpected error fetching historical data for {symbol}: {e}")
                # Try to use cached data if available
                if symbol in self.historical_cache:
                    historical_data[symbol] = self.historical_cache[symbol]
        
        return historical_data
    
    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to historical data."""
        try:
            # Simple moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Exponential moving averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Price change
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
            
            # Clean NaN/inf rows introduced by rolling calculations
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # Optionally trim to a reasonable window to reduce payload size
            if len(df) > 300:
                df = df.iloc[-300:]
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
        
        return df
    
    def _is_cache_valid(self, symbol: str, data_type: str) -> bool:
        """Check if cached data is still valid."""
        if data_type == "real_time":
            if symbol not in self.price_cache or symbol not in self.last_update:
                return False
            last_update_time = self.last_update[symbol]
            
            # Handle both datetime objects and float timestamps
            if isinstance(last_update_time, datetime):
                time_diff = datetime.now() - last_update_time
            else:
                # Convert float timestamp to seconds since last update
                time_diff_seconds = time.time() - last_update_time
                return time_diff_seconds < self.real_time_interval
            
            return time_diff.total_seconds() < self.real_time_interval
        
        elif data_type == "historical":
            # For historical data, use the symbol with _historical suffix
            cache_key = f"{symbol}_historical"
            if symbol not in self.historical_cache or cache_key not in self.last_update:
                return False
            
            last_update_time = self.last_update[cache_key]
            
            # Handle both datetime objects and float timestamps  
            if isinstance(last_update_time, datetime):
                time_diff = datetime.now() - last_update_time
                return time_diff.total_seconds() < 3600  # 1 hour cache for historical
            else:
                # Convert float timestamp to seconds since last update
                time_diff_seconds = time.time() - last_update_time
                return time_diff_seconds < 3600  # 1 hour cache for historical
        
        return False
    
    def _update_cache(self, data: Dict[str, Any], symbols: List[str], data_type: str):
        """Update the data cache."""
        for symbol in symbols:
            if data_type == "real_time" and symbol in data:
                self.price_cache[symbol] = data[symbol]
                self.last_update[symbol] = datetime.now()
            
            elif data_type == "historical" and symbol in data:
                self.historical_cache[symbol] = data[symbol]
                self.last_update[symbol] = datetime.now()
            
            elif data_type == "both":
                if "real_time" in data and symbol in data["real_time"]:
                    self.price_cache[symbol] = data["real_time"][symbol]
                if "historical" in data and symbol in data["historical"]:
                    self.historical_cache[symbol] = data["historical"][symbol]
                self.last_update[symbol] = datetime.now()
    
    def _calculate_data_quality(self, data: Dict[str, Any], symbols: List[str]) -> float:
        """Calculate data quality score (0-1)."""
        if not data or not symbols:
            return 0.0
        
        quality_scores = []
        
        for symbol in symbols:
            score = 0.0
            
            # Check if data exists
            if isinstance(data, dict) and "real_time" in data:
                # Handle "both" data type
                symbol_data = data["real_time"].get(symbol, {})
            else:
                symbol_data = data.get(symbol, {})
            
            if symbol_data:
                score += 0.5  # Data exists
                
                # Check for required fields
                required_fields = ["price", "volume", "timestamp"]
                present_fields = sum(1 for field in required_fields if field in symbol_data)
                score += (present_fields / len(required_fields)) * 0.3
                
                # Check data freshness (real-time data)
                if "timestamp" in symbol_data:
                    try:
                        ts = symbol_data["timestamp"]
                        if isinstance(ts, str):
                            # parse ISO string
                            ts_dt = datetime.fromisoformat(ts)
                        else:
                            ts_dt = ts
                        time_diff = datetime.now() - ts_dt
                        if time_diff.total_seconds() < 300:  # Less than 5 minutes old
                            score += 0.2
                    except Exception:
                        pass
            
            quality_scores.append(score)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _calculate_cache_hit_rate(self, symbols: List[str]) -> float:
        """Calculate cache hit rate."""
        if not symbols:
            return 0.0
        
        hits = sum(1 for symbol in symbols if symbol in self.price_cache)
        return hits / len(symbols)
    
    def _count_data_points(self, data: Dict[str, Any]) -> int:
        """Count total data points fetched."""
        count = 0
        
        if isinstance(data, dict):
            if "real_time" in data and "historical" in data:
                # Both data type
                count += len(data["real_time"])
                for df in data["historical"].values():
                    if isinstance(df, pd.DataFrame):
                        count += len(df)
            elif all(isinstance(v, pd.DataFrame) for v in data.values()):
                # Historical data
                for df in data.values():
                    count += len(df)
            else:
                # Real-time data
                count = len(data)
        
        return count
    
    async def get_live_feed(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get live data feed for symbols."""
        symbols = symbols or self.symbols
        return await self._fetch_real_time_data(symbols)
    
    async def get_historical_data(self, symbols: Optional[List[str]] = None, 
                                days: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Get historical data for symbols."""
        symbols = symbols or self.symbols
        if days:
            original_days = self.historical_days
            self.historical_days = days
            data = await self._fetch_historical_data(symbols)
            self.historical_days = original_days
            return data
        else:
            return await self._fetch_historical_data(symbols)
    
    def get_cached_data(self, symbol: str, data_type: str = "real_time") -> Optional[Any]:
        """Get cached data for a symbol."""
        if data_type == "real_time":
            return self.price_cache.get(symbol)
        elif data_type == "historical":
            return self.historical_cache.get(symbol)
        return None
    
    def clear_cache(self):
        """Clear all cached data."""
        self.price_cache.clear()
        self.historical_cache.clear()
        self.last_update.clear()
        self.logger.info("Cache cleared")
    
    def get_data_source_status(self) -> Dict[str, Any]:
        """Get status of all configured data sources."""
        return self.data_source_manager.get_source_status()
    
    async def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data for a symbol using multi-source failover."""
        try:
            return await self.data_source_manager.fetch_fundamentals(symbol)
        except DataSourceError as e:
            self.logger.error(f"Failed to fetch fundamentals for {symbol}: {e}")
            return {
                "symbol": symbol,
                "source": "none",
                "data": {},
                "error": str(e)
            }