"""
Data Sources Manager

This module provides a unified interface for fetching market data from multiple
sources with automatic failover and rate limiting protection.

Supported Sources:
- yfinance (free, rate limited)
- Alpha Vantage (API key required, professional data)
- Polygon (API key required, high-frequency data)
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import time

import yfinance as yf
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from loguru import logger


class DataSourceError(Exception):
    """Custom exception for data source errors"""
    pass


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 5):
        self.calls_per_minute = calls_per_minute
        self.call_times = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        # If we're at the limit, wait
        if len(self.call_times) >= self.calls_per_minute:
            wait_time = 60 - (now - self.call_times[0]) + 1
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        self.call_times.append(now)


class DataSourceManager:
    """Manages multiple data sources with failover and rate limiting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sources = config.get('sources', ['yfinance'])
        self.api_config = config.get('apis', {})
        
        # Initialize rate limiters for each source
        self.rate_limiters = {
            'alpha_vantage': RateLimiter(calls_per_minute=5),  # Free tier: 5 calls/min
            'polygon': RateLimiter(calls_per_minute=100),      # Depends on tier
            'yfinance': RateLimiter(calls_per_minute=100)      # No official limit, be conservative
        }
        
        # Initialize API clients
        self._init_alpha_vantage()
        
        # Cache for data to avoid duplicate calls
        self.cache = {}
        self.cache_ttl = 60  # Cache for 1 minute
        
        logger.info(f"DataSourceManager initialized with sources: {self.sources}")
    
    def _init_alpha_vantage(self):
        """Initialize Alpha Vantage client"""
        self.alpha_vantage_ts = None
        self.alpha_vantage_fd = None
        
        av_config = self.api_config.get('alpha_vantage', {})
        api_key = av_config.get('key') or os.getenv('ALPHA_VANTAGE_API_KEY')
        
        if api_key and api_key != 'your_alpha_vantage_api_key_here':
            try:
                self.alpha_vantage_ts = TimeSeries(key=api_key, output_format='pandas')
                self.alpha_vantage_fd = FundamentalData(key=api_key, output_format='pandas')
                logger.info("Alpha Vantage client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpha Vantage: {e}")
        else:
            logger.warning("Alpha Vantage API key not found - will skip Alpha Vantage calls")
    
    def _get_cache_key(self, symbol: str, data_type: str, timeframe: str = 'daily') -> str:
        """Generate cache key for data"""
        return f"{symbol}_{data_type}_{timeframe}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time, _ = self.cache[cache_key]
        return time.time() - cached_time < self.cache_ttl
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[cache_key] = (time.time(), data)
    
    async def fetch_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch real-time market data with failover across multiple sources
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Dict with real-time market data
        """
        cache_key = self._get_cache_key(symbol, 'realtime')
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug(f"Using cached real-time data for {symbol}")
            return self.cache[cache_key][1]
        
        # Try each data source in order
        last_error = None
        
        for source in self.sources:
            try:
                logger.debug(f"Trying {source} for real-time data: {symbol}")
                
                if source == 'yfinance':
                    data = await self._fetch_yfinance_realtime(symbol)
                elif source == 'alpha_vantage':
                    data = await self._fetch_alpha_vantage_realtime(symbol)
                elif source == 'polygon':
                    data = await self._fetch_polygon_realtime(symbol)
                else:
                    logger.warning(f"Unknown data source: {source}")
                    continue
                
                if data:
                    self._cache_data(cache_key, data)
                    logger.info(f"Successfully fetched real-time data for {symbol} from {source}")
                    return data
                    
            except Exception as e:
                last_error = e
                logger.warning(f"{source} failed for {symbol}: {e}")
                continue
        
        # All sources failed
        error_msg = f"All data sources failed for {symbol}. Last error: {last_error}"
        logger.error(error_msg)
        raise DataSourceError(error_msg)
    
    async def fetch_historical_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical market data with failover
        
        Args:
            symbol: Stock symbol
            period: Time period ('1y', '2y', '5y', etc.)
            interval: Data interval ('1d', '1h', '5m', etc.)
            
        Returns:
            DataFrame with historical data
        """
        cache_key = self._get_cache_key(symbol, f'historical_{period}_{interval}')
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug(f"Using cached historical data for {symbol}")
            return self.cache[cache_key][1]
        
        last_error = None
        
        for source in self.sources:
            try:
                logger.debug(f"Trying {source} for historical data: {symbol}")
                
                if source == 'yfinance':
                    data = await self._fetch_yfinance_historical(symbol, period, interval)
                elif source == 'alpha_vantage':
                    data = await self._fetch_alpha_vantage_historical(symbol, period)
                elif source == 'polygon':
                    data = await self._fetch_polygon_historical(symbol, period, interval)
                else:
                    continue
                
                if data is not None and not data.empty:
                    self._cache_data(cache_key, data)
                    logger.info(f"Successfully fetched historical data for {symbol} from {source}")
                    return data
                    
            except Exception as e:
                last_error = e
                logger.warning(f"{source} failed for historical {symbol}: {e}")
                continue
        
        error_msg = f"All sources failed for historical data {symbol}. Last error: {last_error}"
        logger.error(error_msg)
        raise DataSourceError(error_msg)
    
    async def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental data with failover
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with fundamental data
        """
        cache_key = self._get_cache_key(symbol, 'fundamentals')
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key][1]
        
        last_error = None
        
        for source in self.sources:
            try:
                if source == 'yfinance':
                    data = await self._fetch_yfinance_fundamentals(symbol)
                elif source == 'alpha_vantage':
                    data = await self._fetch_alpha_vantage_fundamentals(symbol)
                else:
                    continue
                
                if data:
                    self._cache_data(cache_key, data)
                    return data
                    
            except Exception as e:
                last_error = e
                logger.warning(f"{source} failed for fundamentals {symbol}: {e}")
                continue
        
        error_msg = f"All sources failed for fundamentals {symbol}. Last error: {last_error}"
        logger.error(error_msg)
        raise DataSourceError(error_msg)
    
    # yfinance implementations
    async def _fetch_yfinance_realtime(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time data from yfinance"""
        await self.rate_limiters['yfinance'].wait_if_needed()
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1d", interval="1m")
        
        if hist.empty:
            raise DataSourceError("No historical data available")
        
        latest = hist.iloc[-1]
        
        return {
            "symbol": symbol,
            "source": "yfinance",
            "price": float(latest['Close']),
            "volume": int(latest['Volume']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "open": float(latest['Open']),
            "timestamp": datetime.now().isoformat(),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "volume_avg": info.get("averageVolume", 0),
            "52_week_high": info.get("fiftyTwoWeekHigh", 0),
            "52_week_low": info.get("fiftyTwoWeekLow", 0)
        }
    
    async def _fetch_yfinance_historical(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch historical data from yfinance"""
        await self.rate_limiters['yfinance'].wait_if_needed()
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            raise DataSourceError("No historical data available")
        
        return hist
    
    async def _fetch_yfinance_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data from yfinance"""
        await self.rate_limiters['yfinance'].wait_if_needed()
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or len(info) < 5:
            raise DataSourceError("Insufficient fundamental data")
        
        return {
            "symbol": symbol,
            "source": "yfinance",
            "data": info
        }
    
    # Alpha Vantage implementations
    async def _fetch_alpha_vantage_realtime(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time data from Alpha Vantage"""
        if not self.alpha_vantage_ts:
            raise DataSourceError("Alpha Vantage not available")
        
        await self.rate_limiters['alpha_vantage'].wait_if_needed()
        
        # Get intraday data (latest available)
        data, meta_data = self.alpha_vantage_ts.get_intraday(symbol=symbol, interval='1min', outputsize='compact')
        
        if data.empty:
            raise DataSourceError("No Alpha Vantage data available")
        
        # Get the most recent entry
        latest_time = data.index[-1]
        latest = data.iloc[-1]
        
        return {
            "symbol": symbol,
            "source": "alpha_vantage",
            "price": float(latest['4. close']),
            "volume": int(latest['5. volume']),
            "high": float(latest['2. high']),
            "low": float(latest['3. low']),
            "open": float(latest['1. open']),
            "timestamp": latest_time.isoformat(),
            # Alpha Vantage doesn't provide these in real-time, would need separate calls
            "market_cap": 0,
            "pe_ratio": 0,
            "volume_avg": 0,
            "52_week_high": 0,
            "52_week_low": 0
        }
    
    async def _fetch_alpha_vantage_historical(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch historical data from Alpha Vantage"""
        if not self.alpha_vantage_ts:
            raise DataSourceError("Alpha Vantage not available")
        
        await self.rate_limiters['alpha_vantage'].wait_if_needed()
        
        # Alpha Vantage provides different functions based on period
        if period in ['1d', '5d']:
            data, meta_data = self.alpha_vantage_ts.get_intraday(symbol=symbol, interval='60min', outputsize='full')
        else:
            data, meta_data = self.alpha_vantage_ts.get_daily(symbol=symbol, outputsize='full')
        
        if data.empty:
            raise DataSourceError("No Alpha Vantage historical data")
        
        # Normalize column names to match yfinance format
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High', 
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        
        # Sort by date (ascending)
        data = data.sort_index()
        
        return data
    
    async def _fetch_alpha_vantage_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data from Alpha Vantage"""
        if not self.alpha_vantage_fd:
            raise DataSourceError("Alpha Vantage fundamentals not available")
        
        await self.rate_limiters['alpha_vantage'].wait_if_needed()
        
        # Get company overview (contains key fundamental metrics)
        overview, meta_data = self.alpha_vantage_fd.get_company_overview(symbol=symbol)
        
        if overview.empty:
            raise DataSourceError("No Alpha Vantage fundamental data")
        
        # Convert to dictionary format
        fundamental_data = overview.iloc[0].to_dict()
        
        return {
            "symbol": symbol,
            "source": "alpha_vantage",
            "data": fundamental_data
        }
    
    # Polygon implementations (placeholder - requires polygon-api-client)
    async def _fetch_polygon_realtime(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time data from Polygon (placeholder)"""
        # Would implement with polygon-api-client
        raise DataSourceError("Polygon implementation not yet available")
    
    async def _fetch_polygon_historical(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch historical data from Polygon (placeholder)"""
        # Would implement with polygon-api-client  
        raise DataSourceError("Polygon implementation not yet available")
    
    def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all configured data sources"""
        status = {}
        
        # yfinance status
        status['yfinance'] = {
            'available': True,
            'type': 'free',
            'rate_limit': '~100 calls/min (estimated)',
            'features': ['real_time', 'historical', 'fundamentals']
        }
        
        # Alpha Vantage status
        status['alpha_vantage'] = {
            'available': self.alpha_vantage_ts is not None,
            'type': 'api_key_required',
            'rate_limit': '5 calls/min (free tier)',
            'features': ['real_time', 'historical', 'fundamentals']
        }
        
        # Polygon status  
        status['polygon'] = {
            'available': False,
            'type': 'api_key_required',
            'rate_limit': 'varies by plan',
            'features': ['real_time', 'historical']
        }
        
        return status


# Convenience function for backward compatibility
async def create_data_source_manager(config: Dict[str, Any]) -> DataSourceManager:
    """Create and return a configured DataSourceManager"""
    return DataSourceManager(config)