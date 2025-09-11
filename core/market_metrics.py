"""
Market Performance Metrics for Dynamic Thresholds

This module provides market-wide performance analysis including:
- S&P 500 (SPY) returns over various timeframes
- VIX volatility levels and trends
- Market breadth indicators
- Sector rotation analysis

These metrics are used to adjust trading thresholds based on overall
market conditions rather than individual stock signal history.
"""

import logging
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class MarketMetrics:
    """Market performance metrics data structure"""
    spy_return_1d: float
    spy_return_5d: float
    spy_return_30d: float
    vix_current: float
    vix_change_5d: float
    market_breadth: float
    sector_strength: Dict[str, float]
    regime: str  # 'bull', 'bear', 'sideways', 'volatile'
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'spy_return_1d': self.spy_return_1d,
            'spy_return_5d': self.spy_return_5d,
            'spy_return_30d': self.spy_return_30d,
            'vix_current': self.vix_current,
            'vix_change_5d': self.vix_change_5d,
            'market_breadth': self.market_breadth,
            'sector_strength': self.sector_strength,
            'regime': self.regime,
            'timestamp': self.timestamp
        }


class MarketMetricsAnalyzer:
    """Analyzes broad market conditions for dynamic threshold calculation"""
    
    def __init__(self, cache_file: str = "data/market_metrics_cache.json", cache_duration_hours: int = 1):
        """
        Initialize market metrics analyzer
        
        Args:
            cache_file: File to cache market data to reduce API calls
            cache_duration_hours: How long to cache data before refreshing
        """
        self.cache_file = cache_file
        self.cache_duration = timedelta(hours=cache_duration_hours)
        
        # Market symbols for analysis
        self.market_symbols = {
            'spy': 'SPY',      # S&P 500 ETF
            'vix': '^VIX',     # Volatility Index
            'qqq': 'QQQ',      # NASDAQ ETF
            'iwm': 'IWM',      # Russell 2000 (small caps)
        }
        
        # Sector ETFs for breadth analysis
        self.sector_etfs = {
            'technology': 'XLK',
            'healthcare': 'XLV', 
            'financials': 'XLF',
            'energy': 'XLE',
            'industrials': 'XLI',
            'consumer_discretionary': 'XLY',
            'utilities': 'XLU',
            'real_estate': 'XLRE'
        }
        
    def get_market_metrics(self, force_refresh: bool = False) -> Optional[MarketMetrics]:
        """
        Get current market metrics with caching
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            MarketMetrics object or None if fetch fails
        """
        # Try to load from cache first
        if not force_refresh:
            cached_metrics = self._load_from_cache()
            if cached_metrics:
                return cached_metrics
        
        try:
            # Fetch fresh market data
            logger.info("Fetching fresh market metrics...")
            metrics = self._fetch_market_metrics()
            
            if metrics:
                # Save to cache
                self._save_to_cache(metrics)
                logger.info(f"Market metrics updated: regime={metrics.regime}, SPY_30d={metrics.spy_return_30d:.1%}")
                return metrics
            
        except Exception as e:
            logger.error(f"Error fetching market metrics: {e}")
            
            # Try to return cached data even if stale
            cached_metrics = self._load_from_cache(ignore_expiry=True)
            if cached_metrics:
                logger.warning("Using stale cached market metrics due to fetch error")
                return cached_metrics
        
        return None
    
    def _fetch_market_metrics(self) -> Optional[MarketMetrics]:
        """Fetch market metrics from data sources"""
        try:
            # Fetch SPY data for market returns
            spy_data = yf.download('SPY', period='3mo', interval='1d', progress=False, auto_adjust=True)
            if spy_data.empty:
                logger.error("Failed to fetch SPY data")
                return None
            
            # Calculate SPY returns (ensure scalar values)
            spy_return_1d = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-2] - 1)
            spy_return_5d = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-6] - 1) if len(spy_data) >= 6 else 0.0
            spy_return_30d = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-31] - 1) if len(spy_data) >= 31 else 0.0
            
            # Fetch VIX data
            vix_data = yf.download('^VIX', period='1mo', interval='1d', progress=False, auto_adjust=True)
            vix_current = vix_data['Close'].iloc[-1] if not vix_data.empty else 20.0
            vix_change_5d = (vix_current - vix_data['Close'].iloc[-6]) if len(vix_data) >= 6 else 0.0
            
            # Calculate market breadth using sector performance
            market_breadth = self._calculate_market_breadth()
            
            # Analyze sector strength
            sector_strength = self._calculate_sector_strength()
            
            # Determine market regime
            regime = self._determine_market_regime(spy_return_30d, vix_current, market_breadth)
            
            return MarketMetrics(
                spy_return_1d=float(spy_return_1d),
                spy_return_5d=float(spy_return_5d),
                spy_return_30d=float(spy_return_30d),
                vix_current=float(vix_current),
                vix_change_5d=float(vix_change_5d),
                market_breadth=market_breadth,
                sector_strength=sector_strength,
                regime=regime,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in _fetch_market_metrics: {e}")
            return None
    
    def _calculate_market_breadth(self) -> float:
        """Calculate market breadth using sector ETF performance"""
        try:
            positive_sectors = 0
            total_sectors = 0
            
            for sector, etf in self.sector_etfs.items():
                try:
                    data = yf.download(etf, period='1mo', interval='1d', progress=False, auto_adjust=True)
                    if len(data) >= 20:
                        # Check if sector is above 20-day moving average
                        ma_20 = data['Close'].rolling(20).mean().iloc[-1]
                        current_price = data['Close'].iloc[-1]
                        
                        if float(current_price) > float(ma_20):
                            positive_sectors += 1
                        total_sectors += 1
                        
                except Exception as e:
                    logger.debug(f"Failed to get data for {etf}: {e}")
                    continue
            
            breadth = positive_sectors / total_sectors if total_sectors > 0 else 0.5
            logger.debug(f"Market breadth: {positive_sectors}/{total_sectors} = {breadth:.2f}")
            return breadth
            
        except Exception as e:
            logger.warning(f"Error calculating market breadth: {e}")
            return 0.5  # Neutral default
    
    def _calculate_sector_strength(self) -> Dict[str, float]:
        """Calculate relative strength of different sectors"""
        sector_strength = {}
        
        try:
            for sector, etf in self.sector_etfs.items():
                try:
                    data = yf.download(etf, period='1mo', interval='1d', progress=False, auto_adjust=True)
                    if len(data) >= 20:
                        # Calculate 20-day return
                        return_20d = (data['Close'].iloc[-1] / data['Close'].iloc[-21] - 1)
                        sector_strength[sector] = return_20d
                except Exception as e:
                    logger.debug(f"Failed to calculate strength for {sector}: {e}")
                    sector_strength[sector] = 0.0
                    
        except Exception as e:
            logger.warning(f"Error calculating sector strength: {e}")
        
        return sector_strength
    
    def _determine_market_regime(self, spy_return_30d: float, vix_current: float, market_breadth: float) -> str:
        """Determine current market regime based on multiple factors"""
        
        # Thresholds for regime classification
        if spy_return_30d > 0.05 and vix_current < 20 and market_breadth > 0.6:
            return 'bull'
        elif spy_return_30d < -0.05 and vix_current > 25 and market_breadth < 0.4:
            return 'bear'
        elif vix_current > 30:
            return 'volatile'
        else:
            return 'sideways'
    
    def _load_from_cache(self, ignore_expiry: bool = False) -> Optional[MarketMetrics]:
        """Load market metrics from cache if valid"""
        try:
            if not os.path.exists(self.cache_file):
                return None
            
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check cache expiry
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if not ignore_expiry and datetime.now() - cache_time > self.cache_duration:
                logger.debug("Market metrics cache expired")
                return None
            
            return MarketMetrics(
                spy_return_1d=cache_data['spy_return_1d'],
                spy_return_5d=cache_data['spy_return_5d'],
                spy_return_30d=cache_data['spy_return_30d'],
                vix_current=cache_data['vix_current'],
                vix_change_5d=cache_data['vix_change_5d'],
                market_breadth=cache_data['market_breadth'],
                sector_strength=cache_data['sector_strength'],
                regime=cache_data['regime'],
                timestamp=cache_data['timestamp']
            )
            
        except Exception as e:
            logger.debug(f"Error loading market metrics cache: {e}")
            return None
    
    def _save_to_cache(self, metrics: MarketMetrics):
        """Save market metrics to cache"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
                
            logger.debug(f"Market metrics cached to {self.cache_file}")
            
        except Exception as e:
            logger.warning(f"Error saving market metrics cache: {e}")
    
    def get_latest_metrics(self) -> Optional[Dict]:
        """
        Alias for get_market_metrics() that returns a dictionary format
        for compatibility with DynamicThresholdManager
        """
        metrics = self.get_market_metrics()
        if metrics:
            return metrics.to_dict()
        return None


def calculate_market_threshold_component(metrics: MarketMetrics, base_threshold: float = 0.25) -> Tuple[float, str]:
    """
    Calculate threshold component based on market metrics
    
    Args:
        metrics: Market metrics data
        base_threshold: Base threshold value
        
    Returns:
        Tuple of (threshold_component, explanation)
    """
    # Start with base threshold
    threshold = base_threshold
    adjustments = []
    
    # SPY return adjustment (strongest factor)
    if metrics.spy_return_30d > 0.08:  # Very strong market
        threshold *= 1.4
        adjustments.append("very strong SPY returns")
    elif metrics.spy_return_30d > 0.04:  # Strong market
        threshold *= 1.2
        adjustments.append("strong SPY returns")
    elif metrics.spy_return_30d < -0.08:  # Very weak market
        threshold *= 0.6
        adjustments.append("very weak SPY returns")
    elif metrics.spy_return_30d < -0.04:  # Weak market
        threshold *= 0.8
        adjustments.append("weak SPY returns")
    
    # VIX adjustment
    if metrics.vix_current > 30:  # High volatility
        threshold *= 0.8
        adjustments.append("high volatility")
    elif metrics.vix_current < 15:  # Low volatility
        threshold *= 1.1
        adjustments.append("low volatility")
    
    # Market breadth adjustment
    if metrics.market_breadth > 0.7:  # Broad market strength
        threshold *= 1.1
        adjustments.append("broad market strength")
    elif metrics.market_breadth < 0.3:  # Narrow market
        threshold *= 0.9
        adjustments.append("narrow market")
    
    # Regime-based adjustment
    if metrics.regime == 'bull':
        threshold *= 1.1
        adjustments.append("bull regime")
    elif metrics.regime == 'bear':
        threshold *= 0.8
        adjustments.append("bear regime")
    elif metrics.regime == 'volatile':
        threshold *= 0.85
        adjustments.append("volatile regime")
    
    # Generate explanation
    if adjustments:
        explanation = f"Market-adjusted threshold ({', '.join(adjustments)})"
    else:
        explanation = "Market-neutral threshold"
    
    return threshold, explanation