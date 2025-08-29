"""
Technical Analysis Agent

This agent performs sophisticated technical analysis including:
- Trend following strategies (EMA, ADX, Ichimoku)
- Mean reversion strategies (Bollinger Bands, RSI, MACD)
- Momentum analysis
- Volatility analysis
- Statistical arbitrage signals
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ta
from scipy import stats
from loguru import logger

from .base_agent import BaseAgent


class TechnicalAnalystAgent(BaseAgent):
    """
    Technical Analysis Agent that generates trading signals based on technical indicators.
    
    Features:
    - Multiple strategy implementations
    - Signal strength scoring
    - Risk-adjusted signals
    - Performance tracking per strategy
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "TechnicalAnalystAgent")
        
        # Strategy configurations
        self.strategies = config.get("strategies", {})
        
        # Strategy weights for combined signals
        self.strategy_weights = {
            "trend_following": 0.3,
            "mean_reversion": 0.25,
            "momentum": 0.25,
            "volatility": 0.2
        }
        
        # Signal thresholds
        self.signal_thresholds = {
            "strong_buy": 0.7,
            "buy": 0.3,
            "neutral": 0.0,
            "sell": -0.3,
            "strong_sell": -0.7
        }
        
        self.logger.info("Technical Analysis Agent initialized")
    
    async def _execute_logic(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Execute technical analysis on market data.
        
        Args:
            input_data: Contains market data from MarketDataAgent and optionally portfolio context
            
        Returns:
            Tuple of (analysis_results, metrics)
        """
        start_time = time.time()
        
        # Extract market data, handling the structure from the backtester
        market_data_bundle = input_data.get("market_data", {})
        historical_data = market_data_bundle.get("historical", market_data_bundle) # Fallback for single-cycle runs
        symbols = input_data.get("symbols", list(historical_data.keys()))
        
        if not historical_data:
            self.logger.error("Execution failed: No historical market data provided for technical analysis")
            return {}, {}
        
        # Perform analysis for each symbol
        analysis_results = {}
        strategy_performance = {}
        
        for symbol in symbols:
            if symbol in historical_data:
                symbol_data = historical_data[symbol]
                
                if symbol_data.empty:
                    self.logger.warning(f"Skipping empty historical data for {symbol}")
                    continue
                
                # Perform technical analysis with portfolio context
                symbol_analysis = await self._analyze_symbol(symbol, symbol_data)
                analysis_results[symbol] = symbol_analysis
                
                # Track strategy performance
                for strategy, signal in symbol_analysis["strategies"].items():
                    if strategy not in strategy_performance:
                        strategy_performance[strategy] = []
                    strategy_performance[strategy].append(signal["strength"])
        
        # Calculate overall metrics
        execution_time = time.time() - start_time
        
        metrics = {
            "execution_time_seconds": execution_time,
            "symbols_analyzed": len(analysis_results),
            "average_signal_strength": self._calculate_average_signal_strength(analysis_results),
            "strategy_coverage": len(strategy_performance),
            "bullish_signals": self._count_signals(analysis_results, "bullish"),
            "bearish_signals": self._count_signals(analysis_results, "bearish")
        }
        
        # Add strategy-specific metrics
        for strategy, strengths in strategy_performance.items():
            metrics[f"{strategy}_avg_strength"] = np.mean(strengths) if strengths else 0.0
        
        return analysis_results, metrics
    
    async def _analyze_symbol(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive technical analysis for a single symbol with portfolio awareness."""
        
        if data.empty or len(data) < 50:
            return self._create_empty_analysis(symbol)
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            self.logger.warning(f"Missing required columns for {symbol}")
            return self._create_empty_analysis(symbol)
        

        
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "strategies": {},
            "overall_signal": {},
            "risk_metrics": {},
            "market_regime": ""
        }
        
        try:
            # 1. Trend Following Analysis
            trend_analysis = self._trend_following_analysis(data)
            analysis["strategies"]["trend_following"] = trend_analysis
            
            # 2. Mean Reversion Analysis
            mean_reversion_analysis = self._mean_reversion_analysis(data)
            analysis["strategies"]["mean_reversion"] = mean_reversion_analysis
            
            # 3. Momentum Analysis
            momentum_analysis = self._momentum_analysis(data)
            analysis["strategies"]["momentum"] = momentum_analysis
            
            # 4. Volatility Analysis
            volatility_analysis = self._volatility_analysis(data)
            analysis["strategies"]["volatility"] = volatility_analysis
            
            # 5. Statistical Arbitrage Signals
            stat_arb_analysis = self._statistical_arbitrage_analysis(data)
            analysis["strategies"]["statistical_arbitrage"] = stat_arb_analysis
            
            # 6. Combine signals with portfolio awareness
            overall_signal = self._combine_signals(analysis["strategies"], symbol)
            analysis["overall_signal"] = overall_signal
            
            # 7. Risk metrics
            risk_metrics = self._calculate_risk_metrics(data)
            analysis["risk_metrics"] = risk_metrics
            
            # 8. Market regime detection
            market_regime = self._detect_market_regime(data)
            analysis["market_regime"] = market_regime
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return self._create_empty_analysis(symbol)
        
        return analysis
    
    def _trend_following_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Trend following strategy using EMA, ADX, and Ichimoku."""
        
        config = self.strategies.get("trend_following", {})
        ema_short = config.get("ema_short", 12)
        ema_long = config.get("ema_long", 26)
        adx_threshold = config.get("adx_threshold", 25)
        
        # Calculate indicators
        ema_short_values = ta.trend.ema_indicator(data['Close'], window=ema_short)
        ema_long_values = ta.trend.ema_indicator(data['Close'], window=ema_long)
        adx = ta.trend.adx(data['High'], data['Low'], data['Close'], window=14)
        
        # Ichimoku components
        ichimoku_a = ta.trend.ichimoku_a(data['High'], data['Low'])
        ichimoku_b = ta.trend.ichimoku_b(data['High'], data['Low'])
        
        # Current values
        current_price = data['Close'].iloc[-1]
        current_ema_short = ema_short_values.iloc[-1]
        current_ema_long = ema_long_values.iloc[-1]
        current_adx = adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0
        
        # Signal calculation
        signal_strength = 0.0
        
        # EMA crossover signal
        if current_ema_short > current_ema_long:
            signal_strength += 0.4
        else:
            signal_strength -= 0.4
        
        # ADX trend strength
        if current_adx > adx_threshold:
            trend_direction = 1 if current_price > current_ema_long else -1
            signal_strength += trend_direction * 0.3
        
        # Ichimoku cloud
        current_ichimoku_a = ichimoku_a.iloc[-1] if not np.isnan(ichimoku_a.iloc[-1]) else current_price
        current_ichimoku_b = ichimoku_b.iloc[-1] if not np.isnan(ichimoku_b.iloc[-1]) else current_price
        cloud_top = max(current_ichimoku_a, current_ichimoku_b)
        cloud_bottom = min(current_ichimoku_a, current_ichimoku_b)
        
        if current_price > cloud_top:
            signal_strength += 0.3
        elif current_price < cloud_bottom:
            signal_strength -= 0.3
        
        # Normalize signal strength
        signal_strength = np.clip(signal_strength, -1.0, 1.0)
        
        return {
            "name": "Trend Following",
            "strength": signal_strength,
            "direction": "bullish" if signal_strength > 0 else "bearish",
            "confidence": abs(signal_strength),
            "indicators": {
                "ema_short": current_ema_short,
                "ema_long": current_ema_long,
                "adx": current_adx,
                "ichimoku_cloud_position": "above" if current_price > cloud_top else "below" if current_price < cloud_bottom else "inside"
            }
        }
    
    def _mean_reversion_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Mean reversion strategy using Bollinger Bands, RSI, and MACD."""
        
        config = self.strategies.get("mean_reversion", {})
        rsi_oversold = config.get("rsi_oversold", 30)
        rsi_overbought = config.get("rsi_overbought", 70)
        bb_std = config.get("bollinger_std", 2)
        
        # Calculate indicators
        rsi = ta.momentum.rsi(data['Close'], window=14)
        bb_upper = ta.volatility.bollinger_hband(data['Close'], window=20, window_dev=bb_std)
        bb_lower = ta.volatility.bollinger_lband(data['Close'], window=20, window_dev=bb_std)
        bb_middle = ta.volatility.bollinger_mavg(data['Close'], window=20)
        
        macd_line = ta.trend.macd(data['Close'])
        macd_signal = ta.trend.macd_signal(data['Close'])
        
        # Current values
        current_price = data['Close'].iloc[-1]
        current_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
        current_bb_upper = bb_upper.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]
        current_bb_middle = bb_middle.iloc[-1]
        current_macd = macd_line.iloc[-1] if not np.isnan(macd_line.iloc[-1]) else 0
        current_macd_signal = macd_signal.iloc[-1] if not np.isnan(macd_signal.iloc[-1]) else 0
        
        # Signal calculation
        signal_strength = 0.0
        
        # RSI mean reversion
        if current_rsi < rsi_oversold:
            signal_strength += 0.4  # Oversold, expect bounce
        elif current_rsi > rsi_overbought:
            signal_strength -= 0.4  # Overbought, expect pullback
        
        # Bollinger Bands
        bb_position = (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower)
        if bb_position < 0.1:  # Near lower band
            signal_strength += 0.3
        elif bb_position > 0.9:  # Near upper band
            signal_strength -= 0.3
        
        # MACD divergence
        macd_diff = current_macd - current_macd_signal
        if macd_diff > 0:
            signal_strength += 0.3
        else:
            signal_strength -= 0.3
        
        # Normalize signal strength
        signal_strength = np.clip(signal_strength, -1.0, 1.0)
        
        return {
            "name": "Mean Reversion",
            "strength": signal_strength,
            "direction": "bullish" if signal_strength > 0 else "bearish",
            "confidence": abs(signal_strength),
            "indicators": {
                "rsi": current_rsi,
                "bb_position": bb_position,
                "macd_divergence": macd_diff,
                "price_vs_bb_middle": (current_price - current_bb_middle) / current_bb_middle
            }
        }
    
    def _momentum_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Momentum analysis using price and volume dynamics."""
        
        config = self.strategies.get("momentum", {})
        macd_fast = config.get("macd_fast", 12)
        macd_slow = config.get("macd_slow", 26)
        macd_signal_period = config.get("macd_signal", 9)
        
        # Price momentum
        price_change_1d = data['Close'].pct_change(1).iloc[-1]
        price_change_5d = data['Close'].pct_change(5).iloc[-1]
        price_change_20d = data['Close'].pct_change(20).iloc[-1]
        
        # Volume momentum
        volume_sma = data['Volume'].rolling(window=20).mean()
        current_volume = data['Volume'].iloc[-1]
        volume_ratio = current_volume / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1
        
        # MACD momentum
        macd = ta.trend.macd(data['Close'], window_fast=macd_fast, window_slow=macd_slow)
        macd_signal = ta.trend.macd_signal(data['Close'], window_fast=macd_fast, 
                                          window_slow=macd_slow, window_sign=macd_signal_period)
        
        current_macd = macd.iloc[-1] if not np.isnan(macd.iloc[-1]) else 0
        current_macd_signal = macd_signal.iloc[-1] if not np.isnan(macd_signal.iloc[-1]) else 0
        
        # Rate of Change (ROC)
        roc = ta.momentum.roc(data['Close'], window=10)
        current_roc = roc.iloc[-1] if not np.isnan(roc.iloc[-1]) else 0
        
        # Signal calculation
        signal_strength = 0.0
        
        # Price momentum scoring
        momentum_score = 0
        if price_change_1d > 0:
            momentum_score += 1
        if price_change_5d > 0:
            momentum_score += 2
        if price_change_20d > 0:
            momentum_score += 3
        
        signal_strength += (momentum_score - 3) / 3 * 0.4  # Normalize to [-0.4, 0.4]
        
        # Volume confirmation
        if volume_ratio > 1.5:  # High volume
            signal_strength += 0.2 if price_change_1d > 0 else -0.2
        
        # MACD momentum
        if current_macd > current_macd_signal:
            signal_strength += 0.2
        else:
            signal_strength -= 0.2
        
        # ROC momentum
        signal_strength += np.clip(current_roc / 10, -0.2, 0.2)  # Scale ROC contribution
        
        # Normalize signal strength
        signal_strength = np.clip(signal_strength, -1.0, 1.0)
        
        return {
            "name": "Momentum",
            "strength": signal_strength,
            "direction": "bullish" if signal_strength > 0 else "bearish",
            "confidence": abs(signal_strength),
            "indicators": {
                "price_change_1d": price_change_1d,
                "price_change_5d": price_change_5d,
                "price_change_20d": price_change_20d,
                "volume_ratio": volume_ratio,
                "macd_momentum": current_macd - current_macd_signal,
                "roc": current_roc
            }
        }
    
    def _volatility_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Volatility analysis for regime detection and signal adjustment."""
        
        config = self.strategies.get("volatility", {})
        atr_period = config.get("atr_period", 14)
        volatility_threshold = config.get("volatility_threshold", 0.02)
        
        # Calculate volatility indicators
        atr = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=atr_period)
        
        # Historical volatility
        returns = data['Close'].pct_change().dropna()
        historical_vol = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
        
        # Current values
        current_atr = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0
        current_vol = historical_vol.iloc[-1] if not np.isnan(historical_vol.iloc[-1]) else 0
        current_price = data['Close'].iloc[-1]
        
        # ATR as percentage of price
        atr_pct = current_atr / current_price if current_price > 0 else 0
        
        # Volatility regime
        vol_percentile = stats.percentileofscore(historical_vol.dropna(), current_vol)
        
        # Z-score of volatility
        vol_mean = historical_vol.mean()
        vol_std = historical_vol.std()
        vol_zscore = (current_vol - vol_mean) / vol_std if vol_std > 0 else 0
        
        # Signal based on volatility regime
        signal_strength = 0.0
        
        if vol_percentile < 20:  # Low volatility regime
            signal_strength = 0.3  # Favorable for trend following
        elif vol_percentile > 80:  # High volatility regime
            signal_strength = -0.3  # Favorable for mean reversion
        
        # Adjust based on ATR
        if atr_pct > volatility_threshold:
            signal_strength *= 0.7  # Reduce confidence in high volatility
        
        return {
            "name": "Volatility",
            "strength": signal_strength,
            "direction": "low_vol" if vol_percentile < 50 else "high_vol",
            "confidence": abs(signal_strength),
            "indicators": {
                "atr": current_atr,
                "atr_percentage": atr_pct,
                "historical_volatility": current_vol,
                "volatility_percentile": vol_percentile,
                "volatility_zscore": vol_zscore,
                "regime": "low" if vol_percentile < 30 else "high" if vol_percentile > 70 else "normal"
            }
        }
    
    def _statistical_arbitrage_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Statistical arbitrage signals using skewness, kurtosis, and other metrics."""
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 30:
            return self._create_empty_strategy_result("Statistical Arbitrage")
        
        # Statistical measures
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        jarque_bera_stat, jarque_bera_p = stats.jarque_bera(returns)
        
        # Z-score of current price vs. rolling mean
        rolling_mean = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        current_price = data['Close'].iloc[-1]
        price_zscore = (current_price - rolling_mean.iloc[-1]) / rolling_std.iloc[-1] if rolling_std.iloc[-1] > 0 else 0
        
        # Hurst exponent (simplified)
        def hurst_exponent(prices, max_lag=20):
            """Calculate simplified Hurst exponent."""
            if len(prices) < max_lag * 2:
                return 0.5
            
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        hurst = hurst_exponent(data['Close'].values[-100:])  # Use last 100 data points
        
        # Signal calculation
        signal_strength = 0.0
        
        # Price mean reversion signal based on z-score
        if abs(price_zscore) > 2:
            signal_strength += -np.sign(price_zscore) * 0.4  # Mean reversion signal
        
        # Skewness signal
        if abs(skewness) > 0.5:
            signal_strength += -np.sign(skewness) * 0.2  # Counter-skewness signal
        
        # Hurst exponent signal
        if hurst < 0.4:  # Mean reverting
            signal_strength += -np.sign(price_zscore) * 0.2
        elif hurst > 0.6:  # Trending
            signal_strength += np.sign(price_zscore) * 0.2
        
        # Excess kurtosis signal (high kurtosis may indicate regime change)
        if kurtosis > 1:
            signal_strength *= 0.8  # Reduce confidence
        
        # Normalize signal strength
        signal_strength = np.clip(signal_strength, -1.0, 1.0)
        
        return {
            "name": "Statistical Arbitrage",
            "strength": signal_strength,
            "direction": "bullish" if signal_strength > 0 else "bearish",
            "confidence": abs(signal_strength),
            "indicators": {
                "price_zscore": price_zscore,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "hurst_exponent": hurst,
                "jarque_bera_pvalue": jarque_bera_p,
                "normality_test": "normal" if jarque_bera_p > 0.05 else "non_normal"
            }
        }
    
    def _combine_signals(self, strategies: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Combine signals from all strategies into an overall signal with portfolio awareness."""
        
        total_weight = 0
        weighted_strength = 0
        
        for strategy_name, strategy_result in strategies.items():
            weight = self.strategy_weights.get(strategy_name, 0.2)
            strength = strategy_result.get("strength", 0)
            
            weighted_strength += strength * weight
            total_weight += weight
        
        # Normalize by total weight
        overall_strength = weighted_strength / total_weight if total_weight > 0 else 0
        
        # Apply portfolio-aware adjustments
        # Use pure technical signal without portfolio bias
        adjusted_strength = overall_strength
        
        # Determine signal classification
        signal_class = "neutral"
        if adjusted_strength >= self.signal_thresholds["strong_buy"]:
            signal_class = "strong_buy"
        elif adjusted_strength >= self.signal_thresholds["buy"]:
            signal_class = "buy"
        elif adjusted_strength <= self.signal_thresholds["strong_sell"]:
            signal_class = "strong_sell"
        elif adjusted_strength <= self.signal_thresholds["sell"]:
            signal_class = "sell"
        
        return {
            "strength": adjusted_strength,
            "classification": signal_class,
            "direction": "bullish" if adjusted_strength > 0 else "bearish" if adjusted_strength < 0 else "neutral",
            "confidence": abs(adjusted_strength),
            "contributing_strategies": len(strategies)
        }
    
    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics for the symbol."""
        
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 20:
            return {}
        
        # Basic risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        var_95 = np.percentile(returns, 5)  # 5% VaR
        cvar_95 = returns[returns <= var_95].mean()  # Conditional VaR
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            "volatility": volatility,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "current_drawdown": drawdown.iloc[-1]
        }
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime."""
        
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 50:
            return "insufficient_data"
        
        # Calculate regime indicators
        recent_returns = returns.tail(20)
        volatility = recent_returns.std()
        trend = recent_returns.mean()
        
        # Volume analysis
        volume_trend = data['Volume'].tail(20).mean() / data['Volume'].tail(50).mean()
        
        # Regime classification
        if trend > 0.001 and volatility < 0.02 and volume_trend > 1.1:
            return "bullish_trending"
        elif trend < -0.001 and volatility < 0.02 and volume_trend > 1.1:
            return "bearish_trending"
        elif volatility > 0.03:
            return "high_volatility"
        elif abs(trend) < 0.0005 and volatility < 0.015:
            return "low_volatility_sideways"
        else:
            return "normal"
    
    def _create_empty_analysis(self, symbol: str) -> Dict[str, Any]:
        """Create empty analysis result for symbols with insufficient data."""
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "strategies": {
                "trend_following": self._create_empty_strategy_result("Trend Following"),
                "mean_reversion": self._create_empty_strategy_result("Mean Reversion"),
                "momentum": self._create_empty_strategy_result("Momentum"),
                "volatility": self._create_empty_strategy_result("Volatility"),
                "statistical_arbitrage": self._create_empty_strategy_result("Statistical Arbitrage")
            },
            "overall_signal": {
                "strength": 0.0,
                "classification": "neutral",
                "direction": "neutral",
                "confidence": 0.0,
                "contributing_strategies": 0
            },
            "risk_metrics": {},
            "market_regime": "insufficient_data"
        }
    
    def _create_empty_strategy_result(self, strategy_name: str) -> Dict[str, Any]:
        """Create empty strategy result."""
        return {
            "name": strategy_name,
            "strength": 0.0,
            "direction": "neutral",
            "confidence": 0.0,
            "indicators": {}
        }
    
    def _calculate_average_signal_strength(self, results: Dict[str, Any]) -> float:
        """Calculate average signal strength across all symbols."""
        if not results:
            return 0.0
        
        strengths = [
            result["overall_signal"]["strength"] 
            for result in results.values() 
            if "overall_signal" in result
        ]
        
        return np.mean(strengths) if strengths else 0.0
    
    def _count_signals(self, results: Dict[str, Any], signal_type: str) -> int:
        """Count number of bullish or bearish signals."""
        count = 0
        
        for result in results.values():
            if "overall_signal" in result:
                direction = result["overall_signal"].get("direction", "neutral")
                if signal_type == "bullish" and direction == "bullish":
                    count += 1
                elif signal_type == "bearish" and direction == "bearish":
                    count += 1
        
        return count

    async def _calculate_position_risk(self, symbol: str, position: Dict[str, Any], 
                                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics for a single position."""
        
        if symbol not in market_data:
            return self._create_empty_position_risk(symbol)
        
        hist_data = market_data[symbol]
        if not isinstance(hist_data, pd.DataFrame) or "Close" not in hist_data.columns or hist_data.empty:
            return self._create_empty_position_risk(symbol)
        
        returns = hist_data["Close"].pct_change().dropna()
        if returns.empty:
            return self._create_empty_position_risk(symbol)
            
        current_price = hist_data["Close"].iloc[-1]
        
        # Position details