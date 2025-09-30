"""
Enhanced Performance Analytics Module
Provides advanced portfolio performance metrics including Sharpe ratio, Sortino ratio, and drawdown tracking.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import math
from loguru import logger


class PerformanceAnalytics:
    """
    Advanced performance analytics for trading portfolios.
    
    Calculates professional-grade metrics including:
    - Sharpe Ratio (risk-adjusted returns)
    - Sortino Ratio (downside risk-adjusted returns)
    - Maximum Drawdown
    - Calmar Ratio
    - Win/Loss ratios
    - Risk metrics (VaR, volatility)
    """
    
    def __init__(self, initial_capital: float = 100000, risk_free_rate: float = 0.02):
        """
        Initialize performance analytics.
        
        Args:
            initial_capital: Starting portfolio value
            risk_free_rate: Annual risk-free rate (default 2% for current treasury rates)
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = risk_free_rate / 252  # Annualized to daily
        
        # Performance tracking
        self.portfolio_values: List[float] = [initial_capital]
        self.timestamps: List[datetime] = [datetime.now()]
        self.trade_returns: List[float] = []
        self.daily_returns: List[float] = []
        
        logger.info(f"Performance Analytics initialized with ${initial_capital:,.2f} capital")

    def update_portfolio_value(self, current_value: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update portfolio value and calculate returns.
        
        Args:
            current_value: Current total portfolio value
            timestamp: Timestamp of the value (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Store portfolio value
        self.portfolio_values.append(current_value)
        self.timestamps.append(timestamp)
        
        # Calculate daily return if we have previous value
        if len(self.portfolio_values) >= 2:
            prev_value = self.portfolio_values[-2]
            if prev_value > 0:
                daily_return = (current_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
                
                logger.debug(f"Portfolio: ${current_value:,.2f}, Daily Return: {daily_return:.4f}")

    def add_trade_return(self, trade_pnl: float, trade_value: float) -> None:
        """
        Add individual trade return for trade-level analytics.
        
        Args:
            trade_pnl: Profit/loss from the trade
            trade_value: Total value of the trade
        """
        if trade_value > 0:
            trade_return = trade_pnl / trade_value
            self.trade_returns.append(trade_return)
            logger.debug(f"Trade return added: {trade_return:.4f}")

    def calculate_sharpe_ratio(self, period_days: int = 252) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            period_days: Number of days to use for calculation (default 252 = 1 year)
            
        Returns:
            Sharpe ratio (higher is better, >1.0 is good, >2.0 is excellent)
        """
        if len(self.daily_returns) < 2:
            return 0.0
            
        returns = np.array(self.daily_returns[-period_days:])
        
        # Calculate excess returns (returns - risk-free rate)
        excess_returns = returns - self.daily_risk_free_rate
        
        # Sharpe ratio = mean(excess returns) / std(excess returns) * sqrt(252)
        if np.std(excess_returns) == 0:
            return 0.0
            
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe

    def calculate_sortino_ratio(self, period_days: int = 252) -> float:
        """
        Calculate annualized Sortino ratio (focuses only on downside volatility).
        
        Args:
            period_days: Number of days to use for calculation
            
        Returns:
            Sortino ratio (higher is better, typically higher than Sharpe)
        """
        if len(self.daily_returns) < 2:
            return 0.0
            
        returns = np.array(self.daily_returns[-period_days:])
        excess_returns = returns - self.daily_risk_free_rate
        
        # Only consider negative returns for downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
            
        downside_deviation = np.std(downside_returns) * np.sqrt(252)
        
        if downside_deviation == 0:
            return 0.0
            
        sortino = np.mean(excess_returns) * np.sqrt(252) / downside_deviation
        return sortino

    def calculate_maximum_drawdown(self) -> Tuple[float, float, datetime, datetime]:
        """
        Calculate maximum drawdown and related metrics.
        
        Returns:
            Tuple of (max_drawdown_pct, max_drawdown_amount, start_date, end_date)
        """
        if len(self.portfolio_values) < 2:
            return 0.0, 0.0, self.timestamps[0], self.timestamps[0]
            
        values = np.array(self.portfolio_values)
        
        # Calculate running maximum (peak values)
        peaks = np.maximum.accumulate(values)
        
        # Calculate drawdowns
        drawdowns = (values - peaks) / peaks
        
        # Find maximum drawdown
        max_dd_idx = np.argmin(drawdowns)
        max_drawdown_pct = abs(drawdowns[max_dd_idx])
        max_drawdown_amount = abs(values[max_dd_idx] - peaks[max_dd_idx])
        
        # Find the peak before the maximum drawdown
        peak_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if values[i] == peaks[max_dd_idx]:
                peak_idx = i
                break
                
        start_date = self.timestamps[peak_idx]
        end_date = self.timestamps[max_dd_idx]
        
        return max_drawdown_pct, max_drawdown_amount, start_date, end_date

    def calculate_calmar_ratio(self, period_days: int = 252) -> float:
        """
        Calculate Calmar ratio (Annual Return / Maximum Drawdown).
        
        Args:
            period_days: Period for calculation
            
        Returns:
            Calmar ratio (higher is better, >1.0 is good)
        """
        max_drawdown, _, _, _ = self.calculate_maximum_drawdown()
        
        if max_drawdown == 0 or len(self.portfolio_values) < 2:
            return 0.0
            
        # Calculate annualized return
        total_return = (self.portfolio_values[-1] / self.portfolio_values[0]) - 1
        days_elapsed = (self.timestamps[-1] - self.timestamps[0]).days
        
        if days_elapsed == 0:
            return 0.0
            
        annualized_return = (1 + total_return) ** (365.25 / days_elapsed) - 1
        
        calmar = annualized_return / max_drawdown
        return calmar

    def calculate_win_loss_metrics(self) -> Dict[str, float]:
        """
        Calculate win/loss ratios and related metrics.
        
        Returns:
            Dictionary with win rate, avg win, avg loss, profit factor
        """
        if not self.trade_returns:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0
            }
            
        returns = np.array(self.trade_returns)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        
        # Profit factor = total wins / total losses
        total_wins = np.sum(wins) if len(wins) > 0 else 0.0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_trades": len(returns)
        }

    def calculate_value_at_risk(self, confidence_level: float = 0.05, period_days: int = 252) -> float:
        """
        Calculate Value at Risk (VaR) at given confidence level.
        
        Args:
            confidence_level: Confidence level (0.05 = 5% VaR)
            period_days: Period for calculation
            
        Returns:
            VaR as percentage of portfolio value
        """
        if len(self.daily_returns) < 2:
            return 0.0
            
        returns = np.array(self.daily_returns[-period_days:])
        var = np.percentile(returns, confidence_level * 100)
        return abs(var)

    def get_comprehensive_metrics(self, period_days: int = 252) -> Dict[str, Any]:
        """
        Get all performance metrics in a comprehensive report.
        
        Args:
            period_days: Period for calculations
            
        Returns:
            Dictionary with all performance metrics
        """
        if len(self.portfolio_values) < 2:
            logger.warning("Insufficient data for performance metrics calculation")
            return self._get_empty_metrics()
            
        # Calculate basic metrics
        current_value = self.portfolio_values[-1]
        total_return = (current_value / self.initial_capital) - 1
        days_elapsed = (self.timestamps[-1] - self.timestamps[0]).days
        
        # Annualized return
        if days_elapsed > 0:
            annualized_return = (1 + total_return) ** (365.25 / days_elapsed) - 1
        else:
            annualized_return = 0.0
            
        # Risk metrics
        sharpe = self.calculate_sharpe_ratio(period_days)
        sortino = self.calculate_sortino_ratio(period_days)
        calmar = self.calculate_calmar_ratio(period_days)
        
        # Drawdown analysis
        max_dd_pct, max_dd_amount, dd_start, dd_end = self.calculate_maximum_drawdown()
        
        # Volatility
        volatility = np.std(self.daily_returns[-period_days:]) * np.sqrt(252) if self.daily_returns else 0.0
        
        # VaR
        var_5 = self.calculate_value_at_risk(0.05, period_days)
        var_1 = self.calculate_value_at_risk(0.01, period_days)
        
        # Win/Loss metrics
        win_loss_metrics = self.calculate_win_loss_metrics()
        
        metrics = {
            # Basic Performance
            "current_value": current_value,
            "initial_capital": self.initial_capital,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "days_elapsed": days_elapsed,
            
            # Risk-Adjusted Returns
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            
            # Risk Metrics
            "volatility": volatility,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_amount": max_dd_amount,
            "max_drawdown_start": dd_start,
            "max_drawdown_end": dd_end,
            "var_5pct": var_5,
            "var_1pct": var_1,
            
            # Trading Metrics
            **win_loss_metrics,
            
            # Data Quality
            "data_points": len(self.portfolio_values),
            "last_update": self.timestamps[-1] if self.timestamps else None
        }
        
        logger.info(f"Performance metrics calculated: Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, "
                   f"Max DD={max_dd_pct:.2%}, Win Rate={win_loss_metrics['win_rate']:.2%}")
        
        return metrics

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure when insufficient data."""
        return {
            "current_value": self.initial_capital,
            "initial_capital": self.initial_capital,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "days_elapsed": 0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "volatility": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_amount": 0.0,
            "max_drawdown_start": None,
            "max_drawdown_end": None,
            "var_5pct": 0.0,
            "var_1pct": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "data_points": len(self.portfolio_values),
            "last_update": None
        }

    def get_performance_summary(self) -> str:
        """
        Get a formatted performance summary string.
        
        Returns:
            Formatted string with key performance metrics
        """
        metrics = self.get_comprehensive_metrics()
        
        summary = f"""
📊 PERFORMANCE SUMMARY
{'='*50}
💰 Total Return: {metrics['total_return']:+.2%}
📈 Annualized Return: {metrics['annualized_return']:+.2%}
📉 Max Drawdown: {metrics['max_drawdown_pct']:.2%}

🎯 RISK-ADJUSTED METRICS
{'='*50}
⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.2f} {'✅' if metrics['sharpe_ratio'] > 1.0 else '⚠️' if metrics['sharpe_ratio'] > 0.5 else '❌'}
🎯 Sortino Ratio: {metrics['sortino_ratio']:.2f} {'✅' if metrics['sortino_ratio'] > 1.5 else '⚠️' if metrics['sortino_ratio'] > 1.0 else '❌'}
🏆 Calmar Ratio: {metrics['calmar_ratio']:.2f} {'✅' if metrics['calmar_ratio'] > 1.0 else '⚠️' if metrics['calmar_ratio'] > 0.5 else '❌'}

📊 TRADING METRICS
{'='*50}
🎯 Win Rate: {metrics['win_rate']:.1%}
💹 Profit Factor: {metrics['profit_factor']:.2f}
📈 Total Trades: {metrics['total_trades']}
🌪️ Volatility: {metrics['volatility']:.1%}

⚠️ RISK METRICS
{'='*50}
📉 5% VaR: {metrics['var_5pct']:.2%}
🚨 1% VaR: {metrics['var_1pct']:.2%}
📅 Days Tracked: {metrics['days_elapsed']}
        """.strip()
        
        return summary

    def export_performance_data(self) -> Dict[str, Any]:
        """
        Export all performance data for external analysis.
        
        Returns:
            Dictionary with all raw performance data
        """
        return {
            "portfolio_values": self.portfolio_values,
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "daily_returns": self.daily_returns,
            "trade_returns": self.trade_returns,
            "comprehensive_metrics": self.get_comprehensive_metrics()
        }