"""
Portfolio Manager Agent

This agent is responsible for:
- Trade execution and order management
- Capital allocation and position sizing
- Portfolio rebalancing
- Performance tracking
- Integration with trading APIs
"""

import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import pandas as pd
import numpy as np
from loguru import logger

from .base_agent import BaseAgent
from core.utils.json_utils import safe_json_dump
import json
import os
import asyncio


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Position:
    """Represents a trading position."""
    
    def __init__(self, symbol: str, size: float, entry_price: float, entry_date: datetime):
        self.symbol = symbol
        self.size = size  # Positive for long, negative for short
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.current_price = entry_price
        self.last_update = entry_date
        
    @property
    def market_value(self) -> float:
        return self.size * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.size
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price
    
    def update_price(self, new_price: float):
        self.current_price = new_price
        self.last_update = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "size": self.size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "entry_date": self.entry_date,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "last_update": self.last_update
        }


class Order:
    """Represents a trading order."""
    
    def __init__(self, symbol: str, size: float, order_type: OrderType, 
                 price: Optional[float] = None, stop_price: Optional[float] = None):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.size = size  # Positive for buy, negative for sell
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.status = OrderStatus.PENDING
        self.created_at = datetime.now()
        self.filled_at = None
        self.filled_price = None
        self.filled_size = 0.0
        self.remaining_size = abs(size)
        
    @property
    def is_buy(self) -> bool:
        return self.size > 0
    
    @property
    def is_sell(self) -> bool:
        return self.size < 0
    
    def fill(self, price: float, size: float):
        """Fill the order partially or completely."""
        self.filled_price = price
        self.filled_size += size
        self.remaining_size -= size
        
        if self.remaining_size <= 0:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        if not self.filled_at:
            self.filled_at = datetime.now()
    
    def cancel(self):
        """Cancel the order."""
        self.status = OrderStatus.CANCELLED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "size": self.size,
            "order_type": self.order_type.value,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "created_at": self.created_at,
            "filled_at": self.filled_at,
            "filled_price": self.filled_price,
            "filled_size": self.filled_size,
            "remaining_size": self.remaining_size
        }


class PortfolioManagerAgent(BaseAgent):
    """
    Portfolio Manager Agent for trade execution and portfolio management.
    
    Features:
    - Order management and execution
    - Position tracking
    - Capital allocation
    - Portfolio rebalancing
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "PortfolioManagerAgent")
        
        # Configuration
        self.initial_capital = config.get("initial_capital", 100000)
        self.max_positions = config.get("max_positions", 10)
        self.rebalance_frequency = config.get("rebalance_frequency", "daily")
        self.allocation_method = config.get("allocation_method", "equal_weight")
        # Optimizer parameters (used when allocation_method == "optimizer")
        self.total_investment_target = float(config.get("total_investment_target", 0.6))  # fraction of portfolio
        self.risk_aversion = float(config.get("risk_aversion", 5.0))  # higher = more risk averse
        self.mu_scale = float(config.get("mu_scale", 0.10))  # maps signal strength to expected annual return
        
        # Portfolio state
        self.cash_balance = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.portfolio_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Risk limits (inherited from config)
        self.max_position_size = config.get("max_position_size", 0.1)
        self.stop_loss_threshold = config.get("stop_loss_threshold", 0.05)
        self.take_profit_threshold = config.get("take_profit_threshold", 0.15)
        
        # Persistence
        persistence_cfg = config.get("persistence", {})
        self.persistence_enabled = bool(persistence_cfg.get("enabled", False))
        self.persistence_file = str(persistence_cfg.get("state_file", "data/portfolio_state.json"))
        
        self.logger.info(f"Portfolio Manager initialized with ${self.initial_capital:,.2f} capital")
        
        # Load persisted state if enabled
        if self.persistence_enabled:
            self._load_persisted_state()
    
    async def _execute_logic(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Execute portfolio management tasks.
        
        Args:
            input_data: Contains market data, analysis signals, and risk approvals
            
        Returns:
            Tuple of (portfolio_results, metrics)
        """
        start_time = time.time()
        
        # Extract input data
        market_data = input_data.get("market_data", {})
        analysis_signals = input_data.get("analysis_signals", {})
        risk_approvals = input_data.get("risk_approvals", {})
        action = input_data.get("action", "manage")  # manage, rebalance, execute_trades
        is_backtest = input_data.get("is_backtest", False)
        
        # Update portfolio with current market prices
        self._update_portfolio_prices(market_data)
        
        # Perform requested action
        results = {
            "timestamp": datetime.now(),
            "action_performed": action,
            "portfolio_summary": {},
            "orders_executed": [],
            "positions_updated": [],
            "performance_update": {},
            "recommendations": []
        }
        
        # This is the main portfolio management logic
        await self._manage_portfolio(market_data, analysis_signals, risk_approvals, results, is_backtest)
        
        # Persist state after all actions for this cycle are complete
        if self.persistence_enabled and not is_backtest:
            self._persist_state()
            
        # Update and calculate performance metrics
        portfolio_summary = self._get_portfolio_summary()
        performance_update = self._update_performance_metrics()
        
        # Calculate metrics
        execution_time = time.time() - start_time
        
        metrics = {
            "execution_time_seconds": execution_time,
            "total_portfolio_value": portfolio_summary["total_value"],
            "cash_balance": self.cash_balance,
            "active_positions": len(self.positions),
            "pending_orders": len([o for o in self.orders.values() if o.status == OrderStatus.PENDING]),
            "total_unrealized_pnl": portfolio_summary["total_unrealized_pnl"],
            "portfolio_return": performance_update.get("total_return", 0),
            "trades_executed": len(results["orders_executed"])
        }
        
        # Update portfolio history
        self._update_portfolio_history(portfolio_summary, performance_update)
        
        return results, metrics
    
    async def _manage_portfolio(self, market_data: Dict[str, Any], 
                              analysis_signals: Dict[str, Any],
                              risk_approvals: Dict[str, Any], 
                              results: Dict[str, Any],
                              is_backtest: bool = False):
        """Main portfolio management logic."""

        # First, process any stop-loss or take-profit orders based on current prices
        executed_stop_orders = await self._process_stop_orders(market_data)
        if executed_stop_orders:
            results["orders_executed"] = results.get("orders_executed", []) + executed_stop_orders

        # Decide on new trades based on the allocation method
        # SECURITY: Both paths must respect risk manager approvals
        if self.allocation_method == "optimizer":
            trades_to_execute = self._optimizer_allocation_signals(market_data, analysis_signals, risk_approvals)
        else: # Fallback to simpler direct conversion
            trades_to_execute = self._convert_risk_approvals_to_trades(risk_approvals, market_data)
        
        # SECURITY CHECK: Ensure no trades execute without risk approval
        if trades_to_execute and not risk_approvals:
            self.logger.error("SECURITY VIOLATION: Attempted to execute trades without risk manager approval!")
            trades_to_execute = []

        # Execute the trades
        if trades_to_execute:
            await self._execute_trades(trades_to_execute, market_data, results, is_backtest)

    def _convert_risk_approvals_to_trades(self, risk_approvals: Dict[str, Any], 
                                        market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert risk manager approved trades into properly sized portfolio trades.
        
        This is the correct approach: Portfolio manager only handles sizing and execution
        of trades that have already been approved by the risk manager for individual merit.
        """
        
        approved_trades = []
        
        for symbol, approval in (risk_approvals or {}).items():
            if not isinstance(approval, dict) or not approval.get("approved", False):
                if isinstance(approval, dict):
                    rejection_reason = approval.get("reason", "Unknown reason")
                    self.logger.info(f"Risk manager rejected {symbol}: {rejection_reason}")
                continue
                
            # Get the target value from risk approval (already calculated)
            target_value = approval.get("target_value", 0)
            adjusted_size = approval.get("adjusted_size", 0)
            
            if target_value <= 0 or adjusted_size == 0:
                continue
                
            # Get current price for execution
            current_price = self._get_current_price(symbol, market_data)
            if current_price <= 0:
                continue
            
            # Check if we already have a position and calculate incremental buy
            current_position_value = 0
            if symbol in self.positions:
                current_position_value = abs(self.positions[symbol].market_value)
            
            # Only proceed if this is a net new investment (incremental buy)
            incremental_value = target_value - current_position_value
            if incremental_value <= 1000:  # Minimum position threshold
                continue
                
            # Final position sizing based on available cash and risk-approved target
            final_shares = min(adjusted_size, incremental_value / current_price)
            if final_shares <= 0:
                continue
                
            # Create properly sized trade
            trade_signal = {
                "symbol": symbol,
                "action": "buy",
                "size": final_shares,
                "target_value": incremental_value,
                "risk_approved": True,
                "original_risk_approval": approval,
                "execution_price": current_price
            }
            
            approved_trades.append(trade_signal)
            self.logger.info(f"Portfolio manager: Executing risk-approved {final_shares:.2f} shares of {symbol} (${incremental_value:.0f})")
        
        return approved_trades

    def _optimize_portfolio_allocation(self, risk_approvals: Dict[str, Any], 
                                     analysis_signals: Dict[str, Any],
                                     market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Portfolio Manager's main responsibility: 
        - Prioritize trades from risk-approved signals
        - Optimize capital allocation across approved trades  
        - Rebalance entire portfolio including existing positions
        - Ensure diversification and compliance with investment rules
        """
        
        # Get all risk-approved trades with their maximum allowed sizes
        approved_candidates = {}
        for symbol, approval in (risk_approvals or {}).items():
            if not isinstance(approval, dict) or not approval.get("approved", False):
                continue
            
            adjusted_size = approval.get("adjusted_size", 0)
            signal_strength = approval.get("signal_strength", 0)
            
            if adjusted_size > 0 and abs(signal_strength) > 0:
                approved_candidates[symbol] = {
                    "max_size": adjusted_size,
                    "signal_strength": signal_strength,
                    "risk_approval": approval
                }
        
        if not approved_candidates:
            self.logger.info("No risk-approved candidates for portfolio optimization")
            return []
        
        # Run portfolio optimization based on allocation method
        if self.allocation_method == "optimizer":
            return self._run_portfolio_optimizer(approved_candidates, analysis_signals, market_data)
        else:
            return self._run_simple_allocation(approved_candidates, market_data)
    
    def _run_portfolio_optimizer(self, approved_candidates: Dict[str, Any],
                               analysis_signals: Dict[str, Any], 
                               market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run mean-variance optimization across risk-approved trades.
        Considers existing portfolio for full rebalancing.
        """
        
        # Get current portfolio state
        current_portfolio_value = self.cash_balance
        current_weights = {}
        
        # Calculate current position weights
        for symbol, position in self.positions.items():
            current_portfolio_value += abs(position.market_value)
            current_weights[symbol] = abs(position.market_value) / current_portfolio_value if current_portfolio_value > 0 else 0
        
        # Include all symbols: existing positions + new approved candidates
        all_symbols = set(current_weights.keys()) | set(approved_candidates.keys())
        
        # Build expected returns vector from signal strengths
        expected_returns = {}
        for symbol in all_symbols:
            if symbol in approved_candidates:
                strength = approved_candidates[symbol]["signal_strength"]
                expected_returns[symbol] = strength * self.mu_scale  # Convert to annual return expectation
            else:
                # For existing positions without new signals, assume neutral (hold)
                expected_returns[symbol] = 0.0
        
        # Calculate covariance matrix from historical returns
        symbols_list = list(all_symbols)
        returns_data = []
        
        for symbol in symbols_list:
            if symbol in market_data and isinstance(market_data[symbol], pd.DataFrame):
                returns = market_data[symbol]["Close"].pct_change().dropna()
                if len(returns) >= 20:  # Minimum data requirement
                    returns_data.append(returns)
                else:
                    # Not enough data, assume average risk
                    returns_data.append(pd.Series([0.01] * 252, index=pd.date_range('2023-01-01', periods=252)))
            else:
                # No data available, assume average risk
                returns_data.append(pd.Series([0.01] * 252, index=pd.date_range('2023-01-01', periods=252)))
        
        if len(returns_data) < 2:
            # Fall back to simple allocation if insufficient data for optimization
            return self._run_simple_allocation(approved_candidates, market_data)
        
        # Create returns matrix and calculate covariance
        returns_df = pd.concat(returns_data, axis=1, keys=symbols_list).fillna(0)
        covariance_matrix = returns_df.cov().values
        
        # Run optimization (simplified mean-variance)
        mu = np.array([expected_returns[symbol] for symbol in symbols_list])
        
        # Investment constraints
        total_investable = current_portfolio_value * self.total_investment_target
        available_cash = min(self.cash_balance, total_investable)
        
        if available_cash < 1000:  # Minimum threshold
            self.logger.info(f"Insufficient available cash for optimization: ${available_cash:.0f}")
            return []
        
        # Simple optimization: allocate based on risk-adjusted signal strength
        optimized_trades = []
        
        # Calculate allocation weights based on signal strength and diversification
        total_signal_strength = sum(abs(expected_returns[s]) for s in approved_candidates.keys())
        
        for symbol in approved_candidates.keys():
            if total_signal_strength <= 0:
                continue
                
            # Weight based on signal strength
            signal_weight = abs(expected_returns[symbol]) / total_signal_strength
            
            # Apply diversification constraints (max 15% per position for new investments)
            max_position_weight = 0.15
            diversified_weight = min(signal_weight, max_position_weight)
            
            # Calculate target value for this symbol
            target_investment = available_cash * diversified_weight
            
            # Check against risk manager's maximum size
            max_shares = approved_candidates[symbol]["max_size"]
            current_price = self._get_current_price(symbol, market_data)
            
            if current_price <= 0:
                continue
                
            max_investment_by_risk = max_shares * current_price
            final_investment = min(target_investment, max_investment_by_risk)
            
            if final_investment >= 1000:  # Minimum position threshold
                final_shares = final_investment / current_price
                
                optimized_trades.append({
                    "symbol": symbol,
                    "action": "buy",
                    "size": final_shares,
                    "target_value": final_investment,
                    "allocation_weight": diversified_weight,
                    "signal_strength": expected_returns[symbol],
                    "risk_approved": True,
                    "optimization_method": "mean_variance_simplified"
                })
                
                self.logger.info(f"Portfolio optimization: {symbol} allocated {diversified_weight:.1%} "
                               f"(${final_investment:.0f}, {final_shares:.1f} shares)")
        
        return optimized_trades
    
    def _run_simple_allocation(self, approved_candidates: Dict[str, Any],
                             market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simple equal-weight allocation across approved trades.
        """
        
        if not approved_candidates:
            return []
        
        # Equal weight allocation
        weight_per_symbol = 1.0 / len(approved_candidates)
        available_cash = self.cash_balance * self.total_investment_target
        
        simple_trades = []
        
        for symbol, candidate in approved_candidates.items():
            target_value = available_cash * weight_per_symbol
            max_shares = candidate["max_size"]
            current_price = self._get_current_price(symbol, market_data)
            
            if current_price <= 0:
                continue
                
            max_value_by_risk = max_shares * current_price
            final_value = min(target_value, max_value_by_risk)
            
            if final_value >= 1000:
                final_shares = final_value / current_price
                
                simple_trades.append({
                    "symbol": symbol,
                    "action": "buy", 
                    "size": final_shares,
                    "target_value": final_value,
                    "allocation_weight": weight_per_symbol,
                    "risk_approved": True,
                    "optimization_method": "equal_weight"
                })
        
        return simple_trades

    def _optimizer_allocation_signals(self, market_data: Dict[str, Any], 
                                      analysis_signals: Dict[str, Any],
                                      risk_approvals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals using mean-variance optimizer with existing holdings awareness.

        Enhanced to consider:
        - Current portfolio concentration and correlation
        - Position trimming recommendations when over-allocated
        - Diversification benefits vs existing holdings
        - Incremental allocation based on marginal utility
        """

        # Get current portfolio state for analysis
        portfolio_value = self._get_total_portfolio_value()
        current_holdings = {sym: pos.market_value for sym, pos in self.positions.items()}
        current_weights = {sym: val / portfolio_value for sym, val in current_holdings.items()} if portfolio_value > 0 else {}
        
        # Analyze position trimming needs first
        trim_recommendations = []
        for symbol, weight in current_weights.items():
            if weight > self.max_position_size:
                # Calculate shares to trim based on weight difference
                weight_to_trim = weight - self.max_position_size
                total_portfolio_value = self._get_total_portfolio_value()
                value_to_trim = weight_to_trim * total_portfolio_value
                
                # Get current price and calculate shares to sell
                current_price = self._get_current_price(symbol, market_data) 
                shares_to_trim = -abs(value_to_trim / current_price) if current_price > 0 else 0
                
                if abs(shares_to_trim) >= 0.01:  # Only trim if meaningful amount
                    trim_recommendations.append({
                        "symbol": symbol,
                        "action": "trim",
                        "size": shares_to_trim,  # Negative for sell order
                        "current_weight": weight,
                        "target_weight": self.max_position_size,
                        "reason": f"Over-allocated: {weight:.1%} > {self.max_position_size:.1%} limit",
                        "order_type": "market",
                        "signal_strength": -0.5,  # Negative signal for trim operations
                        "rationale": f"Trim {abs(shares_to_trim):.2f} shares to maintain position limits"
                    })
        
        # Candidate symbols: positive signal strength (risk approval comes later)
        candidates: List[str] = []
        signal_strengths: Dict[str, float] = {}
        for symbol, signals in (analysis_signals or {}).items():
            overall = signals.get("overall_signal", {})
            strength = float(overall.get("strength", 0) or 0)
            if strength <= 0:
                continue
            
            # Check if risk manager would likely approve (basic pre-screening)
            # The formal approval happens later, but we can do basic checks here
            approved_by_risk = True
            if symbol in (risk_approvals or {}):
                approval = risk_approvals[symbol]
                if isinstance(approval, dict):
                    approved_by_risk = approval.get("approved", False)
            
            # Enhanced screening: reduce signal strength for over-allocated existing positions
            current_weight = current_weights.get(symbol, 0.0)
            if current_weight > self.max_position_size * 0.8:  # Near limit
                strength *= 0.5  # Reduce by half
                self.logger.info(f"Reducing signal strength for {symbol} due to high allocation ({current_weight:.1%})")
            elif current_weight > self.max_position_size * 0.6:  # Getting high
                strength *= 0.75  # Reduce somewhat
            
            # SECURITY FIX: Only include candidates that are explicitly approved by risk manager
            # If no risk approvals exist, DO NOT proceed with any trades
            if risk_approvals and symbol in risk_approvals and approved_by_risk:
                candidates.append(symbol)
                signal_strengths[symbol] = strength
            elif not risk_approvals:
                # No risk data available - this is NOT permission to trade!
                self.logger.warning(f"No risk approvals available - blocking all trades for safety")
                return []
            else:
                # Symbol was evaluated but rejected by risk manager
                risk_data = risk_approvals.get(symbol, {})
                rejection_reason = risk_data.get("reason", "Unknown reason")
                self.logger.info(f"Risk manager rejected {symbol}: {rejection_reason}")
                continue

        if not candidates:
            return trim_recommendations

        # Enhanced position slot management
        candidates.sort(key=lambda s: abs(signal_strengths.get(s, 0)), reverse=True)
        
        # Prioritize new symbols for diversification
        new_symbols = [s for s in candidates if s not in self.positions]
        existing_symbols = [s for s in candidates if s in self.positions]
        
        available_slots = max(0, self.max_positions - len(self.positions))
        
        # Prefer new symbols when we have room, then consider existing
        if available_slots > 0:
            final_candidates = new_symbols[:available_slots] + existing_symbols
        else:
            # No new slots, only consider topping up existing (with reduced enthusiasm)
            final_candidates = existing_symbols[:max(2, len(existing_symbols)//2)]  # Limited existing additions
        
        if not final_candidates:
            return trim_recommendations

        # Build returns matrix from historical data including existing holdings for correlation analysis
        returns_dict: Dict[str, pd.Series] = {}
        all_symbols_for_covariance = list(set(final_candidates + list(current_holdings.keys())))
        
        for sym in all_symbols_for_covariance:
            md = market_data.get(sym)
            if isinstance(md, pd.DataFrame) and "Close" in md.columns and len(md) > 60:
                rets = md["Close"].pct_change().dropna().tail(252)
                if len(rets) >= 30:
                    returns_dict[sym] = rets

        symbols_final = [s for s in final_candidates if s in returns_dict]
        if len(symbols_final) < 1:
            symbols_final = final_candidates  # fallback when covariance cannot be estimated
        if not symbols_final:
            return trim_recommendations

        # Covariance estimation with fallback to diagonal (20% vol^2)
        try:
            if all(s in returns_dict for s in symbols_final):
                df = pd.DataFrame(returns_dict)
                df = df.dropna()
                Sigma = np.array(df.cov()) if not df.empty else np.eye(len(symbols_final)) * 0.04
            else:
                Sigma = np.eye(len(symbols_final)) * 0.04
        except Exception:
            Sigma = np.eye(len(symbols_final)) * 0.04

        # Expected returns from signals with diversification bonus
        mu = np.array([self.mu_scale * signal_strengths.get(s, 0.0) for s in symbols_final])
        
        # Apply diversification bonus for new symbols
        for i, sym in enumerate(symbols_final):
            if sym not in current_holdings:
                # Bonus for adding new uncorrelated assets
                avg_correlation = 0.0
                if len(current_holdings) > 0 and len(returns_dict) > 1:
                    try:
                        correlations = []
                        for existing_sym in current_holdings.keys():
                            if existing_sym in returns_dict and sym in returns_dict:
                                corr = returns_dict[sym].corr(returns_dict[existing_sym])
                                if not np.isnan(corr):
                                    correlations.append(abs(corr))
                        if correlations:
                            avg_correlation = np.mean(correlations)
                    except Exception:
                        avg_correlation = 0.5  # assume moderate correlation
                
                # Diversification bonus: lower correlation = higher bonus
                diversification_bonus = (1.0 - avg_correlation) * 0.02  # Up to 2% annual return bonus
                mu[i] += diversification_bonus
                self.logger.info(f"Diversification bonus for {sym}: {diversification_bonus:.3f} (avg_corr: {avg_correlation:.2f})")

        # Regularize for numerical stability and invert
        Sigma_reg = Sigma + np.eye(len(symbols_final)) * 1e-6
        try:
            inv_Sigma = np.linalg.pinv(Sigma_reg)
        except Exception:
            inv_Sigma = np.eye(len(symbols_final))

        # Unconstrained solution scaled by risk aversion
        denom = max(2 * self.risk_aversion, 1e-6)
        raw_w = (inv_Sigma @ mu) / denom

        # Long-only and per-position cap considering current holdings
        w = np.clip(raw_w, 0.0, float(self.max_position_size))
        
        # Adjust weights to account for current holdings (incremental approach)
        for i, sym in enumerate(symbols_final):
            current_weight = current_weights.get(sym, 0.0)
            # If already near or above target, reduce significantly
            if current_weight >= w[i]:
                w[i] = max(0.0, w[i] - current_weight) * 0.1  # Very small top-up only
            else:
                w[i] = max(0.0, w[i] - current_weight)  # Incremental to target
        
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            return trim_recommendations
            
        # Scale to available investment capacity (total target minus current allocation)
        current_invested_fraction = sum(current_weights.values())
        available_capacity = max(0.0, self.total_investment_target - current_invested_fraction)
        scale = min(available_capacity / w_sum, 1.0) if w_sum > 0 else 0.0
        w = w * scale

        if portfolio_value <= 0:
            return trim_recommendations

        # Convert weights to signals (incremental shares vs current holdings)
        position_cfg = self.config.get("trading", {}).get("position_sizing", {})
        min_position_value = position_cfg.get("min_position_value", 1000)

        signals: List[Dict[str, Any]] = []
        for i, sym in enumerate(symbols_final):
            weight = float(w[i])
            if weight <= 0:
                continue
            target_value = portfolio_value * weight
            price = self._get_current_price(sym, market_data)
            if price <= 0:
                continue
            current_value = 0.0
            if sym in self.positions:
                current_value = abs(self.positions[sym].market_value)

            # Incremental buy amount; skip if already at/above target
            delta_value = target_value - current_value
            if delta_value <= 0:
                continue
            shares = delta_value / price
            if shares * price < min_position_value:
                continue
                
            # Enhanced rationale with diversification info
            current_weight = current_weights.get(sym, 0.0)
            rationale = f"Optimizer MV: target_w={weight:.3f}, current_w={current_weight:.3f}, " \
                       f"mu={signal_strengths.get(sym,0.0)*self.mu_scale:.3f}"
            if sym not in current_holdings:
                rationale += " [NEW+diversification]"
            
            signals.append({
                "symbol": sym,
                "size": shares,
                "signal_strength": signal_strengths.get(sym, 0.0),
                "direction": "bullish",
                "order_type": "market",
                "confidence": abs(signal_strengths.get(sym, 0.0)),
                "rationale": rationale
            })

        # Add trim recommendations to the signals list
        all_recommendations = trim_recommendations + signals
        all_recommendations.sort(key=lambda x: abs(x.get("signal_strength", 0.0)), reverse=True)
        
        return all_recommendations
    
    async def _rebalance_portfolio(self, market_data: Dict[str, Any], 
                                 analysis_signals: Dict[str, Any],
                                 results: Dict[str, Any]):
        """Rebalance portfolio based on allocation method."""
        
        if self.allocation_method == "equal_weight":
            rebalance_orders = await self._equal_weight_rebalance(market_data, analysis_signals)
        elif self.allocation_method == "risk_parity":
            rebalance_orders = await self._risk_parity_rebalance(market_data, analysis_signals)
        elif self.allocation_method == "market_cap":
            rebalance_orders = await self._market_cap_rebalance(market_data, analysis_signals)
        else:
            rebalance_orders = []
        
        results["orders_executed"] = rebalance_orders
        results["recommendations"].append(f"Portfolio rebalanced using {self.allocation_method} method")
    
    async def _execute_trades(self, trades: List[Dict[str, Any]], 
                            market_data: Dict[str, Any],
                            results: Dict[str, Any],
                            is_backtest: bool = False):
        """Execute specific trades."""
        
        executed_orders = []
        
        for trade in trades:
            try:
                # Safely extract trade parameters
                symbol = trade.get("symbol")
                size = trade.get("size", 0)
                order_type = trade.get("order_type", "market")
                
                if not symbol or size == 0:
                    self.logger.warning(f"Skipping trade with missing symbol or size: {trade}")
                    continue
                
                order = await self._create_and_execute_order(symbol, size, 
                                                            OrderType(order_type), 
                                                            self._get_current_price(symbol, market_data),
                                                            is_backtest)
                if order and order.status == OrderStatus.FILLED:
                    executed_orders.append(order.to_dict())
                    # Update or create position
                    self._update_position_from_order(order, trade.get("rationale", ""))
            except Exception as e:
                self.logger.error(f"Error executing trade {trade}: {e}")
        
        results["orders_executed"] = results.get("orders_executed", []) + executed_orders
    
    def _update_portfolio_prices(self, market_data: Dict[str, Any]):
        """Update current prices for all positions."""
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                # Get current price from market data
                if isinstance(market_data[symbol], dict) and "price" in market_data[symbol]:
                    current_price = market_data[symbol]["price"]
                elif isinstance(market_data[symbol], pd.DataFrame) and "Close" in market_data[symbol].columns:
                    current_price = market_data[symbol]["Close"].iloc[-1]
                else:
                    continue
                
                position.update_price(current_price)
    
    async def _process_stop_orders(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process stop loss and take profit orders."""
        
        executed_orders = []
        
        for symbol, position in list(self.positions.items()):
            current_price = position.current_price
            entry_price = position.entry_price
            
            # Calculate P&L percentage
            pnl_pct = position.unrealized_pnl_pct
            
            # Check stop loss
            if pnl_pct <= -self.stop_loss_threshold:
                self.logger.info(f"Stop loss triggered for {symbol}: {pnl_pct:.2%}")
                order = await self._create_and_execute_order(
                    symbol, -position.size, OrderType.MARKET, current_price
                )
                if order:
                    executed_orders.append(order.to_dict())
                    self._close_position(symbol, current_price, "stop_loss")
            
            # Check take profit
            elif pnl_pct >= self.take_profit_threshold:
                self.logger.info(f"Take profit triggered for {symbol}: {pnl_pct:.2%}")
                order = await self._create_and_execute_order(
                    symbol, -position.size, OrderType.MARKET, current_price
                )
                if order:
                    executed_orders.append(order.to_dict())
                    self._close_position(symbol, current_price, "take_profit")
        
        return executed_orders
    
    def _generate_trading_signals(self, analysis_signals: Dict[str, Any], 
                                risk_approvals: Dict[str, Any],
                                market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on analysis and risk approval."""
        
        trading_signals = []
        
        for symbol, signals in analysis_signals.items():
            if symbol not in risk_approvals:
                continue
            
            risk_approval = risk_approvals[symbol]
            if not risk_approval.get("approved", False):
                continue
            
            # Extract overall signal strength
            overall_signal = signals.get("overall_signal", {})
            signal_strength = overall_signal.get("strength", 0)
            signal_direction = overall_signal.get("direction", "neutral")
            
            # Get suggested position size from risk approval
            suggested_size = risk_approval.get("adjusted_size", 0)
            if suggested_size == 0:
                continue
            
            # Smart position sizing logic
            final_size = self._calculate_optimal_position_size(
                symbol, suggested_size, signal_strength, risk_approval, market_data
            )
            
            if final_size == 0:
                continue
            
            # Create trading signal
            trading_signal = {
                "symbol": symbol,
                "size": final_size,
                "signal_strength": signal_strength,
                "direction": signal_direction,
                "order_type": "market",
                "confidence": overall_signal.get("confidence", 0),
                "rationale": self._create_trade_rationale(signals),
                "original_suggested_size": suggested_size,
                "size_adjustment_reason": self._get_size_adjustment_reason(suggested_size, final_size, risk_approval)
            }
            
            trading_signals.append(trading_signal)
        
        # Sort by signal strength and limit to max positions
        trading_signals.sort(key=lambda x: abs(x["signal_strength"]), reverse=True)
        
        # Limit to available position slots
        available_slots = self.max_positions - len(self.positions)
        if available_slots > 0:
            trading_signals = trading_signals[:available_slots]
        
        return trading_signals
    
    def _calculate_optimal_position_size(self, symbol: str, suggested_size: float, 
                                       signal_strength: float, risk_approval: Dict[str, Any],
                                       market_data: Dict[str, Any]) -> float:
        """Calculate optimal position size using value-based approach considering risk limits."""
        
        # Get current portfolio value
        portfolio_value = self._get_total_portfolio_value()
        if portfolio_value <= 0:
            return 0
        
        # Get current price for the symbol
        current_price = self._get_current_price(symbol, market_data)
        if current_price <= 0:
            return 0
        
        # Get target position value from trade proposal (if available)
        target_value = risk_approval.get("target_value", None)
        if target_value is None:
            # Fallback to calculating from suggested_size (backward compatibility)
            target_value = abs(suggested_size * current_price)
        
        # Get value-based position sizing config
        trading_config = self.config.get("trading", {})
        position_config = trading_config.get("position_sizing", {})
        min_position_value = position_config.get("min_position_value", 1000)
        max_position_value = position_config.get("max_position_value", 15000)
        
        # Apply minimum and maximum value constraints
        target_value = max(min_position_value, min(target_value, max_position_value))
        
        # Calculate position weight relative to portfolio
        target_weight = target_value / portfolio_value
        
        # Check portfolio percentage limit (e.g., 10% max per position)
        if target_weight > self.max_position_size:
            target_value = portfolio_value * self.max_position_size
            target_weight = self.max_position_size
            self.logger.info(f"Position value for {symbol} capped at {self.max_position_size*100:.1f}% of portfolio (${target_value:,.0f})")
        
        # Apply signal strength adjustment (for weak signals)
        adjustment_factor = 1.0
        if abs(signal_strength) < 0.3:
            adjustment_factor = abs(signal_strength) / 0.3
            target_value *= adjustment_factor
            self.logger.info(f"Weak signal for {symbol} ({signal_strength:.3f}), reducing position value by {(1-adjustment_factor)*100:.1f}% to ${target_value:,.0f}")
        
        # Convert target value to shares
        final_shares = target_value / current_price
        
        # Apply direction (buy = positive, sell = negative)
        if suggested_size < 0:
            final_shares = -final_shares
        
        # Final validation - ensure we meet minimum value requirement
        final_value = abs(final_shares * current_price)
        if final_value < min_position_value:
            self.logger.info(f"Final position value too small for {symbol} (${final_value:.0f} < ${min_position_value}), skipping")
            return 0
        
        self.logger.info(f"Position sizing for {symbol}: ${final_value:,.0f} ({final_value/portfolio_value*100:.1f}% of portfolio) = {abs(final_shares):.2f} shares @ ${current_price:.2f}")
        
        return final_shares
    
    def _get_current_price(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Get current price for a symbol from portfolio positions or market data."""
        if symbol in self.positions:
            return self.positions[symbol].current_price
        
        # Get price from market data
        if symbol in market_data:
            if isinstance(market_data[symbol], dict) and "price" in market_data[symbol]:
                return market_data[symbol]["price"]
            elif isinstance(market_data[symbol], pd.DataFrame) and "Close" in market_data[symbol].columns:
                return market_data[symbol]["Close"].iloc[-1]
        
        # Fallback to reasonable estimate if no market data available
        self.logger.warning(f"No price data available for {symbol}, using estimate")
        return 200.0  # Reasonable estimate for most stocks
    
    def _get_size_adjustment_reason(self, original_size: float, final_size: float, 
                                  risk_approval: Dict[str, Any]) -> str:
        """Get human-readable reason for position size adjustment."""
        if abs(final_size - original_size) < 0.01:
            return "No adjustment needed"
        
        adjustment_pct = ((final_size - original_size) / original_size) * 100
        risk_checks = risk_approval.get("risk_checks", [])
        
        reasons = []
        if any("Position size exceeds limit" in check for check in risk_checks):
            reasons.append("Risk limit compliance")
        if abs(risk_approval.get("signal_strength", 0)) < 0.3:
            reasons.append("Weak signal adjustment")
        
        return f"Adjusted by {adjustment_pct:+.1f}%: {', '.join(reasons)}"
    
    def _create_trade_rationale(self, signals: Dict[str, Any]) -> str:
        """Create human-readable rationale for trade."""
        
        rationales = []
        
        # Technical analysis
        if "technical_analyst" in signals:
            tech_signal = signals["technical_analyst"].get("overall_signal", {})
            tech_strength = tech_signal.get("strength", 0)
            if abs(tech_strength) > 0.3:
                rationales.append(f"Technical: {tech_signal.get('classification', 'neutral')}")
        
        # Sentiment analysis
        if "sentiment_analyst" in signals:
            sentiment = signals["sentiment_analyst"].get("overall_sentiment", 0)
            if abs(sentiment) > 0.3:
                sentiment_class = "positive" if sentiment > 0 else "negative"
                rationales.append(f"Sentiment: {sentiment_class}")
        
        # Fundamentals
        if "fundamentals_analyst" in signals:
            fund_rec = signals["fundamentals_analyst"].get("recommendation", "neutral")
            if fund_rec not in ["neutral", "insufficient_data"]:
                rationales.append(f"Fundamentals: {fund_rec}")
        
        return "; ".join(rationales) if rationales else "Multi-factor signal"
    
    async def _execute_signals(self, signals: List[Dict[str, Any]], 
                             market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute trading signals."""
        
        executed_orders = []
        
        for signal in signals:
            try:
                symbol = signal["symbol"]
                size = signal["size"]
                order_type = OrderType(signal.get("order_type", "market"))
                
                # Get current price
                if symbol in market_data:
                    if isinstance(market_data[symbol], dict) and "price" in market_data[symbol]:
                        current_price = market_data[symbol]["price"]
                    elif isinstance(market_data[symbol], pd.DataFrame) and "Close" in market_data[symbol].columns:
                        current_price = market_data[symbol]["Close"].iloc[-1]
                    else:
                        continue
                else:
                    continue
                
                # Create and execute order
                order = await self._create_and_execute_order(symbol, size, order_type, current_price)
                
                if order and order.status == OrderStatus.FILLED:
                    executed_orders.append(order.to_dict())
                    
                    # Update or create position
                    self._update_position_from_order(order, signal.get("rationale", ""))
                
            except Exception as e:
                self.logger.error(f"Error executing signal for {signal.get('symbol', 'unknown')}: {e}")
        
        return executed_orders
    
    async def _create_and_execute_order(self, symbol: str, size: float, 
                                      order_type: OrderType, 
                                      current_price: float,
                                      is_backtest: bool = False) -> Optional[Order]:
        """Create and simulate the execution of an order."""
        if size == 0:
            return None

        # CRITICAL SAFETY CHECK: Validate cash balance for buy orders
        if size > 0:  # Buy order
            trade_value = abs(size) * current_price
            available_cash = self.cash_balance
            
            if available_cash < trade_value:
                self.logger.error(
                    f"🚨 INSUFFICIENT CASH: Cannot buy {abs(size):.2f} shares of {symbol} "
                    f"(${trade_value:,.2f} required, ${available_cash:,.2f} available). "
                    f"Trade REJECTED for financial safety."
                )
                return None
            
            # Additional safety margin check (reserve 1% of portfolio for fees/slippage)
            safety_margin = self.initial_capital * 0.01  # 1% safety margin
            if available_cash - trade_value < safety_margin:
                self.logger.warning(
                    f"⚠️  LOW CASH WARNING: After trade, only ${available_cash - trade_value:,.2f} "
                    f"remaining (below ${safety_margin:,.2f} safety margin)"
                )

        # CRITICAL SAFETY CHECK: Validate sufficient shares for sell orders
        if size < 0:  # Sell order
            shares_to_sell = abs(size)
            if symbol in self.positions:
                available_shares = self.positions[symbol].size
                if available_shares < shares_to_sell:
                    self.logger.error(
                        f"🚨 INSUFFICIENT SHARES: Cannot sell {shares_to_sell:.2f} shares of {symbol} "
                        f"(only {available_shares:.2f} shares owned). Trade REJECTED for financial safety."
                    )
                    return None
            else:
                self.logger.error(
                    f"🚨 NO POSITION: Cannot sell {shares_to_sell:.2f} shares of {symbol} "
                    f"(no position exists). Trade REJECTED for financial safety."
                )
                return None

        # Additional validation: Price sanity check
        if current_price <= 0:
            self.logger.error(f"🚨 INVALID PRICE: Cannot execute trade on {symbol} with price ${current_price:.2f}")
            return None

        order = Order(symbol, size, order_type, price=current_price)
        
        # Simulate execution
        executed = await self._simulate_order_execution(order, current_price, is_backtest)
        
        if executed:
            self.logger.info(f"Executed market order: {abs(order.size):.2f} shares of {order.symbol} at ${order.filled_price:,.2f}")
            order.status = OrderStatus.FILLED
            order.fill(current_price, abs(size))
            
            # Update cash balance (with final validation for extra safety)
            trade_value = abs(size) * current_price
            if size > 0:  # Buy order
                # Double-check cash balance hasn't changed since initial validation
                if self.cash_balance < trade_value:
                    self.logger.error(
                        f"🚨 CASH BALANCE CHANGED: Cannot complete buy order for {symbol}. "
                        f"Cash changed from validation to execution. Trade CANCELLED."
                    )
                    return None
                self.cash_balance -= trade_value
                self.logger.info(f"💰 Cash deducted: ${trade_value:,.2f}. Remaining: ${self.cash_balance:,.2f}")
            else:  # Sell order
                self.cash_balance += trade_value
                self.logger.info(f"💰 Cash added: ${trade_value:,.2f}. New balance: ${self.cash_balance:,.2f}")
            
            # Log trade
            trade_record = {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "size": size,
                "price": current_price,
                "value": trade_value,
                "order_id": order.id,
                "cash_balance": self.cash_balance
            }
            self.trade_history.append(trade_record)
            
            self.logger.info(f"Executed {order_type.value} order: {size} shares of {symbol} at ${current_price:.2f}")
            return order
            
        else:
            self.logger.warning(f"Order for {order.symbol} failed to execute.")
            return None

    async def _simulate_order_execution(self, order: Order, current_price: float, is_backtest: bool = False) -> bool:
        """
        Simulate the execution of an order.
        In a real system, this would interact with a broker's API.
        """
        # For backtesting, assume immediate fill at the given historical price
        if not is_backtest:
            await asyncio.sleep(self.config.get("execution_delay", 0.1)) # Simulate network latency
        
        # Simulate slippage
        slippage = self.config.get("slippage_tolerance", 0.001) * np.random.randn()
        fill_price = current_price * (1 + slippage)
        
        # Update order with execution price
        order.price = fill_price
        order.filled_price = fill_price
        
        return True
    
    def _update_position_from_order(self, order: Order, rationale: str = ""):
        """Update or create position from executed order."""
        
        symbol = order.symbol
        size = order.size
        price = order.filled_price or order.price
        
        if symbol in self.positions:
            # Update existing position
            position = self.positions[symbol]
            
            # Calculate new weighted average price
            total_size = position.size + size
            if total_size != 0:
                new_entry_price = ((position.size * position.entry_price) + (size * price)) / total_size
                position.entry_price = new_entry_price
                position.size = total_size
            else:
                # Position closed
                del self.positions[symbol]
                return
            
        else:
            # Create new position
            position = Position(symbol, size, price, datetime.now())
            self.positions[symbol] = position
        
        position.update_price(price)
        
        self.logger.info(f"Position updated: {symbol} - {position.size} shares at ${position.entry_price:.2f}")
    
    def get_cash_status(self) -> Dict[str, float]:
        """Get current cash status for monitoring and validation."""
        total_value = self._get_total_portfolio_value()
        cash_ratio = self.cash_balance / total_value if total_value > 0 else 1.0
        
        return {
            "cash_balance": self.cash_balance,
            "initial_capital": self.initial_capital,
            "total_portfolio_value": total_value,
            "cash_ratio": cash_ratio,
            "invested_amount": total_value - self.cash_balance,
            "is_healthy": self.cash_balance >= 0,  # Critical health check
            "safety_margin": self.initial_capital * 0.01
        }

    def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position and record the trade."""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate realized P&L
        realized_pnl = (exit_price - position.entry_price) * position.size
        
        # Record closing trade
        closing_trade = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "size": -position.size,  # Opposite of original position
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "realized_pnl": realized_pnl,
            "reason": reason,
            "holding_period": (datetime.now() - position.entry_date).days
        }
        
        self.trade_history.append(closing_trade)
        
        # NOTE: Cash balance is already updated by _create_and_execute_order()
        # Removing duplicate cash update to fix 2X cash increase bug
        
        # Remove position
        del self.positions[symbol]
        
        self.logger.info(f"Position closed: {symbol} - P&L: ${realized_pnl:.2f} ({reason})")
    
    def _update_position_management(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update position management tasks."""
        
        position_updates = []
        
        for symbol, position in self.positions.items():
            # Calculate days held
            days_held = (datetime.now() - position.entry_date).days
            
            # Check for stale positions (held too long without clear trend)
            if days_held > 30 and abs(position.unrealized_pnl_pct) < 0.02:
                position_updates.append({
                    "symbol": symbol,
                    "action": "review",
                    "reason": "Stale position - consider closing",
                    "days_held": days_held,
                    "unrealized_pnl_pct": position.unrealized_pnl_pct
                })
            
            # Check for high-performing positions (consider taking profits)
            if position.unrealized_pnl_pct > 0.20:  # 20% gain
                position_updates.append({
                    "symbol": symbol,
                    "action": "consider_profit_taking",
                    "reason": "High unrealized gains",
                    "unrealized_pnl_pct": position.unrealized_pnl_pct
                })
        
        return position_updates
    
    def _generate_portfolio_recommendations(self, market_data: Dict[str, Any], 
                                          analysis_signals: Dict[str, Any]) -> List[str]:
        """Generate portfolio-level recommendations."""
        
        recommendations = []
        
        # Cash allocation check
        portfolio_value = self._get_total_portfolio_value()
        cash_ratio = self.cash_balance / portfolio_value if portfolio_value > 0 else 1
        
        if cash_ratio > 0.3:
            recommendations.append("High cash allocation - consider deploying capital")
        elif cash_ratio < 0.05:
            recommendations.append("Low cash reserves - consider raising cash for opportunities")
        
        # Position concentration check
        if len(self.positions) > 0:
            position_values = [abs(pos.market_value) for pos in self.positions.values()]
            max_position_pct = max(position_values) / portfolio_value if portfolio_value > 0 else 0
            
            if max_position_pct > 0.2:  # 20% in single position
                recommendations.append("High position concentration - consider diversifying")
        
        # Performance check
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        if total_unrealized_pnl < -portfolio_value * 0.05:  # 5% portfolio loss
            recommendations.append("Portfolio showing losses - review risk management")
        
        return recommendations
    
    async def _equal_weight_rebalance(self, market_data: Dict[str, Any], 
                                    analysis_signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform equal weight rebalancing."""
        
        rebalance_orders = []
        
        if not self.positions:
            return rebalance_orders
        
        portfolio_value = self._get_total_portfolio_value()
        target_position_value = portfolio_value / len(self.positions)
        
        for symbol, position in self.positions.items():
            current_value = abs(position.market_value)
            target_size = target_position_value / position.current_price
            size_difference = target_size - abs(position.size)
            
            # Only rebalance if difference is significant (>5%)
            if abs(size_difference) / abs(position.size) > 0.05:
                try:
                    order = await self._create_and_execute_order(
                        symbol, size_difference, OrderType.MARKET, position.current_price
                    )
                    if order:
                        rebalance_orders.append(order.to_dict())
                except Exception as e:
                    self.logger.error(f"Error rebalancing {symbol}: {e}")
        
        return rebalance_orders
    
    async def _risk_parity_rebalance(self, market_data: Dict[str, Any], 
                                   analysis_signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform risk parity rebalancing (simplified)."""
        
        # Simplified risk parity - equal risk contribution
        # In practice, this would require volatility calculations
        
        return await self._equal_weight_rebalance(market_data, analysis_signals)
    
    async def _market_cap_rebalance(self, market_data: Dict[str, Any], 
                                  analysis_signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform market cap weighted rebalancing."""
        
        rebalance_orders = []
        
        # Get market caps for all positions
        market_caps = {}
        total_market_cap = 0
        
        for symbol in self.positions.keys():
            if symbol in market_data and isinstance(market_data[symbol], dict):
                market_cap = market_data[symbol].get("market_cap", 0)
                if market_cap > 0:
                    market_caps[symbol] = market_cap
                    total_market_cap += market_cap
        
        if total_market_cap == 0:
            return rebalance_orders
        
        # Calculate target weights and rebalance
        portfolio_value = self._get_total_portfolio_value()
        
        for symbol, position in self.positions.items():
            if symbol in market_caps:
                target_weight = market_caps[symbol] / total_market_cap
                target_value = portfolio_value * target_weight
                target_size = target_value / position.current_price
                size_difference = target_size - abs(position.size)
                
                if abs(size_difference) / abs(position.size) > 0.05:
                    try:
                        order = await self._create_and_execute_order(
                            symbol, size_difference, OrderType.MARKET, position.current_price
                        )
                        if order:
                            rebalance_orders.append(order.to_dict())
                    except Exception as e:
                        self.logger.error(f"Error rebalancing {symbol}: {e}")
        
        return rebalance_orders
    
    async def _execute_single_trade(self, trade: Dict[str, Any], 
                                  market_data: Dict[str, Any],
                                  is_backtest: bool = False) -> Optional[Order]:
        """Execute a single trade signal."""
        symbol = trade.get("symbol")
        size = trade.get("size", 0)
        order_type = OrderType(trade.get("order_type", "market"))
        
        if not symbol or size == 0:
            return None
        
        # Get current price if not provided
        if not trade.get("price") and symbol in market_data:
            if isinstance(market_data[symbol], dict) and "price" in market_data[symbol]:
                current_price = market_data[symbol]["price"]
            elif isinstance(market_data[symbol], pd.DataFrame) and "Close" in market_data[symbol].columns:
                current_price = market_data[symbol]["Close"].iloc[-1]
            else:
                current_price = self._get_current_price(symbol, market_data)
        else:
            current_price = trade.get("price")
        
        if not current_price:
            return None
            
        return await self._create_and_execute_order(
            symbol, 
            trade['size'], 
            OrderType.MARKET, 
            current_price,
            is_backtest
        )

    def _get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of the current portfolio state."""
        
        total_market_value = sum(abs(pos.market_value) for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_value = self.cash_balance + total_market_value
        
        position_details = []
        for symbol, position in self.positions.items():
            position_details.append({
                "symbol": symbol,
                "size": position.size,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "market_value": position.market_value,
                "unrealized_pnl": position.unrealized_pnl,
                "unrealized_pnl_pct": position.unrealized_pnl_pct,
                "weight": abs(position.market_value) / total_value if total_value > 0 else 0
            })
        
        return {
            "timestamp": datetime.now(),
            "total_value": total_value,
            "cash_balance": self.cash_balance,
            "total_market_value": total_market_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_unrealized_pnl_pct": total_unrealized_pnl / self.initial_capital if self.initial_capital > 0 else 0,
            "position_count": len(self.positions),
            "cash_allocation": self.cash_balance / total_value if total_value > 0 else 1,
            "positions": position_details
        }

    def _persist_state(self):
        """Persist current cash, positions, and orders to disk."""
        try:
            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
            positions_serialized = {
                sym: {
                    "symbol": pos.symbol,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "entry_date": pos.entry_date,
                    "last_update": pos.last_update,
                }
                for sym, pos in self.positions.items()
            }
            state = {
                "cash_balance": self.cash_balance,
                "initial_capital": self.initial_capital,
                "positions": positions_serialized,
                "trade_history": self.trade_history[-200:],  # keep recent
            }
            with open(self.persistence_file, "w") as f:
                f.write(safe_json_dump(state, indent=2))
            self.logger.info(f"Portfolio state persisted to {self.persistence_file}")
            
            # Log portfolio health status for monitoring
            cash_status = self.get_cash_status()
            if not cash_status["is_healthy"]:
                self.logger.error(f"🚨 PORTFOLIO HEALTH ALERT: Negative cash balance detected: ${cash_status['cash_balance']:,.2f}")
            else:
                self.logger.debug(f"💰 Portfolio health: ${cash_status['cash_balance']:,.2f} cash, {cash_status['cash_ratio']:.1%} ratio")
                
        except Exception as e:
            self.logger.warning(f"Failed to persist portfolio state: {e}")

    def _load_persisted_state(self):
        """Load portfolio state from disk if available."""
        try:
            if not os.path.exists(self.persistence_file):
                return
            with open(self.persistence_file, "r") as f:
                data = json.load(f)
            self.cash_balance = float(data.get("cash_balance", self.initial_capital))
            # Load positions
            positions = data.get("positions", {})
            if isinstance(positions, dict):
                for sym, pdict in positions.items():
                    try:
                        size = float(pdict.get("size", 0))
                        entry_price = float(pdict.get("entry_price", 0))
                        entry_date = pdict.get("entry_date", datetime.now())
                        # Parse ISO datetime strings if present
                        if isinstance(entry_date, str):
                            try:
                                entry_date = datetime.fromisoformat(entry_date)
                            except Exception:
                                entry_date = datetime.now()
                        pos = Position(sym, size, entry_price, entry_date)
                        current_price = pdict.get("current_price", entry_price)
                        try:
                            current_price = float(current_price)
                        except Exception:
                            current_price = entry_price
                        pos.update_price(current_price)
                        self.positions[sym] = pos
                    except Exception:
                        continue
            # Load recent trade history
            th = data.get("trade_history", [])
            if isinstance(th, list):
                self.trade_history = th
            self.logger.info(f"Loaded persisted portfolio state from {self.persistence_file}")
        except Exception as e:
            self.logger.warning(f"Failed to load persisted portfolio state: {e}")
    
    def _get_total_portfolio_value(self) -> float:
        """Get total portfolio value including cash and positions."""
        total_market_value = sum(abs(pos.market_value) for pos in self.positions.values())
        return self.cash_balance + total_market_value
    
    def _update_performance_metrics(self) -> Dict[str, float]:
        """Update and calculate performance metrics."""
        
        current_value = self._get_total_portfolio_value()
        total_return = (current_value - self.initial_capital) / self.initial_capital
        
        # Calculate metrics from trade history
        realized_trades = [t for t in self.trade_history if "realized_pnl" in t]
        
        if realized_trades:
            total_realized_pnl = sum(t["realized_pnl"] for t in realized_trades)
            winning_trades = [t for t in realized_trades if t["realized_pnl"] > 0]
            losing_trades = [t for t in realized_trades if t["realized_pnl"] < 0]
            
            win_rate = len(winning_trades) / len(realized_trades)
            avg_win = np.mean([t["realized_pnl"] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t["realized_pnl"] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(sum(t["realized_pnl"] for t in winning_trades)) / abs(sum(t["realized_pnl"] for t in losing_trades)) if losing_trades else 0
        else:
            total_realized_pnl = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        self.performance_metrics = {
            "total_return": total_return,
            "total_realized_pnl": total_realized_pnl,
            "win_rate": win_rate,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_trades": len(realized_trades),
            "current_positions": len(self.positions)
        }
        
        return self.performance_metrics
    
    def _update_portfolio_history(self, portfolio_summary: Dict[str, Any], 
                                performance_update: Dict[str, float]):
        """Update portfolio history for tracking."""
        
        history_entry = {
            "timestamp": datetime.now(),
            "total_value": portfolio_summary["total_value"],
            "cash_balance": self.cash_balance,
            "total_unrealized_pnl": portfolio_summary["total_unrealized_pnl"],
            "position_count": len(self.positions),
            "total_return": performance_update.get("total_return", 0)
        }
        
        self.portfolio_history.append(history_entry)
        
        # Keep only last 1000 entries
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        return {
            "summary": self._get_portfolio_summary(),
            "performance": self.performance_metrics,
            "active_orders": [order.to_dict() for order in self.orders.values() 
                            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]],
            "recent_trades": self.trade_history[-10:] if self.trade_history else []
        }