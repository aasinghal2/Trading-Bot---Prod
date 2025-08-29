"""
Risk Manager Agent

This agent is the gatekeeper of the trading system, responsible for:
- Value at Risk (VaR) calculations
- Conditional Value at Risk (CVaR) 
- Drawdown monitoring and limits
- Position sizing and portfolio risk management
- Real-time risk assessment and trade approval/rejection
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger

from .base_agent import BaseAgent


class RiskManagerAgent(BaseAgent):
    """
    Risk Manager Agent for comprehensive portfolio risk management.
    
    Features:
    - Multiple VaR calculation methods
    - Real-time risk monitoring
    - Position sizing recommendations
    - Portfolio stress testing
    - Risk-adjusted trade filtering
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "RiskManagerAgent")
        
        # Risk limits and parameters
        self.risk_metrics = config.get("risk_metrics", {})
        self.var_confidence = self.risk_metrics.get("var_confidence", 0.95)
        self.cvar_confidence = self.risk_metrics.get("cvar_confidence", 0.95)
        self.max_drawdown = self.risk_metrics.get("max_drawdown", 0.15)
        self.max_position_size = self.risk_metrics.get("max_position_size", 0.1)
        self.max_portfolio_leverage = self.risk_metrics.get("max_portfolio_leverage", 2.0)
        
        # Risk calculation methods
        self.var_method = config.get("var_method", "historical")  # historical, parametric, monte_carlo
        self.var_window = config.get("var_window", 252)
        
        # Portfolio tracking
        self.portfolio_value = 0.0
        self.positions = {}
        self.risk_history = []
        
        # Risk alerts
        self.risk_alerts = []
        
        self.logger.info("Risk Manager Agent initialized")
    
    async def _execute_logic(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Execute risk management analysis.
        
        Args:
            input_data: Contains portfolio data, market data, and proposed trades
            
        Returns:
            Tuple of (risk_assessment, metrics)
        """
        start_time = time.time()
        
        # Extract input data
        portfolio_data = input_data.get("portfolio_data", {})
        market_data = input_data.get("market_data", {})
        proposed_trades = input_data.get("proposed_trades", [])
        analysis_signals = input_data.get("analysis_signals", {})  # From other agents
        
        # Update portfolio state
        self._update_portfolio_state(portfolio_data)
        
        # Perform comprehensive risk analysis
        risk_assessment = {
            "timestamp": datetime.now(),
            "portfolio_risk": {},
            "position_risks": {},
            "trade_approvals": {},
            "risk_limits_status": {},
            "recommendations": []
        }
        
        # 1. Calculate portfolio-level risk metrics
        portfolio_risk = await self._calculate_portfolio_risk(market_data)
        risk_assessment["portfolio_risk"] = portfolio_risk
        
        # 2. Analyze individual position risks
        try:
            position_risks = await self._analyze_position_risks(market_data)
            # Ensure position_risks is always a dictionary
            if not isinstance(position_risks, dict):
                self.logger.warning(f"_analyze_position_risks returned non-dict: {type(position_risks)}")
                position_risks = {}
        except Exception as e:
            self.logger.error(f"Error analyzing position risks: {e}")
            position_risks = {}
        
        risk_assessment["position_risks"] = position_risks
        
        # 3. Evaluate proposed trades
        try:
            trade_approvals = await self._evaluate_proposed_trades(proposed_trades, market_data, analysis_signals)
            # Ensure trade_approvals is always a dictionary
            if not isinstance(trade_approvals, dict):
                self.logger.warning(f"_evaluate_proposed_trades returned non-dict: {type(trade_approvals)}")
                trade_approvals = {}
        except Exception as e:
            self.logger.error(f"Error evaluating proposed trades: {e}")
            trade_approvals = {}
        
        risk_assessment["trade_approvals"] = trade_approvals
        
        # 4. Check risk limits compliance
        risk_limits_status = self._check_risk_limits(portfolio_risk, position_risks)
        risk_assessment["risk_limits_status"] = risk_limits_status
        
        # 5. Generate risk management recommendations
        recommendations = self._generate_recommendations(portfolio_risk, position_risks, risk_limits_status)
        risk_assessment["recommendations"] = recommendations
        
        # Calculate metrics
        execution_time = time.time() - start_time
        
        # Safe handling of trade_approvals - ensure it's a dictionary
        if isinstance(trade_approvals, dict):
            approvals_list = list(trade_approvals.values())
        else:
            # Fallback: treat as list or convert to safe format
            self.logger.warning(f"trade_approvals is not a dict, got {type(trade_approvals)}: {trade_approvals}")
            approvals_list = trade_approvals if isinstance(trade_approvals, list) else []
        
        metrics = {
            "execution_time_seconds": execution_time,
            "portfolio_var": portfolio_risk.get("var_95", 0),
            "portfolio_cvar": portfolio_risk.get("cvar_95", 0),
            "current_drawdown": portfolio_risk.get("current_drawdown", 0),
            "risk_score": self._calculate_overall_risk_score(portfolio_risk, position_risks),
            "trades_approved": sum(1 for approval in approvals_list if isinstance(approval, dict) and approval.get("approved", False)),
            "trades_rejected": sum(1 for approval in approvals_list if isinstance(approval, dict) and not approval.get("approved", False)),
            "active_risk_alerts": len(self.risk_alerts)
        }
        
        # Update risk history
        self._update_risk_history(portfolio_risk, metrics)
        
        return risk_assessment, metrics
    
    async def _calculate_portfolio_risk(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio risk metrics."""
        
        if not self.positions or not market_data:
            return self._create_empty_portfolio_risk()
        
        # Get historical data for portfolio positions
        portfolio_returns = self._calculate_portfolio_returns(market_data)
        
        if portfolio_returns.empty:
            return self._create_empty_portfolio_risk()
        
        # Calculate Value at Risk (VaR)
        var_95 = self._calculate_var(portfolio_returns, self.var_confidence)
        var_99 = self._calculate_var(portfolio_returns, 0.99)
        
        # Calculate Conditional Value at Risk (CVaR)
        cvar_95 = self._calculate_cvar(portfolio_returns, self.cvar_confidence)
        cvar_99 = self._calculate_cvar(portfolio_returns, 0.99)
        
        # Calculate drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(portfolio_returns)
        
        # Calculate volatility metrics
        volatility_metrics = self._calculate_volatility_metrics(portfolio_returns)
        
        # Calculate correlation metrics
        correlation_metrics = self._calculate_correlation_metrics(market_data)
        
        # Calculate concentration risk
        concentration_risk = self._calculate_concentration_risk()
        
        # Calculate leverage metrics
        leverage_metrics = self._calculate_leverage_metrics()
        
        portfolio_risk = {
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "portfolio_volatility": volatility_metrics["portfolio_volatility"],
            "sharpe_ratio": volatility_metrics["sharpe_ratio"],
            "max_drawdown": drawdown_metrics["max_drawdown"],
            "current_drawdown": drawdown_metrics["current_drawdown"],
            "drawdown_duration": drawdown_metrics["drawdown_duration"],
            "concentration_risk": concentration_risk,
            "leverage_ratio": leverage_metrics["leverage_ratio"],
            "margin_utilization": leverage_metrics["margin_utilization"],
            "portfolio_correlation": correlation_metrics["average_correlation"],
            "diversification_ratio": correlation_metrics["diversification_ratio"]
        }
        
        return portfolio_risk
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk using specified method."""
        
        if returns.empty:
            return 0.0
        
        if self.var_method == "historical":
            return self._historical_var(returns, confidence)
        elif self.var_method == "parametric":
            return self._parametric_var(returns, confidence)
        elif self.var_method == "monte_carlo":
            return self._monte_carlo_var(returns, confidence)
        else:
            return self._historical_var(returns, confidence)
    
    def _historical_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate historical VaR."""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _parametric_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        mean_return = returns.mean()
        std_return = returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        return mean_return + z_score * std_return
    
    def _monte_carlo_var(self, returns: pd.Series, confidence: float, n_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR."""
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random returns
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        return np.percentile(simulated_returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        
        if returns.empty:
            return 0.0
        
        var_threshold = self._calculate_var(returns, confidence)
        tail_returns = returns[returns <= var_threshold]
        
        return tail_returns.mean() if not tail_returns.empty else var_threshold
    
    def _calculate_portfolio_returns(self, market_data: Dict[str, Any]) -> pd.Series:
        """Calculate portfolio returns based on positions and market data."""
        
        if not self.positions:
            return pd.Series()
        
        # This is a simplified calculation - in practice, you'd need historical portfolio values
        # For now, we'll calculate based on position-weighted returns
        
        portfolio_returns = None
        total_weight = sum(abs(pos["size"]) for pos in self.positions.values())
        
        if total_weight == 0:
            return pd.Series()
        
        for symbol, position in self.positions.items():
            if symbol in market_data and isinstance(market_data[symbol], pd.DataFrame):
                hist_data = market_data[symbol]
                if "Close" in hist_data.columns and len(hist_data) > 1:
                    returns = hist_data["Close"].pct_change().dropna()
                    weight = abs(position["size"]) / total_weight
                    
                    if portfolio_returns is None:
                        portfolio_returns = returns * weight
                    else:
                        # Align indices and add
                        common_index = portfolio_returns.index.intersection(returns.index)
                        if not common_index.empty:
                            portfolio_returns = portfolio_returns.loc[common_index] + returns.loc[common_index] * weight
        
        return portfolio_returns if portfolio_returns is not None else pd.Series()
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate drawdown-related metrics."""
        
        if returns.empty:
            return {"max_drawdown": 0.0, "current_drawdown": 0.0, "drawdown_duration": 0}
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1]
        
        # Drawdown duration (simplified)
        drawdown_duration = 0
        if current_drawdown < 0:
            # Count periods since last peak
            for i in range(len(drawdown) - 1, -1, -1):
                if drawdown.iloc[i] < 0:
                    drawdown_duration += 1
                else:
                    break
        
        return {
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "drawdown_duration": drawdown_duration
        }
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate volatility-related metrics."""
        
        if returns.empty:
            return {"portfolio_volatility": 0.0, "sharpe_ratio": 0.0}
        
        # Annualized volatility
        portfolio_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        mean_return = returns.mean() * 252  # Annualized
        sharpe_ratio = mean_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            "portfolio_volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio
        }
    
    def _calculate_correlation_metrics(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate correlation and diversification metrics."""
        
        if len(self.positions) < 2:
            return {"average_correlation": 0.0, "diversification_ratio": 1.0}
        
        # Calculate returns for each position
        returns_matrix = {}
        
        for symbol in self.positions.keys():
            if symbol in market_data and isinstance(market_data[symbol], pd.DataFrame):
                hist_data = market_data[symbol]
                if "Close" in hist_data.columns:
                    returns = hist_data["Close"].pct_change().dropna()
                    returns_matrix[symbol] = returns
        
        if len(returns_matrix) < 2:
            return {"average_correlation": 0.0, "diversification_ratio": 1.0}
        
        # Create DataFrame and calculate correlation matrix
        returns_df = pd.DataFrame(returns_matrix).dropna()
        
        if returns_df.empty:
            return {"average_correlation": 0.0, "diversification_ratio": 1.0}
        
        correlation_matrix = returns_df.corr()
        
        # Calculate average correlation (excluding diagonal)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        average_correlation = correlation_matrix.values[mask].mean()
        
        # Calculate diversification ratio
        weights = np.array([abs(self.positions[symbol]["size"]) for symbol in returns_df.columns])
        weights = weights / weights.sum()
        
        individual_vols = returns_df.std().values
        portfolio_vol = np.sqrt(weights.T @ correlation_matrix.values @ weights)
        weighted_avg_vol = np.sum(weights * individual_vols)
        
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        return {
            "average_correlation": average_correlation,
            "diversification_ratio": diversification_ratio
        }
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration risk using Herfindahl index."""
        
        if not self.positions:
            return 0.0
        
        total_value = sum(abs(pos["market_value"]) for pos in self.positions.values())
        
        if total_value == 0:
            return 0.0
        
        # Calculate Herfindahl index
        weights_squared = [(abs(pos["market_value"]) / total_value) ** 2 for pos in self.positions.values()]
        herfindahl_index = sum(weights_squared)
        
        return herfindahl_index
    
    def _calculate_leverage_metrics(self) -> Dict[str, float]:
        """Calculate leverage-related metrics."""
        
        if not self.positions:
            return {"leverage_ratio": 0.0, "margin_utilization": 0.0}
        
        total_long_value = sum(pos["market_value"] for pos in self.positions.values() if pos["market_value"] > 0)
        total_short_value = sum(abs(pos["market_value"]) for pos in self.positions.values() if pos["market_value"] < 0)
        total_exposure = total_long_value + total_short_value
        
        leverage_ratio = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Simplified margin utilization (assuming 50% margin requirement)
        margin_requirement = total_exposure * 0.5
        margin_utilization = margin_requirement / self.portfolio_value if self.portfolio_value > 0 else 0
        
        return {
            "leverage_ratio": leverage_ratio,
            "margin_utilization": margin_utilization
        }
    
    async def _analyze_position_risks(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk for individual positions."""
        
        position_risks = {}
        
        for symbol, position in self.positions.items():
            try:
                position_risk = await self._calculate_position_risk(symbol, position, market_data)
                position_risks[symbol] = position_risk
            except Exception as e:
                self.logger.error(f"Error calculating risk for position {symbol}: {e}")
                position_risks[symbol] = self._create_empty_position_risk(symbol)
        
        return position_risks
    
    async def _calculate_position_risk(self, symbol: str, position: Dict[str, Any], 
                                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics for a single position."""
        
        if symbol not in market_data:
            return self._create_empty_position_risk(symbol)
        
        hist_data = market_data[symbol]
        if not isinstance(hist_data, pd.DataFrame) or "Close" not in hist_data.columns:
            return self._create_empty_position_risk(symbol)
        
        returns = hist_data["Close"].pct_change().dropna()
        if returns.empty:
            return self._create_empty_position_risk(symbol)
        
        current_price = hist_data["Close"].iloc[-1]
        
        # Position details
        position_size = position.get("size", 0)
        entry_price = position.get("entry_price", current_price)
        market_value = position_size * current_price
        
        # Position VaR
        position_var_95 = self._calculate_var(returns, 0.95) * abs(market_value)
        position_cvar_95 = self._calculate_cvar(returns, 0.95) * abs(market_value)
        
        # Position volatility
        position_volatility = returns.std() * np.sqrt(252)
        
        # Unrealized P&L
        unrealized_pnl = (current_price - entry_price) * position_size
        unrealized_pnl_pct = unrealized_pnl / (entry_price * abs(position_size)) if entry_price > 0 else 0
        
        # Time in position
        entry_date = position.get("entry_date", datetime.now())
        # Handle string datetime from JSON serialization
        if isinstance(entry_date, str):
            try:
                entry_date = datetime.fromisoformat(entry_date)
            except Exception:
                entry_date = datetime.now()
        elif entry_date is None:
            entry_date = datetime.now()
        
        days_held = (datetime.now() - entry_date).days
        
        # Position as percentage of portfolio
        position_weight = abs(market_value) / self.portfolio_value if self.portfolio_value > 0 else 0
        
        return {
            "symbol": symbol,
            "position_size": position_size,
            "market_value": market_value,
            "position_weight": position_weight,
            "var_95": position_var_95,
            "cvar_95": position_cvar_95,
            "volatility": position_volatility,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "days_held": days_held,
            "risk_score": self._calculate_position_risk_score(position_weight, position_volatility, unrealized_pnl_pct)
        }
    
    def _calculate_position_risk_score(self, weight: float, volatility: float, pnl_pct: float) -> float:
        """Calculate a risk score for a position (0-100)."""
        
        # Weight component (higher weight = higher risk)
        weight_score = min(weight * 100, 50)  # Cap at 50
        
        # Volatility component
        volatility_score = min(volatility * 100, 30)  # Cap at 30
        
        # Unrealized loss component
        loss_score = max(-pnl_pct * 50, 0) if pnl_pct < 0 else 0  # Cap at 20
        
        total_score = weight_score + volatility_score + loss_score
        return min(total_score, 100)
    
    async def _evaluate_proposed_trades(self, proposed_trades: List[Dict[str, Any]], 
                                      market_data: Dict[str, Any], 
                                      analysis_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate proposed trades and approve/reject based on risk criteria."""
        
        trade_approvals = {}
        
        for i, trade in enumerate(proposed_trades):
            trade_id = trade.get("id", f"trade_{i}")
            
            try:
                approval = await self._evaluate_single_trade(trade, market_data, analysis_signals)
                trade_approvals[trade_id] = approval
            except Exception as e:
                self.logger.error(f"Error evaluating trade {trade_id}: {e}")
                trade_approvals[trade_id] = {
                    "approved": False,
                    "reason": f"Evaluation error: {str(e)}",
                    "risk_score": 100
                }
        
        return trade_approvals
    
    async def _evaluate_single_trade(self, trade: Dict[str, Any], 
                                   market_data: Dict[str, Any], 
                                   analysis_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single trade for risk compliance."""
        
        symbol = trade.get("symbol")
        size = trade.get("size", 0)
        trade_type = trade.get("type", "market")  # market, limit, stop
        
        if not symbol or size == 0:
            return {"approved": False, "reason": "Invalid trade parameters", "risk_score": 100}
        
        # Calculate trade value
        if symbol in market_data:
            if isinstance(market_data[symbol], dict) and "price" in market_data[symbol]:
                current_price = market_data[symbol]["price"]
            elif isinstance(market_data[symbol], pd.DataFrame) and "Close" in market_data[symbol].columns:
                current_price = market_data[symbol]["Close"].iloc[-1]
            else:
                return {"approved": False, "reason": "No price data available", "risk_score": 100}
        else:
            return {"approved": False, "reason": "No market data for symbol", "risk_score": 100}
        
        trade_value = abs(size * current_price)
        
        # Risk checks
        risk_checks = []
        risk_score = 0
        
        # UNIFIED POSITION WEIGHT CHECK - FIXED LOGIC
        # Calculate what the TOTAL position would be after this trade
        if symbol in self.positions:
            existing_value = abs(self.positions[symbol]["market_value"])
            self.logger.debug(f"Existing {symbol} position: ${existing_value:,.2f}")
        else:
            existing_value = 0
            self.logger.debug(f"No existing {symbol} position")
        
        total_position_value = existing_value + trade_value
        total_position_weight = total_position_value / self.portfolio_value if self.portfolio_value > 0 else 1
        
        # Single, clear check against the actual 10% limit
        if total_position_weight > self.max_position_size:
            risk_checks.append(
                f"Total position would exceed limit: ${total_position_value:,.0f} "
                f"({total_position_weight:.2%} > {self.max_position_size:.2%}). "
                f"Existing: ${existing_value:,.0f}, New trade: ${trade_value:,.0f}"
            )
            risk_score += 35  # Higher penalty for position limit violations
            
            self.logger.warning(
                f"Position limit violation for {symbol}: "
                f"Total {total_position_weight:.2%} > {self.max_position_size:.2%} limit"
            )
        else:
            self.logger.debug(
                f"Position check passed for {symbol}: "
                f"Total {total_position_weight:.2%} ≤ {self.max_position_size:.2%} limit"
            )
        
        # 3. Signal quality check
        signal_strength = 0
        if symbol in analysis_signals:
            # Check technical analysis signals
            if "technical_analyst" in analysis_signals[symbol]:
                tech_signal = analysis_signals[symbol]["technical_analyst"].get("overall_signal", {})
                signal_strength = tech_signal.get("strength", 0)
            
            # Weak signal check
            if abs(signal_strength) < 0.3:
                risk_checks.append("Weak trading signal")
                risk_score += 15
            
            # Signal direction vs trade direction
            signal_direction = 1 if signal_strength > 0 else -1
            trade_direction = 1 if size > 0 else -1
            if signal_direction != trade_direction:
                risk_checks.append("Trade direction opposes analysis signal")
                risk_score += 20
        
        # 4. Volatility check
        if symbol in market_data and isinstance(market_data[symbol], pd.DataFrame):
            returns = market_data[symbol]["Close"].pct_change().dropna()
            if not returns.empty:
                volatility = returns.std() * np.sqrt(252)
                if volatility > 0.5:  # 50% annualized volatility
                    risk_checks.append(f"High volatility asset ({volatility:.1%})")
                    risk_score += 10
        
        # 5. Portfolio leverage check
        current_leverage = self._calculate_leverage_metrics()["leverage_ratio"]
        trade_impact_on_leverage = trade_value / self.portfolio_value if self.portfolio_value > 0 else 1
        projected_leverage = current_leverage + trade_impact_on_leverage
        
        if projected_leverage > self.max_portfolio_leverage:
            risk_checks.append(f"Would exceed leverage limit ({projected_leverage:.1f}x > {self.max_portfolio_leverage:.1f}x)")
            risk_score += 35
        
        # CRITICAL VIOLATIONS - THESE CAN NEVER BE BYPASSED
        critical_violations = []
        for check in risk_checks:
            if any(critical_word in check.lower() for critical_word in 
                   ['leverage limit', 'position would exceed limit', 'position size exceeds limit', 'concentration too high']):
                critical_violations.append(check)
        
        # FIXED DECISION LOGIC - NO BACKDOORS
        # Clean, simple, and safe: Risk score threshold is FINAL
        has_critical_violations = len(critical_violations) > 0
        approved = len(risk_checks) == 0 or (risk_score < 50 and not has_critical_violations)
        
        # Position size is original size - no dangerous size reduction bypasses
        adjusted_size = size
        
        # Enhanced logging for transparency and debugging
        if approved:
            self.logger.info(f"✅ Risk Manager APPROVED {symbol}: Score {risk_score}/100, No critical violations")
        else:
            rejection_reason = []
            if has_critical_violations:
                rejection_reason.append(f"CRITICAL VIOLATIONS: {'; '.join(critical_violations)}")
            if risk_score >= 50:
                rejection_reason.append(f"Risk score {risk_score} >= 50 threshold")
            
            self.logger.warning(
                f"❌ Risk Manager REJECTED {symbol}: {'; '.join(rejection_reason)}. "
                f"All violations: {'; '.join(risk_checks) if risk_checks else 'None listed'}"
            )
        
        # Provide clear, actionable rejection reasons
        if approved:
            reason = f"Approved: Risk score {risk_score}/100, no critical violations"
        else:
            reason_parts = []
            if has_critical_violations:
                reason_parts.append(f"Critical violations: {'; '.join(critical_violations)}")
            if risk_score >= 50:
                reason_parts.append(f"Risk score {risk_score}/100 exceeds threshold (50)")
            reason = "; ".join(reason_parts) if reason_parts else "High risk assessment"
        
        return {
            "approved": approved,
            "adjusted_size": adjusted_size,
            "original_size": size,
            "target_value": trade.get("target_value", trade_value),  # Pass through target value
            "risk_score": risk_score,
            "risk_checks": risk_checks,
            "critical_violations": critical_violations,  # New field for monitoring
            "signal_strength": signal_strength,
            "trade_value": trade_value,
            "position_weight": total_position_weight,  # Now shows TOTAL position weight
            "existing_position_value": existing_value,  # New field for transparency
            "reason": reason
        }
    
    def _check_risk_limits(self, portfolio_risk: Dict[str, Any], 
                          position_risks: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with risk limits."""
        
        limits_status = {
            "overall_status": "compliant",
            "violations": [],
            "warnings": []
        }
        
        # Portfolio-level checks
        max_drawdown = portfolio_risk.get("max_drawdown", 0)
        if max_drawdown < -self.max_drawdown:
            limits_status["violations"].append(f"Max drawdown exceeded: {max_drawdown:.2%}")
            limits_status["overall_status"] = "violation"
        
        current_drawdown = portfolio_risk.get("current_drawdown", 0)
        if current_drawdown < -self.max_drawdown * 0.8:  # Warning at 80% of limit
            limits_status["warnings"].append(f"Approaching drawdown limit: {current_drawdown:.2%}")
        
        leverage_ratio = portfolio_risk.get("leverage_ratio", 0)
        if leverage_ratio > self.max_portfolio_leverage:
            limits_status["violations"].append(f"Leverage exceeds limit: {leverage_ratio:.1f}x")
            limits_status["overall_status"] = "violation"
        
        # Position-level checks
        # Ensure position_risks is a dictionary
        if isinstance(position_risks, dict):
            for symbol, pos_risk in position_risks.items():
                if isinstance(pos_risk, dict):
                    position_weight = pos_risk.get("position_weight", 0)
                    if position_weight > self.max_position_size:
                        limits_status["violations"].append(f"{symbol} position size exceeds limit: {position_weight:.2%}")
                        limits_status["overall_status"] = "violation"
        else:
            self.logger.warning(f"position_risks is not a dict, got {type(position_risks)}: {position_risks}")
            # Add a general warning about risk calculation
            limits_status["warnings"].append("Position risks could not be properly calculated")
        
        # Concentration risk
        concentration = portfolio_risk.get("concentration_risk", 0)
        if concentration > 0.4:  # Herfindahl index above 0.4 indicates high concentration
            limits_status["warnings"].append(f"High portfolio concentration: {concentration:.2f}")
        
        return limits_status
    
    def _generate_recommendations(self, portfolio_risk: Dict[str, Any], 
                                position_risks: Dict[str, Any], 
                                limits_status: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations."""
        
        recommendations = []
        
        # Based on violations
        if limits_status["violations"]:
            recommendations.append("URGENT: Address risk limit violations immediately")
            
            # Specific recommendations for violations
            for violation in limits_status["violations"]:
                if "drawdown" in violation.lower():
                    recommendations.append("Consider reducing position sizes or hedging portfolio")
                elif "leverage" in violation.lower():
                    recommendations.append("Reduce leverage by closing positions or adding capital")
                elif "position size" in violation.lower():
                    recommendations.append("Reduce oversized positions to comply with limits")
        
        # Based on portfolio risk metrics
        portfolio_vol = portfolio_risk.get("portfolio_volatility", 0)
        if portfolio_vol > 0.25:  # 25% annualized volatility
            recommendations.append("Consider reducing portfolio volatility through diversification")
        
        sharpe_ratio = portfolio_risk.get("sharpe_ratio", 0)
        if sharpe_ratio < 0.5:
            recommendations.append("Portfolio risk-adjusted returns are low - review strategy")
        
        diversification_ratio = portfolio_risk.get("diversification_ratio", 1)
        if diversification_ratio < 1.2:
            recommendations.append("Portfolio lacks diversification - consider adding uncorrelated assets")
        
        # Position-specific recommendations
        high_risk_positions = []
        if isinstance(position_risks, dict):
            high_risk_positions = [
                symbol for symbol, risk in position_risks.items() 
                if isinstance(risk, dict) and risk.get("risk_score", 0) > 70
            ]
        
        if high_risk_positions:
            recommendations.append(f"Review high-risk positions: {', '.join(high_risk_positions)}")
        
        # Unrealized loss positions
        losing_positions = []
        if isinstance(position_risks, dict):
            losing_positions = [
                symbol for symbol, risk in position_risks.items() 
                if isinstance(risk, dict) and risk.get("unrealized_pnl_pct", 0) < -0.1  # 10% loss
            ]
        
        if losing_positions:
            recommendations.append(f"Consider stop-losses for losing positions: {', '.join(losing_positions)}")
        
        return recommendations
    
    def _update_portfolio_state(self, portfolio_data: Dict[str, Any]):
        """Update internal portfolio state."""
        
        self.portfolio_value = portfolio_data.get("total_value", self.portfolio_value)
        
        # Update positions
        positions = portfolio_data.get("positions", {})
        if isinstance(positions, dict):
            for symbol, position in positions.items():
                if isinstance(position, dict):
                    self.positions[symbol] = {
                        "size": position.get("size", 0),
                        "entry_price": position.get("entry_price", 0),
                        "market_value": position.get("market_value", 0),
                        "entry_date": position.get("entry_date", datetime.now())
                    }
        else:
            self.logger.warning(f"portfolio_data.positions is not a dict, got {type(positions)}")
        
        # Remove positions that are no longer held
        symbols_to_remove = [
            symbol for symbol in self.positions.keys() 
            if symbol not in positions
        ]
        for symbol in symbols_to_remove:
            del self.positions[symbol]
    
    def _update_risk_history(self, portfolio_risk: Dict[str, Any], metrics: Dict[str, float]):
        """Update risk history for trend analysis."""
        
        risk_entry = {
            "timestamp": datetime.now(),
            "portfolio_var": portfolio_risk.get("var_95", 0),
            "portfolio_cvar": portfolio_risk.get("cvar_95", 0),
            "current_drawdown": portfolio_risk.get("current_drawdown", 0),
            "risk_score": metrics.get("risk_score", 0),
            "portfolio_volatility": portfolio_risk.get("portfolio_volatility", 0)
        }
        
        self.risk_history.append(risk_entry)
        
        # Keep only last 1000 entries
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
    
    def _calculate_overall_risk_score(self, portfolio_risk: Dict[str, Any], 
                                    position_risks: Dict[str, Any]) -> float:
        """Calculate overall portfolio risk score (0-100)."""
        
        score = 0
        
        # Portfolio metrics contribution
        current_drawdown = abs(portfolio_risk.get("current_drawdown", 0))
        score += min(current_drawdown * 200, 30)  # Max 30 points
        
        leverage_ratio = portfolio_risk.get("leverage_ratio", 0)
        score += min(leverage_ratio * 15, 25)  # Max 25 points
        
        concentration_risk = portfolio_risk.get("concentration_risk", 0)
        score += min(concentration_risk * 50, 20)  # Max 20 points
        
        portfolio_vol = portfolio_risk.get("portfolio_volatility", 0)
        score += min(portfolio_vol * 50, 15)  # Max 15 points
        
        # Position risks contribution
        if position_risks:
            avg_position_risk = np.mean([risk.get("risk_score", 0) for risk in position_risks.values()])
            score += min(avg_position_risk * 0.1, 10)  # Max 10 points
        
        return min(score, 100)
    
    def _create_empty_portfolio_risk(self) -> Dict[str, Any]:
        """Create empty portfolio risk result."""
        return {
            "var_95": 0.0, "var_99": 0.0, "cvar_95": 0.0, "cvar_99": 0.0,
            "portfolio_volatility": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "current_drawdown": 0.0, "drawdown_duration": 0,
            "concentration_risk": 0.0, "leverage_ratio": 0.0, "margin_utilization": 0.0,
            "portfolio_correlation": 0.0, "diversification_ratio": 1.0
        }
    
    def _create_empty_position_risk(self, symbol: str) -> Dict[str, Any]:
        """Create empty position risk result."""
        return {
            "symbol": symbol, "position_size": 0, "market_value": 0, "position_weight": 0,
            "var_95": 0, "cvar_95": 0, "volatility": 0, "unrealized_pnl": 0,
            "unrealized_pnl_pct": 0, "days_held": 0, "risk_score": 0
        }
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary."""
        return {
            "portfolio_value": self.portfolio_value,
            "position_count": len(self.positions),
            "active_alerts": len(self.risk_alerts),
            "risk_history_length": len(self.risk_history)
        }