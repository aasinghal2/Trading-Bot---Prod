"""
Multi-Agent Trading System Orchestrator

This module coordinates the execution of all trading agents in a synchronized workflow.
It handles parallel execution of analysis agents and sequential execution based on dependencies.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml
from loguru import logger
import pandas as pd
import numpy as np

from agents import (
    MarketDataAgent,
    TechnicalAnalystAgent, 
    FundamentalsAgent,
    SentimentAgent,
    RiskManagerAgent,
    PortfolioManagerAgent
)
from core.dynamic_thresholds import DynamicThresholdManager


class TradingOrchestrator:
    """
    Orchestrates the multi-agent trading workflow.
    
    Workflow:
    1. Market Data Agent (first, provides data to all)
    2. Parallel execution: Technical, Fundamentals, Sentiment Agents
    3. Risk Manager Agent (waits for analysis agents)
    4. Portfolio Manager Agent (executes approved trades)
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the orchestrator with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.agents = {}
        self.execution_history = []
        
        # Initialize all agents
        self._initialize_agents()
        
        # Initialize dynamic threshold manager
        self.threshold_manager = DynamicThresholdManager(self.config)
        
        # Workflow state
        self.current_execution_id = None
        self.execution_results = {}
        
        self.logger = logger
        
        self.logger.info("Trading Orchestrator initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            "market_data": {
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "real_time_interval": 60
            },
            "agents": {
                "technical_analyst": {"enabled": True},
                "fundamentals_analyst": {"enabled": True},
                "sentiment_analyst": {"enabled": True},
                "risk_manager": {"enabled": True},
                "portfolio_manager": {"enabled": True}
            },
            "risk_management": {
                "max_daily_loss": 0.02,
                "max_single_position": 0.1
            },
            "trading": {
                "execution_delay": 0.1
            }
        }
    
    def _initialize_agents(self):
        """Initialize all trading agents."""
        
        # Market Data Agent
        market_config = self.config.get("market_data", {})
        self.agents["market_data"] = MarketDataAgent(market_config)
        
        # Analysis Agents (run in parallel)
        agents_config = self.config.get("agents", {})
        
        if agents_config.get("technical_analyst", {}).get("enabled", True):
            tech_config = agents_config.get("technical_analyst", {})
            self.agents["technical_analyst"] = TechnicalAnalystAgent(tech_config)
        
        if agents_config.get("fundamentals_analyst", {}).get("enabled", True):
            fund_config = agents_config.get("fundamentals_analyst", {})
            self.agents["fundamentals_analyst"] = FundamentalsAgent(fund_config)
        
        if agents_config.get("sentiment_analyst", {}).get("enabled", True):
            sentiment_config = agents_config.get("sentiment_analyst", {})
            self.agents["sentiment_analyst"] = SentimentAgent(sentiment_config)
        
        # Risk Manager Agent
        if agents_config.get("risk_manager", {}).get("enabled", True):
            risk_config = {**agents_config.get("risk_manager", {}), 
                          **self.config.get("risk_management", {})}
            self.agents["risk_manager"] = RiskManagerAgent(risk_config)
        
        # Portfolio Manager Agent
        if agents_config.get("portfolio_manager", {}).get("enabled", True):
            portfolio_config = {**agents_config.get("portfolio_manager", {}),
                              **self.config.get("trading", {})}
            self.agents["portfolio_manager"] = PortfolioManagerAgent(portfolio_config)
        
        logger.info(f"Initialized {len(self.agents)} agents: {list(self.agents.keys())}")
    
    async def execute_trading_cycle(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute a complete trading cycle.
        
        Args:
            symbols: List of symbols to analyze (uses config default if None)
            
        Returns:
            Dictionary containing results from all agents
        """
        execution_id = f"exec_{int(time.time())}"
        self.current_execution_id = execution_id
        start_time = time.time()
        
        logger.info(f"Starting trading cycle {execution_id}")
        
        # Use configured symbols if none provided
        if symbols is None:
            symbols = self.config.get("market_data", {}).get("symbols", ["AAPL", "GOOGL", "MSFT"])
        
        cycle_results = {
            "execution_id": execution_id,
            "timestamp": datetime.now(),
            "symbols": symbols,
            "agents_executed": [],
            "execution_time": 0,
            "success": False,
            "results": {},
            "errors": []
        }
        
        try:
            # Step 1: Market Data Collection
            logger.info("Step 1: Fetching market data")
            market_data_result = await self._execute_market_data_agent(symbols)
            cycle_results["results"]["market_data"] = market_data_result
            cycle_results["agents_executed"].append("market_data")
            
            if not market_data_result.success:
                raise Exception("Market data collection failed")
            
            # Step 2: Parallel Analysis Execution
            logger.info("Step 2: Running parallel analysis (Technical, Fundamentals, Sentiment)")
            analysis_results = await self._execute_analysis_agents_parallel(
                symbols, market_data_result.data
            )
            cycle_results["results"]["analysis"] = analysis_results
            cycle_results["agents_executed"].extend(analysis_results.keys())
            
            # Step 3: Risk Management
            logger.info("Step 3: Risk assessment")
            risk_result = await self._execute_risk_manager(
                symbols, market_data_result.data, analysis_results
            )
            # Keep pre-trade risk under a dedicated key, and also under the
            # default key for backward compatibility (will be overwritten post-trade)
            cycle_results["results"]["risk_management_pre"] = risk_result
            cycle_results["results"]["risk_management"] = risk_result
            cycle_results["agents_executed"].append("risk_manager")
            
            # Step 4: Portfolio Management
            logger.info("Step 4: Portfolio management and trade execution")
            portfolio_result = await self._execute_portfolio_manager(
                symbols, market_data_result.data, analysis_results, risk_result
            )
            cycle_results["results"]["portfolio_management"] = portfolio_result
            cycle_results["agents_executed"].append("portfolio_manager")

            # Step 5: Post-trade risk recomputation (reflect positions just executed)
            try:
                logger.info("Step 5: Recomputing risk post-trade")
                # Get all symbols in portfolio for complete risk assessment
                all_portfolio_symbols = set(symbols)  # Start with current symbols
                if "portfolio_manager" in self.agents:
                    portfolio_status = self.agents["portfolio_manager"].get_portfolio_status()
                    summary = portfolio_status.get("summary", {}) if isinstance(portfolio_status, dict) else {}
                    positions_list = summary.get("positions", []) if isinstance(summary.get("positions", []), list) else []
                    for pos in positions_list:
                        if pos.get("symbol"):
                            all_portfolio_symbols.add(pos.get("symbol"))
                
                post_trade_risk = await self._execute_post_trade_risk(
                    list(all_portfolio_symbols), market_data_result.data
                )
                # Store both for clarity; keep default key as post-trade
                cycle_results["results"]["risk_management_post"] = post_trade_risk
                cycle_results["results"]["risk_management"] = post_trade_risk
            except Exception as e:
                logger.warning(f"Post-trade risk recompute failed: {e}")
            
            # Calculate total execution time
            execution_time = time.time() - start_time
            cycle_results["execution_time"] = execution_time
            cycle_results["success"] = True
            
            logger.info(f"Trading cycle {execution_id} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            cycle_results["execution_time"] = execution_time
            cycle_results["errors"].append(str(e))
            logger.error(f"Trading cycle {execution_id} failed: {e}")
        
        # Store execution history
        self.execution_history.append(cycle_results)
        
        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        return cycle_results
    
    async def execute_analysis_only(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute ANALYSIS ONLY (no trade execution) for market scanning
        
        This method runs market data collection and analysis agents but skips
        trade execution, risk management, and portfolio management.
        Perfect for market scanning and signal building.
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Dictionary containing analysis results only
        """
        execution_id = f"scan_{int(time.time())}"
        start_time = time.time()
        
        logger.info(f"Starting SCAN-ONLY analysis {execution_id}")
        
        # Use configured symbols if none provided
        if symbols is None:
            symbols = self.config.get("market_data", {}).get("symbols", ["AAPL", "GOOGL", "MSFT"])
        
        scan_results = {
            "execution_id": execution_id,
            "timestamp": datetime.now(),
            "symbols": symbols,
            "agents_executed": [],
            "execution_time": 0,
            "success": False,
            "results": {},
            "errors": []
        }
        
        try:
            # Step 1: Market Data Collection
            logger.info("SCAN Step 1: Fetching market data")
            market_data_result = await self._execute_market_data_agent(symbols)
            scan_results["results"]["market_data"] = market_data_result
            scan_results["agents_executed"].append("market_data")
            
            if not market_data_result.success:
                raise Exception("Market data collection failed")
            
            # Step 2: Analysis Agents Only (NO TRADING)
            logger.info("SCAN Step 2: Running analysis agents only")
            analysis_results = await self._execute_analysis_agents_parallel(
                symbols, market_data_result.data
            )
            scan_results["results"]["analysis"] = analysis_results
            scan_results["results"]["technical"] = analysis_results.get("technical_analyst")
            scan_results["results"]["fundamentals"] = analysis_results.get("fundamentals_analyst") 
            scan_results["results"]["sentiment"] = analysis_results.get("sentiment_analyst")
            
            scan_results["agents_executed"].extend(["technical", "fundamentals", "sentiment"])
            
            # Step 3: Combine signals for ranking (NO TRADE GENERATION)
            logger.info("SCAN Step 3: Processing signals for ranking")
            combined_signals = self._combine_analysis_signals(
                symbols, analysis_results
            )
            scan_results["results"]["combined_signals"] = combined_signals
            
            # Track signals for dynamic thresholds (but don't generate trades)
            try:
                if hasattr(self, 'threshold_manager'):
                    self.threshold_manager.history_tracker.add_signals(execution_id, combined_signals)
                    logger.info(f"Successfully tracked {len(combined_signals)} signals for dynamic thresholds")
                    
                    # Calculate dynamic threshold to populate market metrics cache
                    threshold, threshold_explanation = self.threshold_manager.calculate_threshold()
                    scan_results["dynamic_threshold"] = threshold
                    scan_results["threshold_explanation"] = threshold_explanation
                    logger.info(f"Dynamic threshold calculated: {threshold:.3f} - {threshold_explanation}")
                    
            except Exception as e:
                logger.warning(f"Failed to track signals for dynamic thresholds: {e}")
            
            execution_time = time.time() - start_time
            scan_results["execution_time"] = execution_time
            scan_results["success"] = True
            
            logger.info(f"SCAN-ONLY analysis {execution_id} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            scan_results["execution_time"] = execution_time
            scan_results["errors"].append(str(e))
            logger.error(f"SCAN-ONLY analysis {execution_id} failed: {e}")
        
        return scan_results
    
    async def _execute_market_data_agent(self, symbols: List[str]):
        """Execute market data collection."""
        
        # Include existing portfolio symbols to ensure we get price updates for all positions
        all_symbols = set(symbols)
        if "portfolio_manager" in self.agents:
            try:
                portfolio_status = self.agents["portfolio_manager"].get_portfolio_status()
                summary = portfolio_status.get("summary", {}) if isinstance(portfolio_status, dict) else {}
                positions_list = summary.get("positions", []) if isinstance(summary.get("positions", []), list) else []
                for pos in positions_list:
                    if pos.get("symbol"):
                        all_symbols.add(pos.get("symbol"))
            except Exception as e:
                logger.warning(f"Could not get portfolio symbols for market data: {e}")
        
        agent = self.agents["market_data"]
        input_data = {
            "symbols": list(all_symbols),
            "data_type": "both"  # Get both real-time and historical data
        }
        
        result = await agent.execute(input_data)
        return result
    
    async def _execute_analysis_agents_parallel(self, symbols: List[str], 
                                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis agents in parallel."""
        
        # Include existing portfolio symbols for re-evaluation
        all_symbols = set(symbols)
        if "portfolio_manager" in self.agents:
            try:
                portfolio_status = self.agents["portfolio_manager"].get_portfolio_status()
                summary = portfolio_status.get("summary", {}) if isinstance(portfolio_status, dict) else {}
                positions_list = summary.get("positions", []) if isinstance(summary.get("positions", []), list) else []
                for pos in positions_list:
                    if pos.get("symbol"):
                        all_symbols.add(pos.get("symbol"))
            except Exception as e:
                logger.warning(f"Could not get portfolio symbols for analysis: {e}")
        
        symbols_to_analyze = list(all_symbols)
        logger.info(f"Analyzing {len(symbols_to_analyze)} symbols: {symbols_to_analyze}")
        
        # Prepare tasks for parallel execution
        tasks = []
        agent_names = []
        
        # Technical Analysis Agent - independent stock evaluation
        if "technical_analyst" in self.agents:
            agent = self.agents["technical_analyst"]
            input_data = {
                "symbols": symbols_to_analyze,
                "market_data": market_data.get("historical", {})
            }
            tasks.append(agent.execute(input_data))
            agent_names.append("technical_analyst")
        
        # Fundamentals Agent - independent stock evaluation  
        if "fundamentals_analyst" in self.agents:
            agent = self.agents["fundamentals_analyst"]
            input_data = {
                "symbols": symbols_to_analyze,
                "analysis_depth": "standard"
            }
            tasks.append(agent.execute(input_data))
            agent_names.append("fundamentals_analyst")
        
        # Sentiment Agent - independent stock evaluation
        if "sentiment_analyst" in self.agents:
            agent = self.agents["sentiment_analyst"]
            input_data = {
                "symbols": symbols_to_analyze,
                "analysis_type": "comprehensive"
            }
            tasks.append(agent.execute(input_data))
            agent_names.append("sentiment_analyst")
        
        # Execute all analysis agents in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            analysis_results = {}
            for i, result in enumerate(results):
                agent_name = agent_names[i]
                if isinstance(result, Exception):
                    logger.error(f"Error in {agent_name}: {result}")
                    analysis_results[agent_name] = None
                else:
                    analysis_results[agent_name] = result
        else:
            analysis_results = {}
        
        return analysis_results
    
    async def _execute_risk_manager(self, symbols: List[str], 
                                  market_data: Dict[str, Any],
                                  analysis_results: Dict[str, Any]):
        """Execute risk management assessment."""
        
        if "risk_manager" not in self.agents:
            return None
        
        agent = self.agents["risk_manager"]
        
        # Combine analysis signals
        combined_signals = self._combine_analysis_signals(symbols, analysis_results)
        
        # Track signals for dynamic threshold calculation
        try:
            self.threshold_manager.add_execution_signals(self.current_execution_id, combined_signals)
        except Exception as e:
            logger.warning(f"Failed to track signals for dynamic thresholds: {e}")
        
        # Get current portfolio data from portfolio manager
        portfolio_data = {}
        if "portfolio_manager" in self.agents:
            portfolio_agent = self.agents["portfolio_manager"]
            portfolio_status = portfolio_agent.get_portfolio_status()
            summary = portfolio_status.get("summary", {}) if isinstance(portfolio_status, dict) else {}
            # Transform positions list -> dict for risk manager consumption
            positions_dict: Dict[str, Any] = {}
            positions_list = summary.get("positions", []) if isinstance(summary.get("positions", []), list) else []
            for pos in positions_list:
                try:
                    sym = pos.get("symbol")
                    if not sym:
                        continue
                    positions_dict[sym] = {
                        "size": pos.get("size", 0),
                        "entry_price": pos.get("entry_price", 0),
                        "market_value": pos.get("market_value", 0),
                        "entry_date": pos.get("entry_date"),
                    }
                except Exception:
                    continue
            portfolio_data = {
                "total_value": summary.get("total_value", 0.0),
                "positions": positions_dict,
            }

        # Ensure historical market data includes all current position symbols for portfolio-level risk
        try:
            historical_map = market_data.get("historical", {}) if isinstance(market_data, dict) else {}
            pos_symbols = list(portfolio_data.get("positions", {}).keys()) if isinstance(portfolio_data.get("positions", {}), dict) else []
            missing_syms = [s for s in pos_symbols if s not in historical_map]
            if missing_syms and "market_data" in self.agents:
                md_agent = self.agents["market_data"]
                extra_hist = await md_agent.fetch_historical_data(missing_syms)
                if isinstance(extra_hist, dict):
                    # Merge into a copy to avoid mutating original
                    merged_hist = dict(historical_map)
                    merged_hist.update({k: v for k, v in extra_hist.items() if v is not None})
                    # Replace in market_data copy for downstream usage
                    market_data = dict(market_data)
                    market_data["historical"] = merged_hist
        except Exception as e:
            logger.warning(f"Could not augment historical data for positions: {e}")
        
        # Generate proposed trades based on analysis
        proposed_trades = self._generate_proposed_trades(combined_signals)
        
        input_data = {
            "portfolio_data": portfolio_data,
            "market_data": market_data.get("historical", {}),
            "proposed_trades": proposed_trades,
            "analysis_signals": combined_signals
        }
        
        result = await agent.execute(input_data)
        return result
    
    async def _execute_portfolio_manager(self, symbols: List[str],
                                       market_data: Dict[str, Any],
                                       analysis_results: Dict[str, Any],
                                       risk_result, is_backtest: bool = False):
        """Execute portfolio management and trade execution."""
        
        if "portfolio_manager" not in self.agents:
            return None
        
        agent = self.agents["portfolio_manager"]
        
        # Combine analysis signals
        combined_signals = self._combine_analysis_signals(symbols, analysis_results)
        
        # Extract risk approvals
        risk_approvals = {}
        if risk_result and risk_result.success:
            trade_approvals = risk_result.data.get("trade_approvals", {})
            for trade_id, approval in trade_approvals.items():
                # Extract symbol from trade_id (format: trade_SYMBOL_timestamp)
                if trade_id.startswith("trade_"):
                    symbol = trade_id.split("_")[1]  # Get the symbol part
                else:
                    symbol = approval.get("symbol", trade_id)
                risk_approvals[symbol] = approval
        
        # Prefer real-time data for execution, but gracefully fallback to historical
        # last-close data when real-time is unavailable (e.g., off-market hours).
        real_time_data = market_data.get("real_time", {}) or {}
        historical_data = market_data.get("historical", {}) or {}

        # Merge real-time with historical: if a symbol lacks usable real-time (no price),
        # provide historical DataFrame so the portfolio manager can use last close.
        portfolio_market_data: Dict[str, Any] = dict(real_time_data)
        for sym in symbols:
            rt_entry = portfolio_market_data.get(sym)
            has_realtime_price = isinstance(rt_entry, dict) and ("price" in rt_entry or "close" in rt_entry)
            if not has_realtime_price:
                if sym in historical_data:
                    portfolio_market_data[sym] = historical_data[sym]

        input_data = {
            "market_data": portfolio_market_data,
            "analysis_signals": combined_signals,
            "risk_approvals": risk_approvals,
            "action": "manage"
        }
        
        result = await agent.execute(input_data)
        return result

    async def _execute_post_trade_risk(self, symbols: List[str], market_data: Dict[str, Any]):
        """Recompute risk after portfolio manager has executed trades.

        Uses updated portfolio from portfolio manager and focuses on portfolio-level risk metrics.
        """
        if "risk_manager" not in self.agents or "portfolio_manager" not in self.agents:
            return None

        risk_agent = self.agents["risk_manager"]
        portfolio_agent = self.agents["portfolio_manager"]

        # Get latest portfolio summary
        portfolio_status = portfolio_agent.get_portfolio_status()
        summary = portfolio_status.get("summary", {}) if isinstance(portfolio_status, dict) else {}

        # Transform positions list -> dict keyed by symbol so risk manager can ingest
        positions_dict: Dict[str, Any] = {}
        positions_list = summary.get("positions", []) if isinstance(summary.get("positions", []), list) else []
        held_symbols = set()
        for pos in positions_list:
            try:
                symbol = pos.get("symbol")
                if not symbol:
                    continue
                held_symbols.add(symbol)
                positions_dict[symbol] = {
                    "size": pos.get("size", 0),
                    "entry_price": pos.get("entry_price", 0),
                    "market_value": pos.get("market_value", 0),
                    # entry_date optional; risk manager will default when missing
                }
            except Exception:
                continue

        portfolio_data = {
            "total_value": summary.get("total_value", 0.0),
            "positions": positions_dict,
        }

        # Get all symbols that need historical data (current trade symbols + all in portfolio)
        all_symbols_needed = set(symbols) | set(portfolio_data.get("positions", {}).keys())

        # Check for missing historical data and fetch if needed
        try:
            historical_map = market_data.get("historical", {}) if isinstance(market_data, dict) else {}
            missing_syms = [s for s in all_symbols_needed if s not in historical_map]
            if missing_syms and "market_data" in self.agents:
                md_agent = self.agents["market_data"]
                extra_hist = await md_agent.fetch_historical_data(missing_syms)
                if isinstance(extra_hist, dict):
                    merged_hist = dict(historical_map)
                    merged_hist.update({k: v for k, v in extra_hist.items() if v is not None})
                    market_data["historical"] = merged_hist
        except Exception as e:
            logger.warning(f"Could not augment historical data for post-trade risk: {e}")

        input_data = {
            "portfolio_data": portfolio_data,
            "market_data": market_data.get("historical", {}),
            "proposed_trades": [],  # No new trades to propose in post-trade check
            "analysis_signals": {}
        }

        result = await risk_agent.execute(input_data)
        return result
    
    def _combine_analysis_signals(self, symbols: List[str], 
                                analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine signals from all analysis agents for each symbol."""
        
        combined_signals = {}
        
        for symbol in symbols:
            combined_signals[symbol] = {}
            
            # Technical analysis signals
            if ("technical_analyst" in analysis_results and 
                analysis_results["technical_analyst"] and 
                analysis_results["technical_analyst"].success):
                
                tech_data = analysis_results["technical_analyst"].data.get(symbol, {})
                combined_signals[symbol]["technical_analyst"] = tech_data
            
            # Fundamentals signals
            if ("fundamentals_analyst" in analysis_results and 
                analysis_results["fundamentals_analyst"] and 
                analysis_results["fundamentals_analyst"].success):
                
                fund_data = analysis_results["fundamentals_analyst"].data.get(symbol, {})
                combined_signals[symbol]["fundamentals_analyst"] = fund_data
            
            # Sentiment signals
            if ("sentiment_analyst" in analysis_results and 
                analysis_results["sentiment_analyst"] and 
                analysis_results["sentiment_analyst"].success):
                
                sentiment_data = analysis_results["sentiment_analyst"].data.get(symbol, {})
                combined_signals[symbol]["sentiment_analyst"] = sentiment_data
            
            # Calculate overall signal if we have at least one analysis
            if combined_signals[symbol]:
                combined_signals[symbol]["overall_signal"] = self._calculate_overall_signal(
                    combined_signals[symbol]
                )
        
        return combined_signals
    
    def _calculate_overall_signal(self, symbol_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall trading signal from all analysis types."""
        
        signals = []
        weights = []
        
        # Technical analysis signal
        if "technical_analyst" in symbol_signals:
            tech_signal = symbol_signals["technical_analyst"].get("overall_signal", {})
            tech_strength = tech_signal.get("strength", 0)
            if tech_strength != 0:
                signals.append(tech_strength)
                weights.append(0.4)  # 40% weight for technical
        
        # Fundamentals signal
        if "fundamentals_analyst" in symbol_signals:
            fund_data = symbol_signals["fundamentals_analyst"]
            fund_score = fund_data.get("overall_score", 5.0)
            # Convert 0-10 score to -1 to 1 signal
            fund_signal = (fund_score - 5.0) / 5.0
            if fund_signal != 0:
                signals.append(fund_signal)
                weights.append(0.35)  # 35% weight for fundamentals
        
        # Sentiment signal
        if "sentiment_analyst" in symbol_signals:
            sentiment_data = symbol_signals["sentiment_analyst"]
            sentiment_score = sentiment_data.get("overall_sentiment", 0)
            if sentiment_score != 0:
                signals.append(sentiment_score)
                weights.append(0.25)  # 25% weight for sentiment
        
        # Calculate weighted average
        if signals and weights:
            total_weight = sum(weights)
            weighted_signal = sum(s * w for s, w in zip(signals, weights)) / total_weight
            
            # Classify signal
            if weighted_signal > 0.3:
                classification = "strong_buy"
            elif weighted_signal > 0.1:
                classification = "buy"
            elif weighted_signal < -0.3:
                classification = "strong_sell"
            elif weighted_signal < -0.1:
                classification = "sell"
            else:
                classification = "neutral"
            
            return {
                "strength": weighted_signal,
                "classification": classification,
                "direction": "bullish" if weighted_signal > 0 else "bearish" if weighted_signal < 0 else "neutral",
                "confidence": abs(weighted_signal),
                "contributing_analyses": len(signals)
            }
        
        return {
            "strength": 0.0,
            "classification": "neutral",
            "direction": "neutral",
            "confidence": 0.0,
            "contributing_analyses": 0
        }
    
    def _generate_proposed_trades(self, combined_signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate proposed trades based on combined analysis signals."""
        
        proposed_trades = []
        
        for symbol, signals in combined_signals.items():
            overall_signal = signals.get("overall_signal", {})
            signal_strength = overall_signal.get("strength", 0)
            
            # Get trading thresholds - use dynamic threshold if available
            signal_config = self.config.get("trading", {}).get("signal_thresholds", {})
            position_config = self.config.get("trading", {}).get("position_sizing", {})
            
            # Calculate dynamic threshold
            min_strength, threshold_explanation = self.threshold_manager.calculate_threshold()
            moderate_threshold = signal_config.get("moderate_strength", 0.4)
            strong_multiplier = signal_config.get("strong_multiplier", 1.5)
            base_position_value = position_config.get("base_position_value", 10000)
            max_position_value = position_config.get("max_position_value", 15000)
            max_multiplier = position_config.get("max_size_multiplier", 1.5)
            
            # Generate trades for signals above minimum threshold, with value-based position sizing
            if abs(signal_strength) > min_strength:
                # Calculate target position value based on signal strength
                if abs(signal_strength) < moderate_threshold:
                    # Scale down for moderate signals (0.25-0.4 range)
                    value_multiplier = abs(signal_strength) / moderate_threshold
                else:
                    # Scale up for strong signals, capped at max_multiplier
                    value_multiplier = min(abs(signal_strength) * strong_multiplier, max_multiplier)
                
                target_position_value = min(base_position_value * value_multiplier, max_position_value)
                
                # Convert to shares (will be refined by portfolio manager)
                # This is just for initial trade proposal - final sizing is value-based
                estimated_price = self._estimate_current_price(symbol, combined_signals)
                estimated_shares = target_position_value / estimated_price if estimated_price > 0 else 0
                
                # Determine trade direction (using estimated shares for now)
                if signal_strength > 0:
                    trade_size = estimated_shares
                else:
                    trade_size = -estimated_shares
                
                proposed_trade = {
                    "id": f"trade_{symbol}_{int(time.time())}",
                    "symbol": symbol,
                    "size": trade_size,
                    "target_value": target_position_value,  # New: target dollar amount
                    "type": "market",
                    "signal_strength": signal_strength,
                    "rationale": f"Multi-agent signal: {overall_signal.get('classification', 'neutral')} (strength: {signal_strength:.3f}, target: ${target_position_value:,.0f})"
                }
                
                proposed_trades.append(proposed_trade)
            else:
                logger.info(f"Signal too weak for {symbol}: {signal_strength:.3f} < {min_strength:.3f} minimum threshold")
        
        # Log threshold info only once per cycle
        if combined_signals:
            logger.info(f"Using threshold: {threshold_explanation}")
        
        logger.info(f"Generated {len(proposed_trades)} proposed trades from {len(combined_signals)} analyzed symbols")
        return proposed_trades
    
    def _estimate_current_price(self, symbol: str, combined_signals: Dict[str, Any]) -> float:
        """Estimate current price for a symbol from recent market data."""
        try:
            # Try to get price from market data in signals
            if symbol in combined_signals:
                market_data = combined_signals[symbol].get("market_data", {})
                if isinstance(market_data, dict):
                    if "price" in market_data:
                        price = float(market_data["price"])
                        self.logger.debug(f"Using real-time price for {symbol}: ${price:.2f}")
                        return price
                    elif "close" in market_data:
                        price = float(market_data["close"])
                        self.logger.debug(f"Using close price for {symbol}: ${price:.2f}")
                        return price
                elif hasattr(market_data, "iloc") and len(market_data) > 0:
                    # Pandas DataFrame with Close prices
                    if "Close" in market_data.columns:
                        price = float(market_data["Close"].iloc[-1])
                        self.logger.debug(f"Using historical close for {symbol}: ${price:.2f}")
                        return price
            
            # Try to get price from market data agent directly
            market_agent = self.agents.get("market_data")
            if market_agent:
                try:
                    # Request fresh market data for this symbol
                    fresh_data = market_agent.get_realtime_data([symbol])
                    if symbol in fresh_data and "price" in fresh_data[symbol]:
                        price = float(fresh_data[symbol]["price"])
                        self.logger.info(f"Retrieved fresh price for {symbol}: ${price:.2f}")
                        return price
                except Exception as e:
                    self.logger.warning(f"Failed to get fresh price for {symbol}: {e}")
            
            # IMPROVED FALLBACK: Use stock-specific estimates instead of generic $200
            # Based on common stock price ranges
            fallback_prices = {
                "AAPL": 180.0, "MSFT": 400.0, "GOOGL": 180.0, "AMZN": 180.0,
                "TSLA": 250.0, "META": 500.0, "NVDA": 800.0, "BRK.A": 500000.0,
                "JPM": 200.0, "JNJ": 160.0, "V": 270.0, "PG": 160.0
            }
            
            if symbol in fallback_prices:
                price = fallback_prices[symbol]
                self.logger.warning(f"Using fallback price for {symbol}: ${price:.2f}")
                return price
            
            # Generic fallback
            self.logger.warning(f"Using generic fallback price for {symbol}: $150.00")
            return 150.0
        except Exception as e:
            self.logger.error(f"Error estimating price for {symbol}: {e}")
            return 150.0

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        
        agent_status = {}
        
        for name, agent in self.agents.items():
            agent_status[name] = {
                "agent_type": agent.agent_type,
                "state": agent.get_state().dict(),
                "performance_metrics": agent.get_performance_metrics()
            }
        
        return agent_status
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history."""
        return self.execution_history[-limit:] if self.execution_history else []
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        successful_executions = [e for e in self.execution_history if e["success"]]
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful_executions),
            "success_rate": len(successful_executions) / len(self.execution_history),
            "average_execution_time": sum(e["execution_time"] for e in self.execution_history) / len(self.execution_history),
            "active_agents": len(self.agents),
            "last_execution": self.execution_history[-1]["timestamp"] if self.execution_history else None
        }