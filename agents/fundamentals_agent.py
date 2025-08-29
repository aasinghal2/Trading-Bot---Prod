"""
Fundamentals Analysis Agent

This agent performs fundamental analysis including:
- Financial statement analysis (P/E, EV/EBITDA, ROE, etc.)
- Earnings reports analysis
- Macroeconomic indicators
- Sector and industry comparisons
- Long-term value assessment
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

from .base_agent import BaseAgent
from core.data_sources import DataSourceManager, DataSourceError


class FundamentalsAgent(BaseAgent):
    """
    Fundamentals Analysis Agent for long-term value assessment.
    
    Features:
    - Financial ratios calculation
    - Earnings analysis
    - Growth metrics
    - Sector comparisons
    - Value scoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "FundamentalsAgent")
        
        # Configuration
        self.data_sources = config.get("data_sources", ["financial_statements", "earnings_reports"])
        self.metrics = config.get("metrics", ["pe_ratio", "ev_ebitda", "roe", "debt_to_equity"])
        classification_cfg = config.get("classification", {})
        self.classification_mode = classification_cfg.get("mode", "fixed_thresholds")  # fixed_thresholds | sector_percentiles | zscore_ranks (reserved)
        self.undervalued_pct = float(classification_cfg.get("undervalued_percentile", 0.2))
        self.overvalued_pct = float(classification_cfg.get("overvalued_percentile", 0.8))
        guards_cfg = classification_cfg.get("quality_guards", {})
        self.guard_min_roe = float(guards_cfg.get("min_roe", 0.10))
        self.guard_min_operating_margin = float(guards_cfg.get("min_operating_margin", 0.0))
        self.guard_min_earnings_growth = float(guards_cfg.get("min_earnings_growth", 0.0))
        self.min_sector_peers = int(classification_cfg.get("min_sector_peers", 3))
        
        # Fundamental data cache
        self.fundamentals_cache = {}
        self.sector_benchmarks = {}
        
        # Scoring weights
        self.metric_weights = {
            "valuation": 0.3,
            "profitability": 0.25,
            "growth": 0.25,
            "financial_health": 0.2
        }
        
        self.logger.info("Fundamentals Analysis Agent initialized")
    
    async def _execute_logic(self, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Execute fundamental analysis.
        
        Args:
            input_data: Contains symbols, analysis parameters, and optionally portfolio context
            
        Returns:
            Tuple of (fundamentals_results, metrics)
        """
        start_time = time.time()
        
        symbols = input_data.get("symbols", [])
        analysis_depth = input_data.get("analysis_depth", "standard")  # basic, standard, comprehensive
  # Optional portfolio context
        
        if not symbols:
            raise ValueError("No symbols provided for fundamental analysis")
        
        # Perform fundamental analysis for each symbol with portfolio context
        fundamentals_results = {}
        total_companies_analyzed = 0
        
        for symbol in symbols:
            try:
                symbol_fundamentals = await self._analyze_symbol_fundamentals(symbol, analysis_depth)
                fundamentals_results[symbol] = symbol_fundamentals
                
                if symbol_fundamentals.get("data_available", False):
                    total_companies_analyzed += 1
                    
            except Exception as e:
                self.logger.error(f"Error analyzing fundamentals for {symbol}: {e}")
                fundamentals_results[symbol] = self._create_empty_fundamentals_result(symbol)
        
        # DISABLED: Sector benchmarking causes batch contamination
        # Each symbol should be analyzed independently, not relative to current batch
        # sector_analysis = self._calculate_sector_benchmarks(fundamentals_results)

        # DISABLED: Sector-relative reclassification causes inconsistent results
        # This was causing GOOGL to get different scores based on batch composition
        # if self.classification_mode == "sector_percentiles":
        #     try:
        #         self._reclassify_valuation_sector_relative(fundamentals_results)
        #     except Exception as e:
        #         self.logger.warning(f"Sector-relative valuation reclassification failed: {e}")
        
        self.logger.debug("Sector benchmarking disabled to ensure analysis independence")

        # Calculate overall metrics (after potential reclassification)
        execution_time = time.time() - start_time
        metrics = {
            "execution_time_seconds": execution_time,
            "symbols_analyzed": len(fundamentals_results),
            "companies_with_data": total_companies_analyzed,
            "average_pe_ratio": self._calculate_average_metric(fundamentals_results, "pe_ratio"),
            "average_roe": self._calculate_average_metric(fundamentals_results, "roe"),
            "undervalued_count": self._count_by_valuation(fundamentals_results, "undervalued"),
            "overvalued_count": self._count_by_valuation(fundamentals_results, "overvalued"),
            "sectors_covered": 0  # Disabled sector analysis for independence
        }

        # Sector analysis disabled to ensure consistent results across different batches
        fundamentals_results["_sector_analysis"] = {}
        
        return fundamentals_results, metrics
    
    async def _analyze_symbol_fundamentals(self, symbol: str, analysis_depth: str) -> Dict[str, Any]:
        """Analyze fundamental metrics for a single symbol with portfolio awareness."""
        
        fundamentals_result = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "company_info": {},
            "financial_ratios": {},
            "growth_metrics": {},
            "profitability_metrics": {},
            "financial_health": {},
            "valuation_assessment": {},
            "sector_comparison": {},
            "overall_score": 0.0,
            "recommendation": "neutral",
            "data_available": False
        }
        
        try:
            # Get company information and financial data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or len(info) < 5:  # Basic validation
                self.logger.warning(f"Insufficient data for {symbol}")
                return fundamentals_result
            
            fundamentals_result["data_available"] = True
            
            # Extract company information
            company_info = self._extract_company_info(info)
            fundamentals_result["company_info"] = company_info
            
            # Calculate financial ratios
            financial_ratios = self._calculate_financial_ratios(info)
            fundamentals_result["financial_ratios"] = financial_ratios
            
            # Calculate growth metrics
            if analysis_depth in ["standard", "comprehensive"]:
                growth_metrics = await self._calculate_growth_metrics(ticker, info)
                fundamentals_result["growth_metrics"] = growth_metrics
            
            # Calculate profitability metrics
            profitability_metrics = self._calculate_profitability_metrics(info)
            fundamentals_result["profitability_metrics"] = profitability_metrics
            
            # Assess financial health
            financial_health = self._assess_financial_health(info)
            fundamentals_result["financial_health"] = financial_health
            
            # Valuation assessment
            valuation_assessment = self._assess_valuation(financial_ratios, company_info)
            fundamentals_result["valuation_assessment"] = valuation_assessment
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                financial_ratios, growth_metrics if analysis_depth != "basic" else {},
                profitability_metrics, financial_health, valuation_assessment
            )
            fundamentals_result["overall_score"] = overall_score
            
            # Generate recommendation with portfolio awareness
            recommendation = self._generate_recommendation(overall_score, valuation_assessment, symbol)
            fundamentals_result["recommendation"] = recommendation
            
        except Exception as e:
            self.logger.error(f"Error in fundamental analysis for {symbol}: {e}")
        
        return fundamentals_result
    
    def _extract_company_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic company information."""
        
        return {
            "company_name": info.get("longName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
            "employees": info.get("fullTimeEmployees", 0),
            "country": info.get("country", ""),
            "exchange": info.get("exchange", ""),
            "currency": info.get("currency", "USD")
        }
    
    def _calculate_financial_ratios(self, info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate key financial ratios."""
        
        ratios = {}
        
        # Valuation ratios
        ratios["pe_ratio"] = info.get("trailingPE", 0) or 0
        ratios["forward_pe"] = info.get("forwardPE", 0) or 0
        ratios["peg_ratio"] = info.get("pegRatio", 0) or 0
        ratios["price_to_book"] = info.get("priceToBook", 0) or 0
        ratios["ev_to_ebitda"] = info.get("enterpriseToEbitda", 0) or 0
        ratios["price_to_sales"] = info.get("priceToSalesTrailing12Months", 0) or 0
        
        # Profitability ratios
        ratios["profit_margin"] = info.get("profitMargins", 0) or 0
        ratios["operating_margin"] = info.get("operatingMargins", 0) or 0
        ratios["roe"] = info.get("returnOnEquity", 0) or 0
        ratios["roa"] = info.get("returnOnAssets", 0) or 0
        ratios["roic"] = info.get("returnOnCapital", 0) or 0
        
        # Financial health ratios
        ratios["debt_to_equity"] = info.get("debtToEquity", 0) or 0
        ratios["current_ratio"] = info.get("currentRatio", 0) or 0
        ratios["quick_ratio"] = info.get("quickRatio", 0) or 0
        ratios["interest_coverage"] = self._calculate_interest_coverage(info)
        
        # Dividend ratios
        ratios["dividend_yield"] = info.get("dividendYield", 0) or 0
        ratios["payout_ratio"] = info.get("payoutRatio", 0) or 0
        
        return ratios
    
    def _calculate_interest_coverage(self, info: Dict[str, Any]) -> float:
        """Calculate interest coverage ratio."""
        
        ebit = info.get("ebitda", 0) or 0
        interest_expense = info.get("interestExpense", 0) or 0
        
        if interest_expense > 0:
            return ebit / interest_expense
        return 0.0
    
    async def _calculate_growth_metrics(self, ticker, info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate growth metrics using historical data."""
        
        growth_metrics = {}
        
        try:
            # Get historical financial data
            financials = ticker.financials
            quarterly_financials = ticker.quarterly_financials
            
            if not financials.empty:
                # Revenue growth
                revenue_growth = self._calculate_metric_growth(financials, "Total Revenue")
                growth_metrics["revenue_growth_1y"] = revenue_growth.get("1y", 0)
                growth_metrics["revenue_growth_3y"] = revenue_growth.get("3y", 0)
                
                # Earnings growth
                earnings_growth = self._calculate_metric_growth(financials, "Net Income")
                growth_metrics["earnings_growth_1y"] = earnings_growth.get("1y", 0)
                growth_metrics["earnings_growth_3y"] = earnings_growth.get("3y", 0)
            
            # Analyst estimates
            growth_metrics["earnings_growth_estimate"] = info.get("earningsGrowth", 0) or 0
            growth_metrics["revenue_growth_estimate"] = info.get("revenueGrowth", 0) or 0
            
        except Exception as e:
            self.logger.warning(f"Could not calculate growth metrics: {e}")
            # Fallback to basic estimates
            growth_metrics["earnings_growth_estimate"] = info.get("earningsGrowth", 0) or 0
            growth_metrics["revenue_growth_estimate"] = info.get("revenueGrowth", 0) or 0
        
        return growth_metrics
    
    def _calculate_metric_growth(self, financials: pd.DataFrame, metric_name: str) -> Dict[str, float]:
        """Calculate growth rates for a specific financial metric."""
        
        growth_rates = {}
        
        if metric_name not in financials.index:
            return growth_rates
        
        metric_data = financials.loc[metric_name].dropna()
        
        if len(metric_data) < 2:
            return growth_rates
        
        # Sort by date (most recent first)
        metric_data = metric_data.sort_index(ascending=False)
        
        # 1-year growth
        if len(metric_data) >= 2:
            current = metric_data.iloc[0]
            previous = metric_data.iloc[1]
            if previous != 0:
                growth_rates["1y"] = (current - previous) / abs(previous)
        
        # 3-year CAGR
        if len(metric_data) >= 4:
            current = metric_data.iloc[0]
            three_years_ago = metric_data.iloc[3]
            if three_years_ago != 0:
                growth_rates["3y"] = (current / three_years_ago) ** (1/3) - 1
        
        return growth_rates
    
    def _calculate_profitability_metrics(self, info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate detailed profitability metrics."""
        
        profitability = {}
        
        # Basic profitability
        profitability["gross_margin"] = info.get("grossMargins", 0) or 0
        profitability["operating_margin"] = info.get("operatingMargins", 0) or 0
        profitability["net_margin"] = info.get("profitMargins", 0) or 0
        profitability["ebitda_margin"] = info.get("ebitdaMargins", 0) or 0
        
        # Return metrics
        profitability["roe"] = info.get("returnOnEquity", 0) or 0
        profitability["roa"] = info.get("returnOnAssets", 0) or 0
        profitability["roic"] = info.get("returnOnCapital", 0) or 0
        
        # Efficiency metrics
        profitability["asset_turnover"] = self._calculate_asset_turnover(info)
        profitability["inventory_turnover"] = self._calculate_inventory_turnover(info)
        
        return profitability
    
    def _calculate_asset_turnover(self, info: Dict[str, Any]) -> float:
        """Calculate asset turnover ratio."""
        
        revenue = info.get("totalRevenue", 0) or 0
        total_assets = info.get("totalAssets", 0) or 0
        
        if total_assets > 0:
            return revenue / total_assets
        return 0.0
    
    def _calculate_inventory_turnover(self, info: Dict[str, Any]) -> float:
        """Calculate inventory turnover ratio."""
        
        cogs = info.get("costOfRevenue", 0) or 0
        inventory = info.get("inventory", 0) or 0
        
        if inventory > 0:
            return cogs / inventory
        return 0.0
    
    def _assess_financial_health(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall financial health."""
        
        health_metrics = {}
        
        # Liquidity
        current_ratio = info.get("currentRatio", 0) or 0
        quick_ratio = info.get("quickRatio", 0) or 0
        
        health_metrics["liquidity_score"] = self._score_liquidity(current_ratio, quick_ratio)
        
        # Leverage
        debt_to_equity = info.get("debtToEquity", 0) or 0
        debt_to_assets = self._calculate_debt_to_assets(info)
        
        health_metrics["leverage_score"] = self._score_leverage(debt_to_equity, debt_to_assets)
        
        # Cash position
        cash = info.get("totalCash", 0) or 0
        market_cap = info.get("marketCap", 1) or 1
        cash_ratio = cash / market_cap
        
        health_metrics["cash_score"] = self._score_cash_position(cash_ratio)
        
        # Overall financial health score
        health_metrics["overall_health_score"] = np.mean([
            health_metrics["liquidity_score"],
            health_metrics["leverage_score"],
            health_metrics["cash_score"]
        ])
        
        # Health classification
        overall_score = health_metrics["overall_health_score"]
        if overall_score >= 7:
            health_metrics["health_classification"] = "excellent"
        elif overall_score >= 5:
            health_metrics["health_classification"] = "good"
        elif overall_score >= 3:
            health_metrics["health_classification"] = "fair"
        else:
            health_metrics["health_classification"] = "poor"
        
        return health_metrics
    
    def _calculate_debt_to_assets(self, info: Dict[str, Any]) -> float:
        """Calculate debt to assets ratio."""
        
        total_debt = info.get("totalDebt", 0) or 0
        total_assets = info.get("totalAssets", 0) or 0
        
        if total_assets > 0:
            return total_debt / total_assets
        return 0.0
    
    def _score_liquidity(self, current_ratio: float, quick_ratio: float) -> float:
        """Score liquidity on a scale of 0-10."""
        
        # Current ratio scoring
        if current_ratio >= 2.0:
            current_score = 10
        elif current_ratio >= 1.5:
            current_score = 8
        elif current_ratio >= 1.0:
            current_score = 6
        elif current_ratio >= 0.8:
            current_score = 4
        else:
            current_score = 2
        
        # Quick ratio scoring
        if quick_ratio >= 1.5:
            quick_score = 10
        elif quick_ratio >= 1.0:
            quick_score = 8
        elif quick_ratio >= 0.8:
            quick_score = 6
        elif quick_ratio >= 0.6:
            quick_score = 4
        else:
            quick_score = 2
        
        return (current_score + quick_score) / 2
    
    def _score_leverage(self, debt_to_equity: float, debt_to_assets: float) -> float:
        """Score leverage on a scale of 0-10 (lower leverage = higher score)."""
        
        # Debt to equity scoring
        if debt_to_equity <= 0.3:
            de_score = 10
        elif debt_to_equity <= 0.6:
            de_score = 8
        elif debt_to_equity <= 1.0:
            de_score = 6
        elif debt_to_equity <= 2.0:
            de_score = 4
        else:
            de_score = 2
        
        # Debt to assets scoring
        if debt_to_assets <= 0.2:
            da_score = 10
        elif debt_to_assets <= 0.4:
            da_score = 8
        elif debt_to_assets <= 0.6:
            da_score = 6
        elif debt_to_assets <= 0.8:
            da_score = 4
        else:
            da_score = 2
        
        return (de_score + da_score) / 2
    
    def _score_cash_position(self, cash_ratio: float) -> float:
        """Score cash position on a scale of 0-10."""
        
        if cash_ratio >= 0.15:  # 15% of market cap in cash
            return 10
        elif cash_ratio >= 0.10:
            return 8
        elif cash_ratio >= 0.05:
            return 6
        elif cash_ratio >= 0.02:
            return 4
        else:
            return 2
    
    def _assess_valuation(self, financial_ratios: Dict[str, float], 
                         company_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess company valuation."""
        
        valuation = {}
        
        # PE ratio assessment
        pe_ratio = financial_ratios.get("pe_ratio", 0)
        valuation["pe_assessment"] = self._assess_pe_ratio(pe_ratio, company_info.get("sector", ""))
        
        # PEG ratio assessment
        peg_ratio = financial_ratios.get("peg_ratio", 0)
        valuation["peg_assessment"] = self._assess_peg_ratio(peg_ratio)
        
        # Price to book assessment
        pb_ratio = financial_ratios.get("price_to_book", 0)
        valuation["pb_assessment"] = self._assess_pb_ratio(pb_ratio)
        
        # EV/EBITDA assessment
        ev_ebitda = financial_ratios.get("ev_to_ebitda", 0)
        valuation["ev_ebitda_assessment"] = self._assess_ev_ebitda(ev_ebitda)
        
        # Overall valuation score
        assessments = [
            valuation["pe_assessment"]["score"],
            valuation["peg_assessment"]["score"],
            valuation["pb_assessment"]["score"],
            valuation["ev_ebitda_assessment"]["score"]
        ]
        
        # Filter out zero scores (missing data)
        valid_assessments = [score for score in assessments if score > 0]
        
        if valid_assessments:
            valuation["overall_valuation_score"] = np.mean(valid_assessments)
        else:
            # FALLBACK SCORING: Use basic metrics when primary valuation fails
            fallback_score = self._calculate_fallback_valuation_score(financial_ratios, company_info)
            valuation["overall_valuation_score"] = fallback_score
            self.logger.info(f"Using fallback valuation scoring: {fallback_score}")
        
        # Valuation classification
        avg_score = valuation["overall_valuation_score"]
        if avg_score >= 7:
            valuation["valuation_classification"] = "undervalued"
        elif avg_score >= 4:
            valuation["valuation_classification"] = "fairly_valued"
        else:
            valuation["valuation_classification"] = "overvalued"
        
        return valuation

    def _reclassify_valuation_sector_relative(self, fundamentals_results: Dict[str, Any]) -> None:
        """Reclassify valuation using sector-relative percentiles and apply quality/growth guards.

        - Builds per-sector distributions for key valuation ratios
        - Computes each symbol's percentile (lower is better) per ratio
        - Averages available percentiles to a composite
        - Classifies by composite percentile with guard checks
        - Recomputes overall score and recommendation accordingly
        """
        # Collect sector-wise ratio arrays
        sector_to_values = {}
        ratio_keys = ["pe_ratio", "peg_ratio", "price_to_book", "ev_to_ebitda"]

        for symbol, res in fundamentals_results.items():
            if symbol.startswith("_"):
                continue
            if not isinstance(res, dict) or not res.get("data_available", False):
                continue
            sector = res.get("company_info", {}).get("sector", "Unknown")
            ratios = res.get("financial_ratios", {})
            if sector not in sector_to_values:
                sector_to_values[sector] = {k: [] for k in ratio_keys}
            for rk in ratio_keys:
                val = ratios.get(rk, 0) or 0
                # Keep only positive and reasonable values to avoid garbage
                if isinstance(val, (int, float)) and val > 0 and np.isfinite(val):
                    sector_to_values[sector][rk].append(float(val))

        def percentile_rank(arr: List[float], value: float) -> float:
            if not arr:
                return 0.5
            arr_sorted = sorted(arr)
            # proportion of peers with <= value
            count = np.searchsorted(arr_sorted, value, side="right")
            return count / len(arr_sorted)

        # Now compute per-symbol percentiles and reclassify
        for symbol, res in fundamentals_results.items():
            if symbol.startswith("_"):
                continue
            if not isinstance(res, dict) or not res.get("data_available", False):
                continue
            sector = res.get("company_info", {}).get("sector", "Unknown")
            ratios = res.get("financial_ratios", {})

            # Fallback if not enough peers
            if sector not in sector_to_values or sum(len(v) for v in sector_to_values[sector].values()) == 0:
                # keep existing valuation
                res.setdefault("valuation_assessment", {}).setdefault("method", "fixed_thresholds")
                continue
            # Require a minimum number of peers across ratios
            peer_counts = [len(sector_to_values[sector][rk]) for rk in ratio_keys]
            if max(peer_counts) < self.min_sector_peers:
                res.setdefault("valuation_assessment", {}).setdefault("method", "fixed_thresholds")
                continue

            ratio_percentiles: Dict[str, float] = {}
            for rk in ratio_keys:
                val = ratios.get(rk, 0) or 0
                if isinstance(val, (int, float)) and val > 0 and np.isfinite(val) and sector_to_values[sector][rk]:
                    ratio_percentiles[rk] = float(percentile_rank(sector_to_values[sector][rk], float(val)))
            if not ratio_percentiles:
                res.setdefault("valuation_assessment", {}).setdefault("method", "fixed_thresholds")
                continue

            composite_percentile = float(np.mean(list(ratio_percentiles.values())))

            # Guard checks to avoid value traps
            profitability = res.get("profitability_metrics", {})
            growth = res.get("growth_metrics", {})
            roe = float(profitability.get("roe", 0) or 0)
            op_margin = float(profitability.get("operating_margin", 0) or 0)
            earnings_growth = float(
                (growth.get("earnings_growth_1y") if growth.get("earnings_growth_1y") is not None else growth.get("earnings_growth_estimate", 0))
                or 0
            )

            undervalued = composite_percentile <= self.undervalued_pct
            overvalued = composite_percentile >= self.overvalued_pct

            guard_pass = (roe >= self.guard_min_roe) and (op_margin >= self.guard_min_operating_margin) and (earnings_growth >= self.guard_min_earnings_growth)

            if undervalued and not guard_pass:
                classification = "fairly_valued"
                guard_flag = True
            elif undervalued:
                classification = "undervalued"
                guard_flag = False
            elif overvalued:
                classification = "overvalued"
                guard_flag = False
            else:
                classification = "fairly_valued"
                guard_flag = False

            # Map composite percentile to a 0-10 style score (higher = cheaper)
            overall_val_score = round((1.0 - composite_percentile) * 10.0, 2)

            res["valuation_assessment"] = {
                "method": "sector_percentiles",
                "ratio_percentiles": ratio_percentiles,
                "composite_percentile": composite_percentile,
                "overall_valuation_score": overall_val_score,
                "valuation_classification": classification,
                "guard_failed": guard_flag,
            }

            # Recompute overall score and recommendation to reflect new valuation
            financial_health = res.get("financial_health", {})
            growth_metrics = res.get("growth_metrics", {})
            profitability_metrics = res.get("profitability_metrics", {})
            res["overall_score"] = self._calculate_overall_score(
                res.get("financial_ratios", {}),
                growth_metrics,
                profitability_metrics,
                financial_health,
                res["valuation_assessment"],
            )
            res["recommendation"] = self._generate_recommendation(res["overall_score"], res["valuation_assessment"])
    
    def _assess_pe_ratio(self, pe_ratio: float, sector: str) -> Dict[str, Any]:
        """Assess PE ratio relative to benchmarks."""
        
        # More lenient validity check - only reject clearly invalid data
        if pe_ratio <= 0 or pe_ratio > 200:  # Increased threshold from 100 to 200
            return {"score": 0, "assessment": "invalid", "benchmark": 0}
        
        # Sector-specific benchmarks (simplified)
        sector_pe_benchmarks = {
            "Technology": 30,  # Increased from 25 (tech often trades at premium)
            "Healthcare": 25,  # Increased from 20
            "Financial Services": 12,
            "Consumer Defensive": 18,
            "Utilities": 16,
            "Energy": 15,
            "Industrial": 18,
        }
        
        benchmark = sector_pe_benchmarks.get(sector, 20)  # Increased default from 18
        
        # More sensitive scoring with wider ranges
        if pe_ratio <= benchmark * 0.6:  # Decreased from 0.7 (more generous)
            score = 10
            assessment = "very_attractive"
        elif pe_ratio <= benchmark * 0.9:  # Added intermediate range
            score = 9
            assessment = "attractive"
        elif pe_ratio <= benchmark:
            score = 7  # Reduced from 8 to make room for intermediate
            assessment = "fair"
        elif pe_ratio <= benchmark * 1.2:  # Decreased from 1.3
            score = 6
            assessment = "slightly_expensive"
        elif pe_ratio <= benchmark * 1.5:  # Decreased from 1.6
            score = 4
            assessment = "expensive"
        elif pe_ratio <= benchmark * 2.0:  # Added range for very high PE
            score = 3
            assessment = "very_expensive"
        else:
            score = 2
            assessment = "extremely_expensive"
        
        return {"score": score, "assessment": assessment, "benchmark": benchmark}
    
    def _assess_peg_ratio(self, peg_ratio: float) -> Dict[str, Any]:
        """Assess PEG ratio."""
        
        # More lenient validity check - many value stocks have high PEG
        if peg_ratio <= 0 or peg_ratio > 10:  # Increased threshold from 5 to 10
            return {"score": 0, "assessment": "invalid"}
        
        # More sensitive scoring with additional ranges
        if peg_ratio <= 0.3:
            score = 10
            assessment = "extremely_undervalued"
        elif peg_ratio <= 0.7:  # Increased from 0.5
            score = 9
            assessment = "very_undervalued"
        elif peg_ratio <= 1.0:
            score = 8
            assessment = "undervalued"
        elif peg_ratio <= 1.3:  # Decreased from 1.5
            score = 7
            assessment = "fairly_valued"
        elif peg_ratio <= 1.8:  # Decreased from 2.0
            score = 5
            assessment = "slightly_overvalued"
        elif peg_ratio <= 2.5:  # Added intermediate range
            score = 4
            assessment = "overvalued"
        elif peg_ratio <= 4.0:  # Added range
            score = 3
            assessment = "very_overvalued"
        else:
            score = 2
            assessment = "extremely_overvalued"
        
        return {"score": score, "assessment": assessment}
    
    def _assess_pb_ratio(self, pb_ratio: float) -> Dict[str, Any]:
        """Assess Price-to-Book ratio."""
        
        # More lenient - only reject clearly invalid data
        if pb_ratio <= 0 or pb_ratio > 20:  # Added upper bound for sanity
            return {"score": 0, "assessment": "invalid"}
        
        # More sensitive scoring with additional ranges
        if pb_ratio <= 0.8:
            score = 10
            assessment = "extremely_attractive"
        elif pb_ratio <= 1.2:  # Increased from 1.0
            score = 9
            assessment = "very_attractive"
        elif pb_ratio <= 1.8:  # Increased from 1.5
            score = 7
            assessment = "attractive"
        elif pb_ratio <= 2.8:  # Increased from 2.5
            score = 6
            assessment = "fair"
        elif pb_ratio <= 4.5:  # Increased from 4.0
            score = 4
            assessment = "expensive"
        elif pb_ratio <= 7.0:  # Added intermediate range
            score = 3
            assessment = "very_expensive"
        else:
            score = 2
            assessment = "extremely_expensive"
        
        return {"score": score, "assessment": assessment}
    
    def _assess_ev_ebitda(self, ev_ebitda: float) -> Dict[str, Any]:
        """Assess EV/EBITDA ratio."""
        
        # More lenient validity check
        if ev_ebitda <= 0 or ev_ebitda > 100:  # Increased from 50 to 100
            return {"score": 0, "assessment": "invalid"}
        
        # More sensitive scoring with additional ranges
        if ev_ebitda <= 6:
            score = 10
            assessment = "extremely_attractive"
        elif ev_ebitda <= 10:  # Increased from 8
            score = 9
            assessment = "very_attractive"
        elif ev_ebitda <= 14:  # Increased from 12
            score = 7
            assessment = "attractive"
        elif ev_ebitda <= 20:  # Increased from 18
            score = 6
            assessment = "fair"
        elif ev_ebitda <= 28:  # Increased from 25
            score = 4
            assessment = "expensive"
        elif ev_ebitda <= 40:  # Added intermediate range
            score = 3
            assessment = "very_expensive"
        else:
            score = 2
            assessment = "extremely_expensive"
        
        return {"score": score, "assessment": assessment}
    
    def _calculate_fallback_valuation_score(self, financial_ratios: Dict[str, float], 
                                          company_info: Dict[str, Any]) -> float:
        """Calculate fallback valuation score using basic available metrics."""
        
        scores = []
        
        # Basic profitability check
        profit_margin = financial_ratios.get("profit_margin", 0)
        if profit_margin > 0.15:  # 15%+ margin
            scores.append(8)
        elif profit_margin > 0.08:  # 8%+ margin
            scores.append(6)
        elif profit_margin > 0.03:  # 3%+ margin
            scores.append(5)
        elif profit_margin > 0:
            scores.append(4)
        else:
            scores.append(3)  # Unprofitable
        
        # ROE check
        roe = financial_ratios.get("roe", 0)
        if roe > 0.20:  # 20%+ ROE
            scores.append(8)
        elif roe > 0.15:  # 15%+ ROE
            scores.append(7)
        elif roe > 0.10:  # 10%+ ROE
            scores.append(6)
        elif roe > 0.05:  # 5%+ ROE
            scores.append(5)
        elif roe > 0:
            scores.append(4)
        else:
            scores.append(3)
        
        # Debt-to-equity check (lower is better)
        debt_ratio = financial_ratios.get("debt_to_equity", 0)
        if debt_ratio == 0:  # No debt
            scores.append(8)
        elif debt_ratio < 0.3:  # Low debt
            scores.append(7)
        elif debt_ratio < 0.6:  # Moderate debt
            scores.append(6)
        elif debt_ratio < 1.0:  # Higher debt
            scores.append(4)
        else:  # Very high debt
            scores.append(3)
        
        # Current ratio check (liquidity)
        current_ratio = financial_ratios.get("current_ratio", 0)
        if current_ratio > 2.0:  # Strong liquidity
            scores.append(7)
        elif current_ratio > 1.5:  # Good liquidity
            scores.append(6)
        elif current_ratio > 1.0:  # Adequate liquidity
            scores.append(5)
        else:  # Poor liquidity
            scores.append(3)
        
        # If we have any scores, average them; otherwise use neutral
        if scores:
            fallback_score = np.mean(scores)
            self.logger.info(f"Fallback valuation calculated from {len(scores)} metrics: {fallback_score:.2f}")
            return round(fallback_score, 2)
        else:
            self.logger.warning("No metrics available for fallback valuation - using neutral score")
            return 5.0
    
    def _estimate_growth_from_fundamentals(self, financial_ratios: Dict[str, float], 
                                         profitability_metrics: Dict[str, float]) -> float:
        """Estimate growth potential from basic fundamental metrics when growth data unavailable."""
        
        scores = []
        
        # High ROE suggests potential for growth
        roe = financial_ratios.get("roe", 0)
        if roe > 0.20:  # Very high ROE suggests growth potential
            scores.append(7)
        elif roe > 0.15:
            scores.append(6)
        elif roe > 0.10:
            scores.append(5)
        else:
            scores.append(4)
        
        # Low debt suggests capacity for growth investment
        debt_ratio = financial_ratios.get("debt_to_equity", 1.0)
        if debt_ratio < 0.3:  # Low debt = growth capacity
            scores.append(6)
        elif debt_ratio < 0.6:
            scores.append(5)
        else:
            scores.append(4)
        
        # High profit margins suggest efficiency and growth potential
        profit_margin = financial_ratios.get("profit_margin", 0)
        if profit_margin > 0.15:  # High margins suggest growth efficiency
            scores.append(6)
        elif profit_margin > 0.08:
            scores.append(5)
        else:
            scores.append(4)
        
        # Strong current ratio suggests financial stability for growth
        current_ratio = financial_ratios.get("current_ratio", 1.0)
        if current_ratio > 1.5:
            scores.append(6)
        elif current_ratio > 1.0:
            scores.append(5)
        else:
            scores.append(4)
        
        estimated_growth = np.mean(scores) if scores else 5.0
        return round(estimated_growth, 2)
    
    def _calculate_overall_score(self, financial_ratios: Dict[str, float], 
                               growth_metrics: Dict[str, float],
                               profitability_metrics: Dict[str, float], 
                               financial_health: Dict[str, Any],
                               valuation_assessment: Dict[str, Any]) -> float:
        """Calculate overall fundamental score (0-10)."""
        
        # Valuation component
        valuation_score = valuation_assessment.get("overall_valuation_score", 5.0)
        
        # Profitability component
        roe = profitability_metrics.get("roe", 0)
        operating_margin = profitability_metrics.get("operating_margin", 0)
        profitability_score = self._score_profitability(roe, operating_margin)
        
        # Growth component - use more available data
        if growth_metrics:
            revenue_growth = growth_metrics.get("revenue_growth_1y", 0)
            earnings_growth = growth_metrics.get("earnings_growth_1y", 0)
            growth_score = self._score_growth(revenue_growth, earnings_growth)
        else:
            # Try to estimate growth from basic profitability trends
            growth_score = self._estimate_growth_from_fundamentals(financial_ratios, profitability_metrics)
            self.logger.info(f"Using estimated growth score: {growth_score}")
        
        # Financial health component
        health_score = financial_health.get("overall_health_score", 5.0)
        
        # Weighted average
        overall_score = (
            valuation_score * self.metric_weights["valuation"] +
            profitability_score * self.metric_weights["profitability"] +
            growth_score * self.metric_weights["growth"] +
            health_score * self.metric_weights["financial_health"]
        )
        
        return round(overall_score, 2)
    
    def _score_profitability(self, roe: float, operating_margin: float) -> float:
        """Score profitability metrics (0-10)."""
        
        # ROE scoring
        if roe >= 0.20:  # 20%+
            roe_score = 10
        elif roe >= 0.15:
            roe_score = 8
        elif roe >= 0.10:
            roe_score = 6
        elif roe >= 0.05:
            roe_score = 4
        else:
            roe_score = 2
        
        # Operating margin scoring
        if operating_margin >= 0.20:  # 20%+
            margin_score = 10
        elif operating_margin >= 0.15:
            margin_score = 8
        elif operating_margin >= 0.10:
            margin_score = 6
        elif operating_margin >= 0.05:
            margin_score = 4
        else:
            margin_score = 2
        
        return (roe_score + margin_score) / 2
    
    def _score_growth(self, revenue_growth: float, earnings_growth: float) -> float:
        """Score growth metrics (0-10)."""
        
        # Revenue growth scoring
        if revenue_growth >= 0.20:  # 20%+
            rev_score = 10
        elif revenue_growth >= 0.10:
            rev_score = 8
        elif revenue_growth >= 0.05:
            rev_score = 6
        elif revenue_growth >= 0:
            rev_score = 4
        else:
            rev_score = 2
        
        # Earnings growth scoring
        if earnings_growth >= 0.25:  # 25%+
            earn_score = 10
        elif earnings_growth >= 0.15:
            earn_score = 8
        elif earnings_growth >= 0.05:
            earn_score = 6
        elif earnings_growth >= 0:
            earn_score = 4
        else:
            earn_score = 2
        
        return (rev_score + earn_score) / 2
    
    def _generate_recommendation(self, overall_score: float, 
                               valuation_assessment: Dict[str, Any],
                               symbol: str = None) -> str:
        """Generate investment recommendation with portfolio awareness."""
        
        valuation_class = valuation_assessment.get("valuation_classification", "insufficient_data")
        
        # Base recommendation from fundamentals
        base_recommendation = ""
        if overall_score >= 8 and valuation_class == "undervalued":
            base_recommendation = "strong_buy"
        elif overall_score >= 7 and valuation_class in ["undervalued", "fairly_valued"]:
            base_recommendation = "buy"
        elif overall_score >= 6:
            base_recommendation = "hold"
        elif overall_score >= 4:
            base_recommendation = "weak_hold"
        else:
            base_recommendation = "sell"
        
        # Return pure fundamental recommendation without portfolio bias
        return base_recommendation
    
    def _calculate_sector_benchmarks(self, fundamentals_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sector-level benchmark metrics."""
        
        sector_data = {}
        
        for symbol, result in fundamentals_results.items():
            if symbol.startswith("_"):  # Skip meta keys
                continue
                
            if not result.get("data_available", False):
                continue
            
            sector = result.get("company_info", {}).get("sector", "Unknown")
            
            if sector not in sector_data:
                sector_data[sector] = {
                    "companies": [],
                    "pe_ratios": [],
                    "roe_values": [],
                    "profit_margins": [],
                    "debt_ratios": []
                }
            
            sector_data[sector]["companies"].append(symbol)
            
            # Collect metrics
            ratios = result.get("financial_ratios", {})
            sector_data[sector]["pe_ratios"].append(ratios.get("pe_ratio", 0))
            sector_data[sector]["roe_values"].append(ratios.get("roe", 0))
            sector_data[sector]["profit_margins"].append(ratios.get("profit_margin", 0))
            sector_data[sector]["debt_ratios"].append(ratios.get("debt_to_equity", 0))
        
        # Calculate sector benchmarks
        sector_benchmarks = {}
        
        for sector, data in sector_data.items():
            if len(data["companies"]) >= 2:  # Need at least 2 companies
                sector_benchmarks[sector] = {
                    "company_count": len(data["companies"]),
                    "avg_pe_ratio": np.mean([x for x in data["pe_ratios"] if x > 0]),
                    "avg_roe": np.mean([x for x in data["roe_values"] if x > 0]),
                    "avg_profit_margin": np.mean([x for x in data["profit_margins"] if x > 0]),
                    "avg_debt_ratio": np.mean([x for x in data["debt_ratios"] if x > 0])
                }
        
        return sector_benchmarks
    
    def _calculate_average_metric(self, results: Dict[str, Any], metric: str) -> float:
        """Calculate average value for a specific metric across all symbols."""
        
        values = []
        
        for result in results.values():
            if isinstance(result, dict) and "financial_ratios" in result:
                value = result["financial_ratios"].get(metric, 0)
                if value > 0:  # Filter out invalid values
                    values.append(value)
        
        return np.mean(values) if values else 0.0
    
    def _count_by_valuation(self, results: Dict[str, Any], valuation_type: str) -> int:
        """Count symbols by valuation classification."""
        
        count = 0
        
        for result in results.values():
            if isinstance(result, dict) and "valuation_assessment" in result:
                classification = result["valuation_assessment"].get("valuation_classification", "")
                if classification == valuation_type:
                    count += 1
        
        return count
    
    def _create_empty_fundamentals_result(self, symbol: str) -> Dict[str, Any]:
        """Create empty fundamentals result for error cases."""
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "company_info": {},
            "financial_ratios": {},
            "growth_metrics": {},
            "profitability_metrics": {},
            "financial_health": {},
            "valuation_assessment": {"valuation_classification": "insufficient_data"},
            "sector_comparison": {},
            "overall_score": 0.0,
            "recommendation": "insufficient_data",
            "data_available": False
        }