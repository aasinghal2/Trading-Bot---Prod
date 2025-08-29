"""
Market Scanner Module - S&P 500 Stock Opportunity Scanner

This module provides market scanning capabilities to find trading opportunities
across the S&P 500 index. It's designed as a standalone module that can be
easily removed without affecting core trading functionality.

Features:
- S&P 500 symbol scanning
- P/E ratio filtering
- Lightweight analysis pipeline
- Signal strength ranking
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import yfinance as yf
import pandas as pd
import json
import os

from orchestrator import TradingOrchestrator

# Configure logging
logger = logging.getLogger(__name__)

# S&P 500 Symbols (Top 50 by market cap)
# Source: Major S&P 500 companies as of 2024
SP500_TEST_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'V', 'PG', 'JPM', 'HD', 'MA', 'AVGO', 'PFE', 'ABBV', 'KO', 'COST',
    'MRK', 'ADBE', 'TMO', 'ACN', 'CSCO', 'PEP', 'LLY', 'DHR', 'NFLX', 'VZ',
    'NKE', 'DIS', 'WMT', 'CRM', 'CMCSA', 'ABT', 'TXN', 'CVX', 'NEE', 'ORCL',
    'XOM', 'INTC', 'QCOM', 'AMD', 'BMY', 'UPS', 'T', 'LOW', 'IBM', 'HON'
]


class MarketScanner:
    """Market Scanner for S&P 500 opportunities"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the market scanner"""
        self.config_path = config_path
        self.orchestrator = None
        logger.info(f"Market Scanner initialized with {len(SP500_TEST_SYMBOLS)} S&P 500 symbols")
    
    async def scan_market(self, 
                         top_n: int = 10,
                         min_pe: float = 0,
                         max_pe: float = 50,
                         min_market_cap: float = 1e9) -> Dict[str, Any]:
        """
        Scan S&P 500 for trading opportunities
        
        Args:
            top_n: Number of top opportunities to return
            min_pe: Minimum P/E ratio filter
            max_pe: Maximum P/E ratio filter
            min_market_cap: Minimum market cap filter
            
        Returns:
            Dictionary with scan results and top opportunities
        """
        start_time = datetime.now()
        
        logger.info(f"🔍 Starting S&P 500 market scan...")
        logger.info(f"Filters: P/E {min_pe}-{max_pe}, Min Market Cap: ${min_market_cap:,.0f}")
        
        # Step 1: Filter symbols by fundamentals
        filtered_symbols = await self._filter_by_fundamentals(
            SP500_TEST_SYMBOLS, min_pe, max_pe, min_market_cap
        )
        
        if not filtered_symbols:
            return {
                'success': False,
                'message': 'No symbols passed fundamental filters',
                'scan_time': (datetime.now() - start_time).total_seconds(),
                'symbols_scanned': 0,
                'symbols_filtered': 0,
                'top_opportunities': []
            }
        
        logger.info(f"📊 {len(filtered_symbols)} symbols passed fundamental filters")
        
        # Step 2: Run lightweight analysis on filtered symbols
        opportunities, threshold_info = await self._analyze_filtered_symbols(filtered_symbols)
        
        # Step 3: Rank and select top opportunities
        top_opportunities = self._rank_opportunities(opportunities, top_n)
        
        scan_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate signal statistics for dynamic threshold info
        signal_stats = {}
        if opportunities:
            signals = [opp['combined_signal'] for opp in opportunities]
            signal_stats = {
                'min_signal': min(signals),
                'max_signal': max(signals),
                'avg_signal': sum(signals) / len(signals),
                'signal_count': len(signals),
                'dynamic_threshold': 'Calculated from real-time analysis'
            }
        
        result = {
            'success': True,
            'scan_time': scan_time,
            'symbols_scanned': len(SP500_TEST_SYMBOLS),
            'symbols_filtered': len(filtered_symbols),
            'symbols_analyzed': len(opportunities),
            'signal_stats': signal_stats,
            'top_opportunities': top_opportunities,
            'scan_timestamp': datetime.now().isoformat(),
            'filters_applied': {
                'min_pe': min_pe,
                'max_pe': max_pe,
                'min_market_cap': min_market_cap
            }
        }
        
        # Add dynamic threshold information if available
        if threshold_info:
            result.update(threshold_info)
        
        logger.info(f"✅ Market scan completed in {scan_time:.2f}s")
        logger.info(f"🎯 Found {len(top_opportunities)} top opportunities")
        
        return result
    
    async def _filter_by_fundamentals(self, 
                                    symbols: List[str], 
                                    min_pe: float, 
                                    max_pe: float, 
                                    min_market_cap: float) -> List[str]:
        """Filter symbols by fundamental criteria"""
        filtered_symbols = []
        
        logger.info(f"📈 Filtering {len(symbols)} symbols by fundamentals...")
        
        # Process symbols in batches to avoid overwhelming yfinance
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            for symbol in batch:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Get P/E ratio
                    pe_ratio = info.get('trailingPE', None)
                    if pe_ratio is None:
                        pe_ratio = info.get('forwardPE', None)
                    
                    # Get market cap
                    market_cap = info.get('marketCap', 0)
                    
                    # Apply filters
                    if (pe_ratio is not None and 
                        min_pe <= pe_ratio <= max_pe and 
                        market_cap >= min_market_cap):
                        filtered_symbols.append(symbol)
                        logger.debug(f"✅ {symbol}: P/E={pe_ratio:.2f}, Market Cap=${market_cap:,.0f}")
                    else:
                        logger.debug(f"❌ {symbol}: P/E={pe_ratio}, Market Cap=${market_cap:,.0f} (filtered out)")
                
                except Exception as e:
                    logger.warning(f"⚠️ Error filtering {symbol}: {e}")
                    continue
            
            # Small delay between batches to be respectful to APIs
            await asyncio.sleep(0.5)
        
        return filtered_symbols
    
    async def _analyze_filtered_symbols(self, symbols: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run lightweight analysis on filtered symbols"""
        opportunities = []
        threshold_info = {}
        
        logger.info(f"🔍 Running lightweight analysis on {len(symbols)} symbols...")
        
        # Initialize orchestrator if needed
        if self.orchestrator is None:
            self.orchestrator = TradingOrchestrator(self.config_path)
            # Wait for initialization to complete
            await asyncio.sleep(0.1)
        
        # Analyze symbols in smaller batches for better performance
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            try:
                # Run simplified analysis (just market data and technical)
                result = await self._lightweight_analysis(batch)
                
                if result and result.get('success'):
                    # Capture dynamic threshold info (always use latest/most accurate)
                    if 'dynamic_threshold' in result:
                        threshold_info = {
                            'dynamic_threshold': result['dynamic_threshold'],
                            'threshold_explanation': result.get('threshold_explanation', '')
                        }
                    
                    # Extract signals for each symbol
                    # Get sentiment data for accurate signal calculation
                    sentiment_data = {}
                    sentiment_agent = result.get('results', {}).get('sentiment')
                    if sentiment_agent and hasattr(sentiment_agent, 'data'):
                        sentiment_data = sentiment_agent.data
                    elif isinstance(sentiment_agent, dict):
                        sentiment_data = sentiment_agent.get('data', {})
                    
                    for symbol in batch:
                        signal_data = self._extract_signal_data(symbol, result, sentiment_data)
                        if signal_data:
                            opportunities.append(signal_data)
                
            except Exception as e:
                logger.warning(f"⚠️ Error analyzing batch {batch}: {e}")
                continue
            
            # Progress logging
            logger.info(f"📊 Analyzed {min(i + batch_size, len(symbols))}/{len(symbols)} symbols")
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        return opportunities, threshold_info
    
    async def _lightweight_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Run SCAN-ONLY analysis without trade execution"""
        try:
            # Use analysis-only mode - NO TRADE EXECUTION
            result = await self.orchestrator.execute_analysis_only(symbols)
            return result
        except Exception as e:
            logger.error(f"Error in lightweight analysis: {e}")
            return {}
    
    def _extract_signal_data(self, symbol: str, analysis_result: Dict[str, Any], sentiment_results: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Extract relevant signal data for ranking"""
        try:
            results = analysis_result.get('results', {})
            
            # Debug: Log the structure we're getting
            logger.debug(f"Extracting signals for {symbol}, results keys: {list(results.keys())}")
            
            # Get analysis results from the fixed orchestrator structure
            tech_data = {}
            fund_data = {}
            
            # Get technical data from technical agent result
            tech_agent = results.get('technical')
            if tech_agent and hasattr(tech_agent, 'data'):
                tech_data = tech_agent.data
            elif isinstance(tech_agent, dict):
                tech_data = tech_agent.get('data', {})
            
            # Get fundamentals data from fundamentals agent result
            fund_agent = results.get('fundamentals')
            if fund_agent and hasattr(fund_agent, 'data'):
                fund_data = fund_agent.data
            elif isinstance(fund_agent, dict):
                fund_data = fund_agent.get('data', {})
            
            symbol_tech = tech_data.get(symbol, {})
            tech_signal = symbol_tech.get('overall_signal', {})
                
            symbol_fund = fund_data.get(symbol, {})
            
            # Get market data for current price (handle AgentResult objects)
            market_agent = results.get('market_data')
            market_data = {}
            if market_agent and hasattr(market_agent, 'data'):
                market_data = market_agent.data
            elif isinstance(market_agent, dict):
                market_data = market_agent.get('data', {})
                
            hist_data = market_data.get('historical', {}).get(symbol)
            current_price = None
            if hist_data is not None and hasattr(hist_data, 'iloc') and len(hist_data) > 0:
                current_price = hist_data['Close'].iloc[-1] if 'Close' in hist_data.columns else None
            
            # Calculate combined signal using SAME LOGIC as orchestrator tracking
            tech_strength = tech_signal.get('strength', 0)
            fund_score = symbol_fund.get('overall_score', 5.0)
            fund_signal = (fund_score - 5.0) / 5.0  # Convert to -1 to 1 scale
            
            # Get sentiment signal if available  
            sentiment_signal = 0.0
            if sentiment_results and symbol in sentiment_results:
                sentiment_signal = sentiment_results[symbol].get('overall_sentiment', 0.0)
            
            # Use CORRECT weighted average logic (same as orchestrator)
            signals = []
            weights = []
            
            # Add technical signal if available (non-zero)
            if tech_strength != 0:
                signals.append(tech_strength)
                weights.append(0.4)  # 40% weight for technical
            
            # Add fundamental signal if available (non-zero)
            if fund_signal != 0:
                signals.append(fund_signal)
                weights.append(0.35)  # 35% weight for fundamentals
            
            # Add sentiment signal if available (non-zero)
            if sentiment_signal != 0:
                signals.append(sentiment_signal)
                weights.append(0.25)  # 25% weight for sentiment
            
            # Calculate proper weighted average
            if signals and weights:
                total_weight = sum(weights)
                combined_signal = sum(s * w for s, w in zip(signals, weights)) / total_weight
            else:
                combined_signal = 0.0
            
            # Debug logging
            logger.debug(f"{symbol}: tech={tech_strength}, fund={fund_score}, sentiment={sentiment_signal}, combined={combined_signal}")
            if signals and weights:
                weighted_parts = [f"{s}*{w}" for s, w in zip(signals, weights)]
                logger.info(f"Signal calculation for {symbol}: ({' + '.join(weighted_parts)}) / {sum(weights):.2f} = {combined_signal:.3f}")
            else:
                logger.info(f"Signal calculation for {symbol}: No valid signals available = {combined_signal:.3f}")
            
            # Get valuation summary from valuation_assessment
            valuation_summary = 'Unknown'
            valuation_assessment = symbol_fund.get('valuation_assessment', {})
            if valuation_assessment:
                classification = valuation_assessment.get('valuation_classification', 'unknown')
                overall_score = valuation_assessment.get('overall_valuation_score', 0)
                valuation_summary = f"{classification.title()} ({overall_score:.1f})"
            
            return {
                'symbol': symbol,
                'combined_signal': combined_signal,
                'technical_strength': tech_strength,
                'technical_direction': tech_signal.get('direction', 'neutral'),
                'fundamental_score': fund_score,
                'fundamental_signal': fund_signal,
                'current_price': current_price,
                'valuation_summary': valuation_summary
            }
            
        except Exception as e:
            logger.warning(f"Error extracting signal data for {symbol}: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {e}")
            return None
    
    def _rank_opportunities(self, opportunities: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        """Rank opportunities by combined signal strength"""
        # Sort by combined signal strength (descending)
        ranked = sorted(opportunities, key=lambda x: abs(x['combined_signal']), reverse=True)
        
        # Return top N with additional ranking info
        top_opportunities = []
        for i, opp in enumerate(ranked[:top_n]):
            opp['rank'] = i + 1
            opp['signal_percentile'] = ((len(ranked) - i) / len(ranked)) * 100
            top_opportunities.append(opp)
        
        return top_opportunities


def save_recommendations(results: Dict[str, Any], filename: str = "recommendations.json"):
    """Save scan results as recommendations to a file"""
    if not results.get('success'):
        logger.error("Cannot save recommendations - scan failed")
        return False
    
    # Create recommendations directory if it doesn't exist
    os.makedirs("recommendations", exist_ok=True)
    filepath = os.path.join("recommendations", filename)
    
    # Prepare recommendation data
    recommendations = {
        "scan_timestamp": results['scan_timestamp'],
        "scan_summary": {
            "scan_time_seconds": results['scan_time'],
            "symbols_scanned": results['symbols_scanned'],
            "symbols_filtered": results['symbols_filtered'],
            "symbols_analyzed": results['symbols_analyzed'],
            "filters_applied": results['filters_applied']
        },
        "buy_recommendations": [],
        "sell_recommendations": [],
        "neutral_signals": []
    }
    
    # Categorize opportunities
    for opp in results['top_opportunities']:
        recommendation = {
            "rank": opp['rank'],
            "symbol": opp['symbol'],
            "signal_strength": opp['combined_signal'],
            "technical_direction": opp['technical_direction'],
            "technical_strength": opp['technical_strength'],
            "fundamental_score": opp['fundamental_score'],
            "fundamental_signal": opp['fundamental_signal'],
            "current_price": opp['current_price'],
            "valuation_summary": opp['valuation_summary'],
            "signal_percentile": opp['signal_percentile'],
            "recommendation_reason": _generate_recommendation_reason(opp)
        }
        
        # Categorize based on signal strength and direction
        if opp['combined_signal'] >= 0.25:
            recommendations["buy_recommendations"].append(recommendation)
        elif opp['combined_signal'] <= -0.25:
            recommendations["sell_recommendations"].append(recommendation)
        else:
            recommendations["neutral_signals"].append(recommendation)
    
    # Save to file
    try:
        with open(filepath, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        logger.info(f"📝 Recommendations saved to {filepath}")
        print(f"\n💾 Recommendations saved to: {filepath}")
        
        # Display summary
        print(f"📈 BUY Recommendations: {len(recommendations['buy_recommendations'])}")
        print(f"📉 SELL Recommendations: {len(recommendations['sell_recommendations'])}")
        print(f"🔘 NEUTRAL Signals: {len(recommendations['neutral_signals'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving recommendations: {e}")
        print(f"❌ Error saving recommendations: {e}")
        return False


def _generate_recommendation_reason(opportunity: Dict[str, Any]) -> str:
    """Generate a human-readable recommendation reason"""
    signal = opportunity['combined_signal']
    tech_strength = opportunity['technical_strength']
    fund_signal = opportunity['fundamental_signal']
    direction = opportunity['technical_direction'].lower()
    
    if signal >= 0.4:
        strength_desc = "Very Strong"
    elif signal >= 0.3:
        strength_desc = "Strong"
    elif signal >= 0.25:
        strength_desc = "Moderate"
    elif signal <= -0.25:
        strength_desc = "Strong Bearish"
    else:
        strength_desc = "Weak"
    
    # Determine primary driver
    if abs(tech_strength) > abs(fund_signal):
        driver = f"technical analysis ({direction} trend)"
    else:
        driver = "fundamental analysis"
    
    return f"{strength_desc} signal ({signal:.3f}) driven primarily by {driver}"


def display_scan_results(results: Dict[str, Any]):
    """Display market scan results in a formatted way"""
    print("\n" + "="*80)
    print("🔍 S&P 500 MARKET SCANNER RESULTS")
    print("="*80)
    
    if not results.get('success'):
        print(f"❌ Scan failed: {results.get('message', 'Unknown error')}")
        return
    
    # Summary stats
    print(f"📊 SCAN SUMMARY:")
    print(f"   ⏱️  Scan Time: {results['scan_time']:.2f} seconds")
    print(f"   🎯 Symbols Scanned: {results['symbols_scanned']}")
    print(f"   ✅ Passed Filters: {results['symbols_filtered']}")
    print(f"   📈 Successfully Analyzed: {results['symbols_analyzed']}")
    
    # Filters applied
    filters = results['filters_applied']
    print(f"   🔍 Filters: P/E {filters['min_pe']}-{filters['max_pe']}, Min Cap ${filters['min_market_cap']:,.0f}")
    
    # Show dynamic threshold information if available
    if 'signal_stats' in results:
        stats = results['signal_stats']
        print(f"   🧠 Signal Range: {stats.get('min_signal', 0):.3f} to {stats.get('max_signal', 0):.3f}")
        print(f"   📊 Average Signal: {stats.get('avg_signal', 0):.3f}")
        
        # Show dynamic threshold with explanation if available
        if 'dynamic_threshold' in results:
            threshold = results['dynamic_threshold']
            explanation = results.get('threshold_explanation', 'No explanation available')
            print(f"   🎯 Dynamic Threshold: {threshold:.3f}")
            print(f"   📝 Threshold Info: {explanation}")
        else:
            print(f"   🎯 Dynamic Threshold: {stats.get('dynamic_threshold', 'Unknown')}")
    
    print(f"\n🏆 TOP OPPORTUNITIES:")
    print(f"{'Rank':<4} {'Symbol':<6} {'Signal':<8} {'Direction':<8} {'Price':<10} {'P/E View':<12}")
    print("-" * 60)
    
    for opp in results['top_opportunities']:
        rank = opp['rank']
        symbol = opp['symbol']
        signal = f"{opp['combined_signal']:.3f}"
        direction = opp['technical_direction'].upper()
        price = f"${opp['current_price']:.2f}" if opp['current_price'] else "N/A"
        valuation = opp['valuation_summary'][:10]
        
        print(f"{rank:<4} {symbol:<6} {signal:<8} {direction:<8} {price:<10} {valuation:<12}")
    
    print("="*80)


# CLI Integration Function
async def execute_market_scan(top_n: int = 10, min_pe: float = 0, max_pe: float = 50):
    """Execute market scan, display results, and save recommendations"""
    scanner = MarketScanner()
    results = await scanner.scan_market(top_n=top_n, min_pe=min_pe, max_pe=max_pe)
    display_scan_results(results)
    
    # Save recommendations to file with timestamp
    if results.get('success'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_scan_{timestamp}.json"
        save_recommendations(results, filename)
    
    return results


if __name__ == "__main__":
    # Test the market scanner
    asyncio.run(execute_market_scan(top_n=10, min_pe=5, max_pe=30))