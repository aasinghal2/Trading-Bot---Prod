"""
AI-Powered Multi-Agent Trading System

Main application entry point for the trading system.
Provides CLI interface and continuous trading capabilities.
"""

import asyncio
import argparse
import json
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
import os

from loguru import logger

# Load environment variables from .env file
load_dotenv()

from orchestrator import TradingOrchestrator
from market_scanner import execute_market_scan


async def main():
    """Main function to run the trading system."""
    
    parser = argparse.ArgumentParser(description="AI-Powered Multi-Agent Trading System")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "auto", "scan"], help="Run mode: single cycle, auto-trading, or market scan")
    parser.add_argument("--symbols", nargs="*", help="List of stock symbols to analyze")
    parser.add_argument("--interval", type=int, default=300, help="Interval in seconds for auto-trading")
    parser.add_argument("--output", type=str, help="Output file for backtest results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--classic", action="store_true", help="Classic clean output (suppress intermediate logs)")
    parser.add_argument("--clear-portfolio", action="store_true", help="Clear the persisted portfolio state before running.")
    
    # Market scanner specific arguments
    parser.add_argument("--top", type=int, default=10, help="Number of top opportunities to show (scan mode)")
    parser.add_argument("--min-pe", type=float, default=0, help="Minimum P/E ratio filter (scan mode)")
    parser.add_argument("--max-pe", type=float, default=50, help="Maximum P/E ratio filter (scan mode)")
    
    args = parser.parse_args()
    
    # Handle portfolio clearing
    if args.clear_portfolio:
        clear_portfolio_state()
        # Exit after clearing if no other action is specified.
        # You could also allow it to continue to a trading run.
        if args.mode == "single" and not args.symbols:
             return

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    # Always keep a file sink
    logger.add(
        "logs/trading_system.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="7 days"
    )
    # In classic mode, suppress stdout sink to avoid noisy intermediate logs
    if not args.classic:
        logger.add(
            lambda msg: print(msg, end=""),
            level=log_level,
            format="{time:HH:mm:ss} | {level} | {message}"
        )

    # Optionally suppress third-party warnings for a cleaner classic view
    if args.classic:
        import warnings as _warnings
        _warnings.filterwarnings("ignore")
    
    logger.info("🚀 Starting AI-Powered Multi-Agent Trading System")
    
    try:
        # Initialize orchestrator
        orchestrator = TradingOrchestrator(args.config)
        
        # Execute based on mode
        if args.mode == 'single':
            await execute_single_cycle(orchestrator, args.symbols, args.output)
        elif args.mode == 'auto':
            await run_auto_trader(orchestrator, args.symbols, args.interval)
        elif args.mode == 'scan':
            await execute_market_scan(args.top, args.min_pe, args.max_pe)
        else:
            logger.error(f"❌ Invalid mode: {args.mode}. Please choose 'single', 'auto', or 'scan'.")
        
    except KeyboardInterrupt:
        logger.info("👋 Trading system stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise
    
    logger.info("🏁 Trading system shutdown complete")


async def execute_single_cycle(orchestrator: TradingOrchestrator, symbols: Optional[List[str]], output_file: Optional[str]):
    """Execute a single trading cycle."""
    logger.info("🚀 Starting single trading cycle...")
    
    # Early exit if no symbols provided - prevents unnecessary analysis agent initialization
    if not symbols:
        logger.info("📊 No symbols provided - skipping trading cycle analysis")
        logger.info("✅ Trading cycle completed (no analysis needed)")
        return
    
    logger.info("📊 Executing single trading cycle")
    
    # Execute trading cycle
    result = await orchestrator.execute_trading_cycle(symbols)
    
    # Display results
    display_cycle_results(result)
    
    # Save results if output file specified
    if output_file:
        save_results_to_file(result, output_file)
        logger.info(f"💾 Backtest results saved to {output_file}")



async def execute_continuous_trading(orchestrator: TradingOrchestrator,
                                   symbols: Optional[List[str]],
                                   interval: int,
                                   max_cycles: int,
                                   output_file: Optional[str]):
    """Execute continuous trading cycles."""
    
    logger.info(f"🔄 Starting continuous trading: {max_cycles} cycles, {interval}s interval")
    
    all_results = []
    
    for cycle in range(max_cycles):
        logger.info(f"📊 Executing cycle {cycle + 1}/{max_cycles}")
        
        try:
            # Execute trading cycle
            result = await orchestrator.execute_trading_cycle(symbols)
            all_results.append(result)
            
            # Display brief results
            display_brief_results(result, cycle + 1)
            
            # Wait for next cycle (except on last iteration)
            if cycle < max_cycles - 1:
                logger.info(f"⏳ Waiting {interval} seconds until next cycle...")
                await asyncio.sleep(interval)
                
        except Exception as e:
            logger.error(f"❌ Error in cycle {cycle + 1}: {e}")
            continue
    
    # Display summary
    display_continuous_summary(all_results)
    
    # Save results if output file specified
    if output_file:
        save_results_to_file({"cycles": all_results, "summary": get_summary_stats(all_results)}, output_file)
        logger.info(f"💾 Results saved to {output_file}")


async def execute_backtest(orchestrator: TradingOrchestrator,
                         symbols: Optional[List[str]],
                         start_date: Optional[str],
                         end_date: Optional[str],
                         output_file: Optional[str]):
    """Execute backtesting."""
    
    if not start_date or not end_date:
        logger.error("❌ Backtest requires --backtest-start and --backtest-end dates")
        return
    
    logger.info(f"📈 Starting backtest: {start_date} to {end_date}")
    
    # Execute backtest
    result = await orchestrator.execute_backtest(start_date, end_date, symbols)
    
    # Display results
    display_backtest_results(result)
    
    # Save results if output file specified
    if output_file:
        save_results_to_file(result, output_file)
        logger.info(f"💾 Backtest results saved to {output_file}")


def clear_portfolio_state():
    """Deletes the portfolio state file to reset the system."""
    state_file = "data/portfolio_state.json"
    try:
        if os.path.exists(state_file):
            os.remove(state_file)
            logger.info(f"✅ Portfolio state file '{state_file}' has been cleared.")
        else:
            logger.info("ℹ️ No portfolio state file to clear.")
    except Exception as e:
        logger.error(f"❌ Error clearing portfolio state file: {e}")


def display_cycle_results(result: dict):
    """Display detailed results from a single trading cycle."""
    
    print("\n" + "="*80)
    print("🎯 TRADING CYCLE RESULTS")
    print("="*80)
    
    print(f"📅 Execution ID: {result['execution_id']}")
    print(f"🕐 Timestamp: {result['timestamp']}")
    print(f"📊 Symbols: {', '.join(result['symbols'])}")
    print(f"⏱️  Execution Time: {result['execution_time']:.2f}s")
    print(f"✅ Success: {'Yes' if result['success'] else 'No'}")
    
    if result['errors']:
        print(f"❌ Errors: {'; '.join(result['errors'])}")
    
    print(f"\n🤖 Agents Executed: {len(result['agents_executed'])}")
    for agent in result['agents_executed']:
        print(f"   • {agent}")
    
    # Display agent results
    if result['success'] and result['results']:
        print(f"\n📈 AGENT RESULTS:")
        
        # Market Data
        if 'market_data' in result['results']:
            market_result = result['results']['market_data']
            if market_result and market_result.success:
                metrics = market_result.metrics
                print(f"   📊 Market Data: {metrics.get('symbols_processed', 0)} symbols, "
                      f"{metrics.get('data_quality_score', 0):.2f} quality score")
        
        # Analysis Results
        if 'analysis' in result['results']:
            analysis = result['results']['analysis']
            for agent_name, agent_result in analysis.items():
                if agent_result and agent_result.success:
                    metrics = agent_result.metrics
                    if agent_name == 'technical_analyst':
                        print(f"   🔍 Technical: {metrics.get('symbols_analyzed', 0)} symbols, "
                              f"{metrics.get('bullish_signals', 0)} bullish, "
                              f"{metrics.get('bearish_signals', 0)} bearish")
                    elif agent_name == 'sentiment_analyst':
                        print(f"   🗞️  Sentiment: {metrics.get('total_articles_processed', 0)} articles, "
                              f"{metrics.get('average_sentiment', 0):.2f} avg sentiment")
                    elif agent_name == 'fundamentals_analyst':
                        print(f"   💰 Fundamentals: {metrics.get('companies_with_data', 0)} companies analyzed, "
                              f"{metrics.get('undervalued_count', 0)} undervalued")
        
        # Risk Management (pre-trade if available)
        def _print_risk_block(risk_result, label_prefix: str = "⚠️  Risk"):
            if risk_result and getattr(risk_result, 'success', False):
                metrics = risk_result.metrics
                print(f"   {label_prefix}: Score {metrics.get('risk_score', 0):.1f}/100, "
                      f"{metrics.get('trades_approved', 0)} approved, "
                      f"{metrics.get('trades_rejected', 0)} rejected")
                try:
                    data = risk_result.data if isinstance(risk_result.data, dict) else {}
                    pr = data.get('portfolio_risk', {}) if isinstance(data.get('portfolio_risk', {}), dict) else {}
                    limits = data.get('risk_limits_status', {}) if isinstance(data.get('risk_limits_status', {}), dict) else {}
                    var95 = pr.get('var_95', 0)
                    cvar95 = pr.get('cvar_95', 0)
                    lev = pr.get('leverage_ratio', 0)
                    conc = pr.get('concentration_risk', 0)
                    vios = len(limits.get('violations', [])) if isinstance(limits.get('violations', []), list) else 0
                    warns = len(limits.get('warnings', [])) if isinstance(limits.get('warnings', []), list) else 0
                    print(f"      · VaR95: {var95:.4f}, CVaR95: {cvar95:.4f}, Leverage: {lev:.2f}x, Concentration: {conc:.2f}")
                    if vios or warns:
                        print(f"      · Limits: {limits.get('overall_status', 'n/a')} | Violations: {vios}, Warnings: {warns}")
                except Exception:
                    pass

        pre_key = 'risk_management_pre'
        post_key = 'risk_management_post'
        has_pre = pre_key in result['results'] and result['results'][pre_key]
        has_post = post_key in result['results'] and result['results'][post_key]

        if has_pre:
            _print_risk_block(result['results'][pre_key], label_prefix="⚠️  Risk (Pre-Trade)")
        elif 'risk_management' in result['results'] and not has_post:
            # Fallback to single risk block if no pre/post split exists
            _print_risk_block(result['results']['risk_management'])
        
        # Portfolio Management (prints between pre and post risk)
        if 'portfolio_management' in result['results']:
            portfolio_result = result['results']['portfolio_management']
            if portfolio_result and portfolio_result.success:
                metrics = portfolio_result.metrics
                print(f"   💼 Portfolio: ${metrics.get('total_portfolio_value', 0):,.2f} value, "
                      f"{metrics.get('active_positions', 0)} positions, "
                      f"{metrics.get('trades_executed', 0)} trades executed")

                # Classic output enhancement: show executed trades and cash remaining
                try:
                    pdata = portfolio_result.data if isinstance(portfolio_result.data, dict) else {}
                    orders = pdata.get('orders_executed', []) if isinstance(pdata.get('orders_executed', []), list) else []
                    if orders:
                        print("   🧾 Executions:")
                        total_bought_shares = 0.0
                        for o in orders:
                            try:
                                symbol = o.get('symbol', 'UNKNOWN')
                                size = float(o.get('size', 0) or 0)
                                filled_size = float(o.get('filled_size', abs(size)) or 0)
                                filled_price = float(o.get('filled_price', o.get('price', 0)) or 0)
                                side = 'BUY' if size > 0 else 'SELL'
                                print(f"      · {side} {filled_size:.2f} {symbol} @ ${filled_price:,.2f}")
                                if size > 0:
                                    total_bought_shares += filled_size
                            except Exception:
                                continue

                        # Cash remaining
                        cash_remaining = None
                        # Prefer metrics if provided
                        if isinstance(metrics, dict) and 'cash_balance' in metrics:
                            cash_remaining = metrics.get('cash_balance')
                        else:
                            # Fallback to portfolio summary
                            summary = pdata.get('portfolio_summary', {}) if isinstance(pdata.get('portfolio_summary', {}), dict) else {}
                            cash_remaining = summary.get('cash_balance')

                        if total_bought_shares > 0:
                            print(f"   📦 Total Shares Bought (this cycle): {total_bought_shares:.2f}")
                        if cash_remaining is not None:
                            print(f"   💵 Cash Remaining: ${cash_remaining:,.2f}")
                except Exception:
                    pass

        # Risk Management (post-trade if available)
        if has_post:
            _print_risk_block(result['results'][post_key], label_prefix="⚠️  Risk (Post-Trade)")
    
    print("="*80)


def display_brief_results(result: dict, cycle_num: int):
    """Display brief results from a trading cycle."""
    
    status = "✅" if result['success'] else "❌"
    exec_time = result['execution_time']
    
    brief_info = f"{status} Cycle {cycle_num}: {exec_time:.1f}s"
    
    if result['success'] and result['results']:
        # Add key metrics
        portfolio_result = result['results'].get('portfolio_management')
        if portfolio_result and portfolio_result.success:
            portfolio_value = portfolio_result.metrics.get('total_portfolio_value', 0)
            trades_executed = portfolio_result.metrics.get('trades_executed', 0)
            brief_info += f" | Portfolio: ${portfolio_value:,.0f} | Trades: {trades_executed}"
    
    print(brief_info)


def display_continuous_summary(results: list):
    """Display summary of continuous trading results."""
    
    print("\n" + "="*80)
    print("📊 CONTINUOUS TRADING SUMMARY")
    print("="*80)
    
    successful_cycles = [r for r in results if r['success']]
    
    print(f"🔄 Total Cycles: {len(results)}")
    print(f"✅ Successful: {len(successful_cycles)} ({len(successful_cycles)/len(results)*100:.1f}%)")
    print(f"⏱️  Avg Execution Time: {sum(r['execution_time'] for r in results)/len(results):.2f}s")
    
    # Portfolio performance
    portfolio_values = []
    for result in successful_cycles:
        portfolio_result = result['results'].get('portfolio_management')
        if portfolio_result and portfolio_result.success:
            value = portfolio_result.metrics.get('total_portfolio_value', 0)
            if value > 0:
                portfolio_values.append(value)
    
    if portfolio_values:
        print(f"💼 Portfolio Range: ${min(portfolio_values):,.2f} - ${max(portfolio_values):,.2f}")
        if len(portfolio_values) > 1:
            performance = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
            print(f"📈 Performance: {performance:+.2f}%")
    
    print("="*80)


def display_backtest_results(result: dict):
    """Display backtest results."""
    
    print("\n" + "="*80)
    print("📈 BACKTEST RESULTS")
    print("="*80)
    
    if not result or "total_return" not in result:
        print("Backtest did not produce valid results. No trades may have been executed.")
        print("="*80)
        return

    print(f"📅 Period: {result.get('start_date')} to {result.get('end_date')}")
    print(f"📊 Symbols: {', '.join(result.get('symbols', []))}")
    print(f"🔄 Total Trading Days: {result.get('total_cycles')}")
    print(f"💰 Initial Portfolio Value: ${result.get('initial_portfolio_value', 0):,.2f}")
    print(f"🏁 Final Portfolio Value: ${result.get('final_portfolio_value', 0):,.2f}")
    print(f"📈 Total Return: {result.get('total_return', 0):.2%}")
    print(f"📊 Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
    print(f"📉 Max Drawdown: {result.get('max_drawdown', 0):.2%}")
    print(f"🎯 Win Rate: {result.get('win_rate', 0):.2%}")
    print(f" trades executed: {result.get('total_trades', 0)}")
    
    print("="*80)



def get_summary_stats(results: list) -> dict:
    """Calculate summary statistics from multiple results."""
    
    successful_results = [r for r in results if r['success']]
    
    return {
        "total_cycles": len(results),
        "successful_cycles": len(successful_results),
        "success_rate": len(successful_results) / len(results) if results else 0,
        "avg_execution_time": sum(r['execution_time'] for r in results) / len(results) if results else 0,
        "total_errors": sum(len(r['errors']) for r in results)
    }


def save_results_to_file(results: dict, filename: str):
    """Save analysis results to a JSON file."""
    try:
        # Attach risk explanations to help understand decisions
        def build_risk_explanation(cycle_result: dict) -> dict:
            out = {}
            try:
                res = cycle_result.get('results', {}) if isinstance(cycle_result, dict) else {}
                risk = res.get('risk_management')
                if not risk or not getattr(risk, 'success', False):
                    return out
                data = risk.data if isinstance(risk.data, dict) else {}
                metrics = risk.metrics if isinstance(risk.metrics, dict) else {}
                portfolio = data.get('portfolio_risk', {}) if isinstance(data.get('portfolio_risk', {}), dict) else {}
                limits = data.get('risk_limits_status', {}) if isinstance(data.get('risk_limits_status', {}), dict) else {}
                approvals = data.get('trade_approvals', {})
                if isinstance(approvals, dict):
                    approvals_list = [
                        {
                            'trade_id': k,
                            'approved': v.get('approved'),
                            'risk_score': v.get('risk_score'),
                            'risk_checks': v.get('risk_checks'),
                            'original_size': v.get('original_size'),
                            'adjusted_size': v.get('adjusted_size'),
                            'position_weight': v.get('position_weight'),
                            'trade_value': v.get('trade_value'),
                            'signal_strength': v.get('signal_strength'),
                            'reason': v.get('reason'),
                        }
                        for k, v in approvals.items() if isinstance(v, dict)
                    ]
                else:
                    approvals_list = []
                pos_risks = data.get('position_risks', {}) if isinstance(data.get('position_risks', {}), dict) else {}
                top_positions = []
                try:
                    top_positions = sorted(
                        [
                            {
                                'symbol': s,
                                'risk_score': r.get('risk_score'),
                                'position_weight': r.get('position_weight'),
                                'volatility': r.get('volatility'),
                                'unrealized_pnl_pct': r.get('unrealized_pnl_pct'),
                            }
                            for s, r in pos_risks.items() if isinstance(r, dict)
                        ],
                        key=lambda x: (x.get('risk_score') or 0),
                        reverse=True,
                    )[:5]
                except Exception:
                    top_positions = []
                approved_count = sum(1 for a in approvals_list if a.get('approved'))
                rejected_count = sum(1 for a in approvals_list if a.get('approved') is False)
                adjusted_count = sum(1 for a in approvals_list if a.get('adjusted_size') not in (None, a.get('original_size')))
                out = {
                    'portfolio_risk': portfolio,
                    'risk_limits': limits,
                    'approvals': approvals_list,
                    'position_risks_top': top_positions,
                    'decision_summary': {
                        'risk_score': metrics.get('risk_score'),
                        'trades_approved': approved_count,
                        'trades_rejected': rejected_count,
                        'trades_adjusted': adjusted_count,
                    },
                }
            except Exception:
                pass
            return out

        def attach_risk_explanations(obj: dict) -> dict:
            if not isinstance(obj, dict):
                return obj
            # Single cycle result
            if 'results' in obj:
                obj['risk_explanation'] = build_risk_explanation(obj)
            # Continuous results
            if 'cycles' in obj and isinstance(obj['cycles'], list):
                for i, cyc in enumerate(obj['cycles']):
                    if isinstance(cyc, dict):
                        obj['cycles'][i]['risk_explanation'] = build_risk_explanation(cyc)
            return obj

        results = attach_risk_explanations(results)

        # Convert datetime objects to strings for JSON serialization
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj
        
        serializable_results = convert_datetime(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
            
    except Exception as e:
        logger.error(f"Error saving results to file: {e}")


if __name__ == "__main__":
    # Run the main application
    asyncio.run(main())