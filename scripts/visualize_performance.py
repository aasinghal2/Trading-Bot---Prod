#!/usr/bin/env python3
"""
Performance Visualization Script

Generate visual charts and analytics for your trading bot performance.
Includes equity curves, drawdown charts, and performance metrics dashboard.

Usage:
    python scripts/visualize_performance.py
    python scripts/visualize_performance.py --output charts/
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.portfolio_manager_agent import PortfolioManagerAgent
import yaml

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Warning: matplotlib not installed. Install it with: pip install matplotlib")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  Warning: pandas not installed. Install it with: pip install pandas")


def load_portfolio_manager():
    """Load the portfolio manager with current state."""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    portfolio_config = config.get('agents', {}).get('portfolio_manager', {})
    return PortfolioManagerAgent(portfolio_config)


def print_text_summary(pm: PortfolioManagerAgent):
    """Print a text-based performance summary (no visualization required)."""
    print("\n" + "="*60)
    print("📊 TRADING BOT PERFORMANCE REPORT")
    print("="*60)
    
    # Get performance summary
    summary = pm.get_performance_summary()
    print(summary)
    
    print("\n" + "="*60)
    print("💼 CURRENT PORTFOLIO STATUS")
    print("="*60)
    
    portfolio_summary = pm._get_portfolio_summary()
    total_value = portfolio_summary['total_value']
    total_return_pct = ((total_value - pm.initial_capital) / pm.initial_capital) * 100
    
    print(f"\n💰 Cash Balance: ${portfolio_summary['cash_balance']:,.2f}")
    print(f"📊 Portfolio Market Value: ${portfolio_summary['total_market_value']:,.2f}")
    print(f"🏦 Total Account Value: ${total_value:,.2f}")
    print(f"📈 Total Return: {total_return_pct:+.2f}%")
    print(f"📊 Number of Positions: {len(portfolio_summary['positions'])}")
    
    if portfolio_summary['positions']:
        print("\n📈 CURRENT POSITIONS:")
        print("-" * 60)
        for pos in portfolio_summary['positions']:
            pnl_emoji = "🟢" if pos['unrealized_pnl'] >= 0 else "🔴"
            print(f"  {pnl_emoji} {pos['symbol']}: {pos['size']:.2f} shares @ ${pos['current_price']:.2f}")
            print(f"     Entry: ${pos['entry_price']:.2f} | P&L: ${pos['unrealized_pnl']:+,.2f} ({pos['unrealized_pnl_pct']:+.2f}%)")
    
    # Trade history
    if pm.trade_history:
        print(f"\n📊 TRADE HISTORY: {len(pm.trade_history)} total trades")
        recent_trades = pm.trade_history[-5:] if len(pm.trade_history) > 5 else pm.trade_history
        print("  Most recent 5 trades:")
        for trade in reversed(recent_trades):
            timestamp = trade.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            print(f"  • {timestamp.strftime('%Y-%m-%d %H:%M')} - {trade.get('symbol', 'N/A')}: "
                  f"{trade.get('size', 0):.2f} @ ${trade.get('price', 0):.2f}")


def create_visualizations(pm: PortfolioManagerAgent, output_dir: str = "charts"):
    """Create performance visualization charts."""
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Cannot create visualizations: matplotlib not installed")
        print("   Install with: pip install matplotlib")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get performance data
    perf_data = pm.export_performance_data()
    
    if not perf_data['portfolio_values']:
        print("⚠️  Not enough data for visualizations yet. Trade for a few days first!")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    if perf_data['portfolio_values']:
        values = perf_data['portfolio_values']
        dates = [datetime.fromisoformat(v['date']) if isinstance(v['date'], str) else v['date'] for v in values]
        portfolio_values = [v['value'] for v in values]
        
        ax1.plot(dates, portfolio_values, linewidth=2, color='#2E7D32', label='Portfolio Value')
        ax1.axhline(y=pm.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('📈 Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Drawdown Chart
    ax2 = fig.add_subplot(gs[1, :])
    if perf_data['drawdown_history']:
        dd_data = perf_data['drawdown_history']
        dates = [datetime.fromisoformat(d['date']) if isinstance(d['date'], str) else d['date'] for d in dd_data]
        drawdowns = [d['drawdown'] * 100 for d in dd_data]  # Convert to percentage
        
        ax2.fill_between(dates, drawdowns, 0, color='#D32F2F', alpha=0.3)
        ax2.plot(dates, drawdowns, linewidth=2, color='#D32F2F')
        ax2.set_title('📉 Drawdown Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Returns Distribution
    ax3 = fig.add_subplot(gs[2, 0])
    if perf_data['returns']:
        returns = [r * 100 for r in perf_data['returns']]  # Convert to percentage
        ax3.hist(returns, bins=30, color='#1976D2', alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('📊 Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Daily Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
    
    # 4. Performance Metrics Table
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    metrics = perf_data['metrics']
    
    # Create metrics table
    metrics_text = f"""
    📊 PERFORMANCE METRICS
    {'='*40}
    
    ⚡ Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f}
    🎯 Sortino Ratio:       {metrics.get('sortino_ratio', 0):.2f}
    🏆 Calmar Ratio:        {metrics.get('calmar_ratio', 0):.2f}
    
    📉 Max Drawdown:        {metrics.get('max_drawdown', 0)*100:.2f}%
    🌪️  Volatility:          {metrics.get('volatility', 0)*100:.2f}%
    ⚠️  VaR (5%):            {metrics.get('var_95', 0)*100:.2f}%
    
    🎯 Win Rate:            {metrics.get('win_rate', 0)*100:.1f}%
    💹 Profit Factor:       {metrics.get('profit_factor', 0):.2f}
    📈 Total Trades:        {metrics.get('total_trades', 0)}
    
    💰 Total Return:        {metrics.get('total_return', 0)*100:+.2f}%
    📈 Annualized Return:   {metrics.get('annualized_return', 0)*100:+.2f}%
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save the figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'performance_report_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Performance chart saved to: {output_file}")
    
    # Also save a latest version
    latest_file = os.path.join(output_dir, 'performance_latest.png')
    plt.savefig(latest_file, dpi=300, bbox_inches='tight')
    print(f"✅ Latest chart saved to: {latest_file}")
    
    plt.close()


def export_to_csv(pm: PortfolioManagerAgent, output_dir: str = "charts"):
    """Export performance data to CSV files."""
    
    if not PANDAS_AVAILABLE:
        print("⚠️  pandas not installed, exporting to JSON instead")
        export_to_json(pm, output_dir)
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    perf_data = pm.export_performance_data()
    
    # Export portfolio values
    if perf_data['portfolio_values']:
        df_values = pd.DataFrame(perf_data['portfolio_values'])
        values_file = os.path.join(output_dir, 'portfolio_values.csv')
        df_values.to_csv(values_file, index=False)
        print(f"✅ Portfolio values exported to: {values_file}")
    
    # Export drawdowns
    if perf_data['drawdown_history']:
        df_dd = pd.DataFrame(perf_data['drawdown_history'])
        dd_file = os.path.join(output_dir, 'drawdown_history.csv')
        df_dd.to_csv(dd_file, index=False)
        print(f"✅ Drawdown history exported to: {dd_file}")
    
    # Export metrics
    metrics_file = os.path.join(output_dir, 'performance_metrics.csv')
    df_metrics = pd.DataFrame([perf_data['metrics']])
    df_metrics.to_csv(metrics_file, index=False)
    print(f"✅ Performance metrics exported to: {metrics_file}")


def export_to_json(pm: PortfolioManagerAgent, output_dir: str = "charts"):
    """Export performance data to JSON files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    perf_data = pm.export_performance_data()
    
    # Convert datetime objects to strings for JSON serialization
    def serialize_dates(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: serialize_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_dates(item) for item in obj]
        return obj
    
    perf_data = serialize_dates(perf_data)
    
    output_file = os.path.join(output_dir, 'performance_data.json')
    with open(output_file, 'w') as f:
        json.dump(perf_data, f, indent=2)
    
    print(f"✅ Performance data exported to: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize trading bot performance')
    parser.add_argument('--output', '-o', default='charts', help='Output directory for charts')
    parser.add_argument('--export-csv', action='store_true', help='Export data to CSV files')
    parser.add_argument('--export-json', action='store_true', help='Export data to JSON files')
    parser.add_argument('--no-charts', action='store_true', help='Skip chart generation')
    
    args = parser.parse_args()
    
    print("\n🚀 Loading trading bot performance data...")
    
    try:
        pm = load_portfolio_manager()
        
        # Always show text summary
        print_text_summary(pm)
        
        # Create visualizations if requested
        if not args.no_charts:
            print("\n📊 Generating performance charts...")
            create_visualizations(pm, args.output)
        
        # Export data if requested
        if args.export_csv:
            print("\n💾 Exporting data to CSV...")
            export_to_csv(pm, args.output)
        
        if args.export_json:
            print("\n💾 Exporting data to JSON...")
            export_to_json(pm, args.output)
        
        print("\n✅ Performance analysis complete!")
        
        if not MATPLOTLIB_AVAILABLE and not args.no_charts:
            print("\n💡 Tip: Install matplotlib to see visual charts:")
            print("   pip install matplotlib pandas")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())