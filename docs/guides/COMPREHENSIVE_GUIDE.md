# AI-Powered Multi-Agent Trading System - Comprehensive Guide

*Last Updated: August 2025 - Current System Status*

## Table of Contents
1. [System Overview](#system-overview)
2. [Installation & Setup](#installation--setup)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Trading Modes & Commands](#trading-modes--commands)
6. [Market Scanner](#market-scanner)
7. [Dynamic Thresholds](#dynamic-thresholds)
8. [Portfolio Management](#portfolio-management)
9. [Risk Management](#risk-management)
10. [Social Media Integration](#social-media-integration)
11. [Advanced Features](#advanced-features)
12. [Troubleshooting](#troubleshooting)

## System Overview

The AI-Powered Multi-Agent Trading System is a sophisticated trading platform that uses multiple specialized AI agents to analyze markets and make trading decisions. The system operates with a **mock portfolio** using real market data for safe testing and development.

### 🤖 **Agent Architecture**
Each agent focuses on a specific aspect of market analysis:

- **Market Data Agent**: Real-time market data collection via Yahoo Finance
- **Technical Analyst Agent**: Technical analysis with 15+ indicators (RSI, MACD, Bollinger Bands, etc.)
- **Fundamentals Agent**: Company financial analysis and valuation metrics
- **Sentiment Agent**: News and social media sentiment analysis (Reddit integration)
- **Risk Manager Agent**: Risk assessment, position sizing, and portfolio protection
- **Portfolio Manager Agent**: Trade execution, position tracking, and portfolio optimization

### 🎯 **Current System Status (August 2025)**
- ✅ **Fully Operational**: All agents working correctly
- ✅ **Portfolio Tracking**: Persistent state management
- ✅ **Risk Management**: Multi-layer protection systems
- ✅ **Dynamic Thresholds**: Market-aware adaptive signal thresholds
- ✅ **Market Scanner**: S&P 500 opportunity discovery engine
- ✅ **Multi-Symbol Support**: Analyze unlimited symbols simultaneously
- ✅ **Hybrid Position Sizing**: Signal-based and optimizer-based allocation methods
- ✅ **Market Metrics**: SPY, VIX, and sector breadth integration

### 🚀 **New Features (August 2025)**
- **Market Scanner**: Discover opportunities across S&P 500 with P/E filtering
- **Dynamic Thresholds**: Hybrid market performance + signal history adaptation
- **Market Metrics**: Real-time SPY returns, VIX volatility, sector breadth analysis
- **Recommendation Engine**: Save scan results to structured JSON files
- **Signal History Tracking**: 60-day rolling signal database for trend analysis

## Installation & Setup

1. **Clone the Repository**
```bash
git clone <repository-url>
cd TradingBot
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Required: Set Up Environment Variables**
```bash
# Copy template and configure APIs (required for full functionality)
cp env_template.txt .env

# Required API keys for production use:
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key      # Market data
# REDDIT_CLIENT_ID=your_reddit_client_id            # Sentiment analysis  
# REDDIT_CLIENT_SECRET=your_reddit_client_secret    # Sentiment analysis
# REDDIT_USER_AGENT=TradingBot/1.0 (by u/username)  # Reddit identification
# EMAIL_USERNAME=your_email@gmail.com               # Trade notifications
# EMAIL_PASSWORD=your_gmail_app_password            # Trade notifications
```

5. **Verify Installation**
```bash
python3 main.py --help
```

## Configuration

The system uses `config/config.yaml` for all configuration settings:

### **Core Settings**
```yaml
# Market Data
market_data:
  symbols: ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
  historical_lookback_days: 252

# Trading Configuration
trading:
  signal_thresholds:
    minimum_strength: 0.25           # Static fallback threshold
    dynamic:
      enabled: true                  # Enable hybrid dynamic thresholds
      market_weight: 0.6            # 60% market metrics weight
      signal_weight: 0.4            # 40% signal history weight
      static_blend: 0.2             # 20% static baseline blend
      floor_threshold: 0.15         # Minimum threshold (safety)
      ceiling_threshold: 0.40       # Maximum threshold (safety)
  
  position_sizing:
    base_position_value: 10000      # Base position size ($)
    max_position_value: 15000       # Maximum position size ($)

# Risk Management
risk_manager:
  max_position_size: 0.1            # 10% max per position
  max_portfolio_leverage: 2.0       # 2x max leverage
```

### **Market Scanner Settings**
The scanner uses the first 50 S&P 500 symbols with configurable P/E filtering:
- Scans across major market cap stocks
- Filters by P/E ratios, market cap thresholds
- Saves recommendations to `recommendations/` folder

## Running the System

### **Basic Commands**

#### **1. Single Analysis Cycle**
```bash
# Analyze specific symbols
python3 main.py --mode single --symbols AAPL GOOGL MSFT --classic

# Clear portfolio first, then analyze
python3 main.py --clear-portfolio --mode single --symbols TSLA NVDA --classic
```

#### **2. Market Scanner (NEW)**
```bash
# Scan S&P 500 for opportunities
python3 main.py --mode scan --classic

# Custom P/E filtering
python3 main.py --mode scan --top 10 --min-pe 5 --max-pe 25 --classic

# Conservative value scanning
python3 main.py --mode scan --top 5 --min-pe 5 --max-pe 15 --classic
```

#### **3. Auto-Trading Mode**
```bash
# Continuous trading (5-minute intervals)
python3 main.py --mode auto --symbols AAPL GOOGL MSFT --interval 300 --classic

# Faster intervals for active trading
python3 main.py --mode auto --symbols META VZ HD --interval 180 --classic
```

### **Command Arguments**
- `--mode`: `single`, `auto`, or `scan`
- `--symbols`: List of stock symbols to analyze (not needed for scan mode)
- `--top`: Number of opportunities to show (scan mode only)
- `--min-pe`, `--max-pe`: P/E ratio filters (scan mode only)
- `--interval`: Seconds between cycles (auto mode only)
- `--classic`: Clean output format (recommended)
- `--clear-portfolio`: Reset portfolio state before running

## Market Scanner

### **Overview**
The Market Scanner is a powerful new feature that discovers trading opportunities across the S&P 500:

```bash
python3 main.py --mode scan --classic
```

### **Features**
- **S&P 500 Coverage**: Analyzes 50 major stocks automatically
- **P/E Filtering**: Screens by valuation metrics
- **Signal Ranking**: Orders opportunities by strength
- **Recommendation Export**: Saves detailed analysis to JSON files

### **Scanner Output**
```
🔍 S&P 500 MARKET SCANNER RESULTS
📊 SCAN SUMMARY:
   ⏱️  Scan Time: 156.60 seconds
   🎯 Symbols Scanned: 50
   ✅ Passed Filters: 30
   📈 Successfully Analyzed: 30

🏆 TOP OPPORTUNITIES:
Rank Symbol Signal   Direction Price      P/E View    
------------------------------------------------------------
1    GOOGL  0.407    BULLISH  $201.42    Unknown     
2    VZ     0.353    BULLISH  $43.15     Unknown     
3    META   0.328    BULLISH  $769.30    Unknown
```

### **Recommendation Files**
Scanner results are automatically saved to `recommendations/market_scan_YYYYMMDD_HHMMSS.json`:

```json
{
  "scan_timestamp": "2025-08-13T17:45:19.923885",
  "buy_recommendations": [
    {
      "rank": 1,
      "symbol": "GOOGL",
      "signal_strength": 0.407,
      "technical_direction": "bullish",
      "recommendation_reason": "Very Strong signal (0.407) driven by fundamental analysis"
    }
  ]
}
```

### **Scanner Workflow**
1. **Fundamental Filtering**: P/E ratios, market cap thresholds
2. **Technical Analysis**: Full indicator suite on filtered stocks
3. **Signal Ranking**: Sort by combined signal strength
4. **Export Results**: Structured recommendations for review

## Dynamic Thresholds

### **Overview**
The system now uses **Hybrid Dynamic Thresholds** that adapt to market conditions:

```
Final Threshold = 60% Market Metrics + 40% Signal History + 20% Static Baseline
```

### **Market Metrics Integration**
- **SPY Returns**: 1-day, 5-day, 30-day S&P 500 performance
- **VIX Volatility**: Current level and trend analysis
- **Market Breadth**: Sector rotation strength (8 major sectors)
- **Regime Detection**: Bull/bear/sideways/volatile classification

### **Threshold Adaptation Examples**

#### **Bull Market Scenario**
- **SPY 30-day return**: +8%
- **VIX Level**: 12 (low volatility)
- **Market Breadth**: 75% sectors positive
- **Result**: Threshold raises to ~0.35 (selective)

#### **Bear Market Scenario**
- **SPY 30-day return**: -8%
- **VIX Level**: 32 (high volatility)
- **Market Breadth**: 25% sectors positive
- **Result**: Threshold lowers to ~0.18 (opportunity capture)

### **Threshold Explanations**
The system provides clear explanations for threshold decisions:
```
Using threshold: Hybrid: 0.294 (market strong, signals neutral)
Using threshold: Hybrid: 0.188 (market weak, signals weak) floor-limited
```

### **Signal History Tracking**
- **Automatic**: Every cycle stores signals in `data/signal_history.json`
- **Rolling Window**: Maintains 60 days of signal data
- **Non-Zero Focus**: Filters out failed analysis signals
- **Percentile Calculation**: Uses 80th percentile for dynamic component

## Portfolio Management

### **Portfolio State**
- **File**: `data/portfolio_state.json`
- **Starting Capital**: $100,000 (configurable)
- **Position Tracking**: Real-time share counts and values
- **Trade History**: Complete execution log with timestamps

### **Position Sizing Methods**

#### **1. Signal-Based Sizing (Orchestrator)**
```yaml
position_sizing:
  base_position_value: 10000        # Base amount per trade
  max_position_value: 15000         # Maximum per position
  max_size_multiplier: 1.5          # Strong signal multiplier
```

- **Weak Signals** (0.25-0.4): Scale down proportionally
- **Strong Signals** (0.4+): Scale up to maximum multiplier
- **Value-Based**: Targets dollar amounts, not share counts

#### **2. Optimizer-Based Sizing (Portfolio Manager)**
```yaml
portfolio_manager:
  allocation_method: "optimizer"     # Enable mean-variance optimization
  total_investment_target: 0.8       # 80% of capital deployed
```

- **Mean-Variance Optimization**: Calculates optimal weights
- **Risk-Adjusted**: Considers correlations and volatility
- **Overrides**: Takes precedence over signal-based sizing

### **Portfolio Operations**
```bash
# Clear portfolio (reset to $100,000 cash)
python3 main.py --clear-portfolio

# Check current status
python3 main.py --mode single --symbols CURRENT_HOLDINGS --classic
```

## Risk Management

### **Multi-Layer Protection**

#### **1. Position Limits**
```yaml
risk_manager:
  max_position_size: 0.1            # 10% max per position
  max_portfolio_leverage: 2.0       # 2x leverage limit
  max_drawdown: 0.15               # 15% portfolio drawdown limit
```

#### **2. Risk Metrics**
- **VaR (Value at Risk)**: 95% confidence daily loss estimate
- **CVaR (Conditional VaR)**: Expected loss beyond VaR threshold
- **Portfolio Correlation**: Diversification analysis
- **Concentration Risk**: Single position exposure limits

#### **3. Dynamic Risk Assessment**
- **Pre-Trade Analysis**: Risk evaluation before execution
- **Post-Trade Recomputation**: Updated risk after trades
- **Violation Detection**: Automatic limit breach alerts

### **Risk Approval Process**
1. **Signal Generation**: Technical + fundamental + sentiment analysis
2. **Risk Evaluation**: Position sizing and portfolio impact
3. **Approval Gates**: Multiple checks before execution
4. **Execution Control**: Only approved trades proceed

## Social Media Integration

### **Reddit Integration (Fully Operational)**
- **Data Source**: Reddit financial subreddits
- **Technology**: Async PRAW (Python Reddit API Wrapper)
- **Sentiment Analysis**: RoBERTa-based NLP models
- **Real-Time**: Fresh posts analyzed each cycle

### **Setup (Required for Sentiment Analysis)**
```bash
# Set environment variables in .env file
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret 
REDDIT_USER_AGENT=TradingBot/1.0 (by u/username)
```

**How to get Reddit API credentials:**
1. Visit [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Click "Create App" → Select "script"
3. Get Client ID and Client Secret
4. Set user agent to identify your bot

### **News Sentiment**
- **Sources**: Financial news APIs and RSS feeds
- **Processing**: Multi-model sentiment analysis
- **Weight**: Time-decayed based on article freshness

## Advanced Features

### **1. Execution Modes**
- **Single**: One-time analysis and potential execution
- **Auto**: Continuous trading with configurable intervals
- **Scan**: Market opportunity discovery (no execution)

### **2. Signal Processing**
- **Combined Signals**: Multi-agent weighted combination
- **Threshold Adaptation**: Market-aware signal filtering
- **Portfolio Awareness**: Existing positions influence decisions

### **3. Data Persistence**
- **Portfolio State**: `data/portfolio_state.json`
- **Signal History**: `data/signal_history.json`
- **Market Metrics Cache**: `data/market_metrics_cache.json`
- **Recommendations**: `recommendations/market_scan_*.json`

### **4. Logging & Monitoring**
```bash
# View logs
tail -f logs/trading_system.log

# Trading-specific logs
grep "threshold\|signal\|trade" logs/trading_system.log

# Error tracking
grep "ERROR\|WARNING" logs/trading_system.log
```

### **5. Configuration Management**
- **YAML-based**: Human-readable configuration
- **Hot-reload**: Some settings update without restart
- **Validation**: Configuration sanity checks on startup

## Troubleshooting

### **Common Issues**

#### **1. No Trades Executed**
**Symptoms**: Analysis runs but no trades occur
**Causes**:
- Signals below dynamic threshold
- Risk manager rejections
- Insufficient cash

**Solutions**:
```bash
# Check recent threshold levels
grep "Using threshold" logs/trading_system.log | tail -5

# Review risk rejections
grep "Risk manager rejected" logs/trading_system.log | tail -5

# Clear portfolio if needed
python3 main.py --clear-portfolio
```

#### **2. Market Scanner Slow**
**Symptoms**: Scanner takes >5 minutes
**Causes**:
- Network connectivity issues
- API rate limiting
- Large symbol universe

**Solutions**:
```bash
# Reduce scan scope in market_scanner.py
# Check network connectivity
# Use --classic flag for cleaner output
```

#### **3. Dynamic Thresholds Not Working**
**Symptoms**: Always using static threshold
**Causes**:
- Insufficient signal history (<10 samples)
- Market data fetch failures
- Configuration disabled

**Solutions**:
```bash
# Build signal history
python3 main.py --mode scan --classic

# Check market metrics
grep "Market metrics" logs/trading_system.log

# Verify configuration
grep -A10 "dynamic:" config/config.yaml
```

#### **4. Memory/Performance Issues**
**Symptoms**: Slow execution, high memory usage
**Causes**:
- Large signal history files
- Unclosed network connections
- Model loading overhead

**Solutions**:
```bash
# Clean old signal history
# Restart Python process periodically
# Monitor with system tools
```

### **System Health Checks**

#### **1. Agent Status**
```bash
# Check agent initialization
grep "agent.*initialized" logs/trading_system.log | tail -6
```

#### **2. Data Quality**
```bash
# Verify market data
grep "Market Data.*quality score" logs/trading_system.log | tail -3
```

#### **3. Threshold Evolution**
```bash
# Track threshold changes
grep "threshold.*Hybrid" logs/trading_system.log | tail -10
```

### **Performance Optimization**

#### **1. Fast Analysis**
```bash
# Use classic mode for speed
python3 main.py --mode single --symbols AAPL --classic

# Reduce symbol count for testing
python3 main.py --mode single --symbols AAPL GOOGL --classic
```

#### **2. Scanner Optimization**
```bash
# Fewer top results
python3 main.py --mode scan --top 5 --classic

# Stricter P/E filtering
python3 main.py --mode scan --min-pe 10 --max-pe 20 --classic
```

## Support & Development

### **File Structure**
```
TradingBot/
├── agents/                 # AI agent implementations
├── core/                   # Core system components
│   ├── dynamic_thresholds.py
│   ├── market_metrics.py
│   └── signal_history.py
├── config/                 # Configuration files
├── data/                   # Persistent data storage
├── recommendations/        # Scanner output files
├── logs/                   # System logs
├── market_scanner.py       # S&P 500 scanner module
└── main.py                # Entry point
```

### **Recent Updates (August 2025)**
- ✅ Dynamic threshold system with market metrics
- ✅ S&P 500 market scanner with P/E filtering
- ✅ Signal history tracking and analysis
- ✅ Recommendation export system
- ✅ Hybrid market performance integration
- ✅ Enhanced portfolio awareness
- ✅ Documentation overhaul

The system is now production-ready with intelligent market adaptation, comprehensive opportunity discovery, and robust risk management capabilities.