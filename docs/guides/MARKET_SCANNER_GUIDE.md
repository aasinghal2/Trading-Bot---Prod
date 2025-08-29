# Market Scanner - Complete Guide

*Last Updated: August 2025*

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Features & Capabilities](#features--capabilities)
4. [Command Reference](#command-reference)
5. [Understanding Results](#understanding-results)
6. [Recommendation Files](#recommendation-files)
7. [Filtering & Customization](#filtering--customization)
8. [Performance & Optimization](#performance--optimization)
9. [Integration with Trading System](#integration-with-trading-system)
10. [Troubleshooting](#troubleshooting)

## Overview

The Market Scanner is a powerful opportunity discovery engine that analyzes the S&P 500 to find trading opportunities you might otherwise miss. Instead of manually guessing which stocks to analyze, the scanner systematically evaluates 50 major stocks and ranks them by signal strength.

### **Key Benefits**
- **Automated Discovery**: No more guessing which stocks to analyze
- **Systematic Coverage**: Scans 50 S&P 500 stocks automatically
- **Data-Driven Rankings**: Orders opportunities by signal strength
- **Time Efficient**: 2-3 minutes vs hours of manual research
- **Recommendation Export**: Structured output for further analysis

### **How It Differs from Manual Analysis**
| Manual Approach | Market Scanner |
|----------------|----------------|
| Pick 2-3 stocks manually | Analyzes 50 stocks automatically |
| Hit-or-miss opportunity discovery | Systematic ranking by signal strength |
| 33% success rate (1 of 3 approved) | 100% focused on strong signals |
| Time consuming research | 2-3 minute automated scan |
| No structured output | JSON export with detailed analysis |

## Quick Start

### **Basic Scan**
```bash
# Default scan - Top 10 opportunities across S&P 500
python3 main.py --mode scan --classic
```

### **Custom P/E Filtering**
```bash
# Value opportunities (lower P/E ratios)
python3 main.py --mode scan --top 5 --min-pe 5 --max-pe 15 --classic

# Growth stocks with reasonable valuations
python3 main.py --mode scan --top 10 --min-pe 10 --max-pe 30 --classic
```

### **Conservative Screening**
```bash
# Very selective - only top 3 opportunities
python3 main.py --mode scan --top 3 --min-pe 5 --max-pe 20 --classic
```

## Features & Capabilities

### **1. S&P 500 Coverage**
The scanner analyzes 50 major S&P 500 stocks including:
- **Technology**: AAPL, MSFT, GOOGL, META, NVDA, TSLA
- **Healthcare**: JNJ, PFE, ABT, UNH
- **Finance**: JPM, V, MA
- **Consumer**: KO, PG, HD, WMT
- **Energy**: XOM, CVX
- **And 30+ more major companies**

### **2. Multi-Stage Filtering**

#### **Stage 1: Fundamental Screening**
- **P/E Ratio Filter**: Configurable min/max P/E ratios
- **Market Cap Filter**: Minimum $1B market capitalization
- **Financial Data Quality**: Excludes stocks with missing data

#### **Stage 2: Technical Analysis**
For stocks passing fundamental filters:
- **15+ Technical Indicators**: RSI, MACD, Bollinger Bands, ADX, etc.
- **Trend Analysis**: EMA crossovers, momentum indicators
- **Volume Analysis**: Volume ratios, accumulation/distribution

#### **Stage 3: Fundamental Analysis**
- **Valuation Metrics**: P/E, P/B, EV/EBITDA analysis
- **Financial Health**: Revenue growth, profit margins
- **Quality Scores**: Overall fundamental rating (1-10 scale)

#### **Stage 4: Sentiment Analysis**
- **News Sentiment**: Recent financial news analysis
- **Social Media**: Reddit sentiment (if configured)
- **Time-Weighted**: Recent news gets higher importance

### **3. Signal Combination & Ranking**
```
Combined Signal = 0.6 × Technical + 0.4 × Fundamental
```
- **Technical Emphasis**: 60% weight on technical indicators
- **Fundamental Support**: 40% weight on fundamental quality
- **Sentiment Enhancement**: Modifies base score
- **Percentile Ranking**: Relative strength vs all analyzed stocks

## Command Reference

### **Core Commands**

#### **Basic Scanner**
```bash
python3 main.py --mode scan --classic
```
- Scans 50 S&P 500 stocks
- P/E filter: 0-50 (wide range)
- Shows top 10 opportunities
- Saves results to recommendations folder

#### **Custom Parameters**
```bash
python3 main.py --mode scan --top 5 --min-pe 10 --max-pe 25 --classic
```

### **Parameter Details**

| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|----------------|
| `--top` | Number of opportunities to show | 10 | 3, 5, 15, 20 |
| `--min-pe` | Minimum P/E ratio filter | 0 | 5, 10, 15 |
| `--max-pe` | Maximum P/E ratio filter | 50 | 15, 25, 30, 40 |
| `--classic` | Clean output format | - | Always recommended |

### **Example Use Cases**

#### **Value Investing**
```bash
# Deep value opportunities
python3 main.py --mode scan --top 5 --min-pe 5 --max-pe 12 --classic
```

#### **Growth at Reasonable Price (GARP)**
```bash
# Growth stocks with reasonable valuations
python3 main.py --mode scan --top 10 --min-pe 8 --max-pe 25 --classic
```

#### **Quality Focus**
```bash
# Best overall opportunities regardless of valuation
python3 main.py --mode scan --top 7 --min-pe 0 --max-pe 50 --classic
```

#### **Conservative Screening**
```bash
# Only highest conviction opportunities
python3 main.py --mode scan --top 3 --min-pe 10 --max-pe 20 --classic
```

## Understanding Results

### **Scanner Output Format**
```
🔍 S&P 500 MARKET SCANNER RESULTS
📊 SCAN SUMMARY:
   ⏱️  Scan Time: 156.60 seconds
   🎯 Symbols Scanned: 50
   ✅ Passed Filters: 30
   📈 Successfully Analyzed: 30
   🔍 Filters: P/E 5.0-30.0, Min Cap $1,000,000,000

🏆 TOP OPPORTUNITIES:
Rank Symbol Signal   Direction Price      P/E View    
------------------------------------------------------------
1    GOOGL  0.407    BULLISH  $201.42    Unknown     
2    VZ     0.353    BULLISH  $43.15     Unknown     
3    META   0.328    BULLISH  $769.30    Unknown     
4    ABT    0.303    BULLISH  $134.28    Unknown     
5    MA     0.281    BULLISH  $574.32    Unknown
```

### **Key Metrics Explained**

#### **Scan Summary**
- **Scan Time**: Total processing time (2-4 minutes typical)
- **Symbols Scanned**: Total S&P 500 stocks attempted (50)
- **Passed Filters**: Stocks meeting P/E and market cap criteria
- **Successfully Analyzed**: Stocks with complete analysis data

#### **Opportunity Rankings**
- **Rank**: Ordered by signal strength (1 = strongest)
- **Symbol**: Stock ticker symbol
- **Signal**: Combined signal strength (0.000 to 1.000 scale)
- **Direction**: Technical trend (BULLISH/BEARISH/NEUTRAL)
- **Price**: Current stock price
- **P/E View**: Fundamental valuation assessment

#### **Signal Strength Interpretation**
| Signal Range | Quality | Description |
|-------------|---------|-------------|
| 0.400+ | **Excellent** | Very strong signals, high conviction |
| 0.300-0.399 | **Strong** | Good signals, solid opportunities |
| 0.250-0.299 | **Moderate** | Above threshold, consider carefully |
| 0.200-0.249 | **Weak** | Below threshold, typically rejected |
| Below 0.200 | **Very Weak** | Poor signals, avoid |

#### **Direction Analysis**
- **BULLISH**: Technical indicators suggest upward momentum
- **BEARISH**: Technical indicators suggest downward pressure
- **NEUTRAL**: Mixed or sideways technical signals

## Recommendation Files

### **Automatic Export**
Every scan automatically saves detailed results to:
```
recommendations/market_scan_YYYYMMDD_HHMMSS.json
```

### **File Structure**
```json
{
  "scan_timestamp": "2025-08-13T17:45:19.923885",
  "scan_summary": {
    "scan_time_seconds": 156.6,
    "symbols_scanned": 50,
    "symbols_filtered": 30,
    "symbols_analyzed": 30,
    "filters_applied": {
      "min_pe": 5.0,
      "max_pe": 30.0,
      "min_market_cap": 1000000000.0
    }
  },
  "buy_recommendations": [
    {
      "rank": 1,
      "symbol": "GOOGL",
      "signal_strength": 0.407,
      "technical_direction": "bullish",
      "technical_strength": 0.342,
      "fundamental_score": 6.77,
      "fundamental_signal": 0.354,
      "current_price": 201.42,
      "recommendation_reason": "Very Strong signal (0.407) driven by fundamental analysis"
    }
  ],
  "sell_recommendations": [],
  "neutral_signals": []
}
```

### **Using Recommendation Files**

#### **View Latest Results**
```bash
# List recent scans
ls -la recommendations/

# View latest scan results
cat recommendations/market_scan_*.json | jq '.buy_recommendations'
```

#### **Extract Top Symbols**
```bash
# Get top 5 symbols for further analysis
cat recommendations/market_scan_*.json | jq -r '.buy_recommendations[:5] | .[].symbol'
```

#### **Analyze Specific Recommendation**
```bash
# Run detailed analysis on scanner pick
python3 main.py --mode single --symbols GOOGL --classic
```

## Filtering & Customization

### **P/E Ratio Strategies**

#### **Deep Value (P/E 5-12)**
- **Target**: Undervalued companies, potential turnarounds
- **Risk**: Value traps, declining businesses
- **Command**: `--min-pe 5 --max-pe 12`

#### **Quality Value (P/E 10-20)**
- **Target**: Quality companies at reasonable prices
- **Risk**: Moderate, balanced approach
- **Command**: `--min-pe 10 --max-pe 20`

#### **Growth Reasonable (P/E 15-30)**
- **Target**: Growing companies with reasonable valuations
- **Risk**: Growth premium, higher volatility
- **Command**: `--min-pe 15 --max-pe 30`

#### **Growth Aggressive (P/E 20-50)**
- **Target**: High-growth companies, momentum plays
- **Risk**: High valuations, market-dependent
- **Command**: `--min-pe 20 --max-pe 50`

### **Result Set Sizing**

#### **Conservative Approach**
```bash
# Top 3 highest conviction only
python3 main.py --mode scan --top 3 --classic
```

#### **Balanced Approach**
```bash
# Top 5-7 for diversified selection
python3 main.py --mode scan --top 5 --classic
```

#### **Comprehensive Approach**
```bash
# Top 10-15 for full opportunity spectrum
python3 main.py --mode scan --top 15 --classic
```

### **Market Cap Considerations**
- **Fixed Filter**: $1B minimum market cap
- **Rationale**: Ensures liquidity and stability
- **Coverage**: Includes large and mid-cap stocks
- **Exclusions**: Small-cap and micro-cap stocks

## Performance & Optimization

### **Typical Performance**
- **Scan Time**: 2-4 minutes for 50 stocks
- **Success Rate**: 60-80% stocks pass fundamental filters
- **Analysis Rate**: ~2-3 seconds per stock
- **Network Usage**: Moderate (market data fetching)

### **Optimization Strategies**

#### **Faster Scans**
```bash
# Reduce result count
python3 main.py --mode scan --top 5 --classic

# Stricter filtering (fewer stocks to analyze)
python3 main.py --mode scan --min-pe 15 --max-pe 25 --classic
```

#### **Network Optimization**
- **Caching**: Market data cached for 1 hour
- **Batch Processing**: Processes 5-10 stocks per batch
- **Rate Limiting**: Built-in delays to respect API limits

#### **Memory Management**
- **Cleanup**: Automatic cleanup of old scan data
- **Garbage Collection**: Proper resource management
- **Connection Pooling**: Efficient network resource usage

### **Performance Monitoring**
```bash
# Monitor scan progress
tail -f logs/trading_system.log | grep "Analyzing\|symbols"

# Check scan completion times
grep "Market scan completed" logs/trading_system.log | tail -5
```

## Integration with Trading System

### **Scanner to Analysis Workflow**

#### **1. Discover Opportunities**
```bash
python3 main.py --mode scan --top 5 --classic
```

#### **2. Detailed Analysis**
```bash
# Analyze scanner recommendations
python3 main.py --mode single --symbols GOOGL VZ META --classic
```

#### **3. Execute Trades**
Based on detailed analysis, the system will automatically execute approved trades.

### **Why Scanner + Analysis?**

#### **Scanner Benefits**
- **Broad Coverage**: 50 stocks vs 2-3 manual picks
- **Systematic**: No emotional or biased stock selection
- **Efficient**: 3 minutes vs hours of research
- **Documented**: Structured output for review

#### **Detailed Analysis Benefits**
- **Portfolio Awareness**: Considers existing positions
- **Risk Assessment**: Full risk management evaluation
- **Fresh Data**: Real-time prices and indicators
- **Execution Control**: Final approval gates

### **Combined Workflow Example**
```bash
# 1. Weekly opportunity discovery
python3 main.py --mode scan --top 10 --classic

# 2. Review recommendations file
cat recommendations/market_scan_*.json | jq '.buy_recommendations[:5]'

# 3. Analyze top picks
python3 main.py --mode single --symbols GOOGL VZ META ABT MA --classic

# 4. Review execution results and portfolio
```

## Troubleshooting

### **Common Issues**

#### **1. Slow Scan Performance**
**Symptoms**: Scan takes >5 minutes
**Causes**:
- Network connectivity issues
- API rate limiting
- System resource constraints

**Solutions**:
```bash
# Check network connectivity
ping google.com

# Reduce scan scope
python3 main.py --mode scan --top 5 --classic

# Monitor system resources
top -p $(pgrep python)
```

#### **2. Few Stocks Passing Filters**
**Symptoms**: "Passed Filters: 8" (very low)
**Causes**:
- Very restrictive P/E filters
- Market conditions (high valuations)
- Data quality issues

**Solutions**:
```bash
# Widen P/E range
python3 main.py --mode scan --min-pe 0 --max-pe 50 --classic

# Check market conditions
python3 main.py --mode single --symbols SPY QQQ --classic
```

#### **3. No Strong Signals Found**
**Symptoms**: All signals below 0.25 threshold
**Causes**:
- Weak market conditions
- High dynamic thresholds
- Poor overall market sentiment

**Solutions**:
```bash
# Check dynamic threshold levels
grep "Using threshold" logs/trading_system.log | tail -3

# Analyze market metrics
grep "Market metrics" logs/trading_system.log | tail -1

# Consider market scanner recommendations regardless
# They're still ranked by relative strength
```

#### **4. Market Data Errors**
**Symptoms**: "Failed to fetch data for XYZ"
**Causes**:
- Yahoo Finance API issues
- Network connectivity problems
- Invalid ticker symbols

**Solutions**:
```bash
# Check logs for specific errors
grep "Failed to fetch\|Error" logs/trading_system.log | tail -10

# Test individual symbol fetch
python3 -c "import yfinance as yf; print(yf.download('AAPL', period='1d'))"

# Retry scan later if widespread issues
```

### **Performance Diagnostics**

#### **Monitor Scan Progress**
```bash
# Real-time scan monitoring
tail -f logs/trading_system.log | grep -E "Filtering|Analyzing|symbols"
```

#### **Check Completion Stats**
```bash
# Recent scan statistics
grep -A5 "SCAN SUMMARY" logs/trading_system.log | tail -10
```

#### **Network Performance**
```bash
# Check for network-related errors
grep -i "timeout\|connection\|network" logs/trading_system.log | tail -5
```

### **Configuration Tuning**

#### **Market Scanner Symbol List**
To modify the scanned universe, edit `market_scanner.py`:
```python
# In market_scanner.py, modify SP500_TEST_SYMBOLS
SP500_TEST_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', ...  # Add/remove symbols as needed
]
```

#### **Batch Size Optimization**
```python
# In market_scanner.py, adjust batch processing
batch_size = 5  # Increase for faster scans, decrease for stability
```

#### **Cache Settings**
```python
# In core/market_metrics.py, adjust cache duration
cache_duration_hours = 1  # Increase to reduce API calls
```

## Best Practices

### **1. Regular Scanning Schedule**
```bash
# Daily opportunity discovery
python3 main.py --mode scan --top 10 --classic

# Weekly comprehensive scan
python3 main.py --mode scan --top 20 --min-pe 0 --max-pe 50 --classic
```

### **2. Filter Strategy Evolution**
- **Bull Markets**: Tighten P/E filters (avoid overvaluation)
- **Bear Markets**: Widen P/E filters (capture opportunities)
- **Sideways Markets**: Focus on quality (moderate P/E ranges)

### **3. Integration with Portfolio Management**
```bash
# Before major portfolio changes
python3 main.py --mode scan --classic

# After scanning, analyze current positions
python3 main.py --mode single --symbols CURRENT_HOLDINGS --classic
```

### **4. Documentation & Review**
- **Save Recommendations**: Automatic JSON export provides audit trail
- **Review Decisions**: Compare executed trades vs scanner recommendations
- **Performance Tracking**: Monitor scanner pick performance over time

The Market Scanner transforms stock selection from guesswork into a systematic, data-driven process. By automatically analyzing 50 S&P 500 stocks and ranking them by signal strength, it ensures you never miss high-quality opportunities while focusing your time on the most promising trades.