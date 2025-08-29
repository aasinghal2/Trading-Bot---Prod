# 🎯 Multi-Ticker Trading Bot Guide

*Last Updated: August 2025 - Enhanced with Market Scanner & Dynamic Thresholds*

## 🚀 **Quick Start Commands**

**Note**: The system now includes automatic market scanning and dynamic thresholds that adapt to market conditions.

### **Tech Giants (FAANG+)**
```bash
# Standard analysis with clean output
python3 main.py --mode single --symbols AAPL GOOGL MSFT AMZN META --classic

# Auto-trading mode with dynamic thresholds
python3 main.py --mode auto --symbols AAPL GOOGL MSFT --interval 300 --classic
```

### **AI & Semiconductor Leaders**
```bash
# Single analysis
python3 main.py --mode single --symbols NVDA AMD INTC TSM AVGO --classic

# Auto-trading with faster intervals
python3 main.py --mode auto --symbols NVDA AMD --interval 180 --classic
```

### **Electric Vehicle & Clean Energy**
```bash
python3 main.py --mode single --symbols TSLA NIO LCID RIVN PLUG --classic
```

### **Banking & Finance**
```bash
python3 main.py --mode single --symbols JPM BAC WFC GS MS --classic
```

### **Healthcare & Biotechnology**
```bash
python3 main.py --mode single --symbols JNJ PFE UNH ABBV BMY --classic
```

### **Energy & Oil**
```bash
python3 main.py --mode single --symbols XOM CVX COP EOG SLB --classic
```

### **Consumer Staples**
```bash
python3 main.py --mode single --symbols KO PG WMT PEP COST --classic
```

### **Media & Entertainment**
```bash
python3 main.py --mode single --symbols DIS NFLX CMCSA T VZ --classic
```

---

## 🆕 **New Market Scanner Approach (Recommended)**

Instead of manually selecting symbols, use the market scanner to discover opportunities:

### **Automatic S&P 500 Discovery**
```bash
# Scan 50 major S&P 500 stocks for opportunities
python3 main.py --mode scan --classic

# Focus on top 5 value opportunities
python3 main.py --mode scan --top 5 --min-pe 5 --max-pe 20 --classic

# Growth stocks with reasonable valuations
python3 main.py --mode scan --top 10 --min-pe 10 --max-pe 30 --classic
```

### **Scanner-to-Trading Workflow**
```bash
# 1. Discover opportunities
python3 main.py --mode scan --top 10 --classic

# 2. Review recommendations
cat recommendations/market_scan_*.json | jq '.buy_recommendations[:5]'

# 3. Analyze top picks in detail
python3 main.py --mode single --symbols GOOGL VZ META ABT MA --classic

# 4. Auto-trade on highest conviction picks
python3 main.py --mode auto --symbols GOOGL VZ META --interval 300 --classic
```

---

## 🎛️ **Multi-Ticker Trading Modes**

### **1. Single Analysis Mode**
**Purpose**: Analyze multiple stocks simultaneously for immediate insights

```bash
# Analyze 3-5 stocks
python3 main.py --mode single --symbols AAPL GOOGL MSFT --classic

# Analyze 8-10 stocks (comprehensive)
python3 main.py --mode single --symbols AAPL GOOGL MSFT TSLA NVDA META JPM HD --classic
```

**Benefits:**
- See comparative signal strengths
- Identify relative opportunities
- Portfolio-aware analysis
- Dynamic threshold application

### **2. Auto-Trading Mode (Enhanced)**
**Purpose**: Continuous automated trading across multiple positions

```bash
# Conservative approach (3-5 stocks)
python3 main.py --mode auto --symbols AAPL GOOGL MSFT --interval 300 --classic

# Diversified approach (6-8 stocks)
python3 main.py --mode auto --symbols AAPL GOOGL MSFT TSLA NVDA META --interval 300 --classic

# Sector-focused approach
python3 main.py --mode auto --symbols AAPL GOOGL MSFT NVDA AMD --interval 300 --classic
```

**New Features:**
- **Dynamic Thresholds**: Adapts to market conditions automatically
- **Portfolio Intelligence**: Considers existing positions
- **Risk Balancing**: Prevents over-concentration
- **Signal Learning**: Improves over time

### **3. Market Scanner Mode (NEW)**
**Purpose**: Systematic opportunity discovery across the market

```bash
# Broad market scan
python3 main.py --mode scan --classic

# Value-focused scan
python3 main.py --mode scan --min-pe 5 --max-pe 15 --classic

# Quality growth scan
python3 main.py --mode scan --min-pe 15 --max-pe 25 --classic
```

---

## 📊 **Dynamic Threshold Impact on Multi-Ticker Trading**

### **Market Adaptation Examples**

#### **Bull Market Scenario**
- **Market Conditions**: SPY +6%, VIX 14, Strong breadth
- **Threshold Adjustment**: Raises to ~0.35 (more selective)
- **Multi-Ticker Impact**: Only 1-2 of 5 stocks may qualify for trades
- **Example Output**:
```
AAPL: Signal 0.387 > Threshold 0.35 ✅ (Trade)
GOOGL: Signal 0.324 < Threshold 0.35 ❌ (Skip)
MSFT: Signal 0.298 < Threshold 0.35 ❌ (Skip)
TSLA: Signal 0.401 > Threshold 0.35 ✅ (Trade)
NVDA: Signal 0.312 < Threshold 0.35 ❌ (Skip)
```

#### **Bear Market Scenario**
- **Market Conditions**: SPY -7%, VIX 31, Weak breadth
- **Threshold Adjustment**: Lowers to ~0.19 (opportunity capture)
- **Multi-Ticker Impact**: 3-4 of 5 stocks may qualify for trades
- **Example Output**:
```
AAPL: Signal 0.234 > Threshold 0.19 ✅ (Trade)
GOOGL: Signal 0.198 > Threshold 0.19 ✅ (Trade)
MSFT: Signal 0.156 < Threshold 0.19 ❌ (Skip)
TSLA: Signal 0.221 > Threshold 0.19 ✅ (Trade)
NVDA: Signal 0.203 > Threshold 0.19 ✅ (Trade)
```

---

## 🎯 **Strategic Multi-Ticker Approaches**

### **Strategy 1: Scanner-Driven Diversification**
```bash
# Weekly opportunity discovery
python3 main.py --mode scan --top 10 --classic

# Extract top symbols for analysis
cat recommendations/market_scan_*.json | jq -r '.buy_recommendations[:6] | .[].symbol'

# Deep analysis on scanner picks
python3 main.py --mode single --symbols SCANNER_PICKS --classic

# Auto-trade diversified portfolio
python3 main.py --mode auto --symbols SCANNER_PICKS --interval 300 --classic
```

### **Strategy 2: Sector Rotation**
```bash
# Technology focus
python3 main.py --mode auto --symbols AAPL GOOGL MSFT NVDA --interval 300 --classic

# Healthcare focus
python3 main.py --mode auto --symbols JNJ PFE UNH ABBV --interval 300 --classic

# Financial focus
python3 main.py --mode auto --symbols JPM BAC GS MS --interval 300 --classic
```

### **Strategy 3: Market Cap Diversification**
```bash
# Large cap core
python3 main.py --mode auto --symbols AAPL GOOGL MSFT JPM --interval 300 --classic

# Mid cap blend
python3 main.py --mode auto --symbols AMD SQ ROKU ZM --interval 300 --classic
```

### **Strategy 4: Risk-Tiered Approach**
```bash
# Conservative tier (blue chips)
python3 main.py --mode auto --symbols AAPL MSFT JPM PG --interval 300 --classic

# Growth tier (higher volatility)
python3 main.py --mode auto --symbols TSLA NVDA AMD ROKU --interval 300 --classic
```

---

## 📈 **Multi-Ticker Performance Optimization**

### **Symbol Selection Best Practices**

#### **Optimal Portfolio Size**
- **3-5 Stocks**: Deep focus, easier monitoring
- **6-8 Stocks**: Balanced diversification
- **9-12 Stocks**: Maximum diversification (may reduce performance)

#### **Correlation Considerations**
```bash
# Avoid over-correlation
# Good: AAPL GOOGL JPM XOM (different sectors)
# Bad: AAPL GOOGL MSFT NVDA (all tech-heavy)

# Mix sectors for better diversification
python3 main.py --mode single --symbols AAPL JPM XOM JNJ PG --classic
```

#### **Volatility Balance**
```bash
# Mix high and low volatility
# High vol: TSLA NVDA AMD
# Low vol: KO PG JNJ
# Balanced: AAPL GOOGL MSFT

python3 main.py --mode auto --symbols TSLA AAPL KO --interval 300 --classic
```

### **Market Scanner Integration**

#### **Identify Best Multi-Ticker Combinations**
```bash
# 1. Run comprehensive scan
python3 main.py --mode scan --top 15 --classic

# 2. Filter by sectors
cat recommendations/market_scan_*.json | jq '.buy_recommendations[] | select(.symbol | test("AAPL|GOOGL|MSFT|JPM|XOM"))'

# 3. Build diversified portfolio
python3 main.py --mode auto --symbols DIVERSIFIED_PICKS --interval 300 --classic
```

#### **Dynamic Symbol Rotation**
```bash
# Weekly: Update symbol list based on scanner results
# Daily: Run scanner to validate current positions
# Hourly: Auto-trade with current symbol list
```

---

## 🛡️ **Risk Management in Multi-Ticker Trading**

### **Portfolio-Level Protections**

#### **Concentration Limits**
- **Per Position**: Maximum 10% of portfolio
- **Per Sector**: Maximum 30% of portfolio
- **Correlation**: Automatic correlation analysis
- **Cash Buffer**: Maintains 10-20% cash position

#### **Dynamic Risk Adjustment**
```bash
# System automatically:
# - Reduces position sizes in volatile markets
# - Increases diversification bonuses
# - Applies correlation penalties
# - Maintains VaR limits
```

### **Multi-Ticker Risk Scenarios**

#### **Concentration Risk Example**
```
Portfolio: $100,000
AAPL: $15,000 (15%) ✅ Under limit
GOOGL: $12,000 (12%) ✅ Under limit  
MSFT: $11,000 (11%) ✅ Under limit
TSLA: $8,000 (8%) ✅ Under limit
Tech Sector: $46,000 (46%) ⚠️ High concentration
```

#### **Correlation Risk Example**
```
High Correlation Detected:
AAPL ↔ GOOGL: 0.78
AAPL ↔ MSFT: 0.82
GOOGL ↔ MSFT: 0.85

Action: Reduce tech positions or add uncorrelated assets
Suggestion: Add JPM, XOM, or JNJ for diversification
```

---

## 📊 **Multi-Ticker Monitoring & Analysis**

### **Real-Time Monitoring Commands**

#### **Portfolio Overview**
```bash
# Current positions
cat data/portfolio_state.json | jq '.positions'

# Portfolio value progression
grep "Portfolio value" logs/trading_system.log | tail -10

# Recent trades across all symbols
cat data/portfolio_state.json | jq '.trade_history[-10:]'
```

#### **Symbol-Specific Analysis**
```bash
# Signal strengths for all symbols
grep "Signal.*>" logs/trading_system.log | tail -10

# Threshold comparisons
grep "Threshold.*\|Signal.*" logs/trading_system.log | tail -20

# Execution results
grep "Executed.*BUY\|Executed.*SELL" logs/trading_system.log | tail -10
```

### **Performance Tracking**

#### **Multi-Ticker Metrics**
```bash
# Win rate by symbol
grep "Executed" logs/trading_system.log | awk '{print $NF}' | sort | uniq -c

# Average signal strength by symbol
grep "Signal" logs/trading_system.log | awk '{print $2, $4}' | sort

# Portfolio composition changes
cat data/portfolio_state.json | jq '.positions | to_entries | map({symbol: .key, value: .value.current_value})'
```

---

## 🔧 **Troubleshooting Multi-Ticker Issues**

### **Common Problems**

#### **1. Only Some Symbols Trading**
**Symptoms**: 5 symbols analyzed, only 1-2 execute trades
**Cause**: Dynamic thresholds filtering lower-quality signals
**Solution**: This is normal and healthy behavior
```bash
# Check threshold level
grep "Using threshold" logs/trading_system.log | tail -1

# Review signal distribution
grep "Signal.*" logs/trading_system.log | tail -10
```

#### **2. High Correlation Warning**
**Symptoms**: Risk manager flags correlation risk
**Cause**: Too many symbols from same sector
**Solution**: Add uncorrelated assets
```bash
# Add symbols from different sectors
python3 main.py --mode auto --symbols AAPL JPM XOM JNJ --interval 300 --classic
```

#### **3. Scanner Picks Not Executing**
**Symptoms**: Scanner finds opportunities but auto-trading skips them
**Cause**: Market conditions changed between scan and execution
**Solution**: Use recent scanner results
```bash
# Fresh scan before auto-trading
python3 main.py --mode scan --top 5 --classic
# Immediately follow with auto-trading
python3 main.py --mode auto --symbols FRESH_PICKS --interval 300 --classic
```

### **Performance Optimization**

#### **Speed Improvements**
```bash
# Reduce symbol count for faster cycles
--symbols AAPL GOOGL MSFT  # vs 8+ symbols

# Increase intervals for complex portfolios
--interval 300  # vs --interval 180 for 6+ symbols

# Use market scanner to pre-filter quality opportunities
```

#### **Quality Improvements**
```bash
# Let signal history build across all symbols
# Monitor portfolio correlation metrics
# Use scanner to refresh symbol selection weekly
```

---

## 📚 **Advanced Multi-Ticker Features**

### **Portfolio Optimization**
The system now includes mean-variance optimization:
```yaml
# Enable in config/config.yaml
portfolio_manager:
  allocation_method: "optimizer"
  total_investment_target: 0.8
```
- Calculates optimal position weights
- Considers correlation matrix
- Balances risk vs return
- Overrides signal-based sizing

### **Sector Rebalancing**
```bash
# System automatically:
# - Monitors sector exposures
# - Suggests rebalancing opportunities
# - Applies sector diversification bonuses
# - Prevents sector over-concentration
```

### **Dynamic Symbol Management**
```bash
# Weekly symbol refresh workflow:
# 1. Run market scan
python3 main.py --mode scan --top 20 --classic

# 2. Analyze current positions
python3 main.py --mode single --symbols CURRENT_HOLDINGS --classic

# 3. Update auto-trading with new opportunities
python3 main.py --mode auto --symbols REFRESHED_LIST --interval 300 --classic
```

---

## 🎉 **Multi-Ticker Success Patterns**

### **What Good Multi-Ticker Trading Looks Like**
- **Balanced Execution**: 40-60% of symbols execute trades per cycle
- **Sector Diversification**: No single sector >40% of portfolio
- **Dynamic Adaptation**: Threshold adjustments visible in logs
- **Steady Growth**: Portfolio value trending upward
- **Controlled Risk**: VaR under 2%, max drawdown under 10%

### **Example Successful Portfolio Progression**
```
Week 1: Scanner finds GOOGL, VZ, META, ABT, MA
Week 2: Auto-trading builds positions based on signals
Week 3: Dynamic thresholds adapt to market volatility
Week 4: Portfolio rebalances automatically
Result: 15% diversified portfolio across 5 sectors
```

### **Key Performance Indicators**
```bash
# Portfolio diversity score
cat data/portfolio_state.json | jq '.positions | length'

# Sector concentration (should be <40% any single sector)
# Average signal strength (should trend upward over time)
# Execution rate (40-60% signals leading to trades)
# Sharpe ratio improvement over time
```

The enhanced multi-ticker system combines systematic opportunity discovery through market scanning, intelligent threshold adaptation, and sophisticated portfolio management to deliver superior diversified trading performance. Start with scanner-driven symbol selection, then graduate to automatic multi-ticker trading as you become comfortable with the system's portfolio intelligence.

**Ready to discover your next multi-ticker portfolio?**
```bash
python3 main.py --mode scan --top 10 --classic
```