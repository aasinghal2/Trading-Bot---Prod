# 📈 AI Trading Bot - Portfolio Maintenance Guide

## 📊 Current Portfolio Status
- **7 Active Positions**: MRK, PFE, META, VZ, ABT, GOOGL, QCOM
- **Capital Deployed**: $70,000 (70%)
- **Cash Reserve**: $30,000 (30%)
- **Risk Level**: Moderate (33.7/100)

---

## 🗓️ DAILY ROUTINE (5-10 minutes)

### 1. **Morning Portfolio Health Check**
```bash
# Quick portfolio status
python3 main.py --mode single --symbols MRK PFE META VZ ABT GOOGL QCOM --classic
```

**What to Monitor:**
- Current portfolio value vs. entry prices
- Any positions down >5% (consider review)
- Risk metrics (VaR, leverage, concentration)
- Cash balance changes

### 2. **Signal Data Update** (Every 2-3 days)
```bash
# Light market scan to keep signal history fresh
python3 main.py --mode scan --top 25 --classic
```

**Purpose:**
- Maintains dynamic threshold accuracy
- Keeps market metrics cache current
- Builds signal history for better decisions

### 3. **News & Sentiment Monitoring**
- Check for major news on your 7 holdings
- Look for earnings announcements, FDA approvals, regulatory changes
- Monitor overall market sentiment shifts

---

## 📅 WEEKLY ROUTINE (30-45 minutes)

### 1. **Full Market Opportunity Scan** (Mondays)
```bash
# Comprehensive scan for new opportunities
python3 main.py --mode scan --top 50 --classic
```

**Action Items:**
- Review top opportunities above dynamic threshold
- Compare with current holdings
- Identify potential new positions or replacements

### 2. **Portfolio Rebalancing Check** (Wednesdays)
```bash
# Detailed analysis of current holdings
python3 main.py --mode single --symbols MRK PFE META VZ ABT GOOGL QCOM --verbose
```

**Key Decisions:**
- **Hold**: Signals still strong (>0.25)
- **Reduce**: Signals weakening (0.10-0.25)
- **Exit**: Signals negative (<0.10) or fundamental deterioration
- **Add**: Strong signals with available cash

### 3. **New Position Analysis** (Fridays)
If weekly scan identified strong opportunities:
```bash
# Analyze potential new positions
python3 main.py --mode single --symbols [NEW_CANDIDATES] --verbose
```

**Capital Allocation Rules:**
- Keep 20-30% cash reserve
- Max 10% per position
- Target 8-12 total positions when fully deployed

---

## 🚨 EMERGENCY PROTOCOLS

### **Market Crash (>3% down)**
```bash
# Emergency portfolio assessment
python3 main.py --mode single --symbols MRK PFE META VZ ABT GOOGL QCOM --emergency
```

**Actions:**
1. Check which positions are defensive vs. cyclical
2. Consider increasing cash if risk score >70
3. Look for opportunities in high-quality names

### **Individual Stock Crisis (>10% down)**
```bash
# Individual stock deep dive
python3 main.py --mode single --symbols [AFFECTED_STOCK] --verbose --fundamental-deep
```

**Decision Framework:**
- **Temporary**: Add to position if fundamentals intact
- **Fundamental**: Exit position immediately
- **Unknown**: Reduce position by 50%, investigate

### **Risk Alert (Score >60)**
```bash
# Risk assessment and rebalancing
python3 main.py --risk-only --rebalance
```

---

## 📊 PERFORMANCE TRACKING

### **Monthly Review Checklist**
- [ ] Portfolio vs. S&P 500 performance
- [ ] Individual stock performance vs. entry prices
- [ ] Dynamic threshold effectiveness (false positives/negatives)
- [ ] Risk-adjusted returns (Sharpe ratio)
- [ ] Capital deployment efficiency

### **Quarterly Deep Dive**
- [ ] Fundamental analysis refresh for all holdings
- [ ] Sector allocation review
- [ ] Market regime assessment
- [ ] System parameter optimization

---

## 🎯 KEY PERFORMANCE INDICATORS (KPIs)

### **Daily Monitoring**
- Portfolio value change (%)
- Worst performing position
- Risk score trend
- Cash deployment ratio

### **Weekly Analysis**
- Signal strength trends for holdings
- New opportunities vs. current positions
- Dynamic threshold evolution
- Market breadth indicators

### **Monthly Evaluation**
- Total return vs. benchmark
- Win/loss ratio on positions
- Average holding period
- Maximum drawdown

---

## 💡 OPTIMIZATION TIPS

### **Signal Quality**
- Maintain 50+ signals in history for robust thresholds
- Run scans during different market conditions
- Monitor threshold sensitivity to market regimes

### **Portfolio Construction**
- Aim for 8-12 positions when fully deployed
- Diversify across sectors but focus on quality
- Keep 20-30% cash for opportunities and protection

### **Risk Management**
- Never exceed 10% in any single position
- Exit positions with deteriorating fundamentals quickly
- Use cash as a risk management tool during uncertainty

---

## 🔧 MAINTENANCE COMMANDS CHEAT SHEET

```bash
# Daily quick check
python3 main.py --mode single --symbols MRK PFE META VZ ABT GOOGL QCOM --classic

# Weekly opportunity scan
python3 main.py --mode scan --top 50 --classic

# Monthly deep analysis
python3 main.py --mode single --symbols MRK PFE META VZ ABT GOOGL QCOM --verbose

# Emergency assessment
python3 main.py --mode single --symbols [SYMBOL] --emergency --verbose

# Clean historical data (monthly)
rm -f data/signal_history.json data/market_metrics_cache.json

# Check system health
python3 main.py --system-check
```

---

**🎯 Remember**: The AI system is designed to be your intelligent assistant, not a replacement for judgment. Always consider:
- Market context and news events
- Your personal risk tolerance
- Long-term investment goals
- Economic and sector cycles

**Your AI trading bot + human oversight = Optimal results! 📈**