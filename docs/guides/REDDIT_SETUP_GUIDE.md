# 🔗 Reddit API Setup Guide

*Last Updated: August 2025 - Enhanced System Integration*

**Status**: Reddit integration is fully operational with async PRAW for improved performance and non-blocking sentiment analysis. Now integrated with dynamic thresholds and signal history tracking.

## 📋 **Complete Step-by-Step Instructions**

### **1. Create Reddit Application**

1. **Go to Reddit Apps Page**:
   - Visit: https://www.reddit.com/prefs/apps
   - Log in to your Reddit account

2. **Create New App**:
   - Click **"Create App"** or **"Create Another App"**
   - Fill out the form:
     - **Name**: `TradingBot-Sentiment`
     - **App type**: Select **"script"** (important!)
     - **Description**: `Stock sentiment analysis bot with dynamic thresholds`
     - **About URL**: Leave blank
     - **Redirect URI**: `http://localhost:8080`
   - Click **"Create app"**

3. **Get Your Credentials**:
   - **Client ID**: Look for a string under your app name (e.g., `abc123xyz`)
   - **Secret**: Click "secret" to reveal it (e.g., `def456uvw`)

### **2. Update Your .env File**

Open your `.env` file and replace these lines:
```bash
REDDIT_CLIENT_ID=your_actual_client_id_here
REDDIT_CLIENT_SECRET=your_actual_secret_here
REDDIT_USER_AGENT=TradingBot/1.0 by /u/YourRedditUsername
```

**Example:**
```bash
REDDIT_CLIENT_ID=abc123xyz
REDDIT_CLIENT_SECRET=def456uvw
REDDIT_USER_AGENT=TradingBot/1.0 by /u/myusername
```

### **3. Test Your Setup**

Test Reddit integration with the enhanced system:
```bash
# Test single stock with Reddit sentiment
python3 main.py --mode single --symbols AAPL --classic

# Test with multiple stocks
python3 main.py --mode single --symbols AAPL GOOGL TSLA --classic

# Test with market scanner (includes Reddit sentiment in analysis)
python3 main.py --mode scan --top 5 --classic
```

## 🚀 **How Reddit Integration Works (Enhanced)**

### **Data Sources**
The system monitors these subreddits for stock sentiment:
- `/r/investing` - General investment discussions
- `/r/stocks` - Stock-specific conversations  
- `/r/SecurityAnalysis` - Fundamental analysis discussions
- `/r/StockMarket` - Market news and sentiment
- `/r/ValueInvesting` - Value investing perspectives

### **Enhanced Sentiment Analysis Pipeline**

#### **1. Post Collection (Async)**
```python
# Asynchronous Reddit data collection
async with asyncpraw.Reddit(...) as reddit:
    subreddit = await reddit.subreddit("investing+stocks")
    async for post in subreddit.hot(limit=50):
        # Collect relevant posts mentioning target symbols
```

#### **2. Advanced NLP Processing**
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **FinBERT**: `ProsusAI/finbert` for financial context
- **Time Weighting**: Recent posts get higher importance
- **Relevance Filtering**: Stock ticker mentions and context analysis

#### **3. Signal Integration**
```python
# Reddit sentiment now contributes to dynamic threshold calculation
combined_signal = (
    0.6 * technical_strength +
    0.4 * fundamental_score +
    0.2 * reddit_sentiment  # Reddit sentiment modifier
)
```

#### **4. Signal History Tracking**
- Reddit sentiment scores are tracked in signal history
- Contributes to dynamic threshold learning
- Improves threshold adaptation over time

## 📊 **Reddit Sentiment in Dynamic Thresholds**

### **Sentiment Impact on Thresholds**

#### **Positive Reddit Sentiment**
- **Market Euphoria**: May raise thresholds to prevent overconfidence
- **Sector Enthusiasm**: Adjusts sector-specific analysis
- **Individual Stock Buzz**: Influences specific symbol evaluation

#### **Negative Reddit Sentiment**
- **Market Fear**: May lower thresholds to capture oversold opportunities
- **Sector Pessimism**: Considers broader sentiment context
- **Individual Stock Criticism**: Validates technical/fundamental concerns

### **Example Integration**
```
AAPL Analysis:
├── Technical Signal: 0.34
├── Fundamental Score: 6.5/10
├── Reddit Sentiment: 0.78 (positive)
├── Combined Signal: 0.347
├── Dynamic Threshold: 0.294 (market-adapted)
└── Decision: Execute trade (0.347 > 0.294) ✅
```

## 🔍 **Market Scanner + Reddit Integration**

### **Enhanced Scanner Analysis**
The market scanner now includes Reddit sentiment for all 50 S&P 500 stocks:

```bash
# Scanner with Reddit sentiment integration
python3 main.py --mode scan --top 10 --classic
```

**Scanner Output Now Includes:**
- Technical analysis
- Fundamental valuation
- **Reddit sentiment score**
- Combined signal with sentiment weighting
- Market-adapted threshold comparison

### **Sentiment-Driven Opportunities**
```
🔍 S&P 500 MARKET SCANNER RESULTS (with Reddit Sentiment)
🏆 TOP OPPORTUNITIES:
Rank Symbol Signal Direction Reddit  Price    P/E View    
------------------------------------------------------------
1    GOOGL  0.407  BULLISH   +0.65   $201.42  Unknown     
2    VZ     0.353  BULLISH   +0.23   $43.15   Unknown     
3    META   0.328  BULLISH   -0.12   $769.30  Unknown   
```

## ⚡ **Performance & Efficiency**

### **Async Processing Benefits**
- **Non-blocking**: Reddit calls don't delay other agents
- **Parallel Execution**: Multiple subreddit queries simultaneously
- **Timeout Protection**: Prevents hanging on slow Reddit responses
- **Error Recovery**: Graceful handling of Reddit API failures

### **Rate Limiting & Respect**
```python
# Built-in rate limiting
reddit_config = {
    'requests_per_minute': 60,    # Within Reddit's limits
    'timeout_seconds': 10,        # Prevent hanging
    'retry_attempts': 3,          # Graceful error handling
    'cache_duration': 300         # 5-minute cache
}
```

### **Caching Strategy**
- **Post Caching**: Avoid re-processing same posts
- **Sentiment Caching**: Store processed sentiment scores
- **Time-based Expiry**: Fresh data every 5 minutes
- **Memory Efficient**: Cleanup old cache entries

## 🛡️ **Privacy & Security**

### **Data Handling**
- **Public Data Only**: Only accesses public Reddit posts
- **No Personal Info**: No user data collection
- **Minimal Storage**: Sentiment scores only, not post content
- **Temporary Processing**: Posts processed and discarded

### **API Security**
```bash
# Secure credential management
REDDIT_CLIENT_ID=abc123xyz      # App identifier
REDDIT_CLIENT_SECRET=def456uvw  # Secret key (keep private)
REDDIT_USER_AGENT=TradingBot/1.0 by /u/username  # Identification
```

### **Rate Limiting Compliance**
- **60 requests/minute**: Well within Reddit's 100/minute limit
- **Respectful Delays**: Built-in delays between requests
- **Error Handling**: Proper HTTP status code handling
- **User-Agent**: Clear identification of bot purpose

## 🔧 **Configuration & Customization**

### **Subreddit Configuration**
Edit `agents/sentiment_agent.py` to customize subreddits:
```python
DEFAULT_SUBREDDITS = [
    'investing',
    'stocks', 
    'SecurityAnalysis',
    'StockMarket',
    'ValueInvesting',
    # Add custom subreddits:
    # 'wallstreetbets',  # More volatile sentiment
    # 'financialindependence',  # Long-term focus
]
```

### **Sentiment Weight Tuning**
Adjust Reddit sentiment influence:
```python
# In sentiment_agent.py
sentiment_weights = {
    'technical': 0.6,      # Technical analysis weight
    'fundamental': 0.4,    # Fundamental analysis weight  
    'reddit': 0.2,         # Reddit sentiment modifier
    'news': 0.15          # News sentiment modifier
}
```

### **Time Decay Settings**
```python
# Recent posts get higher weight
time_decay_config = {
    'hours_recent': 24,     # Posts from last 24 hours
    'recent_weight': 1.0,   # Full weight for recent posts
    'decay_rate': 0.5,      # 50% weight reduction per day
    'min_weight': 0.1       # Minimum weight for old posts
}
```

## 📈 **Monitoring Reddit Integration**

### **Log Monitoring**
```bash
# Check Reddit sentiment processing
grep "Reddit\|sentiment" logs/trading_system.log | tail -10

# Monitor Reddit API calls
grep "reddit.*request\|reddit.*response" logs/trading_system.log | tail -5

# Check sentiment scores
grep "sentiment.*score" logs/trading_system.log | tail -10
```

### **Performance Metrics**
```bash
# Reddit processing time
grep "Reddit sentiment.*seconds" logs/trading_system.log | tail -5

# Success rate
grep "Reddit.*success\|Reddit.*failed" logs/trading_system.log | tail -10

# Cache hit rate
grep "Reddit.*cache" logs/trading_system.log | tail -5
```

### **Signal History Impact**
```bash
# Check Reddit sentiment in signal history
cat data/signal_history.json | jq '.signals[] | select(.sentiment_score != null)'

# Average sentiment impact
cat data/signal_history.json | jq '.signals | map(.sentiment_score) | add / length'
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Authentication Errors**
**Symptoms**: `401 Unauthorized` or `Invalid client_id`
**Causes**: Incorrect credentials in `.env` file
**Solutions**:
```bash
# Verify credentials
cat .env | grep REDDIT

# Test credentials manually
python3 -c "
import praw
reddit = praw.Reddit(
    client_id='your_id',
    client_secret='your_secret', 
    user_agent='test'
)
print(reddit.user.me())
"
```

#### **2. Rate Limiting**
**Symptoms**: `429 Too Many Requests`
**Causes**: Exceeding Reddit's API limits
**Solutions**:
```bash
# Check request frequency in logs
grep "Reddit.*request" logs/trading_system.log | tail -20

# Reduce request frequency in code
# Increase cache duration
```

#### **3. No Sentiment Data**
**Symptoms**: Sentiment scores always 0.0 or null
**Causes**: No relevant posts found, network issues
**Solutions**:
```bash
# Test Reddit connectivity
python3 -c "
import asyncpraw
# Test connection and data retrieval
"

# Check subreddit activity for your symbols
# Verify symbol ticker mentions in posts
```

#### **4. Slow Performance**
**Symptoms**: Reddit sentiment processing takes >30 seconds
**Causes**: Network latency, large subreddit queries
**Solutions**:
```bash
# Reduce number of posts analyzed
# Implement more aggressive caching
# Use Reddit's faster endpoints
```

## 🔄 **Without Reddit (Optional)**

### **System Works Without Reddit**
If you prefer not to set up Reddit integration:
- System continues to work perfectly
- Uses technical + fundamental analysis only
- Dynamic thresholds still adapt to market conditions
- Market scanner still discovers opportunities

### **Disable Reddit Integration**
```python
# In config/config.yaml
sentiment_agent:
  reddit:
    enabled: false  # Disable Reddit integration
```

### **Alternative Sentiment Sources**
- **News APIs**: Financial news sentiment analysis
- **Twitter/X APIs**: Social media sentiment (if available)
- **Financial Blog RSS**: Blog post sentiment analysis
- **SEC Filings**: Corporate filing sentiment

## 📚 **Advanced Reddit Features**

### **Keyword Expansion**
```python
# Enhanced ticker detection
symbol_patterns = {
    'AAPL': ['apple', 'iphone', 'tim cook', 'cupertino'],
    'TSLA': ['tesla', 'elon', 'cybertruck', 'model 3'],
    'GOOGL': ['google', 'alphabet', 'search', 'android']
}
```

### **Sentiment Trend Analysis**
```python
# Track sentiment changes over time
sentiment_history = {
    'daily_average': [],     # Daily sentiment averages
    'volatility': [],        # Sentiment volatility
    'trend_direction': '',   # Improving/declining
    'momentum': 0.0          # Rate of change
}
```

### **Subreddit Weighting**
```python
# Different subreddits have different reliability
subreddit_weights = {
    'SecurityAnalysis': 1.0,    # High quality analysis
    'investing': 0.8,           # Good general discussion
    'stocks': 0.6,              # Mixed quality
    'wallstreetbets': 0.3       # High noise, some signal
}
```

## 🎯 **Best Practices**

### **Setup Recommendations**
1. **Use Real Account**: Don't use throwaway accounts
2. **Descriptive User Agent**: Clear bot identification
3. **Respect Rate Limits**: Stay well under API limits
4. **Monitor Performance**: Track sentiment processing times
5. **Regular Updates**: Keep PRAW library updated

### **Analysis Quality**
1. **Multiple Subreddits**: Diversify sentiment sources
2. **Time Weighting**: Recent posts more important
3. **Relevance Filtering**: Focus on stock-specific discussions
4. **Context Understanding**: Consider discussion context
5. **Sentiment Validation**: Cross-check with other indicators

### **Integration Strategy**
1. **Start Simple**: Basic sentiment integration first
2. **Monitor Impact**: Track sentiment influence on trades
3. **Tune Weights**: Adjust sentiment importance over time
4. **Validate Results**: Compare sentiment vs actual performance
5. **Continuous Learning**: Let signal history improve sentiment use

The enhanced Reddit integration now contributes to the system's learning and adaptation capabilities, helping improve trading decisions through community sentiment analysis while maintaining respect for Reddit's platform and user privacy.

**Ready to enhance your trading with Reddit sentiment?**
```bash
# Test your Reddit setup
python3 main.py --mode single --symbols AAPL TSLA --classic
```