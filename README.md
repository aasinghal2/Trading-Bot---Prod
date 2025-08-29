# 🤖 AI-Powered Smart Trading Bot

A sophisticated, **streamlined trading system** that leverages multiple AI agents to analyze markets, manage portfolios, and execute trades automatically with advanced risk management and email notifications.

## 🌟 **ONE-COMMAND Trading Automation**

```bash
# Start your complete trading automation
./start_smart_trader.sh
```

**That's it!** Your bot will:
- 🌅 **8:30 AM**: Pre-market portfolio analysis & trades
- 🚀 **9:30 AM**: Scan top 50 stocks for opportunities  
- 📊 **Every 15min**: Monitor portfolio during market hours
- 🔍 **Hourly**: Scan for opportunities when portfolio is empty
- 📧 **Email you**: After every buy/sell transaction

---

## 🎯 **Smart Features**

### 🤖 **Multi-Agent AI System**
- **Market Data Agent**: Real-time data collection and processing
- **Technical Analyst**: Advanced technical analysis and pattern recognition
- **Fundamentals Agent**: Company financials and valuation analysis
- **Sentiment Agent**: Reddit sentiment analysis from financial subreddits (/r/investing, /r/stocks, etc.)
- **Risk Manager**: Portfolio risk assessment and management
- **Portfolio Manager**: Intelligent position sizing and allocation

### 📧 **Email Notifications**
- Detailed transaction alerts after every trade
- Portfolio status updates (empty → filled)
- Trading summaries with P&L information
- Smart scheduling based on portfolio status

### 🧠 **Adaptive Behavior**
- **Portfolio Mode**: 15-minute monitoring when you have positions
- **Scanner Mode**: Hourly opportunity scanning when portfolio is empty
- **Market Hours**: Only trades during open market hours
- **Dynamic Thresholds**: Adapts to market conditions automatically

---

## 🚀 **Quick Start (2 Minutes)**

### **Step 1: Set Up API Keys**
```bash
# Create your environment file from template
cp env_template.txt .env
nano .env

# Add your API keys:
EMAIL_NOTIFICATIONS=true
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_gmail_app_password
ALPHA_VANTAGE_API_KEY=your_key_here
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=TradingBot/1.0 (by u/YourUsername)
```

### **Step 2: Start Trading**
```bash
# Test individual components first
./start_smart_trader.sh --test-morning   # Test pre-market routine
./start_smart_trader.sh --test-check     # Test portfolio monitoring

# Start full automation (runs 24/7)
./start_smart_trader.sh
```

### **Alternative: Manual Trading**
```bash
# Single analysis cycle
python main.py --mode single --symbols AAPL GOOGL MSFT

# Market scan for opportunities
python main.py --mode scan --top 20

# Continuous trading (legacy mode)
python main.py --mode auto --interval 300
```

---

## 🌐 **Cloud Deployment (Always-On)**

### **Deploy in 5 Minutes**
```bash
# Deploy to Railway (easiest)
./scripts/deploy.sh --platform railway

# Monitor deployment
./scripts/monitor.sh --url https://your-bot.railway.app --continuous
```

### **Other Platforms**
- **Render**: `./scripts/deploy.sh --platform render`
- **DigitalOcean**: `./scripts/deploy.sh --platform digitalocean`
- **AWS**: `./scripts/deploy.sh --platform aws`
- **Google Cloud**: `./scripts/deploy.sh --platform gcp`

---

## 📊 **How It Works - SWING TRADER MODE**

### **🎯 Portfolio Mode** (When you have positions) - 7 operations/day
```
8:30 AM  → Pre-market portfolio analysis & trades
9:30 AM  → Market opens: scan for new opportunities
11:00 AM → Strategic portfolio check & rebalancing
12:00 PM → Midday opportunity scan
1:00 PM  → Afternoon portfolio check
3:30 PM  → Pre-close portfolio check
4:00 PM  → End-of-day review & summary email
```

### **🔍 Scanner Mode** (When portfolio is empty) - 4 operations/day
```
8:30 AM  → Pre-market preparation
9:30 AM  → Market opens: scan top 50 stocks
12:00 PM → Strategic midday scan & execute
2:30 PM  → Afternoon opportunity scan & execute
```

**💰 Efficiency**: 77% reduction from 30 operations/day → optimized for swing trading (3-7 day positions)

### **Email Alerts**
Every time your bot makes a trade, you get an email like:
```
🟢 BOUGHT 25.00 shares of AAPL @ $150.25 = $3,756.25
🔴 SOLD 15.00 shares of TSLA @ $220.50 = $3,307.50

Summary: Net Change: -$448.75
Portfolio: 3 stocks (AAPL, MSFT, NVDA)
```

---

## 📁 **Project Structure**

See **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** for complete directory organization.

### **Key Files**
- `start_smart_trader.sh` - **Main automation startup**
- `scripts/smart_trader.py` - Complete automation logic
- `main.py` - Manual trading operations
- `config/config.yaml` - Trading configuration
- `.env` - API keys and credentials

### **Cloud Deployment**
- `scripts/deploy.sh` - Deploy to any cloud platform
- `scripts/monitor.sh` - Monitor cloud deployment
- `Dockerfile` & `docker-compose.yml` - Containerization

---

## 📚 **Documentation**

### **Essential Guides**
- **[DEPLOYMENT_README.md](DEPLOYMENT_README.md)** - Quick deployment guide
- **[docs/guides/CLOUD_DEPLOYMENT_GUIDE.md](docs/guides/CLOUD_DEPLOYMENT_GUIDE.md)** - Complete deployment guide
- **[docs/guides/COST_OPTIMIZATION_GUIDE.md](docs/guides/COST_OPTIMIZATION_GUIDE.md)** - Reduce monthly costs

### **Advanced Documentation**
- [Comprehensive Guide](docs/guides/COMPREHENSIVE_GUIDE.md) - Complete system documentation
- [Multi-Ticker Guide](docs/guides/MULTI_TICKER_GUIDE.md) - Trading multiple symbols
- [Dynamic Thresholds Guide](docs/guides/DYNAMIC_THRESHOLDS_GUIDE.md) - Adaptive trading
- [Market Scanner Guide](docs/guides/MARKET_SCANNER_GUIDE.md) - Opportunity scanning

---

## 🎛️ **Configuration**

### **Trading Configuration** (`config/config.yaml`)
- AI agent settings and strategies
- Risk management parameters  
- Trading symbols and intervals
- Market hours and schedules

### **Environment Variables** (`.env`)
- API keys (Alpha Vantage, OpenAI, etc.)
- Email notifications setup
- Database connections
- Cloud deployment settings

---

## 💰 **Cost Overview**

### **Free Components**
- Alpha Vantage API (5 calls/min free)
- Yahoo Finance data (free) 
- Reddit API (free for sentiment analysis)
- Basic email notifications (free)

### **Paid Components** 
- Cloud hosting (~$5-30/month depending on platform)
- Optional premium APIs (Polygon, etc.)

### **Total Estimated Cost**
- **Development**: $0/month (run locally with free APIs)
- **Production**: $5-30/month (primarily cloud hosting)

---

## 🔧 **Development & Testing**

### **Local Development**
```bash
# Virtual environment setup
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Test individual components
python main.py --mode single --symbols AAPL --verbose
```

### **Docker Development**
```bash
# Build and run locally
docker-compose up --build

# Test health endpoints
curl http://localhost:8081/health
curl http://localhost:8081/health/detailed
```

---

## ⚠️ **Important Notes**

### **Risk Disclaimer**
This trading bot is for educational and research purposes. Always:
- Start with paper trading
- Use small amounts initially
- Monitor performance closely  
- Understand the risks involved

### **API Requirements**
- **Alpha Vantage**: Free tier (5 calls/min, 500/day) - Market data
- **Reddit API**: Free (100 calls/min) - Sentiment analysis from financial subreddits  
- **Email**: Gmail app password required for trade notifications
- **Optional**: Polygon API (paid) for premium market data

### **Market Hours**
- Bot respects NYSE trading hours (9:30 AM - 4:00 PM EST)
- Automatically adjusts for weekends and holidays
- Pre-market analysis starts at 8:30 AM EST

---

## 🎉 **Get Started Now!**

1. **Configure**: Edit `.env` with your API keys
2. **Test**: `./start_smart_trader.sh --test-check`
3. **Deploy**: `./scripts/deploy.sh --platform railway`
4. **Monitor**: Check your email for trade notifications!

**Your AI trading bot is ready to work 24/7! 🚀📧**