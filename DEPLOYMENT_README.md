# 🚀 PRODUCTION Cloud Deployment Guide - SWING TRADER

Deploy your AI-powered trading bot with **$100K live capital** to the cloud in under 10 minutes!

## ⚠️ **PRODUCTION DEPLOYMENT NOTICE**
**This deployment uses REAL MONEY ($100,000) for live trading!**
- ✅ SWING TRADER Mode: 7 operations/day (optimized)
- ✅ Error alerting: Critical errors sent via email
- ✅ Portfolio monitoring: End-of-day summaries
- ✅ API optimization: 70% cost reduction vs high-frequency trading

## 🎯 Fastest Deployment (Railway - Recommended)

### 1. Prepare Your $100K Production Bot
```bash
# 1. Set up PRODUCTION environment variables
cp env_template.txt .env
nano .env  # Add your PRODUCTION API keys

# PRODUCTION Configuration (REQUIRED):
ENVIRONMENT=production
DEPLOYMENT_MODE=production
SWING_TRADER_MODE=true
INITIAL_CAPITAL=100000

# Required APIs for PRODUCTION:
EMAIL_NOTIFICATIONS=true
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_gmail_app_password
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=TradingBot/1.0 (by u/YourUsername)

# 2. Test SWING TRADER locally first
docker-compose up --build
curl http://localhost:8081/health  # Should return "OK"

# 3. Test SWING TRADER schedule (RECOMMENDED)
./start_smart_trader.sh --test-morning
./start_smart_trader.sh --test-check
```

### 2. Deploy to Railway
```bash
# Use our automated deployment script
./scripts/deploy.sh --platform railway

# Or manually:
# 1. Push to GitHub
# 2. Visit railway.app
# 3. "Deploy from GitHub"
# 4. Select your repo
# 5. Add environment variables
```

### 3. Verify Deployment
```bash
# Check health (replace with your Railway URL)
curl https://your-bot.railway.app/health

# Monitor your bot
./scripts/monitor.sh --url https://your-bot.railway.app --check-once
```

**Done! Your bot is now running 24/7!** 🎉

---

## 📋 Platform Comparison

| Platform | Setup Time | Monthly Cost | Best For |
|----------|------------|--------------|----------|
| **Railway** | 5 min | $5-20 | Beginners |
| **Render** | 10 min | $7-25 | Free tier testing |
| **DigitalOcean** | 15 min | $12-40 | Full control |
| **AWS ECS** | 30 min | $15-50 | Enterprise |
| **Google Cloud** | 20 min | $10-30 | Pay-per-use |

---

## 🛠️ What We've Created

### 🌟 **Streamlined Core Files**
- **`start_smart_trader.sh`** - **ONE-COMMAND startup** (your main automation)
- **`scripts/smart_trader.py`** - Complete trading automation logic
- **`Dockerfile`** - Container configuration
- **`docker-compose.yml`** - Local development setup
- **`.dockerignore`** - Optimized builds

### 🚀 **Deployment & Monitoring**
- **`scripts/deploy.sh`** - Automated cloud deployment
- **`scripts/monitor.sh`** - Production monitoring
- **`scripts/health_check.py`** - Advanced health endpoints
- **`.github/workflows/deploy.yml`** - CI/CD automation

### 📚 **Documentation**
- **`PROJECT_STRUCTURE.md`** - **Complete project organization**
- **`docs/guides/CLOUD_DEPLOYMENT_GUIDE.md`** - Comprehensive deployment guide
- **`docs/guides/COST_OPTIMIZATION_GUIDE.md`** - Cost reduction strategies

### 📊 **Monitoring Endpoints**
- **`/health`** - Basic health check
- **`/health/detailed`** - System metrics
- **`/metrics`** - Prometheus metrics
- **`/status`** - Application status

---

## 🚨 Pre-Deployment Checklist

### 🔑 **Required API Keys**
- [ ] **Alpha Vantage API** - Free tier (5 calls/min, 500/day) from [alphavantage.co](https://www.alphavantage.co/support/#api-key)
- [ ] **Reddit API** - Free (100 calls/min) from [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
  - Create "script" type app
  - Get Client ID and Client Secret
- [ ] **Gmail App Password** - For trade notifications
  - Enable 2-Factor Authentication
  - Generate App Password in Google Account Settings

### ✅ **Pre-Deployment Tests**
- [ ] **API Keys Set** - All required keys in `.env`
- [ ] **Docker Works** - `docker-compose up` succeeds
- [ ] **Health Check** - `curl localhost:8081/health` returns OK
- [ ] **Test Trading** - `python main.py --mode single --symbols AAPL --classic` works
- [ ] **Git Repo** - Code pushed to GitHub
- [ ] **Platform Account** - Railway/Render/etc. account created
- [ ] **Monitoring** - Health check URLs bookmarked

---

## 🔧 Quick Commands

```bash
# Deploy to Railway
./scripts/deploy.sh --platform railway

# Deploy to Render  
./scripts/deploy.sh --platform render

# Deploy to DigitalOcean
./scripts/deploy.sh --platform digitalocean

# Monitor your deployment
./scripts/monitor.sh --url https://your-bot-url.com --continuous

# Check single health status
./scripts/monitor.sh --url https://your-bot-url.com --check-once

# Test Docker build locally
docker-compose up --build -d
```

---

## 💰 Cost Optimization

### Start Small
- Begin with Railway Starter ($5/month) or Render Free
- Monitor actual resource usage
- Scale up only when needed

### API Optimization
- Use free tiers: Alpha Vantage (5 calls/min), Reddit API (100 calls/min), Yahoo Finance
- Increase intervals: `real_time_interval: 300` (5 minutes)
- Cache data when possible
- Reddit sentiment: uses cached results to minimize API calls

### Schedule Trading
- Only run during market hours (9:30 AM - 4:00 PM EST)
- Scale down on weekends
- Use hibernation features

### Monitor Costs
- Set up billing alerts
- Track API usage daily
- Clean up old logs regularly

---

## 🔍 Troubleshooting

### Common Issues

**❌ Health Check Fails**
```bash
# Check logs
docker-compose logs trading-bot

# Verify environment variables
grep -v "^#" .env | grep -v "^$"

# Test locally first
python main.py --mode single --symbols AAPL
```

**❌ Out of Memory**
```bash
# Check resource usage
docker stats trading-bot

# Increase memory in deployment:
# Railway: Dashboard > Settings > Resources
# Render: Plan upgrade
# DigitalOcean: Resize droplet
```

**❌ API Rate Limits**
```bash
# Check API usage
grep "rate limit" logs/trading_system.log

# Increase intervals in config.yaml:
real_time_interval: 300  # 5 minutes
```

**❌ Trading Not Happening**
```bash
# Check market hours
date -u  # Current UTC time
# Market: 14:30-21:00 UTC (9:30 AM - 4:00 PM EST)

# Check configuration
grep -A 5 "market_hours" config/config.yaml
```

---

## 📞 Support & Resources

### Documentation
- **[Complete Deployment Guide](docs/guides/CLOUD_DEPLOYMENT_GUIDE.md)** - Detailed instructions for all platforms
- **[Cost Optimization Guide](docs/guides/COST_OPTIMIZATION_GUIDE.md)** - Reduce your monthly costs
- **[Architecture Overview](docs/architecture/)** - Understanding the system

### Monitoring URLs
After deployment, bookmark these:
- `https://your-bot-url.com/health` - Basic health
- `https://your-bot-url.com/health/detailed` - Detailed metrics  
- `https://your-bot-url.com/metrics` - Prometheus metrics
- `https://your-bot-url.com/status` - Application status

### Community
- **Reddit**: r/algotrading
- **Discord**: Trading bot communities
- **GitHub Issues**: For bug reports

---

## 🎉 Success Criteria

Your deployment is successful when:
- ✅ Health endpoint returns 200 OK
- ✅ Bot executes trading cycles automatically
- ✅ Logs show successful API calls
- ✅ Portfolio state is persisted
- ✅ No critical errors in logs
- ✅ Resource usage is within limits

---

## ⚠️ Important Reminders

1. **Start with Paper Trading** - Test thoroughly before using real money
2. **Monitor Closely** - Check logs and performance daily initially
3. **Set Stop Losses** - Configure maximum daily loss limits
4. **Keep API Keys Secure** - Never commit `.env` files
5. **Stay Within Budgets** - Set billing alerts and monitor costs
6. **Understand Risks** - Trading involves financial risk

---

**🚀 Ready to deploy? Run `./scripts/deploy.sh --platform railway` and get your bot running in the cloud!**

*Remember: Trading involves risk. Past performance doesn't guarantee future results. Always test with paper trading first.*