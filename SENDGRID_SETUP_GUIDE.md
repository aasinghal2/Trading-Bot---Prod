# 📧 SendGrid Setup for Railway Deployment

## 🚨 **Why SendGrid?**

Railway has **blocked outbound SMTP connections** to prevent spam. Your trading bot needs SendGrid API instead of Gmail SMTP for notifications.

**Benefits:**
- ✅ **Works on Railway** (HTTP-based, not SMTP)
- ✅ **100 free emails/day** (plenty for trading alerts)
- ✅ **Professional delivery** (better than Gmail for automated emails)
- ✅ **Reliable & fast** (no timeouts or blocks)

---

## 🚀 **Quick Setup (5 minutes)**

### **Step 1: Create SendGrid Account**
1. Go to [sendgrid.com/pricing](https://sendgrid.com/pricing/)
2. Click **"Start for free"**
3. Sign up with your email
4. Verify your email address

### **Step 2: Get API Key**
1. Login to SendGrid dashboard
2. Go to **Settings** > **API Keys**
3. Click **"Create API Key"**
4. Choose **"Full Access"** (for simplicity)
5. Name it: `Trading Bot Railway`
6. **Copy the API key** (you won't see it again!)

### **Step 3: Set Environment Variables**

**In Railway Dashboard:**
```bash
# Go to your Railway project > Variables tab
SENDGRID_API_KEY=SG.your_actual_api_key_here
FROM_EMAIL=trading-bot@yourdomain.com
TO_EMAIL=your_personal_email@gmail.com
EMAIL_NOTIFICATIONS=true
```

**Or in local .env file:**
```bash
SENDGRID_API_KEY=SG.your_actual_api_key_here
FROM_EMAIL=trading-bot@yourdomain.com  
TO_EMAIL=your_personal_email@gmail.com
EMAIL_NOTIFICATIONS=true
```

### **Step 4: Deploy to Railway**
```bash
git add .
git commit -m "Add SendGrid notifications for Railway"
git push origin main
```

**Done!** 🎉 Your bot will now send notifications via SendGrid.

---

## 🔧 **Configuration Details**

### **Environment Variables Explained**

| Variable | Description | Example |
|----------|-------------|---------|
| `SENDGRID_API_KEY` | Your SendGrid API key | `SG.ABcdef123...` |
| `FROM_EMAIL` | Sender email address | `trading-bot@yourdomain.com` |
| `TO_EMAIL` | Your email for notifications | `your-email@gmail.com` |
| `EMAIL_NOTIFICATIONS` | Enable notifications | `true` |

### **Email Addresses**

**FROM_EMAIL**: Can be any email (doesn't need to exist)
- ✅ `trading-bot@yourdomain.com`
- ✅ `noreply@yourproject.com`
- ✅ `alerts@tradingbot.app`

**TO_EMAIL**: Your real email where you want alerts
- ✅ Your Gmail, Yahoo, Outlook, etc.
- ✅ Multiple emails: `email1@gmail.com,email2@yahoo.com`

---

## 📧 **What Notifications You'll Get**

### **Trade Alerts**
```
🚀 TRADE EXECUTED

Symbol: AAPL
Action: BUY
Quantity: 50 shares
Price: $150.25
Total Value: $7,512.50

📊 Portfolio Status:
Total Value: $98,500
Cash Remaining: $42,487.50
Active Positions: 3
```

### **Portfolio Updates**
```
📊 PORTFOLIO UPDATE

💰 Total Value: $101,250.00
💵 Cash: $25,750.00
📈 Daily P&L: +$1,250.00 (+1.25%)
📊 Total P&L: +$1,250.00 (+1.25%)

🎯 Active Positions (3):
• AAPL: 50 shares @ $150.25 = $7,512.50
• GOOGL: 10 shares @ $2,650.00 = $26,500.00
• MSFT: 25 shares @ $420.80 = $10,520.00
```

### **Error Alerts**
```
🚨 CRITICAL ALERT

Error Type: Portfolio Risk Exceeded
Message: Daily loss limit reached (-5%)

🕐 Time: 2024-01-15 14:30:25
```

---

## 🧪 **Testing Your Setup**

### **Test 1: Check Configuration**
```bash
# Run locally to test
python -c "
import os
print('SendGrid API Key:', 'SET' if os.getenv('SENDGRID_API_KEY') else 'MISSING')
print('From Email:', os.getenv('FROM_EMAIL', 'NOT SET'))
print('To Email:', os.getenv('TO_EMAIL', 'NOT SET'))
"
```

### **Test 2: Send Test Email**
```bash
# Test SendGrid connection
python -c "
from core.notifications import send_error_alert
import asyncio
asyncio.run(send_error_alert('Test Alert', 'SendGrid setup test - ignore this'))
print('Test email sent!')
"
```

### **Test 3: Check Railway Logs**
```bash
# After deployment, check Railway logs for:
✅ "SendGrid notifications enabled (Railway-compatible)"
✅ "SendGrid notification sent: [subject]"
❌ "SendGrid API error: [error]"
```

---

## 🔍 **Troubleshooting**

### **"SendGrid API error: 401"**
- ❌ **Problem**: Invalid API key
- ✅ **Solution**: Double-check your `SENDGRID_API_KEY` in Railway variables

### **"SendGrid API error: 403"**
- ❌ **Problem**: API key doesn't have email permissions
- ✅ **Solution**: Create new API key with "Full Access"

### **No emails received**
- ❌ **Problem**: Wrong `TO_EMAIL` or in spam
- ✅ **Solution**: Check spam folder, verify `TO_EMAIL` address

### **"Method 'smtp' not working"**
- ✅ **Expected**: Railway blocks SMTP - this is normal
- ✅ **Solution**: Use SendGrid (automatic when `SENDGRID_API_KEY` is set)

### **Missing `requests` module**
- ❌ **Problem**: HTTP library not installed
- ✅ **Solution**: Already included in `requirements.txt`

---

## 💰 **SendGrid Pricing**

### **Free Tier** (Perfect for Trading Bot)
- ✅ **100 emails/day** forever
- ✅ **No credit card required**
- ✅ **All features included**

**Daily Usage Estimate:**
- Morning trade: 1-3 emails
- Portfolio updates: 1-2 emails  
- Market scans: 0-1 emails
- Error alerts: 0-1 emails
- **Total: ~5-7 emails/day** (well under 100 limit)

### **If You Need More**
- **$14.95/month**: 15,000 emails
- **$89.95/month**: 100,000 emails

---

## 🎯 **Migration from SMTP**

Your existing configuration still works! The bot automatically detects SendGrid:

```bash
# If SENDGRID_API_KEY is set → Uses SendGrid ✅
# If only EMAIL_USERNAME is set → Uses SMTP (blocked on Railway) ❌
# Both set → Prefers SendGrid ✅
```

**Migration steps:**
1. Add `SENDGRID_API_KEY` to Railway
2. Keep existing `EMAIL_USERNAME` for local development
3. Deploy - notifications switch to SendGrid automatically

---

## 🏆 **Success Verification**

Your SendGrid setup is working when you see:

1. ✅ **Railway logs**: `"SendGrid notifications enabled (Railway-compatible)"`
2. ✅ **After trades**: `"SendGrid notification sent: Trade Executed"`
3. ✅ **In your inbox**: Professional trading alert emails
4. ✅ **No SMTP errors** in Railway logs

---

**🚀 Ready to trade with reliable Railway notifications!**

*Your trading bot will now send professional email alerts that work perfectly on Railway's infrastructure.*