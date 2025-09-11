# 🚀 Discord Webhook Setup (2 Minutes)

## 🎯 **Why Discord?**

- ✅ **100% Free** forever
- ✅ **Works on Railway** (HTTP-based)
- ✅ **Rich notifications** with colors and formatting
- ✅ **Mobile + desktop** alerts
- ✅ **No API limits** or restrictions
- ✅ **Easy setup** - just need a webhook URL

---

## 🚀 **Super Quick Setup**

### **Step 1: Create Discord Server** (30 seconds)
1. Open Discord (web/app)
2. Click the **"+"** to create a server
3. Choose **"Create My Own"**
4. Name it: `Trading Bot Alerts`
5. Click **"Create"**

### **Step 2: Create Webhook** (30 seconds)
1. Right-click your server name
2. **Server Settings** > **Integrations**
3. Click **"Webhooks"** > **"New Webhook"**
4. Name it: `Trading Bot`
5. Click **"Copy Webhook URL"**

### **Step 3: Add to Railway** (30 seconds)
```bash
# In Railway project > Variables tab:
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ACTUAL_URL_HERE
EMAIL_NOTIFICATIONS=true
```

### **Step 4: Deploy** (30 seconds)
```bash
git add .
git commit -m "Add Discord notifications"
git push origin main
```

**Done!** 🎉 You'll get rich notifications like this:

---

## 📱 **What You'll See**

### **Trade Alerts**
```
🤖 Trade Alert - AAPL BUY

🚀 Trade Executed: AAPL
📊 Trade Details          💼 Portfolio Status
Action: BUY               Total Value: $98,500
Quantity: 50 shares       Cash Remaining: $42,487
Price: $150.25            Active Positions: 3
Total Value: $7,512.50

AI Trading Bot • Railway Deployment
```

### **Portfolio Updates**
```
📊 Portfolio Update - $101,250 (+1,250)

📈 Portfolio Update
💰 Portfolio Summary
Total Value: $101,250.00
Cash Balance: $25,750.00
Daily P&L: +$1,250.00 (+1.25%)
Total P&L: +$1,250.00 (+1.25%)

🎯 Top Positions (3)
🟢 AAPL: 50 @ $150.25 (+125)
🟢 GOOGL: 10 @ $2,650.00 (+350)
🔴 MSFT: 25 @ $420.80 (-75)

Active Positions: 3 • AI Trading Bot
```

### **Error Alerts**
```
🚨 CRITICAL ALERT - Portfolio Risk Exceeded

🚨 CRITICAL Alert
Error Type: Portfolio Risk Exceeded
Message: Daily loss limit reached (-5%)

🖥️ Environment Info
Railway Deployment
Check logs for details

AI Trading Bot Alert System
```

---

## 🎨 **Visual Features**

- **🟢 Green**: Profitable trades/updates
- **🔴 Red**: Losses or sell orders
- **🟡 Orange**: Warnings
- **🔵 Blue**: Market scans and info
- **Rich embeds**: Professional formatting
- **Timestamps**: Exact execution times
- **Mobile push**: Instant notifications

---

## 🧪 **Test Your Setup**

### **Check Configuration**
```bash
python -c "
import os
webhook = os.getenv('DISCORD_WEBHOOK_URL')
print('Discord Webhook:', 'CONFIGURED' if webhook else 'MISSING')
if webhook:
    print('Webhook domain:', webhook.split('/')[2])
"
```

### **Send Test Message**
```bash
python -c "
from core.discord_notifications import send_simple_message
import asyncio
asyncio.run(send_simple_message('Test message from trading bot!', 'Setup Test'))
print('Test sent to Discord!')
"
```

### **Check Railway Logs**
```bash
# Look for these in Railway logs:
✅ "Discord notifications enabled"
✅ "Discord notification sent successfully"
❌ "Discord webhook error: [error]"
```

---

## 🔧 **Advanced Features**

### **Multiple Servers**
You can create webhooks for different types of alerts:
```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/TRADES_CHANNEL
DISCORD_ERRORS_WEBHOOK=https://discord.com/api/webhooks/ERRORS_CHANNEL
```

### **Custom Bot Name & Avatar**
In Discord webhook settings:
- **Name**: `AI Trading Bot`
- **Avatar**: Upload a robot/chart icon
- **Channel**: Create `#trading-alerts`

### **Notification Roles**
Tag yourself for critical alerts:
1. Discord Server > Roles > Create `@trader`
2. Assign role to yourself
3. Critical alerts will mention `@trader`

---

## 🔍 **Troubleshooting**

### **"Discord webhook error: 404"**
- ❌ **Problem**: Invalid webhook URL
- ✅ **Solution**: Re-copy webhook URL from Discord

### **"Discord webhook error: 401/403"**
- ❌ **Problem**: Webhook deleted or permissions changed
- ✅ **Solution**: Create new webhook

### **No notifications appearing**
- ❌ **Problem**: Wrong channel or webhook URL
- ✅ **Solution**: Check webhook channel in Discord server

### **Messages truncated**
- ✅ **Expected**: Discord has 2000 character limit
- ✅ **Handled**: Messages auto-truncate with "..."

---

## 💡 **Pro Tips**

### **Organization**
- Create separate channels: `#trades`, `#portfolio`, `#errors`
- Use different webhooks for each channel
- Mute channels you don't need constantly

### **Mobile Notifications**
- Enable Discord mobile notifications
- Set custom notification sounds
- Use "Priority Speaker" role for critical alerts

### **Privacy**
- Create private server (invite-only)
- Don't share webhook URLs
- Use DM channel for sensitive data

---

## 🎯 **Why Discord > Email?**

| Feature | Discord | Email | SMTP |
|---------|---------|--------|------|
| **Railway Compatible** | ✅ Yes | ✅ Yes | ❌ Blocked |
| **Setup Time** | 2 min | 5 min | 10 min |
| **Rich Formatting** | ✅ Embeds | ✅ HTML | ✅ HTML |
| **Mobile Notifications** | ✅ Instant | ⚠️ Delayed | ⚠️ Delayed |
| **Cost** | 🆓 Free | 🆓 100/day | 🆓 Free |
| **Reliability** | ✅ 99.9% | ✅ Good | ❌ Blocked |
| **Real-time** | ✅ Instant | ⚠️ Minutes | ❌ Fails |

---

## 🏆 **Success Checklist**

Your Discord setup is working when:

1. ✅ **Discord server created** with webhook
2. ✅ **Environment variable set** in Railway  
3. ✅ **Railway logs show**: `"Discord notifications enabled"`
4. ✅ **Test message received** in Discord channel
5. ✅ **Rich embeds displaying** with colors and formatting
6. ✅ **Mobile notifications working** (if app installed)

---

**🚀 Your trading bot now has beautiful, instant Discord notifications that work perfectly on Railway!**

*No more email issues - just clean, professional trading alerts in Discord.* ⚡