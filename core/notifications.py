"""
Railway-Compatible Notification System
Replaces SMTP with SendGrid API for email notifications
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger


class NotificationManager:
    """Manages notifications using Railway-compatible services"""
    
    def __init__(self):
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        self.from_email = os.getenv("FROM_EMAIL", "trading-bot@yourdomain.com")
        self.to_email = os.getenv("TO_EMAIL", os.getenv("EMAIL_USERNAME"))
        
        # Fallback to environment variables for backwards compatibility
        if not self.to_email:
            self.to_email = os.getenv("NOTIFICATION_EMAIL", "your-email@example.com")
            
        self.enabled = bool(self.sendgrid_api_key and self.to_email)
        
        if self.enabled:
            logger.info(f"✅ SendGrid notifications enabled - sending to {self.to_email}")
        else:
            logger.warning("⚠️ SendGrid not configured - using log-only notifications")
    
    async def send_trade_alert(self, trade_details: Dict[str, Any]):
        """Send trade execution notification"""
        
        subject = f"🤖 Trade Executed: {trade_details.get('symbol', 'Unknown')}"
        
        # Create detailed trade message
        symbol = trade_details.get('symbol', 'Unknown')
        action = trade_details.get('action', 'Unknown')
        quantity = trade_details.get('quantity', 0)
        price = trade_details.get('price', 0)
        total_value = trade_details.get('total_value', quantity * price)
        
        body = f"""
🚀 TRADE EXECUTED

Symbol: {symbol}
Action: {action.upper()}
Quantity: {quantity:,.2f} shares
Price: ${price:,.2f}
Total Value: ${total_value:,.2f}

📊 Portfolio Status:
Total Value: ${trade_details.get('portfolio_value', 'N/A')}
Cash Remaining: ${trade_details.get('cash_remaining', 'N/A')}
Active Positions: {trade_details.get('active_positions', 'N/A')}

🕐 Executed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
AI-Powered Trading Bot on Railway
        """.strip()
        
        await self._send_notification(subject, body)
        
        # Always log the trade
        logger.info(f"🚨 TRADE_EXECUTED: {action.upper()} {quantity} {symbol} @ ${price:.2f} | Total: ${total_value:,.2f}")
    
    async def send_portfolio_update(self, portfolio_data: Dict[str, Any]):
        """Send portfolio status update"""
        
        subject = f"📊 Portfolio Update - ${portfolio_data.get('total_value', 0):,.2f}"
        
        positions = portfolio_data.get('positions', [])
        daily_pnl = portfolio_data.get('daily_pnl', 0)
        total_pnl = portfolio_data.get('total_pnl', 0)
        
        body = f"""
📊 PORTFOLIO UPDATE

💰 Total Value: ${portfolio_data.get('total_value', 0):,.2f}
💵 Cash: ${portfolio_data.get('cash_balance', 0):,.2f}
📈 Daily P&L: ${daily_pnl:,.2f} ({daily_pnl/portfolio_data.get('total_value', 1)*100:+.2f}%)
📊 Total P&L: ${total_pnl:,.2f} ({total_pnl/100000*100:+.2f}%)

🎯 Active Positions ({len(positions)}):
        """
        
        for pos in positions[:10]:  # Show top 10 positions
            symbol = pos.get('symbol', 'Unknown')
            quantity = pos.get('quantity', 0)
            current_price = pos.get('current_price', 0)
            value = pos.get('market_value', quantity * current_price)
            pnl = pos.get('unrealized_pnl', 0)
            
            body += f"""
• {symbol}: {quantity:,.0f} shares @ ${current_price:.2f} = ${value:,.2f} ({pnl:+.2f})"""
        
        body += f"""

🕐 Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
AI-Powered Trading Bot on Railway
        """.strip()
        
        await self._send_notification(subject, body)
        
        # Always log the portfolio update
        logger.info(f"📊 PORTFOLIO_UPDATE: ${portfolio_data.get('total_value', 0):,.2f} | Daily P&L: ${daily_pnl:+,.2f}")
    
    async def send_error_alert(self, error_type: str, error_message: str, critical: bool = False):
        """Send error/alert notification"""
        
        priority = "🚨 CRITICAL" if critical else "⚠️ WARNING"
        subject = f"{priority}: {error_type}"
        
        body = f"""
{priority} ALERT

Error Type: {error_type}
Message: {error_message}

🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🖥️ Environment: Railway Deployment

Please check the Railway logs for more details:
https://railway.app/project/your-project/deployments

---
AI-Powered Trading Bot Alert System
        """.strip()
        
        await self._send_notification(subject, body)
        
        # Always log the error
        log_level = logger.critical if critical else logger.warning
        log_level(f"🚨 {priority}: {error_type} - {error_message}")
    
    async def send_market_scan_results(self, opportunities: list):
        """Send market scanning results"""
        
        if not opportunities:
            return
            
        subject = f"🔍 Market Scan: {len(opportunities)} Opportunities Found"
        
        body = f"""
🔍 MARKET SCAN RESULTS

Found {len(opportunities)} trading opportunities:

"""
        
        for i, opp in enumerate(opportunities[:5], 1):  # Show top 5
            symbol = opp.get('symbol', 'Unknown')
            signal_strength = opp.get('signal_strength', 0)
            price = opp.get('current_price', 0)
            
            body += f"""
{i}. {symbol}
   Signal Strength: {signal_strength:.3f}
   Current Price: ${price:.2f}
   Recommendation: {opp.get('recommendation', 'Hold')}

"""
        
        body += f"""
🕐 Scan completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
AI-Powered Trading Bot Scanner
        """.strip()
        
        await self._send_notification(subject, body)
        
        # Always log scan results
        logger.info(f"🔍 MARKET_SCAN: Found {len(opportunities)} opportunities")
    
    async def _send_notification(self, subject: str, body: str):
        """Send notification via SendGrid API"""
        
        # Always log the notification
        logger.info(f"📧 NOTIFICATION: {subject}")
        
        if not self.enabled:
            logger.warning("SendGrid not configured - notification logged only")
            return
        
        try:
            # SendGrid API v3 payload
            payload = {
                "personalizations": [
                    {
                        "to": [{"email": self.to_email}],
                        "subject": subject
                    }
                ],
                "from": {"email": self.from_email},
                "content": [
                    {
                        "type": "text/plain",
                        "value": body
                    }
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {self.sendgrid_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 202:
                logger.info(f"✅ Email sent successfully via SendGrid")
            else:
                logger.error(f"❌ SendGrid API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Failed to send notification via SendGrid: {e}")


# Global notification manager instance
notification_manager = NotificationManager()


# Convenience functions for easy usage
async def send_trade_alert(trade_details: Dict[str, Any]):
    """Send trade execution alert"""
    await notification_manager.send_trade_alert(trade_details)


async def send_portfolio_update(portfolio_data: Dict[str, Any]):
    """Send portfolio status update"""
    await notification_manager.send_portfolio_update(portfolio_data)


async def send_error_alert(error_type: str, error_message: str, critical: bool = False):
    """Send error alert"""
    await notification_manager.send_error_alert(error_type, error_message, critical)


async def send_market_scan_results(opportunities: list):
    """Send market scan results"""
    await notification_manager.send_market_scan_results(opportunities)