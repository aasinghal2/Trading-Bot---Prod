"""
Discord Webhook Notifications for Trading Bot
Railway-compatible, no account setup needed beyond Discord webhook
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger


class DiscordNotifier:
    """Discord webhook notification system for trading alerts"""
    
    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        self.enabled = bool(self.webhook_url)
        
        if self.enabled:
            logger.info("✅ Discord notifications enabled")
        else:
            logger.warning("⚠️ Discord webhook not configured - using log-only notifications")
    
    async def send_trade_alert(self, trade_details: Dict[str, Any]):
        """Send trade execution notification to Discord"""
        
        symbol = trade_details.get('symbol', 'Unknown')
        action = trade_details.get('action', 'Unknown')
        quantity = trade_details.get('quantity', 0)
        price = trade_details.get('price', 0)
        total_value = trade_details.get('total_value', quantity * price)
        
        # Create rich Discord embed
        embed = {
            "title": f"🚀 Trade Executed: {symbol}",
            "color": 0x00ff00 if action.upper() == "BUY" else 0xff4444,  # Green for buy, red for sell
            "timestamp": datetime.now().isoformat(),
            "fields": [
                {
                    "name": "📊 Trade Details",
                    "value": f"""
**Action:** {action.upper()}
**Quantity:** {quantity:,.2f} shares
**Price:** ${price:,.2f}
**Total Value:** ${total_value:,.2f}
                    """.strip(),
                    "inline": True
                },
                {
                    "name": "💼 Portfolio Status",
                    "value": f"""
**Total Value:** ${trade_details.get('portfolio_value', 'N/A')}
**Cash Remaining:** ${trade_details.get('cash_remaining', 'N/A')}
**Active Positions:** {trade_details.get('active_positions', 'N/A')}
                    """.strip(),
                    "inline": True
                }
            ],
            "footer": {
                "text": "AI Trading Bot • Railway Deployment"
            }
        }
        
        await self._send_discord_message(
            content=f"🤖 **Trade Alert** - {symbol} {action.upper()}",
            embeds=[embed]
        )
        
        # Always log the trade
        logger.info(f"🚨 TRADE_EXECUTED: {action.upper()} {quantity} {symbol} @ ${price:.2f} | Total: ${total_value:,.2f}")
    
    async def send_portfolio_update(self, portfolio_data: Dict[str, Any]):
        """Send portfolio status update to Discord"""
        
        total_value = portfolio_data.get('total_value', 0)
        cash_balance = portfolio_data.get('cash_balance', 0)
        daily_pnl = portfolio_data.get('daily_pnl', 0)
        total_pnl = portfolio_data.get('total_pnl', 0)
        positions = portfolio_data.get('positions', [])
        
        # Determine color based on P&L
        if daily_pnl > 0:
            color = 0x00ff00  # Green
            trend_emoji = "📈"
        elif daily_pnl < 0:
            color = 0xff4444  # Red
            trend_emoji = "📉"
        else:
            color = 0x888888  # Gray
            trend_emoji = "➡️"
        
        # Format positions
        position_text = ""
        for i, pos in enumerate(positions[:8]):  # Show top 8 positions
            symbol = pos.get('symbol', 'Unknown')
            quantity = pos.get('quantity', 0)
            current_price = pos.get('current_price', 0)
            unrealized_pnl = pos.get('unrealized_pnl', 0)
            pnl_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
            
            position_text += f"{pnl_emoji} **{symbol}**: {quantity:,.0f} @ ${current_price:.2f} ({unrealized_pnl:+.0f})\n"
        
        if len(positions) > 8:
            position_text += f"... and {len(positions) - 8} more positions"
        
        embed = {
            "title": f"{trend_emoji} Portfolio Update",
            "color": color,
            "timestamp": datetime.now().isoformat(),
            "fields": [
                {
                    "name": "💰 Portfolio Summary",
                    "value": f"""
**Total Value:** ${total_value:,.2f}
**Cash Balance:** ${cash_balance:,.2f}
**Daily P&L:** ${daily_pnl:,.2f} ({daily_pnl/total_value*100:+.2f}%)
**Total P&L:** ${total_pnl:,.2f} ({total_pnl/100000*100:+.2f}%)
                    """.strip(),
                    "inline": False
                }
            ],
            "footer": {
                "text": f"Active Positions: {len(positions)} • AI Trading Bot"
            }
        }
        
        # Add positions field if we have any
        if position_text:
            embed["fields"].append({
                "name": f"🎯 Top Positions ({min(len(positions), 8)})",
                "value": position_text.strip(),
                "inline": False
            })
        
        await self._send_discord_message(
            content=f"📊 **Portfolio Update** - ${total_value:,.2f} ({daily_pnl:+.2f})",
            embeds=[embed]
        )
        
        # Always log the portfolio update
        logger.info(f"📊 PORTFOLIO_UPDATE: ${total_value:,.2f} | Daily P&L: ${daily_pnl:+,.2f}")
    
    async def send_error_alert(self, error_type: str, error_message: str, critical: bool = False):
        """Send error/alert notification to Discord"""
        
        color = 0xff0000 if critical else 0xffaa00  # Red for critical, orange for warning
        emoji = "🚨" if critical else "⚠️"
        priority = "CRITICAL" if critical else "WARNING"
        
        embed = {
            "title": f"{emoji} {priority} Alert",
            "description": f"**Error Type:** {error_type}\n**Message:** {error_message}",
            "color": color,
            "timestamp": datetime.now().isoformat(),
            "fields": [
                {
                    "name": "🖥️ Environment Info",
                    "value": "Railway Deployment\nCheck logs for details",
                    "inline": True
                }
            ],
            "footer": {
                "text": "AI Trading Bot Alert System"
            }
        }
        
        await self._send_discord_message(
            content=f"{emoji} **{priority} ALERT** - {error_type}",
            embeds=[embed]
        )
        
        # Always log the error
        log_level = logger.critical if critical else logger.warning
        log_level(f"🚨 {priority}: {error_type} - {error_message}")
    
    async def send_market_scan_results(self, opportunities: list):
        """Send market scanning results to Discord"""
        
        if not opportunities:
            return
        
        embed = {
            "title": f"🔍 Market Scan Results",
            "description": f"Found **{len(opportunities)}** trading opportunities",
            "color": 0x0099ff,  # Blue
            "timestamp": datetime.now().isoformat(),
            "fields": [],
            "footer": {
                "text": "AI Trading Bot Scanner"
            }
        }
        
        # Add top 5 opportunities
        for i, opp in enumerate(opportunities[:5], 1):
            symbol = opp.get('symbol', 'Unknown')
            signal_strength = opp.get('signal_strength', 0)
            price = opp.get('current_price', 0)
            recommendation = opp.get('recommendation', 'Hold')
            
            # Determine emoji based on signal strength
            if signal_strength > 0.7:
                strength_emoji = "🟢"
            elif signal_strength > 0.4:
                strength_emoji = "🟡"
            else:
                strength_emoji = "🔴"
            
            embed["fields"].append({
                "name": f"{strength_emoji} {i}. {symbol}",
                "value": f"**Signal:** {signal_strength:.3f}\n**Price:** ${price:.2f}\n**Action:** {recommendation}",
                "inline": True
            })
        
        # Add summary field
        if len(opportunities) > 5:
            embed["fields"].append({
                "name": "📈 Summary",
                "value": f"Showing top 5 of {len(opportunities)} opportunities\nCheck logs for complete list",
                "inline": False
            })
        
        await self._send_discord_message(
            content=f"🔍 **Market Scan Complete** - {len(opportunities)} opportunities found",
            embeds=[embed]
        )
        
        # Always log scan results
        logger.info(f"🔍 MARKET_SCAN: Found {len(opportunities)} opportunities")
    
    async def send_simple_message(self, message: str, title: str = None):
        """Send a simple text message to Discord"""
        
        if title:
            embed = {
                "title": title,
                "description": message,
                "color": 0x0099ff,
                "timestamp": datetime.now().isoformat(),
                "footer": {
                    "text": "AI Trading Bot"
                }
            }
            await self._send_discord_message(content=f"📢 **{title}**", embeds=[embed])
        else:
            await self._send_discord_message(content=f"🤖 {message}")
    
    async def _send_discord_message(self, content: str = None, embeds: list = None):
        """Send message to Discord webhook"""
        
        # Always log the notification
        logger.info(f"📢 DISCORD: {content or 'Embed message'}")
        
        if not self.enabled:
            logger.warning("Discord webhook not configured - notification logged only")
            return
        
        try:
            payload = {}
            
            if content:
                payload["content"] = content
            
            if embeds:
                payload["embeds"] = embeds
            
            # Discord webhook limits
            if content and len(content) > 2000:
                payload["content"] = content[:1997] + "..."
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info("✅ Discord notification sent successfully")
            else:
                logger.error(f"❌ Discord webhook error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Failed to send Discord notification: {e}")


# Global Discord notifier instance
discord_notifier = DiscordNotifier()


# Convenience functions for easy usage
async def send_trade_alert(trade_details: Dict[str, Any]):
    """Send trade execution alert to Discord"""
    await discord_notifier.send_trade_alert(trade_details)


async def send_portfolio_update(portfolio_data: Dict[str, Any]):
    """Send portfolio status update to Discord"""
    await discord_notifier.send_portfolio_update(portfolio_data)


async def send_error_alert(error_type: str, error_message: str, critical: bool = False):
    """Send error alert to Discord"""
    await discord_notifier.send_error_alert(error_type, error_message, critical)


async def send_market_scan_results(opportunities: list):
    """Send market scan results to Discord"""
    await discord_notifier.send_market_scan_results(opportunities)


async def send_simple_message(message: str, title: str = None):
    """Send simple message to Discord"""
    await discord_notifier.send_simple_message(message, title)