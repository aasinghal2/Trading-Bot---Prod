#!/usr/bin/env python3
"""
Quick test script for Discord notifications
Run this to verify your Discord webhook is working
"""

import asyncio
import os
from core.discord_notifications import send_simple_message, send_trade_alert, send_portfolio_update

async def test_discord_notifications():
    """Test Discord notification system"""
    
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    
    if not webhook_url:
        print("❌ DISCORD_WEBHOOK_URL not set in environment")
        print("Set it in .env file or Railway variables")
        print("Example: DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_URL")
        return
    
    print("✅ Discord webhook configured")
    print(f"Webhook domain: {webhook_url.split('/')[2]}")
    print()
    
    try:
        # Test 1: Simple message
        print("📢 Sending test message...")
        await send_simple_message("Discord notifications are working! 🚀", "Test Setup")
        print("✅ Simple message sent")
        
        # Test 2: Trade alert
        print("📊 Sending test trade alert...")
        trade_details = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 50.0,
            'price': 150.25,
            'total_value': 7512.50,
            'portfolio_value': 98500.00,
            'cash_remaining': 42487.50,
            'active_positions': 3
        }
        await send_trade_alert(trade_details)
        print("✅ Trade alert sent")
        
        # Test 3: Portfolio update
        print("💼 Sending test portfolio update...")
        portfolio_data = {
            'total_value': 101250.00,
            'cash_balance': 25750.00,
            'daily_pnl': 1250.00,
            'total_pnl': 1250.00,
            'positions': [
                {'symbol': 'AAPL', 'quantity': 50, 'current_price': 150.25, 'unrealized_pnl': 125},
                {'symbol': 'GOOGL', 'quantity': 10, 'current_price': 2650.00, 'unrealized_pnl': 350},
                {'symbol': 'MSFT', 'quantity': 25, 'current_price': 420.80, 'unrealized_pnl': -75}
            ]
        }
        await send_portfolio_update(portfolio_data)
        print("✅ Portfolio update sent")
        
        print()
        print("🎉 All Discord tests passed!")
        print("Check your Discord channel for the test messages")
        print()
        print("Your trading bot notifications are ready for Railway! 🚀")
        
    except Exception as e:
        print(f"❌ Discord test failed: {e}")
        print("Check your DISCORD_WEBHOOK_URL and try again")

if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Note: python-dotenv not installed, using system environment variables")
    
    asyncio.run(test_discord_notifications())