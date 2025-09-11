#!/usr/bin/env python3
"""
Smart Trading Automation - Streamlined Daily Trading Bot

This script implements your exact requirements:
1. Every morning before trading begins -> scan current portfolio and execute trades
2. Every morning post trading begins, check portfolio every 15 minutes  
3. Before first check, scan top 50 stocks and update dynamic threshold for new opportunities
4. Send new opportunities for in-depth analysis and execute based on recommendations
"""

import asyncio
import schedule
import time
import signal
import sys
import pytz
from datetime import datetime, time as dt_time
from typing import List, Optional, Dict, Any
import json
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from main import main as trading_main
from market_scanner import execute_market_scan
# Discord notifications disabled - using email notifications only

class SmartTrader:
    """Streamlined daily trading automation"""
    
    def __init__(self):
        self.running = True
        
        # Set up timezone for US Eastern Time (market timezone)
        self.market_tz = pytz.timezone('US/Eastern')
        
        # Detect if we're running in Railway (UTC) or local (SGT)
        import os
        if os.getenv('RAILWAY_ENVIRONMENT') == 'true' or os.getenv('DEPLOYMENT_MODE') == 'production':
            self.local_tz = pytz.timezone('UTC')  # Railway uses UTC
            logger.info("🌍 Detected Railway deployment - using UTC timezone")
        else:
            self.local_tz = pytz.timezone('Asia/Singapore')  # SGT for local development
            logger.info("🌍 Detected local development - using Singapore timezone")
        
        self.market_open = dt_time(9, 30)  # 9:30 AM EST
        self.market_close = dt_time(16, 0)  # 4:00 PM EST
        self.current_symbols = self._load_current_symbols()
        self.portfolio_empty = len(self.current_symbols) == 0
        self.last_portfolio_check = None
        
        # Email configuration
        self.email_config = self._load_email_config()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Log current times for verification
        now_local = datetime.now(self.local_tz)
        now_market = now_local.astimezone(self.market_tz)
        logger.info(f"📊 Smart Trader initialized")
        logger.info(f"🌍 Market timezone: US/Eastern")
        local_tz_display = 'UTC' if self.local_tz.zone == 'UTC' else 'SGT'
        logger.info(f"🕐 Local time ({local_tz_display}): {now_local.strftime('%H:%M:%S')}")
        logger.info(f"🕐 Market time (ET): {now_market.strftime('%H:%M:%S')}")
        logger.info(f"Portfolio empty: {self.portfolio_empty}")
        logger.info(f"Current symbols: {self.current_symbols}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _load_current_symbols(self) -> List[str]:
        """Load current portfolio symbols from portfolio state"""
        try:
            if os.path.exists('data/portfolio_state.json'):
                with open('data/portfolio_state.json', 'r') as f:
                    portfolio_data = json.load(f)
                    positions = portfolio_data.get('positions', {})
                    symbols = [symbol for symbol, data in positions.items() 
                              if data.get('size', 0) > 0]
                    if symbols:
                        logger.info(f"📊 Current portfolio symbols: {', '.join(symbols)}")
                        return symbols
                    else:
                        logger.info("📊 Portfolio is empty - no current positions")
                        return []
        except Exception as e:
            logger.warning(f"Could not load portfolio symbols: {e}")
        
        # Return empty list if no portfolio file or error - will trigger scanner mode
        logger.info("📊 No portfolio found - starting in scanner mode")
        return []
    
    def _load_email_config(self) -> Dict[str, str]:
        """Load notification configuration (Discord + SendGrid + Mailgun + SMTP fallbacks)"""
        # Check notification methods in order of preference for Railway
        discord_enabled = bool(os.getenv('DISCORD_WEBHOOK_URL'))
        sendgrid_enabled = bool(os.getenv('SENDGRID_API_KEY'))
        mailgun_enabled = bool(os.getenv('MAILGUN_API_KEY') and os.getenv('MAILGUN_DOMAIN'))
        smtp_requested = os.getenv('EMAIL_NOTIFICATIONS', 'false').lower() == 'true'
        
        config = {
            'enabled': discord_enabled or sendgrid_enabled or mailgun_enabled or smtp_requested,
            'method': 'discord' if discord_enabled else ('sendgrid' if sendgrid_enabled else ('mailgun' if mailgun_enabled else 'smtp')),
            'discord_webhook': os.getenv('DISCORD_WEBHOOK_URL', ''),
            'sendgrid_api_key': os.getenv('SENDGRID_API_KEY', ''),
            'from_email': os.getenv('FROM_EMAIL', 'trading-bot@yourdomain.com'),
            'to_email': os.getenv('TO_EMAIL', os.getenv('EMAIL_USERNAME', '')),
            # Legacy SMTP config for fallback
            'smtp_server': os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', '587')),
            'username': os.getenv('EMAIL_USERNAME', ''),
            'password': os.getenv('EMAIL_PASSWORD', '')
        }
        
        if config['enabled']:
            if config['method'] == 'discord':
                logger.info("✅ Discord notifications enabled (Railway-compatible)")
            elif config['method'] == 'sendgrid':
                logger.info("✅ SendGrid notifications enabled (Railway-compatible)")
            elif config['method'] == 'mailgun':
                logger.info("✅ Mailgun notifications enabled (Railway-compatible)")
            else:
                logger.info("📧 SMTP notifications enabled (may not work on Railway)")
                if not config['username']:
                    logger.warning("📧 Email notifications enabled but EMAIL_USERNAME not set")
                    config['enabled'] = False
        
        return config
    
    def is_market_open(self) -> bool:
        """Check if market is currently open (weekdays 9:30-16:00 ET)"""
        # First check if it's a market day (Monday-Friday)
        if not self._is_market_day():
            return False
        
        # Then check if it's within market hours
        now_et = datetime.now(self.market_tz)
        current_time = now_et.time()
        return self.market_open <= current_time <= self.market_close
    
    def _schedule_async_task(self, coro_func):
        """Helper method to schedule async tasks with proper error handling"""
        try:
            # Get the current event loop if running, or create a new task
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a task in the running loop
                task = loop.create_task(coro_func())
                # Add done callback for error handling
                task.add_done_callback(self._handle_task_result)
            else:
                # If no loop is running, run in a new thread
                import threading
                thread = threading.Thread(target=lambda: asyncio.run(coro_func()))
                thread.start()
        except Exception as e:
            logger.error(f"❌ Error scheduling async task {coro_func.__name__}: {e}")
    
    def _handle_task_result(self, task):
        """Handle the result of a scheduled async task"""
        try:
            if task.exception():
                logger.error(f"❌ Scheduled task failed: {task.exception()}")
            else:
                logger.debug(f"✅ Scheduled task completed successfully")
        except Exception as e:
            logger.error(f"❌ Error handling task result: {e}")
    
    async def morning_pre_market_routine(self):
        """
        Step 1: Morning pre-market routine (before 9:30 AM)
        - Scan current portfolio
        - Execute trades based on recommendations
        """
        logger.info("🌅 Starting morning pre-market routine...")
        
        # Check if it's a market day
        if not self._is_market_day():
            logger.info("🏖️ Skipping pre-market routine - market closed on weekends")
            return
        
        try:
            # Scan and analyze current portfolio
            await self._execute_portfolio_analysis()
            
            logger.success("✅ Morning pre-market routine completed")
            
        except Exception as e:
            logger.error(f"❌ Error in morning pre-market routine: {e}")
            await self._send_error_alert("Morning Pre-Market Error", str(e), critical=True)
    
    async def market_opening_routine(self):
        """
        Step 3: Market opening routine (at 9:30 AM)
        - Scan top 50 stocks
        - Update dynamic thresholds  
        - Find new opportunities
        - Execute in-depth analysis and trades
        """
        logger.info("🚀 Starting market opening routine...")
        
        # Check if it's a market day
        if not self._is_market_day():
            logger.info("🏖️ Skipping market opening routine - market closed on weekends")
            return
        
        try:
            # Step 3a: Scan top 50 stocks and update dynamic thresholds
            logger.info("🔍 Scanning top 50 stocks for new opportunities...")
            
            # Execute market scan using existing functionality
            await self._execute_market_scan()
            
            # Step 3b: Get new opportunities and execute in-depth analysis
            await self._analyze_new_opportunities()
            
            logger.success("✅ Market opening routine completed")
            
        except Exception as e:
            logger.error(f"❌ Error in market opening routine: {e}")
            await self._send_error_alert("Market Opening Error", str(e), critical=True)
    
    async def portfolio_check_routine(self):
        """
        Step 2: Regular portfolio monitoring (every 15 minutes during market hours)
        - Check portfolio status
        - Execute trades if needed
        """
        if not self.is_market_open():
            logger.info("🌙 Market closed, skipping portfolio check")
            return
        
        logger.info("📊 Running 15-minute portfolio check...")
        
        try:
            # Check if portfolio is empty and switch to scanner mode if needed
            if self.portfolio_empty:
                logger.info("📊 Portfolio empty - skipping regular check, using hourly scanner instead")
                return
            
            # Quick portfolio analysis and rebalancing
            await self._execute_portfolio_analysis()
            
            logger.info("✅ Portfolio check completed")
            
        except Exception as e:
            logger.error(f"❌ Error in portfolio check: {e}")
            await self._send_error_alert("Portfolio Check Error", str(e), critical=False)
    
    async def hourly_scanner_routine(self):
        """
        Strategic scanner routine (SWING TRADER MODE)
        - Scan market for opportunities
        - Execute trades on recommended stocks
        """
        logger.info("🔍 Running strategic market scan (SWING TRADER MODE)...")
        
        # Check if it's a market day
        if not self._is_market_day():
            logger.info("🏖️ Skipping strategic scan - market closed on weekends")
            return
        
        try:
            # Execute market scan
            await self._execute_market_scan()
            
            # Analyze and execute on opportunities
            await self._analyze_new_opportunities()
            
            # Check if we now have positions
            old_portfolio_empty = self.portfolio_empty
            self.current_symbols = self._load_current_symbols()
            self.portfolio_empty = len(self.current_symbols) == 0
            
            if old_portfolio_empty and not self.portfolio_empty:
                logger.success("🎯 Portfolio no longer empty - switching to SWING TRADER monitoring mode")
                await self._send_email_notification(
                    "Portfolio Status Change - Swing Trader",
                    f"Portfolio is no longer empty. New positions: {', '.join(self.current_symbols)}"
                )
            
            logger.info("✅ Strategic scan completed")
            
        except Exception as e:
            logger.error(f"❌ Error in strategic scanner: {e}")
            await self._send_error_alert("Strategic Scanner Error", str(e), critical=False)

    async def end_of_day_review(self):
        """
        End-of-day review routine (SWING TRADER MODE)
        - Final portfolio check
        - Performance summary
        - Position evaluation for next day
        """
        logger.info("📊 Running end-of-day review (SWING TRADER MODE)...")
        
        # Check if it's a market day (but allow EOD review to run on weekends for portfolio status)
        if not self._is_market_day():
            logger.info("🏖️ Weekend end-of-day review - showing portfolio status only")
        
        try:
            # Quick portfolio analysis without trading (review only)
            if self.current_symbols:
                logger.info(f"📈 End-of-day portfolio review: {', '.join(self.current_symbols)}")
                
                # Get portfolio state for review
                portfolio_state = await self._get_portfolio_state()
                
                # Send end-of-day summary email
                await self._send_end_of_day_summary(portfolio_state)
                
                # Debug portfolio state for Railway inspection
                self.debug_portfolio_state()
            else:
                logger.info("📊 End-of-day: Portfolio empty, no positions to review")
                # Still debug even if empty to show current state
                self.debug_portfolio_state()
            
            logger.info("✅ End-of-day review completed")
            
        except Exception as e:
            logger.error(f"❌ Error in end-of-day review: {e}")
            await self._send_error_alert("End-of-Day Review Error", str(e), critical=False)
    
    async def _execute_market_scan(self):
        """Execute market scan for top 50 stocks"""
        logger.info("🔍 Scanning top 50 stocks...")
        
        # Use the existing market scanner
        original_argv = sys.argv.copy()
        try:
            sys.argv = [
                'main.py',
                '--mode', 'scan',
                '--top', '50',
                '--min-pe', '0',
                '--max-pe', '50'
            ]
            await trading_main()
        finally:
            sys.argv = original_argv
    
    async def _analyze_new_opportunities(self):
        """Analyze new opportunities from market scan"""
        logger.info("📈 Analyzing new opportunities...")
        
        # Check if market scan found new opportunities
        scan_results_file = self._find_latest_market_scan_file()
        if scan_results_file and os.path.exists(scan_results_file):
            try:
                with open(scan_results_file, 'r') as f:
                    scan_data = json.load(f)
                    
                buy_recommendations = scan_data.get('buy_recommendations', [])
                if buy_recommendations:
                    # Get symbols of strong buy opportunities
                    new_symbols = [rec['symbol'] for rec in buy_recommendations[:10]]  # Top 10 buy recommendations
                    logger.info(f"🎯 Found {len(new_symbols)} buy opportunities: {', '.join(new_symbols)}")
                    
                    # Store portfolio state before trading
                    old_portfolio_state = await self._get_portfolio_state()
                    
                    # Execute in-depth analysis on new opportunities
                    await self._execute_trading_cycle(new_symbols)
                    
                    # Check for transactions and send notifications
                    await self._check_and_notify_transactions(old_portfolio_state, "Market Scan")
                else:
                    logger.info("📊 No buy recommendations found in market scan")
                    
            except Exception as e:
                logger.warning(f"Could not read market scan results: {e}")
        else:
            logger.info("📊 No market scan results found, proceeding with portfolio analysis")
    
    def _find_latest_market_scan_file(self):
        """Find the most recent market scan file"""
        try:
            recommendations_dir = 'recommendations'
            if not os.path.exists(recommendations_dir):
                return None
                
            # Find all market scan files with timestamp pattern
            scan_files = []
            for filename in os.listdir(recommendations_dir):
                if filename.startswith('market_scan_') and filename.endswith('.json'):
                    scan_files.append(filename)
            
            if not scan_files:
                return None
                
            # Sort by filename (which includes timestamp) to get most recent
            scan_files.sort(reverse=True)
            latest_file = os.path.join(recommendations_dir, scan_files[0])
            logger.info(f"📁 Found latest market scan file: {scan_files[0]}")
            return latest_file
            
        except Exception as e:
            logger.warning(f"Error finding market scan file: {e}")
            return None
    
    async def _execute_portfolio_analysis(self):
        """Execute trading cycle on current portfolio symbols"""
        logger.info(f"📊 Analyzing portfolio symbols: {', '.join(self.current_symbols)}")
        
        # Store portfolio state before trading
        old_portfolio_state = await self._get_portfolio_state()
        
        await self._execute_trading_cycle(self.current_symbols)
        
        # Refresh portfolio symbols after trading
        old_symbols = self.current_symbols.copy()
        self.current_symbols = self._load_current_symbols()
        old_portfolio_empty = self.portfolio_empty
        self.portfolio_empty = len(self.current_symbols) == 0
        
        # Check for transactions and send email notifications
        await self._check_and_notify_transactions(old_portfolio_state, "Portfolio Analysis")
        
        # Check if portfolio became empty
        if not old_portfolio_empty and self.portfolio_empty:
            logger.warning("📊 Portfolio became empty - switching to hourly scanner mode")
            await self._send_email_notification(
                "Portfolio Empty",
                "Portfolio is now empty. Switching to hourly scanner mode for new opportunities."
            )
    
    async def _execute_trading_cycle(self, symbols: List[str]):
        """Execute a single trading cycle for given symbols"""
        if not symbols:
            logger.warning("No symbols to analyze")
            return
        
        original_argv = sys.argv.copy()
        try:
            sys.argv = [
                'main.py',
                '--mode', 'single',
                '--symbols'] + symbols + [
                '--verbose'
            ]
            await trading_main()
        finally:
            sys.argv = original_argv
    
    async def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state for transaction comparison"""
        try:
            if os.path.exists('data/portfolio_state.json'):
                with open('data/portfolio_state.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load portfolio state: {e}")
        return {}
    
    async def _check_and_notify_transactions(self, old_portfolio_state: Dict[str, Any], scan_type: str = "Unknown"):
        """Check for transactions and send email notifications"""
        try:
            current_portfolio_state = await self._get_portfolio_state()
            
            old_positions = old_portfolio_state.get('positions', {})
            current_positions = current_portfolio_state.get('positions', {})
            
            transactions = []
            
            # Check for new positions (buys)
            for symbol, data in current_positions.items():
                current_qty = data.get('size', 0)
                old_qty = old_positions.get(symbol, {}).get('size', 0)
                
                if current_qty > old_qty:
                    bought = current_qty - old_qty
                    avg_price = data.get('entry_price', 0)
                    transactions.append({
                        'type': 'BUY',
                        'symbol': symbol,
                        'quantity': bought,
                        'price': avg_price,
                        'value': bought * avg_price
                    })
            
            # Check for reduced positions (sells)
            for symbol, data in old_positions.items():
                old_qty = data.get('size', 0)
                current_qty = current_positions.get(symbol, {}).get('size', 0)
                
                if current_qty < old_qty:
                    sold = old_qty - current_qty
                    # Estimate sell price from current market data
                    current_price = current_positions.get(symbol, {}).get('current_price', data.get('entry_price', 0))
                    transactions.append({
                        'type': 'SELL',
                        'symbol': symbol,
                        'quantity': sold,
                        'price': current_price,
                        'value': sold * current_price
                    })
            
            # Send email notifications for transactions
            if transactions:
                await self._send_transaction_email(transactions, scan_type, current_portfolio_state)
                
        except Exception as e:
            logger.error(f"Error checking transactions: {e}")
    
    async def _send_transaction_email(self, transactions: List[Dict[str, Any]], scan_type: str, current_portfolio_state: Dict[str, Any]):
        """Send comprehensive email notification for transactions"""
        if not self.email_config['enabled']:
            logger.info("📧 Email notifications disabled")
            return
        
        try:
            from datetime import datetime
            import yfinance as yf
            
            execution_time = datetime.now()
            subject = f"🤖 Trading Bot Execution Report - {len(transactions)} Transaction(s)"
            
            # Get current market prices for unrealized P&L calculation
            current_prices = {}
            if current_portfolio_state.get('positions'):
                symbols = list(current_portfolio_state['positions'].keys())
                try:
                    tickers = yf.Tickers(' '.join(symbols))
                    for symbol in symbols:
                        ticker_data = tickers.tickers[symbol].info
                        current_prices[symbol] = ticker_data.get('currentPrice', 
                                                               ticker_data.get('regularMarketPrice', 0))
                except Exception as e:
                    logger.warning(f"Could not fetch current prices for unrealized P&L: {e}")
            
            # Build comprehensive email body
            body_lines = [
                "🤖 AI TRADING BOT - EXECUTION REPORT",
                "=" * 60,
                "",
                f"📅 Execution Date & Time: {execution_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
                f"🔍 Trigger: {scan_type}",
                f"📊 Total Transactions: {len(transactions)}",
                "",
                "=" * 60,
                "📈 TRANSACTION DETAILS",
                "=" * 60
            ]
            
            total_bought_value = 0
            total_sold_value = 0
            total_realized_pnl = 0
            
            for i, txn in enumerate(transactions, 1):
                action_emoji = "🟢 BUY" if txn['type'] == 'BUY' else "🔴 SELL"
                symbol = txn['symbol']
                quantity = txn['quantity']
                price = txn['price']
                value = txn['value']
                
                body_lines.extend([
                    f"",
                    f"{i}. {action_emoji} {symbol}",
                    f"   Units: {quantity:.4f} shares",
                    f"   Price: ${price:.2f} per share",
                    f"   Total Value: ${value:.2f}"
                ])
                
                if txn['type'] == 'BUY':
                    total_bought_value += value
                else:
                    total_sold_value += value
                    # Calculate realized P&L for sells using trade history
                    realized_pnl = 0
                    if 'trade_history' in current_portfolio_state:
                        # Find the most recent buy price for this symbol from trade history
                        for trade in reversed(current_portfolio_state['trade_history']):
                            if trade.get('symbol') == symbol and trade.get('size', 0) > 0:
                                buy_price = trade.get('price', price)
                                realized_pnl = (price - buy_price) * quantity
                                total_realized_pnl += realized_pnl
                                break
                    
                    pnl_emoji = "🟢" if realized_pnl >= 0 else "🔴"
                    body_lines.append(f"   {pnl_emoji} Realized P&L: ${realized_pnl:,.2f}")
            
            # Transaction Summary
            body_lines.extend([
                "",
                "=" * 60,
                "💰 TRANSACTION SUMMARY",
                "=" * 60,
                f"💵 Total Purchased: ${total_bought_value:,.2f}",
                f"💵 Total Sold: ${total_sold_value:,.2f}",
                f"📊 Net Cash Flow: ${total_sold_value - total_bought_value:,.2f}",
                f"💰 Total Realized P&L: ${total_realized_pnl:,.2f}",
                f"💰 Closing Cash Balance: ${current_portfolio_state.get('cash_balance', 0):,.2f}",
                ""
            ])
            
            # Current Portfolio Status
            positions = current_portfolio_state.get('positions', {})
            if positions:
                body_lines.extend([
                    "=" * 60,
                    "📊 CURRENT PORTFOLIO POSITIONS",
                    "=" * 60
                ])
                
                total_portfolio_value = 0
                total_unrealized_pnl = 0
                
                for symbol, position in positions.items():
                    shares = position.get('size', 0)
                    entry_price = position.get('entry_price', 0)
                    current_price = current_prices.get(symbol, entry_price)
                    
                    position_value = shares * current_price
                    position_cost = shares * entry_price
                    unrealized_pnl = position_value - position_cost
                    unrealized_pnl_pct = (unrealized_pnl / position_cost * 100) if position_cost > 0 else 0
                    
                    total_portfolio_value += position_value
                    total_unrealized_pnl += unrealized_pnl
                    
                    pnl_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
                    
                    body_lines.extend([
                        f"",
                        f"📈 {symbol}",
                        f"   Shares: {shares:.4f}",
                        f"   Entry Price: ${entry_price:.2f}",
                        f"   Current Price: ${current_price:.2f}",
                        f"   Position Value: ${position_value:,.2f}",
                        f"   {pnl_emoji} Unrealized P&L: ${unrealized_pnl:,.2f} ({unrealized_pnl_pct:+.2f}%)"
                    ])
                
                # Portfolio Summary
                total_account_value = total_portfolio_value + current_portfolio_state.get('cash_balance', 0)
                initial_capital = current_portfolio_state.get('initial_capital', 100000)
                total_account_pnl = total_account_value - initial_capital
                total_account_pnl_pct = (total_account_pnl / initial_capital * 100) if initial_capital > 0 else 0
                
                account_pnl_emoji = "🟢" if total_account_pnl >= 0 else "🔴"
                
                body_lines.extend([
                    "",
                    "=" * 60,
                    "💼 ACCOUNT SUMMARY", 
                    "=" * 60,
                    f"💰 Cash Balance: ${current_portfolio_state.get('cash_balance', 0):,.2f}",
                    f"📊 Portfolio Value: ${total_portfolio_value:,.2f}",
                    f"🏦 Total Account Value: ${total_account_value:,.2f}",
                    f"💵 Initial Capital: ${initial_capital:,.2f}",
                    f"{account_pnl_emoji} Total Account P&L: ${total_account_pnl:,.2f} ({total_account_pnl_pct:+.2f}%)",
                    f"📈 Total Unrealized P&L: ${total_unrealized_pnl:,.2f}",
                    f"📊 Number of Positions: {len(positions)}"
                ])
            else:
                body_lines.extend([
                    "=" * 60,
                    "📊 CURRENT PORTFOLIO",
                    "=" * 60,
                    "💼 Portfolio is currently empty",
                    f"💰 Cash Balance: ${current_portfolio_state.get('cash_balance', 0):,.2f}"
                ])
            
            body_lines.extend([
                "",
                "=" * 60,
                "🤖 This is an automated notification from your AI Trading Bot",
                f"⏰ Report generated at: {execution_time.strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 60
            ])
            
            body = "\n".join(body_lines)
            
            await self._send_email_notification(subject, body)
            logger.success(f"📧 Comprehensive transaction email sent for {len(transactions)} transactions")
            
        except Exception as e:
            logger.error(f"Error sending transaction email: {e}")

    async def _send_end_of_day_summary(self, portfolio_state: Dict[str, Any]):
        """Send end-of-day portfolio summary email"""
        if not self.email_config['enabled']:
            logger.info("📧 Email notifications disabled")
            return

        try:
            from datetime import datetime
            import yfinance as yf

            execution_time = datetime.now()
            subject = f"📊 End-of-Day Portfolio Summary - {execution_time.strftime('%Y-%m-%d')}"

            # Get current market prices
            current_prices = {}
            positions = portfolio_state.get('positions', {})
            if positions:
                symbols = list(positions.keys())
                try:
                    tickers = yf.Tickers(' '.join(symbols))
                    for symbol in symbols:
                        ticker_data = tickers.tickers[symbol].info
                        current_prices[symbol] = ticker_data.get('currentPrice', 
                                                               ticker_data.get('regularMarketPrice', 0))
                except Exception as e:
                    logger.warning(f"Could not fetch current prices for EOD summary: {e}")

            # Build EOD summary
            body_lines = [
                "📊 SWING TRADER - END-OF-DAY SUMMARY",
                "=" * 50,
                "",
                f"📅 Date: {execution_time.strftime('%Y-%m-%d')}",
                f"⏰ Review Time: {execution_time.strftime('%H:%M:%S %Z')}",
                f"🎯 Mode: SWING TRADER (7 operations/day)",
                "",
                "=" * 50,
                "💼 PORTFOLIO OVERVIEW",
                "=" * 50
            ]

            if positions:
                total_portfolio_value = 0
                total_unrealized_pnl = 0

                for symbol, position in positions.items():
                    shares = position.get('size', 0)
                    entry_price = position.get('entry_price', 0)
                    current_price = current_prices.get(symbol, entry_price)
                    
                    position_value = shares * current_price
                    position_cost = shares * entry_price
                    unrealized_pnl = position_value - position_cost
                    unrealized_pnl_pct = (unrealized_pnl / position_cost * 100) if position_cost > 0 else 0
                    
                    total_portfolio_value += position_value
                    total_unrealized_pnl += unrealized_pnl
                    
                    pnl_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
                    
                    body_lines.extend([
                        f"",
                        f"📈 {symbol}",
                        f"   Shares: {shares:.4f}",
                        f"   Entry: ${entry_price:.2f} → Current: ${current_price:.2f}",
                        f"   Value: ${position_value:,.2f}",
                        f"   {pnl_emoji} P&L: ${unrealized_pnl:,.2f} ({unrealized_pnl_pct:+.2f}%)"
                    ])

                # Portfolio summary
                total_account_value = total_portfolio_value + portfolio_state.get('cash_balance', 0)
                initial_capital = portfolio_state.get('initial_capital', 100000)
                total_account_pnl = total_account_value - initial_capital
                total_account_pnl_pct = (total_account_pnl / initial_capital * 100) if initial_capital > 0 else 0

                account_pnl_emoji = "🟢" if total_account_pnl >= 0 else "🔴"

                body_lines.extend([
                    "",
                    "=" * 50,
                    "💰 ACCOUNT PERFORMANCE",
                    "=" * 50,
                    f"💰 Cash Balance: ${portfolio_state.get('cash_balance', 0):,.2f}",
                    f"📊 Portfolio Value: ${total_portfolio_value:,.2f}",
                    f"🏦 Total Account Value: ${total_account_value:,.2f}",
                    f"💵 Initial Capital: ${initial_capital:,.2f}",
                    f"{account_pnl_emoji} Total P&L: ${total_account_pnl:,.2f} ({total_account_pnl_pct:+.2f}%)",
                    f"📈 Unrealized P&L: ${total_unrealized_pnl:,.2f}",
                    f"📊 Active Positions: {len(positions)}"
                ])
            else:
                body_lines.extend([
                    "💼 Portfolio is currently empty",
                    f"💰 Cash Balance: ${portfolio_state.get('cash_balance', 0):,.2f}",
                    "🔍 SWING TRADER will scan for opportunities tomorrow"
                ])

            body_lines.extend([
                "",
                "=" * 50,
                "🤖 SWING TRADER - Automated Daily Summary",
                f"⏰ Generated: {execution_time.strftime('%Y-%m-%d %H:%M:%S')}",
                "🎯 Next scan: Tomorrow's pre-market routine",
                "=" * 50
            ])

            body = "\n".join(body_lines)
            await self._send_email_notification(subject, body)
            logger.success("📧 End-of-day summary email sent")

        except Exception as e:
            logger.error(f"Error sending end-of-day summary email: {e}")

    async def _send_error_alert(self, error_type: str, error_message: str, critical: bool = False):
        """Send critical error alerts via email for production monitoring"""
        if not self.email_config['enabled']:
            logger.warning("📧 Email notifications disabled - cannot send error alert")
            return

        try:
            from datetime import datetime
            
            priority = "🚨 CRITICAL" if critical else "⚠️ WARNING"
            subject = f"{priority} Trading Bot Error - {error_type}"
            
            body_lines = [
                f"🚨 TRADING BOT ERROR ALERT",
                "=" * 50,
                "",
                f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}",
                f"🎯 Error Type: {error_type}",
                f"🔥 Priority: {priority}",
                f"🤖 Mode: SWING TRADER (Production)",
                "",
                "=" * 50,
                "❌ ERROR DETAILS",
                "=" * 50,
                f"{error_message}",
                "",
                "=" * 50,
                "🔧 RECOMMENDED ACTIONS",
                "=" * 50
            ]
            
            if critical:
                body_lines.extend([
                    "⚠️ This is a CRITICAL error that may affect trading operations",
                    "🔍 Please check logs immediately",
                    "💻 Consider manual intervention if needed",
                    "📊 Monitor portfolio status"
                ])
            else:
                body_lines.extend([
                    "ℹ️ This is a non-critical error",
                    "🔍 Please review when convenient",
                    "📊 Trading operations should continue normally"
                ])
            
            body_lines.extend([
                "",
                f"🤖 Automated alert from SWING TRADER",
                f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 50
            ])
            
            body = "\n".join(body_lines)
            await self._send_email_notification(subject, body)
            logger.info(f"📧 Error alert sent: {error_type}")
            
        except Exception as e:
            logger.error(f"Failed to send error alert: {e}")
    
    async def _send_email_notification(self, subject: str, body: str):
        """Send notification using SendGrid, Mailgun API, or SMTP fallback"""
        if not self.email_config['enabled']:
            return
        
        try:
            if self.email_config['method'] == 'sendgrid':
                # Use SendGrid API (Railway-compatible)
                await self._send_sendgrid_notification(subject, body)
            elif self.email_config['method'] == 'mailgun':
                # Use Mailgun API (Railway-compatible)
                await self._send_mailgun_notification(subject, body)
            else:
                # Fallback to SMTP (may not work on Railway)
                await self._send_smtp_notification(subject, body)
                
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    async def _send_sendgrid_notification(self, subject: str, body: str):
        """Send notification via SendGrid API (Railway-compatible)"""
        import requests
        
        try:
            payload = {
                "personalizations": [
                    {
                        "to": [{"email": self.email_config['to_email']}],
                        "subject": subject
                    }
                ],
                "from": {"email": self.email_config['from_email']},
                "content": [
                    {
                        "type": "text/plain",
                        "value": body
                    }
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {self.email_config['sendgrid_api_key']}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 202:
                logger.info(f"✅ SendGrid notification sent: {subject}")
            else:
                logger.error(f"❌ SendGrid API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"❌ SendGrid notification failed: {e}")
    
    async def _send_mailgun_notification(self, subject: str, body: str):
        """Send notification via Mailgun API (Railway-compatible)"""
        import requests
        
        try:
            # Get Mailgun configuration from environment
            api_key = os.getenv('MAILGUN_API_KEY')
            domain = os.getenv('MAILGUN_DOMAIN')
            from_email = os.getenv('MAILGUN_FROM_EMAIL', f'trading-bot@{domain}')
            to_email = os.getenv('EMAIL_USERNAME', self.email_config['username'])
            
            if not api_key or not domain:
                logger.error("❌ Mailgun API key or domain not configured")
                return
            
            # Mailgun API payload
            data = {
                'from': from_email,
                'to': to_email,
                'subject': subject,
                'text': body
            }
            
            # Send via Mailgun API
            response = requests.post(
                f"https://api.mailgun.net/v3/{domain}/messages",
                auth=("api", api_key),
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Mailgun notification sent: {subject}")
            else:
                logger.error(f"❌ Mailgun API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Mailgun notification failed: {e}")
    
    async def _send_smtp_notification(self, subject: str, body: str):
        """Send notification via SMTP (fallback method)"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = self.email_config['username']  # Send to self
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['username'], self.email_config['username'], text)
            server.quit()
            
            logger.info(f"📧 SMTP notification sent: {subject}")
            
        except Exception as e:
            logger.error(f"❌ SMTP notification failed (expected on Railway): {e}")
    
    def _get_market_time_in_local(self, market_time_str: str) -> str:
        """Convert US Eastern Time to local Singapore time for scheduling"""
        # Parse market time (ET)
        market_hour, market_minute = map(int, market_time_str.split(':'))
        
        # Create a datetime in market timezone (today)
        now_et = datetime.now(self.market_tz)
        market_dt = now_et.replace(hour=market_hour, minute=market_minute, second=0, microsecond=0)
        
        # Convert to local timezone
        local_dt = market_dt.astimezone(self.local_tz)
        
        return local_dt.strftime('%H:%M')
    
    def _is_market_day(self) -> bool:
        """Check if today is a trading day (Monday-Friday in ET timezone)"""
        now_et = datetime.now(self.market_tz)
        weekday = now_et.weekday()  # Monday=0, Sunday=6
        
        # Check if it's weekend (Saturday=5, Sunday=6)
        is_weekend = weekday >= 5
        
        if is_weekend:
            day_name = now_et.strftime('%A')
            logger.info(f"🏖️ Market closed: Today is {day_name} - no trading")
            return False
        
        # Could add holiday checking here in the future
        # For now, just check weekends
        return True
    
    def _should_run_scheduled_tasks(self) -> bool:
        """Check if scheduled tasks should run (only on market days)"""
        if not self._is_market_day():
            return False
        
        return True
    
    def schedule_daily_tasks(self):
        """Schedule all daily tasks using market timezone converted to local time"""
        logger.info("📅 Setting up daily trading schedule (timezone-aware)...")
        
        # Convert US market times to local Singapore times
        premarket_local = self._get_market_time_in_local("08:30")  # 8:30 AM ET -> SGT
        market_open_local = self._get_market_time_in_local("09:30")  # 9:30 AM ET -> SGT
        
        logger.info(f"🌍 Timezone conversion:")
        local_tz_name = 'UTC' if self.local_tz.zone == 'UTC' else 'SGT'
        logger.info(f"  Pre-market: 08:30 ET → {premarket_local} {local_tz_name}")
        logger.info(f"  Market open: 09:30 ET → {market_open_local} {local_tz_name}")
        
        # Morning pre-market routine (8:30 AM ET)
        schedule.every().day.at(premarket_local).do(
            self._schedule_async_task, self.morning_pre_market_routine
        )
        
        # Market opening routine (9:30 AM ET)
        schedule.every().day.at(market_open_local).do(
            self._schedule_async_task, self.market_opening_routine
        )
        
        if self.portfolio_empty:
            # SWING TRADER MODE - Empty Portfolio: 4 operations/day
            logger.info("📊 SWING TRADER MODE - Portfolio empty")
            
            # Schedule strategic scans: midday and afternoon
            scan_times_et = ["12:00", "14:30"]  # Midday scan, Afternoon scan
            scan_times_local = []
            for scan_time_et in scan_times_et:
                scan_time_local = self._get_market_time_in_local(scan_time_et)
                scan_times_local.append(scan_time_local)
                schedule.every().day.at(scan_time_local).do(
                    self._schedule_async_task, self.hourly_scanner_routine
                )
                
            logger.info(f"🌍 SWING TRADER - Empty Portfolio Schedule:")
            logger.info(f"  Pre-market: 8:30 AM ET → {premarket_local} {local_tz_name}")
            logger.info(f"  Market opening: 9:30 AM ET → {market_open_local} {local_tz_name}")
            for et_time, sgt_time in zip(scan_times_et, scan_times_local):
                logger.info(f"  {et_time} ET → {sgt_time} {local_tz_name}")
            
            logger.info(f"📅 Scheduled {len(scan_times_local) + 2} tasks (SWING TRADER - empty portfolio)")
            
        else:
            # SWING TRADER MODE - Normal Portfolio: 7 operations/day
            logger.info("📊 SWING TRADER MODE - Portfolio monitoring")
            
            # Strategic portfolio checks: Morning, Midday, Afternoon, Pre-close, End-of-day
            portfolio_check_times_et = ["11:00", "13:00", "15:30"]  # Morning, Afternoon, Pre-close
            portfolio_check_times_local = []
            for check_time_et in portfolio_check_times_et:
                check_time_local = self._get_market_time_in_local(check_time_et)
                portfolio_check_times_local.append(check_time_local)
                schedule.every().day.at(check_time_local).do(
                    self._schedule_async_task, self.portfolio_check_routine
                )
            
            # Opportunity scan: Once at midday
            opportunity_scan_et = "12:00"  # Midday opportunity scan
            opportunity_scan_local = self._get_market_time_in_local(opportunity_scan_et)
            schedule.every().day.at(opportunity_scan_local).do(
                self._schedule_async_task, self.hourly_scanner_routine
            )
            
            # End-of-day review
            eod_review_et = "16:00"  # Market close review
            eod_review_local = self._get_market_time_in_local(eod_review_et)
            schedule.every().day.at(eod_review_local).do(
                self._schedule_async_task, self.end_of_day_review
            )
                
            logger.info(f"🌍 SWING TRADER - Normal Portfolio Schedule:")
            logger.info(f"  Pre-market: 8:30 AM ET → {premarket_local} {local_tz_name}")
            logger.info(f"  Market opening: 9:30 AM ET → {market_open_local} {local_tz_name}")
            for et_time, sgt_time in zip(portfolio_check_times_et, portfolio_check_times_local):
                logger.info(f"  Portfolio check: {et_time} ET → {sgt_time} {local_tz_name}")
            logger.info(f"  Opportunity scan: {opportunity_scan_et} ET → {opportunity_scan_local} {local_tz_name}")
            logger.info(f"  End-of-day review: {eod_review_et} ET → {eod_review_local} {local_tz_name}")
            
            total_tasks = len(portfolio_check_times_local) + 4  # +4 for pre-market, opening, opportunity, EOD
            logger.info(f"📅 Scheduled {total_tasks} tasks (SWING TRADER - normal mode)")
            logger.info("🎯 SWING TRADER: 7 operations/day (77% reduction from 30/day)")
            logger.info("💰 Estimated savings: 70% on cloud costs, safe API usage")
    
    async def run_forever(self):
        """Run the smart trader continuously"""
        logger.info("🤖 Smart Trader starting up...")
        
        # Set up schedule
        self.schedule_daily_tasks()
        
        # If portfolio is empty, run immediate scan to find opportunities
        if self.portfolio_empty:
            logger.info("🔍 Portfolio is empty - running immediate opportunity scan...")
            try:
                await self._run_immediate_scan()
            except Exception as e:
                logger.error(f"❌ Error in immediate scan: {e}")
        
        logger.info("✅ Smart Trader is running! Press Ctrl+C to stop.")
        
        # Run indefinitely
        while self.running:
            # Only run scheduled tasks on market days (Monday-Friday)
            if self._should_run_scheduled_tasks():
                schedule.run_pending()
            else:
                # On weekends, just sleep and check again periodically
                await asyncio.sleep(300)  # Check every 5 minutes on weekends
                continue
                
            await asyncio.sleep(1)  # Check every second on market days
        
        logger.info("👋 Smart Trader stopped")
    
    async def _run_immediate_scan(self):
        """Run immediate scan when portfolio is empty"""
        logger.info("🚀 Starting immediate market scan for empty portfolio...")
        
        try:
            # Execute market scan
            await self._execute_market_scan()
            
            # Analyze and execute on opportunities
            await self._analyze_new_opportunities()
            
            # Check if we now have positions after the immediate scan
            old_portfolio_empty = self.portfolio_empty
            self.current_symbols = self._load_current_symbols()
            self.portfolio_empty = len(self.current_symbols) == 0
            
            if old_portfolio_empty and not self.portfolio_empty:
                logger.success("🎯 Immediate scan successful! Portfolio now has positions")
                logger.info(f"📊 New positions: {', '.join(self.current_symbols)}")
                await self._send_email_notification(
                    "Immediate Scan Success",
                    f"Smart Trader found opportunities immediately! New positions: {', '.join(self.current_symbols)}"
                )
                
                # Reschedule for regular portfolio mode
                logger.info("🔄 Rescheduling for regular portfolio monitoring mode...")
                schedule.clear()
                self.schedule_daily_tasks()
            else:
                logger.info("📊 Immediate scan completed - no positions acquired yet, will continue hourly scanning")
            
        except Exception as e:
            logger.error(f"❌ Error in immediate scan: {e}")
            await self._send_email_notification(
                "Immediate Scan Error",
                f"Error during immediate portfolio scan: {str(e)}"
            )

    def debug_portfolio_state(self):
        """Debug method to log current portfolio state for Railway inspection"""
        try:
            portfolio_file = "data/portfolio_state.json"
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
                
                logger.info("🔍 DEBUG: Current Portfolio State")
                logger.info("=" * 50)
                logger.info(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"💰 Cash Balance: ${portfolio_data.get('cash_balance', 0):,.2f}")
                logger.info(f"🏦 Initial Capital: ${portfolio_data.get('initial_capital', 0):,.2f}")
                
                positions = portfolio_data.get('positions', {})
                logger.info(f"📊 Total Positions: {len(positions)}")
                
                total_value = 0
                for symbol, pos in positions.items():
                    size = pos.get('size', 0)
                    current_price = pos.get('current_price', 0)
                    entry_price = pos.get('entry_price', 0)
                    market_value = size * current_price
                    pnl = (current_price - entry_price) * size
                    total_value += market_value
                    
                    logger.info(f"  {symbol}: {size:.2f} shares @ ${current_price:.2f} = ${market_value:,.2f} (P&L: ${pnl:,.2f})")
                
                total_account = portfolio_data.get('cash_balance', 0) + total_value
                total_pnl = total_account - portfolio_data.get('initial_capital', 0)
                
                logger.info(f"💼 Total Portfolio Value: ${total_value:,.2f}")
                logger.info(f"🏆 Total Account Value: ${total_account:,.2f}")
                logger.info(f"📈 Total P&L: ${total_pnl:,.2f} ({total_pnl/portfolio_data.get('initial_capital', 1)*100:.2f}%)")
                logger.info("=" * 50)
                
                # Also log the raw JSON for technical inspection
                logger.info("🔍 DEBUG: Raw Portfolio JSON")
                logger.info(json.dumps(portfolio_data, indent=2, default=str))
                
            else:
                logger.warning("⚠️ DEBUG: Portfolio state file not found")
                
        except Exception as e:
            logger.error(f"❌ DEBUG: Error reading portfolio state: {e}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Trading Automation")
    parser.add_argument("--test-morning", action="store_true", 
                       help="Test morning pre-market routine")
    parser.add_argument("--test-opening", action="store_true",
                       help="Test market opening routine")
    parser.add_argument("--test-check", action="store_true",
                       help="Test portfolio check routine")
    parser.add_argument("--debug-portfolio", action="store_true",
                       help="Debug portfolio state and display current JSON data")
    parser.add_argument("--run", action="store_true",
                       help="Run continuous smart trading")
    
    args = parser.parse_args()
    
    trader = SmartTrader()
    
    if args.test_morning:
        logger.info("🧪 Testing morning pre-market routine...")
        await trader.morning_pre_market_routine()
    elif args.test_opening:
        logger.info("🧪 Testing market opening routine...")
        await trader.market_opening_routine()
    elif args.test_check:
        logger.info("🧪 Testing portfolio check routine...")
        await trader.portfolio_check_routine()
    elif args.debug_portfolio:
        logger.info("🧪 Debugging portfolio state...")
        trader.debug_portfolio_state()
    elif args.run:
        await trader.run_forever()
    else:
        logger.info("🤖 Smart Trader")
        logger.info("Usage:")
        logger.info("  --test-morning    Test morning pre-market routine")
        logger.info("  --test-opening    Test market opening routine") 
        logger.info("  --test-check      Test portfolio check routine")
        logger.info("  --debug-portfolio Debug and display current portfolio JSON data")
        logger.info("  --run             Run continuous smart trading")


if __name__ == "__main__":
    asyncio.run(main())