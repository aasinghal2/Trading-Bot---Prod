#!/usr/bin/env python3
"""
Simple script to debug portfolio state on Railway
Usage: python scripts/debug_portfolio.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.smart_trader import SmartTrader

def main():
    """Debug portfolio state"""
    print("🔍 Portfolio Debug Tool")
    print("=" * 40)
    
    trader = SmartTrader()
    trader.debug_portfolio_state()

if __name__ == "__main__":
    main()