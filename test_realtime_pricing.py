#!/usr/bin/env python3
"""
Test script to verify real-time pricing functionality
"""
import asyncio
import sys
import os
from loguru import logger

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.market_data_agent import MarketDataAgent
from orchestrator import TradingOrchestrator

async def test_realtime_pricing():
    """Test real-time pricing from market data agent"""
    
    logger.info("🧪 Testing Real-time Pricing Fix")
    logger.info("=" * 50)
    
    # Test 1: Direct market data agent test
    logger.info("Test 1: Direct Market Data Agent")
    
    config = {
        "symbols": ["GOOGL", "AAPL", "MSFT"],
        "market_data": {
            "sources": ["yfinance"],
            "cache_duration": 60
        }
    }
    
    market_agent = MarketDataAgent(config)
    
    try:
        # Test get_live_feed method
        live_data = await market_agent.get_live_feed(["GOOGL"])
        
        if "GOOGL" in live_data and "price" in live_data["GOOGL"]:
            price = live_data["GOOGL"]["price"]
            logger.info(f"✅ Market Agent: GOOGL price = ${price:.2f}")
        else:
            logger.error("❌ Market Agent: No GOOGL price data")
            logger.info(f"Data received: {live_data}")
            
    except Exception as e:
        logger.error(f"❌ Market Agent failed: {e}")
    
    # Test 2: Orchestrator price estimation
    logger.info("\nTest 2: Orchestrator Price Estimation")
    
    orchestrator_config = {
        "agents": {
            "market_data": {
                "symbols": ["GOOGL"],
                "market_data": {"sources": ["yfinance"]}
            }
        }
    }
    
    try:
        orchestrator = TradingOrchestrator(orchestrator_config)
        
        # Mock combined_signals to test price estimation
        mock_signals = {
            "GOOGL": {
                "market_data": {
                    # Empty market data to force fresh fetch
                }
            }
        }
        
        estimated_price = await orchestrator._estimate_current_price("GOOGL", mock_signals)
        logger.info(f"✅ Orchestrator: GOOGL estimated price = ${estimated_price:.2f}")
        
        # Check if it's using real data (not fallback)
        if estimated_price == 180.0:
            logger.warning("⚠️  Still using fallback price - real-time fetch may have failed")
        else:
            logger.info("🎉 Using real-time price data!")
            
    except Exception as e:
        logger.error(f"❌ Orchestrator test failed: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(test_realtime_pricing())