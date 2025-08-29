#!/bin/bash
# Smart Trader Startup Script
# Streamlined automation for your exact requirements

set -e

echo "🤖 Smart Trading Bot"
echo "==================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found. Please run this script from the TradingBot directory"
    exit 1
fi

# Install required package if not present
if ! python3 -c "import schedule" &> /dev/null; then
    echo "📦 Installing schedule package..."
    pip3 install schedule
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Parse command line arguments
MODE="run"
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-morning)
            MODE="test-morning"
            shift
            ;;
        --test-opening)
            MODE="test-opening"
            shift
            ;;
        --test-check)
            MODE="test-check"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --test-morning    Test morning pre-market routine"
            echo "  --test-opening    Test market opening routine"
            echo "  --test-check      Test portfolio check routine"
            echo "  --help            Show this help message"
            echo ""
            echo "Default: Run continuous smart trading"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "📊 Smart Trading Schedule:"
echo "  🌅 8:30 AM  - Pre-market portfolio scan & trades"
echo "  🚀 9:30 AM  - Market opening: scan top 50 stocks, find opportunities"
echo "  📊 9:45 AM+ - Portfolio checks every 15 minutes until 4:00 PM"
echo ""

case $MODE in
    test-morning)
        echo "🧪 Testing morning pre-market routine..."
        python3 scripts/smart_trader.py --test-morning
        ;;
    test-opening)
        echo "🧪 Testing market opening routine..."
        python3 scripts/smart_trader.py --test-opening
        ;;
    test-check)
        echo "🧪 Testing portfolio check routine..."
        python3 scripts/smart_trader.py --test-check
        ;;
    run)
        echo "🤖 Starting Smart Trader..."
        echo "Press Ctrl+C to stop"
        python3 scripts/smart_trader.py --run
        ;;
esac

echo ""
echo "🏁 Smart Trader stopped"