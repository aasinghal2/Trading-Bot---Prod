#!/usr/bin/env python3
"""
Simple script to start the Trading Bot UI
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if required directories exist
os.makedirs('logs', exist_ok=True)

# Start the Flask app
if __name__ == '__main__':
    try:
        from app import app, socketio
        print("🚀 Starting Trading Bot UI...")
        print("📊 Navigate to: http://localhost:5002")
        print("Press Ctrl+C to stop")
        socketio.run(app, debug=False, port=5002, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\n🛑 Trading Bot UI stopped")
    except Exception as e:
        print(f"❌ Error starting UI: {e}")
        sys.exit(1)