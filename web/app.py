from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import asyncio
import json

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Fallback JSON encoder (not used after sanitization)
        return super().default(obj)
import logging
from datetime import datetime, date
from typing import List, Dict, Any
import os
import pandas as pd
import numpy as np
import math
import traceback

# Monkey-patch pandas fillna to be safe if called without value/method by dependencies
try:
    _orig_df_fillna = pd.DataFrame.fillna
    _orig_sr_fillna = pd.Series.fillna
    from pandas.core.indexes.base import Index as _PD_Index
    _orig_idx_fillna = _PD_Index.fillna if hasattr(_PD_Index, 'fillna') else None

    def _safe_df_fillna(self, value=None, method=None, *args, **kwargs):
        if value is None and method is None:
            return _orig_df_fillna(self, value=0)
        return _orig_df_fillna(self, value=value, method=method, *args, **kwargs)

    def _safe_sr_fillna(self, value=None, method=None, *args, **kwargs):
        if value is None and method is None:
            return _orig_sr_fillna(self, value=0)
        return _orig_sr_fillna(self, value=value, method=method, *args, **kwargs)

    pd.DataFrame.fillna = _safe_df_fillna
    pd.Series.fillna = _safe_sr_fillna
    if _orig_idx_fillna:
        def _safe_idx_fillna(self, value=None, method=None, *args, **kwargs):
            if value is None and method is None:
                return _orig_idx_fillna(self, value=0)
            return _orig_idx_fillna(self, value=value, method=method, *args, **kwargs)
        _PD_Index.fillna = _safe_idx_fillna
except Exception:
    pass

# Import our trading system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator import TradingOrchestrator
from core.utils.json_utils import sanitize_for_json
import yaml


# Using sanitize_for_json from core.utils.json_utils
app = Flask(__name__)
socketio = SocketIO(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_ui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
orchestrator = None
trade_history = []
portfolio_positions = {}

def initialize_system():
    global orchestrator
    if orchestrator is None:
        # Load configuration
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        orchestrator = TradingOrchestrator(config)
    return orchestrator

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_stocks():
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        if not symbols:
            return jsonify({'error': 'No symbols provided'}), 400

        # Initialize trading system if needed
        initialize_system()
        
        # Run analysis (synchronously for now)
        step = 'execute_trading_cycle'
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(orchestrator.execute_trading_cycle(symbols))
        loop.close()
        
        # Extract relevant information
        results = result.get('results', {})
        market_data_result = results.get('market_data', {})
        analysis_block = results.get('analysis', {}) if isinstance(results.get('analysis', {}), dict) else {}
        tech_result = analysis_block.get('technical_analyst', {})
        fund_result = analysis_block.get('fundamentals_analyst', {})
        sentiment_result = analysis_block.get('sentiment_analyst', {})
        risk_result = results.get('risk_management', {})
        portfolio_result = results.get('portfolio_management', {})
        
        # Extract data from agent results
        market_data = market_data_result.data if hasattr(market_data_result, 'data') else {}
        technical_data = tech_result.data if hasattr(tech_result, 'data') else {}
        fundamentals_data = fund_result.data if hasattr(fund_result, 'data') else {}
        sentiment_data = sentiment_result.data if hasattr(sentiment_result, 'data') else {}
        risk_data = risk_result.data if hasattr(risk_result, 'data') else {}
        
        step = 'build_analysis_results_raw'
        # Use underlying data dicts for result objects where possible
        portfolio_action_raw = portfolio_result.data if hasattr(portfolio_result, 'data') else portfolio_result
        risk_assessment_raw = risk_data

        analysis_results_raw = {
            'timestamp': datetime.now().isoformat(),
            'symbols': symbols,
            'market_data': market_data,
            'technical_analysis': technical_data,
            'fundamentals': fundamentals_data,
            'sentiment': sentiment_data,
            'risk_assessment': risk_assessment_raw,
            'portfolio_action': portfolio_action_raw
        }
        # Sanitize entire payload for JSON
        step = 'sanitize_for_json'
        analysis_results = sanitize_for_json(analysis_results_raw)

        # Debug: detect remaining datetime-like objects
        def _find_datetime_paths(obj, path="root"):
            hits = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    hits.extend(_find_datetime_paths(v, f"{path}.{k}"))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    hits.extend(_find_datetime_paths(v, f"{path}[{i}]"))
            else:
                if isinstance(obj, (datetime, date)):
                    hits.append((path, type(obj).__name__))
            return hits

        dt_hits = _find_datetime_paths(analysis_results)
        if dt_hits:
            logger.warning(f"Datetime remnants before json.dumps: {dt_hits[:5]} (showing up to 5)")

        # Update trade history if any trades were executed
        if portfolio_result and hasattr(portfolio_result, 'data') and 'orders_executed' in portfolio_result.data:
            for order in portfolio_result.data['orders_executed']:
                trade_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': order['symbol'],
                    'size': order['size'],
                    'price': order['filled_price'],
                    'type': order['order_type'],
                    'status': order['status']
                })

        # Update portfolio positions
        if portfolio_result and hasattr(portfolio_result, 'data') and 'portfolio_summary' in portfolio_result.data:
            portfolio_positions.update(portfolio_result.data['portfolio_summary'])

        # Serialize with strict JSON (no NaN allowed)
        step = 'json_dumps'
        json_str = json.dumps(
            analysis_results,
            allow_nan=False,
            default=lambda o: (
                o.isoformat() if hasattr(o, 'isoformat') else str(o)
            )
        )
        return json_str, 200, {'Content-Type': 'application/json'}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error in analyze_stocks (step={step}): {e}\n{tb}")
        err_payload = {'error': str(e), 'step': step, 'traceback': tb}
        return json.dumps(err_payload), 500, {'Content-Type': 'application/json'}

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    try:
        initialize_system()
        return jsonify({
            'positions': portfolio_positions,
            'trade_history': trade_history
        })
    except Exception as e:
        logger.error(f"Error in get_portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades', methods=['GET'])
def get_trades():
    try:
        return jsonify(trade_history)
    except Exception as e:
        logger.error(f"Error in get_trades: {str(e)}")
        return jsonify({'error': str(e)}), 500

# WebSocket for real-time updates
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

def emit_portfolio_update():
    socketio.emit('portfolio_update', {
        'positions': portfolio_positions,
        'trade_history': trade_history
    })

if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Initialize the system
    initialize_system()
    
    # Run the app
    socketio.run(app, debug=True, port=5002)