#!/usr/bin/env python3
"""
Health Check Server for Trading Bot
Provides endpoints for monitoring the trading bot's health and status.
"""

import json
import time
import psutil
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone
import os
import sqlite3
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints."""
    
    def __init__(self, *args, **kwargs):
        self.start_time = time.time()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests for health check endpoints."""
        try:
            if self.path == '/health':
                self._handle_basic_health()
            elif self.path == '/health/detailed':
                self._handle_detailed_health()
            elif self.path == '/metrics':
                self._handle_metrics()
            elif self.path == '/status':
                self._handle_status()
            else:
                self._send_response(404, {'error': 'Endpoint not found'})
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self._send_response(500, {'error': 'Internal server error'})
    
    def _handle_basic_health(self):
        """Basic health check - just returns OK if service is running."""
        self._send_response(200, {'status': 'OK', 'timestamp': datetime.now(timezone.utc).isoformat()})
    
    def _handle_detailed_health(self):
        """Detailed health check with system metrics."""
        try:
            health_data = {
                'status': 'OK',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'uptime_seconds': time.time() - self.start_time,
                'system': self._get_system_metrics(),
                'application': self._get_application_health(),
                'trading': self._get_trading_health()
            }
            
            # Determine overall health status
            if health_data['system']['memory_usage_percent'] > 90:
                health_data['status'] = 'WARNING'
            if health_data['system']['cpu_usage_percent'] > 95:
                health_data['status'] = 'CRITICAL'
            
            status_code = 200 if health_data['status'] == 'OK' else 503
            self._send_response(status_code, health_data)
            
        except Exception as e:
            logger.error(f"Error in detailed health check: {e}")
            self._send_response(503, {'status': 'ERROR', 'error': str(e)})
    
    def _handle_metrics(self):
        """Prometheus-style metrics endpoint."""
        try:
            metrics = self._get_prometheus_metrics()
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(metrics.encode())
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            self._send_response(500, {'error': 'Metrics generation failed'})
    
    def _handle_status(self):
        """Application status with recent activity."""
        try:
            status_data = {
                'application': 'AI Trading Bot',
                'version': '1.0.0',
                'environment': os.getenv('ENVIRONMENT', 'development'),
                'status': 'RUNNING',
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'components': {
                    'market_data': self._check_component_health('market_data'),
                    'analysis': self._check_component_health('analysis'),
                    'portfolio': self._check_component_health('portfolio'),
                    'risk_management': self._check_component_health('risk_management')
                }
            }
            self._send_response(200, status_data)
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            self._send_response(500, {'error': 'Status check failed'})
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage_percent': psutil.cpu_percent(interval=1),
                'memory_usage_percent': memory.percent,
                'memory_available_mb': memory.available // (1024 * 1024),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free // (1024 * 1024 * 1024),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {'error': str(e)}
    
    def _get_application_health(self) -> Dict[str, Any]:
        """Get application-specific health metrics."""
        try:
            return {
                'log_file_exists': os.path.exists('logs/trading_system.log'),
                'config_file_exists': os.path.exists('config/config.yaml'),
                'data_directory_exists': os.path.exists('data'),
                'portfolio_state_exists': os.path.exists('data/portfolio_state.json'),
                'recent_log_size_mb': self._get_file_size_mb('logs/trading_system.log'),
                'python_version': f"{psutil.Process().exe()}"
            }
        except Exception as e:
            logger.error(f"Error getting application health: {e}")
            return {'error': str(e)}
    
    def _get_trading_health(self) -> Dict[str, Any]:
        """Get trading-specific health metrics."""
        try:
            trading_health = {
                'last_execution': 'unknown',
                'portfolio_value': 0,
                'active_positions': 0,
                'last_error': None
            }
            
            # Check portfolio state
            if os.path.exists('data/portfolio_state.json'):
                try:
                    with open('data/portfolio_state.json', 'r') as f:
                        portfolio_data = json.load(f)
                        trading_health['portfolio_value'] = portfolio_data.get('total_value', 0)
                        trading_health['active_positions'] = len(portfolio_data.get('positions', {}))
                        trading_health['last_execution'] = portfolio_data.get('last_updated', 'unknown')
                except json.JSONDecodeError:
                    trading_health['portfolio_state_error'] = 'Invalid JSON'
            
            # Check recent logs for errors
            trading_health['last_error'] = self._get_recent_error()
            
            return trading_health
        except Exception as e:
            logger.error(f"Error getting trading health: {e}")
            return {'error': str(e)}
    
    def _check_component_health(self, component: str) -> str:
        """Check health of specific component."""
        try:
            # This is a simplified check - in practice, you'd check component-specific metrics
            log_file = 'logs/trading_system.log'
            if os.path.exists(log_file):
                # Check for recent successful operations
                with open(log_file, 'r') as f:
                    recent_lines = f.readlines()[-100:]  # Last 100 lines
                    recent_text = ''.join(recent_lines)
                    
                    if component in recent_text and 'ERROR' not in recent_text[-500:]:
                        return 'HEALTHY'
                    elif 'ERROR' in recent_text[-500:] and component in recent_text[-500:]:
                        return 'ERROR'
                    else:
                        return 'UNKNOWN'
            return 'NO_DATA'
        except Exception:
            return 'ERROR'
    
    def _get_recent_error(self) -> str:
        """Get the most recent error from logs."""
        try:
            log_file = 'logs/trading_system.log'
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Look for ERROR in the last 50 lines
                    for line in reversed(lines[-50:]):
                        if 'ERROR' in line:
                            return line.strip()
            return None
        except Exception:
            return None
    
    def _get_file_size_mb(self, filepath: str) -> float:
        """Get file size in MB."""
        try:
            if os.path.exists(filepath):
                return os.path.getsize(filepath) / (1024 * 1024)
            return 0
        except Exception:
            return 0
    
    def _get_prometheus_metrics(self) -> str:
        """Generate Prometheus-style metrics."""
        try:
            system = self._get_system_metrics()
            application = self._get_application_health()
            trading = self._get_trading_health()
            
            metrics = []
            
            # System metrics
            if 'cpu_usage_percent' in system:
                metrics.append(f"trading_bot_cpu_usage_percent {system['cpu_usage_percent']}")
            if 'memory_usage_percent' in system:
                metrics.append(f"trading_bot_memory_usage_percent {system['memory_usage_percent']}")
            if 'disk_usage_percent' in system:
                metrics.append(f"trading_bot_disk_usage_percent {system['disk_usage_percent']}")
            
            # Application metrics
            metrics.append(f"trading_bot_uptime_seconds {time.time() - self.start_time}")
            metrics.append(f"trading_bot_log_size_mb {application.get('recent_log_size_mb', 0)}")
            
            # Trading metrics
            if isinstance(trading.get('portfolio_value'), (int, float)):
                metrics.append(f"trading_bot_portfolio_value {trading['portfolio_value']}")
            if isinstance(trading.get('active_positions'), int):
                metrics.append(f"trading_bot_active_positions {trading['active_positions']}")
            
            return '\n'.join(metrics) + '\n'
        except Exception as e:
            logger.error(f"Error generating Prometheus metrics: {e}")
            return f"# Error generating metrics: {e}\n"
    
    def _send_response(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def log_message(self, format, *args):
        """Override to reduce noise in logs."""
        pass


def start_health_check_server(port: int = 8081, host: str = "0.0.0.0"):
    """Start the health check server."""
    try:
        server = HTTPServer((host, port), HealthCheckHandler)
        logger.info(f"Health check server starting on {host}:{port}")
        
        # Set server start time for uptime calculation
        HealthCheckHandler.start_time = time.time()
        
        logger.info("Health check endpoints available:")
        logger.info(f"  Basic health: http://{host}:{port}/health")
        logger.info(f"  Detailed health: http://{host}:{port}/health/detailed")
        logger.info(f"  Metrics: http://{host}:{port}/metrics")
        logger.info(f"  Status: http://{host}:{port}/status")
        
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Health check server stopped")
    except Exception as e:
        logger.error(f"Error starting health check server: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Health Check Server for Trading Bot")
    parser.add_argument("--port", type=int, default=8081, help="Port to run health check server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind health check server to")
    
    args = parser.parse_args()
    
    start_health_check_server(port=args.port, host=args.host)