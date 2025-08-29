"""
Signal History Management for Dynamic Thresholds

This module tracks signal strengths over time to enable dynamic threshold
calculation based on recent market conditions and signal distributions.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """Individual signal record for history tracking"""
    timestamp: str
    symbol: str
    signal_strength: float
    technical_strength: float
    fundamental_signal: float
    execution_id: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SignalRecord':
        return cls(**data)


class SignalHistoryTracker:
    """Tracks and manages signal history for dynamic threshold calculation"""
    
    def __init__(self, history_file: str = "data/signal_history.json", max_days: int = 60):
        self.history_file = history_file
        self.max_days = max_days
        self.signals: List[SignalRecord] = []
        self._load_history()
    
    def _load_history(self):
        """Load existing signal history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.signals = [SignalRecord.from_dict(record) for record in data.get('signals', [])]
                logger.info(f"Loaded {len(self.signals)} signal records from {self.history_file}")
            else:
                logger.info("No existing signal history found - starting fresh")
        except Exception as e:
            logger.error(f"Error loading signal history: {e}")
            self.signals = []
    
    def _save_history(self):
        """Save signal history to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            # Prepare data for saving
            data = {
                'last_updated': datetime.now().isoformat(),
                'total_records': len(self.signals),
                'signals': [signal.to_dict() for signal in self.signals]
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved {len(self.signals)} signal records to {self.history_file}")
        except Exception as e:
            logger.error(f"Error saving signal history: {e}")
    
    def add_signals(self, execution_id: str, combined_signals: Dict[str, Dict]) -> int:
        """
        Add new signals from a trading cycle
        
        Args:
            execution_id: Unique identifier for the trading cycle
            combined_signals: Dictionary of symbol -> signal data
            
        Returns:
            Number of signals added
        """
        timestamp = datetime.now().isoformat()
        added_count = 0
        
        for symbol, signal_data in combined_signals.items():
            try:
                # Extract signal components from the overall_signal
                overall_signal = signal_data.get('overall_signal', {})
                signal_strength = overall_signal.get('strength', 0.0)
                
                # Get technical strength from technical_analyst data
                tech_strength = 0.0
                if 'technical_analyst' in signal_data:
                    tech_data = signal_data['technical_analyst']
                    # tech_data is already the extracted symbol data, not an AgentResult
                    tech_signal = tech_data.get('overall_signal', {})
                    tech_strength = tech_signal.get('strength', 0.0)
                
                # Get fundamental signal from fundamentals_analyst data
                fund_signal = 0.0
                if 'fundamentals_analyst' in signal_data:
                    fund_data = signal_data['fundamentals_analyst']
                    # fund_data is already the extracted symbol data, not an AgentResult
                    fund_score = fund_data.get('overall_score', 5.0)
                    fund_signal = (fund_score - 5.0) / 5.0  # Convert to -1 to 1 scale
                
                # Create and add record
                record = SignalRecord(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_strength=signal_strength,
                    technical_strength=tech_strength,
                    fundamental_signal=fund_signal,
                    execution_id=execution_id
                )
                
                self.signals.append(record)
                added_count += 1
                
            except Exception as e:
                logger.warning(f"Error adding signal for {symbol}: {e}")
                continue
        
        # Clean old records and save
        self._cleanup_old_records()
        self._save_history()
        
        logger.info(f"Added {added_count} new signal records (execution: {execution_id})")
        return added_count
    
    def _cleanup_old_records(self):
        """Remove records older than max_days"""
        cutoff_date = datetime.now() - timedelta(days=self.max_days)
        
        original_count = len(self.signals)
        self.signals = [
            signal for signal in self.signals 
            if datetime.fromisoformat(signal.timestamp) > cutoff_date
        ]
        
        cleaned_count = original_count - len(self.signals)
        if cleaned_count > 0:
            logger.info(f"Cleaned {cleaned_count} old signal records (older than {self.max_days} days)")
    
    def get_recent_signals(self, days: int = 30) -> List[float]:
        """
        Get signal strengths from the last N days
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of signal strengths from recent periods
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_signals = [
            signal.signal_strength for signal in self.signals
            if datetime.fromisoformat(signal.timestamp) > cutoff_date
        ]
        
        logger.debug(f"Retrieved {len(recent_signals)} signals from last {days} days")
        return recent_signals
    
    def calculate_dynamic_threshold(self, 
                                  lookback_days: int = 30,
                                  percentile: float = 80,
                                  min_samples: int = 10) -> Optional[float]:
        """
        Calculate dynamic threshold based on recent signal distribution
        
        Args:
            lookback_days: Days to look back for signal history
            percentile: Percentile threshold (80 = top 20% of signals)
            min_samples: Minimum number of samples required for calculation
            
        Returns:
            Dynamic threshold value or None if insufficient data
        """
        recent_signals = self.get_recent_signals(days=lookback_days)
        
        if len(recent_signals) < min_samples:
            logger.warning(f"Insufficient signal history: {len(recent_signals)} < {min_samples} required")
            return None
        
        # Remove zero signals for threshold calculation (failed analyses)
        non_zero_signals = [s for s in recent_signals if abs(s) > 0.001]
        
        if len(non_zero_signals) < min_samples:
            logger.warning(f"Insufficient non-zero signals: {len(non_zero_signals)} < {min_samples} required")
            return None
        
        # Calculate percentile threshold
        threshold = np.percentile(np.abs(non_zero_signals), percentile)
        
        logger.info(f"Calculated dynamic threshold: {threshold:.3f} "
                   f"(percentile {percentile}% of {len(non_zero_signals)} signals)")
        
        return float(threshold)
    
    def get_signal_statistics(self, days: int = 30) -> Dict:
        """Get statistics about recent signals for monitoring"""
        recent_signals = self.get_recent_signals(days=days)
        
        if not recent_signals:
            return {
                'count': 0,
                'days_analyzed': days,
                'message': 'No recent signals available'
            }
        
        abs_signals = [abs(s) for s in recent_signals if abs(s) > 0.001]
        
        if not abs_signals:
            return {
                'count': len(recent_signals),
                'days_analyzed': days,
                'message': 'No non-zero signals in period'
            }
        
        return {
            'count': len(recent_signals),
            'non_zero_count': len(abs_signals),
            'days_analyzed': days,
            'mean': np.mean(abs_signals),
            'median': np.median(abs_signals),
            'std': np.std(abs_signals),
            'min': np.min(abs_signals),
            'max': np.max(abs_signals),
            'percentiles': {
                '25th': np.percentile(abs_signals, 25),
                '50th': np.percentile(abs_signals, 50),
                '75th': np.percentile(abs_signals, 75),
                '80th': np.percentile(abs_signals, 80),
                '90th': np.percentile(abs_signals, 90)
            }
        }