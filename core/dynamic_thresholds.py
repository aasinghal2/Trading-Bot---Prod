"""
Dynamic Threshold Management System

This module provides dynamic threshold calculation based on recent signal
distributions and market conditions, replacing static thresholds with
adaptive ones that respond to market volatility and signal patterns.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
from core.signal_history import SignalHistoryTracker
from core.market_metrics import MarketMetricsAnalyzer, calculate_market_threshold_component, MarketMetrics

logger = logging.getLogger(__name__)


class DynamicThresholdManager:
    """
    Manages dynamic threshold calculation for trading signals.
    
    This class combines signal history tracking with market metrics analysis
    to provide adaptive thresholds that respond to changing market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Dynamic Threshold Manager.
        
        Args:
            config: Configuration dictionary containing threshold settings
        """
        self.config = config
        # Get the dynamic threshold config from the correct path
        trading_config = config.get('trading', {})
        signal_config = trading_config.get('signal_thresholds', {})
        self.threshold_config = signal_config.get('dynamic', {})
        
        # Initialize components
        self.history_tracker = SignalHistoryTracker(
            history_file=self.threshold_config.get('history_file', 'data/signal_history.json'),
            max_days=self.threshold_config.get('max_history_days', 60)
        )
        
        self.market_analyzer = MarketMetricsAnalyzer(
            cache_file=self.threshold_config.get('metrics_cache_file', 'data/market_metrics_cache.json')
        )
        
        # Threshold calculation parameters
        self.default_threshold = self.threshold_config.get('default_threshold', 0.3)
        self.min_signals_required = self.threshold_config.get('min_samples', 10)
        self.lookback_days = self.threshold_config.get('lookback_days', 30)
        self.confidence_level = self.threshold_config.get('percentile', 80)
        self.volatility_adjustment = self.threshold_config.get('volatility_adjustment', True)
        
        logger.info("Dynamic Threshold Manager initialized")
    
    def add_execution_signals(self, execution_id: str, combined_signals: Dict[str, Dict]) -> int:
        """
        Add signals from a trading execution to the history.
        
        Args:
            execution_id: Unique identifier for the execution
            combined_signals: Dictionary of symbol -> signal data
            
        Returns:
            Number of signals added
        """
        return self.history_tracker.add_signals(execution_id, combined_signals)
    
    def calculate_threshold(self, 
                          lookback_days: int = 30,
                          market_regime: Optional[str] = None) -> Tuple[float, str]:
        """
        Calculate dynamic threshold based on recent signal distribution and market conditions.
        
        Args:
            lookback_days: Number of days to look back for signal analysis
            market_regime: Optional market regime override ('bull', 'bear', 'neutral')
            
        Returns:
            Tuple of (threshold_value, explanation_string)
        """
        try:
            # Get recent signal statistics
            signal_stats = self.history_tracker.get_signal_statistics(days=lookback_days)
            recent_signals = self.history_tracker.get_recent_signals(days=lookback_days)
            
            # Check if we have enough data
            if len(recent_signals) < self.min_signals_required:
                explanation = f"Insufficient signal history ({len(recent_signals)} < {self.min_signals_required}), using default threshold"
                logger.warning(explanation)
                return self.default_threshold, explanation
            
            # Get signal component from signal distribution (this becomes 40% of result)
            signal_component = self.history_tracker.calculate_dynamic_threshold(
                lookback_days=lookback_days,
                percentile=self.confidence_level
            )
            
            # Get market component and apply hybrid formula
            if self.volatility_adjustment:
                market_metrics = self.market_analyzer.get_latest_metrics()
                adjusted_threshold = self._calculate_hybrid_threshold(
                    signal_component, 
                    market_metrics, 
                    market_regime
                )
            else:
                adjusted_threshold = signal_component
            
            # Ensure threshold is within configured bounds
            floor = self.threshold_config.get("floor_threshold", 0.15)
            ceiling = self.threshold_config.get("ceiling_threshold", 0.40)
            adjusted_threshold = max(floor, min(ceiling, adjusted_threshold))
            
            # Create explanation
            explanation = self._create_threshold_explanation(
                signal_component, 
                adjusted_threshold, 
                signal_stats, 
                len(recent_signals)
            )
            
            logger.info(f"Dynamic threshold calculated: {adjusted_threshold:.3f}")
            return adjusted_threshold, explanation
            
        except Exception as e:
            logger.error(f"Error calculating dynamic threshold: {e}")
            explanation = f"Error in calculation, using default threshold: {str(e)}"
            return self.default_threshold, explanation
    
    def _calculate_hybrid_threshold(self, 
                                   signal_component: float, 
                                   market_metrics: Dict[str, Any],
                                   market_regime: Optional[str] = None) -> float:
        """
        Calculate hybrid threshold using sophisticated market analysis.
        Implements: final = (0.8 * hybrid) + (0.2 * static) where hybrid = 60% market + 40% signal
        
        Args:
            signal_component: Threshold from signal distribution analysis
            market_metrics: Current market metrics
            market_regime: Optional market regime override
            
        Returns:
            Hybrid threshold value
        """
        try:
            # Get configuration weights
            market_weight = self.threshold_config.get('market_weight', 0.6)
            signal_weight = self.threshold_config.get('signal_weight', 0.4)
            static_blend = self.threshold_config.get('static_blend', 0.2)
            static_threshold = 0.25  # Static baseline as documented
            
            # Calculate market component using sophisticated analysis
            market_component = static_threshold  # Default fallback
            
            if market_metrics:
                try:
                    # Convert dict to MarketMetrics object for the sophisticated function
                    metrics_obj = MarketMetrics(
                        spy_return_1d=market_metrics.get('spy_return_1d', 0.0),
                        spy_return_5d=market_metrics.get('spy_return_5d', 0.0),
                        spy_return_30d=market_metrics.get('spy_return_30d', 0.0),
                        vix_current=market_metrics.get('vix_current', 20.0),
                        vix_change_5d=market_metrics.get('vix_change_5d', 0.0),
                        market_breadth=market_metrics.get('market_breadth', 0.5),
                        sector_strength=market_metrics.get('sector_strength', {}),
                        regime=market_metrics.get('regime', 'neutral'),
                        timestamp=market_metrics.get('timestamp', '')
                    )
                    
                    # Use the sophisticated market analysis function
                    market_component, market_explanation = calculate_market_threshold_component(
                        metrics_obj, base_threshold=static_threshold
                    )
                    
                    logger.info(f"Market component: {market_component:.3f} ({market_explanation})")
                    
                except Exception as e:
                    logger.warning(f"Error using sophisticated market analysis, using fallback: {e}")
                    market_component = static_threshold
            
            # Apply hybrid formula as documented
            # hybrid = (market_weight * market_component) + (signal_weight * signal_component)
            hybrid_component = (market_weight * market_component) + (signal_weight * signal_component)
            
            # Static baseline blend: final = (1-static_blend) * hybrid + static_blend * static
            final_threshold = ((1 - static_blend) * hybrid_component) + (static_blend * static_threshold)
            
            logger.info(f"Hybrid threshold calculation: "
                       f"Market({market_weight:.1f}×{market_component:.3f}) + "
                       f"Signal({signal_weight:.1f}×{signal_component:.3f}) = "
                       f"Hybrid({hybrid_component:.3f}) → "
                       f"Final({final_threshold:.3f})")
            
            return final_threshold
            
        except Exception as e:
            logger.error(f"Error in hybrid threshold calculation: {e}")
            return signal_component  # Fallback to signal component only
    
    def _create_threshold_explanation(self, 
                                    base_threshold: float,
                                    final_threshold: float,
                                    signal_stats: Dict[str, Any],
                                    signal_count: int) -> str:
        """
        Create a human-readable explanation of the threshold calculation.
        
        Args:
            base_threshold: Base threshold from signal distribution
            final_threshold: Final adjusted threshold
            signal_stats: Signal statistics dictionary
            signal_count: Number of signals used in calculation
            
        Returns:
            Explanation string
        """
        explanation_parts = [
            f"Calculated from {signal_count} recent signals",
            f"Signal range: {signal_stats.get('min', 0):.3f} to {signal_stats.get('max', 0):.3f}",
            f"Average signal: {signal_stats.get('mean', 0):.3f}"
        ]
        
        if abs(final_threshold - base_threshold) > 0.01:
            adjustment_percent = ((final_threshold / base_threshold) - 1) * 100
            if adjustment_percent > 0:
                explanation_parts.append(f"Adjusted +{adjustment_percent:.1f}% for market conditions")
            else:
                explanation_parts.append(f"Adjusted {adjustment_percent:.1f}% for market conditions")
        
        return " | ".join(explanation_parts)
    
    def get_threshold_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive threshold calculation metrics.
        
        Returns:
            Dictionary containing threshold calculation details
        """
        try:
            recent_signals = self.history_tracker.get_recent_signals(days=30)
            signal_stats = self.history_tracker.get_signal_statistics(days=30)
            current_threshold, explanation = self.calculate_threshold()
            
            return {
                'current_threshold': current_threshold,
                'explanation': explanation,
                'signal_count': len(recent_signals),
                'signal_statistics': signal_stats,
                'config': {
                    'confidence_level': self.confidence_level,
                    'min_signals_required': self.min_signals_required,
                    'volatility_adjustment': self.volatility_adjustment
                }
            }
        except Exception as e:
            logger.error(f"Error getting threshold metrics: {e}")
            return {
                'current_threshold': self.default_threshold,
                'explanation': f"Error retrieving metrics: {str(e)}",
                'signal_count': 0,
                'signal_statistics': {},
                'config': {}
            }