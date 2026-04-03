"""
Layer Aggregator
Combines signals from multiple analysis layers using Confidence-Based Dynamic Weighting
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import layer_weight_config, safety_config

# Setup logging
logger = logging.getLogger(__name__)


# ==================== DATA CLASSES ====================

@dataclass
class LayerSignal:
    """Standardized output from each analysis layer"""
    layer_name: str
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0-1
    base_weight: float  # From config
    raw_score: float  # -1 to +1
    is_valid: bool  # Has sufficient data
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerContribution:
    """Contribution of a single layer to final signal"""
    layer_name: str
    display_name: str
    signal: str
    confidence: float
    base_weight: float
    adjusted_weight: float
    contribution: float
    is_valid: bool


@dataclass
class AggregatedSignal:
    """Final aggregated signal from all layers"""
    final_signal: str  # "BUY", "SELL", "HOLD"
    final_confidence: float  # 0-1
    raw_score: float  # Weighted average (-1 to +1)
    
    # Layer breakdown
    layer_contributions: List[LayerContribution] = field(default_factory=list)
    
    # Agreement metrics
    agreement_level: float = 0.0  # 0-1, how much layers agree
    dominant_layer: str = ""  # Most influential layer
    
    # Metadata
    timestamp: str = ""
    layers_used: int = 0
    layers_total: int = 0


# ==================== LAYER AGGREGATOR CLASS ====================

class LayerAggregator:
    """
    Aggregates signals from multiple layers using Confidence-Based Dynamic Weighting
    
    Weight adjustment logic:
    - Each layer's weight = base_weight × confidence
    - Weights are normalized to sum to 1.0
    - Invalid/missing layers get 0 weight and their weight redistributes
    """
    
    def __init__(self):
        """Initialize Layer Aggregator with config weights"""
        self.config = layer_weight_config
        
        # Base weights from config
        self.base_weights = {
            'ml_model': self.config.WEIGHT_ML_MODEL,
            'whale_engine': self.config.WEIGHT_WHALE_ENGINE,
            'technical': self.config.WEIGHT_TECHNICAL,
            'holder_metrics': self.config.WEIGHT_HOLDER_METRICS,
            'user_profile': self.config.WEIGHT_USER_PROFILE
        }
        
        # Display names
        self.display_names = self.config.LAYER_NAMES
        
        logger.info("Layer Aggregator initialized")
        logger.info(f"Base weights: {self.base_weights}")
    
    def aggregate(
        self,
        layer_signals: Dict[str, LayerSignal],
        excluded_from_coverage: Optional[set] = None
    ) -> AggregatedSignal:
        """
        Aggregate signals from all layers into final recommendation
        
        Uses Confidence-Based Dynamic Weighting:
        - adjusted_weight = base_weight × confidence
        - All weights normalized to sum to 1.0
        
        Args:
            layer_signals: Dictionary of layer name -> LayerSignal
            excluded_from_coverage: Layer names that are structurally unavailable
                for this token type and should not count toward coverage penalty
            
        Returns:
            AggregatedSignal with final recommendation
        """
        logger.debug(f"Aggregating {len(layer_signals)} layer signals...")
        
        # Step 1: Calculate adjusted weights
        adjusted_weights = self._calculate_adjusted_weights(layer_signals)
        
        # Step 2: Calculate weighted score
        weighted_score = 0.0
        total_weight_used = 0.0
        contributions = []
        
        for layer_name, signal in layer_signals.items():
            weight = adjusted_weights.get(layer_name, 0.0)
            
            if signal.is_valid and weight > 0:
                # Convert signal to numeric value
                signal_value = self._signal_to_numeric(signal.signal, signal.confidence)
                
                # Calculate contribution
                contribution = signal_value * weight
                weighted_score += contribution
                total_weight_used += weight
                
                contributions.append(LayerContribution(
                    layer_name=layer_name,
                    display_name=self.display_names.get(layer_name, layer_name),
                    signal=signal.signal,
                    confidence=signal.confidence,
                    base_weight=self.base_weights.get(layer_name, 0.0),
                    adjusted_weight=weight,
                    contribution=contribution,
                    is_valid=signal.is_valid
                ))
            else:
                contributions.append(LayerContribution(
                    layer_name=layer_name,
                    display_name=self.display_names.get(layer_name, layer_name),
                    signal=signal.signal if signal else "N/A",
                    confidence=signal.confidence if signal else 0.0,
                    base_weight=self.base_weights.get(layer_name, 0.0),
                    adjusted_weight=0.0,
                    contribution=0.0,
                    is_valid=False
                ))
        
        # Step 3: Convert score to signal
        final_signal, final_confidence = self._score_to_signal(weighted_score)
        
        # Step 4: Apply agreement bonus/penalty
        agreement_level = self._calculate_agreement(layer_signals)
        final_confidence = self._apply_agreement_adjustment(final_confidence, agreement_level)
        
        # Step 5: Apply coverage penalty when too few layers are valid
        # Exclude structurally-unavailable layers (e.g. holder_metrics for perps)
        _excluded = excluded_from_coverage or set()
        applicable_signals = {k: v for k, v in layer_signals.items() if k not in _excluded}
        valid_layers = sum(1 for s in applicable_signals.values() if s.is_valid)
        total_layers = len(applicable_signals)
        if total_layers > 0:
            coverage_ratio = valid_layers / total_layers
            min_coverage = self.config.MIN_COVERAGE_RATIO
            if coverage_ratio < min_coverage:
                coverage_factor = 0.5 + 0.5 * coverage_ratio
                final_confidence = max(0.3, final_confidence * coverage_factor)
                logger.info(
                    f"Coverage penalty applied: {valid_layers}/{total_layers} applicable layers valid "
                    f"(ratio {coverage_ratio:.2f} < {min_coverage}), "
                    f"confidence scaled by {coverage_factor:.2f}"
                )

        # Step 6: Find dominant layer
        dominant_layer = ""
        max_contribution = 0.0
        for contrib in contributions:
            if abs(contrib.contribution) > abs(max_contribution):
                max_contribution = contrib.contribution
                dominant_layer = contrib.layer_name
        
        result = AggregatedSignal(
            final_signal=final_signal,
            final_confidence=final_confidence,
            raw_score=weighted_score,
            layer_contributions=contributions,
            agreement_level=agreement_level,
            dominant_layer=dominant_layer,
            timestamp=datetime.now().isoformat(),
            layers_used=valid_layers,
            layers_total=total_layers
        )
        
        logger.info(f"Aggregation result: {final_signal} ({final_confidence:.2f}), "
                   f"score: {weighted_score:.3f}, agreement: {agreement_level:.2f}")
        
        return result
    
    def _calculate_adjusted_weights(
        self, 
        layer_signals: Dict[str, LayerSignal]
    ) -> Dict[str, float]:
        """
        Calculate confidence-adjusted weights
        
        Weight adjustment:
        - adjusted = base_weight × (MIN_MULT + confidence × (MAX_MULT - MIN_MULT))
        - Invalid layers get 0
        - All weights normalized to sum to 1.0
        """
        adjusted = {}
        
        for layer_name, base_weight in self.base_weights.items():
            signal = layer_signals.get(layer_name)
            
            if not signal or not signal.is_valid:
                adjusted[layer_name] = 0.0
                continue
            
            # Quality multiplier based on confidence
            # Maps confidence (0-1) to (MIN_MULT - MAX_MULT)
            min_mult = self.config.MIN_CONFIDENCE_MULTIPLIER
            max_mult = self.config.MAX_CONFIDENCE_MULTIPLIER
            quality_mult = min_mult + (signal.confidence * (max_mult - min_mult))
            
            adjusted[layer_name] = base_weight * quality_mult
        
        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            for layer_name in adjusted:
                adjusted[layer_name] /= total
        
        return adjusted
    
    def _signal_to_numeric(self, signal: str, confidence: float) -> float:
        """
        Convert signal to numeric value for aggregation
        
        Returns:
            -1 to +1 where:
            - +confidence for BUY
            - -confidence for SELL
            - 0 for HOLD
        """
        if signal == "BUY":
            return confidence
        elif signal == "SELL":
            return -confidence
        else:  # HOLD
            return 0.0
    
    def _score_to_signal(self, score: float) -> Tuple[str, float]:
        """
        Convert numeric score back to signal with confidence tapering
        near thresholds to avoid false conviction on borderline scores.

        Args:
            score: Weighted score (-1 to +1)

        Returns:
            (signal, confidence)
        """
        buy_threshold = self.config.SIGNAL_BUY_THRESHOLD
        sell_threshold = self.config.SIGNAL_SELL_THRESHOLD
        taper_zone = safety_config.THRESHOLD_TAPER_ZONE
        taper_min = safety_config.TAPER_MIN_FACTOR

        confidence_base = self.config.CONFIDENCE_BASE

        if score >= buy_threshold:
            signal = "BUY"
            confidence = min(0.95, confidence_base + score)
            distance = score - buy_threshold
            if distance < taper_zone:
                taper_factor = taper_min + (1.0 - taper_min) * (distance / taper_zone)
                confidence *= taper_factor
        elif score <= sell_threshold:
            signal = "SELL"
            confidence = min(0.95, confidence_base + abs(score))
            distance = abs(score) - abs(sell_threshold)
            if distance < taper_zone:
                taper_factor = taper_min + (1.0 - taper_min) * (distance / taper_zone)
                confidence *= taper_factor
        else:
            signal = "HOLD"
            confidence = 0.6 - abs(score)

        return signal, max(0.3, min(0.95, confidence))
    
    def _calculate_agreement(self, layer_signals: Dict[str, LayerSignal]) -> float:
        """
        Calculate how much the layers agree
        
        Returns:
            0-1 where 1 = perfect agreement
        """
        valid_signals = [s.signal for s in layer_signals.values() if s.is_valid]
        
        if len(valid_signals) < 2:
            return 1.0  # Single signal or none = perfect agreement
        
        # Count occurrences
        from collections import Counter
        counts = Counter(valid_signals)
        
        # Agreement = most common signal count / total signals
        most_common_count = counts.most_common(1)[0][1]
        agreement = most_common_count / len(valid_signals)
        
        return agreement
    
    def _apply_agreement_adjustment(
        self, 
        confidence: float, 
        agreement: float
    ) -> float:
        """
        Apply bonus/penalty based on layer agreement
        
        High agreement → boost confidence
        Low agreement → reduce confidence
        """
        high_threshold = self.config.HIGH_AGREEMENT_THRESHOLD
        low_threshold = self.config.LOW_AGREEMENT_THRESHOLD
        bonus = self.config.AGREEMENT_CONFIDENCE_BONUS
        penalty = self.config.DISAGREEMENT_CONFIDENCE_PENALTY
        
        if agreement >= high_threshold:
            # Bonus for high agreement
            confidence = min(0.95, confidence + bonus)
        elif agreement <= low_threshold:
            # Penalty for low agreement
            confidence = max(0.3, confidence - penalty)
        
        return confidence


# ==================== FACTORY FUNCTIONS ====================

def create_layer_signal(
    layer_name: str,
    signal: str,
    confidence: float,
    is_valid: bool = True,
    details: Dict[str, Any] = None
) -> LayerSignal:
    """
    Factory function to create a LayerSignal
    
    Args:
        layer_name: Name of the layer
        signal: BUY/SELL/HOLD
        confidence: 0-1
        is_valid: Whether layer has valid data
        details: Additional layer-specific details
        
    Returns:
        LayerSignal object
    """
    config = layer_weight_config
    
    base_weights = {
        'ml_model': config.WEIGHT_ML_MODEL,
        'whale_engine': config.WEIGHT_WHALE_ENGINE,
        'technical': config.WEIGHT_TECHNICAL,
        'holder_metrics': config.WEIGHT_HOLDER_METRICS,
        'user_profile': config.WEIGHT_USER_PROFILE
    }
    
    base_weight = base_weights.get(layer_name, 0.0)
    
    # Calculate raw score
    if signal == "BUY":
        raw_score = confidence
    elif signal == "SELL":
        raw_score = -confidence
    else:
        raw_score = 0.0
    
    return LayerSignal(
        layer_name=layer_name,
        signal=signal,
        confidence=confidence,
        base_weight=base_weight,
        raw_score=raw_score,
        is_valid=is_valid,
        details=details or {}
    )


def aggregation_to_dict(aggregation: AggregatedSignal) -> Dict[str, Any]:
    """Convert AggregatedSignal to dictionary for JSON serialization"""
    return {
        'final_signal': aggregation.final_signal,
        'final_confidence': round(aggregation.final_confidence, 4),
        'raw_score': round(aggregation.raw_score, 4),
        'agreement_level': round(aggregation.agreement_level, 4),
        'dominant_layer': aggregation.dominant_layer,
        'layers_used': aggregation.layers_used,
        'layers_total': aggregation.layers_total,
        'layer_contributions': [
            {
                'layer': c.layer_name,
                'display_name': c.display_name,
                'signal': c.signal,
                'confidence': round(c.confidence, 4),
                'base_weight': round(c.base_weight, 4),
                'adjusted_weight': round(c.adjusted_weight, 4),
                'contribution': round(c.contribution, 4),
                'is_valid': c.is_valid
            }
            for c in aggregation.layer_contributions
        ],
        'timestamp': aggregation.timestamp
    }


if __name__ == "__main__":
    print("Layer Aggregator")
    print("=" * 60)
    
    # Test with mock signals
    aggregator = LayerAggregator()
    
    # Create mock layer signals
    test_signals = {
        'ml_model': create_layer_signal('ml_model', 'BUY', 0.72, True),
        'whale_engine': create_layer_signal('whale_engine', 'BUY', 0.65, True),
        'technical': create_layer_signal('technical', 'HOLD', 0.55, True),
        'holder_metrics': create_layer_signal('holder_metrics', 'SELL', 0.70, True),
        'user_profile': create_layer_signal('user_profile', 'BUY', 0.80, True)
    }
    
    # Aggregate
    result = aggregator.aggregate(test_signals)
    
    print(f"\nTest Aggregation Result:")
    print(f"  Final Signal: {result.final_signal}")
    print(f"  Confidence: {result.final_confidence:.2f}")
    print(f"  Raw Score: {result.raw_score:.3f}")
    print(f"  Agreement: {result.agreement_level:.2f}")
    print(f"  Dominant Layer: {result.dominant_layer}")
    
    print(f"\nLayer Contributions:")
    for contrib in result.layer_contributions:
        print(f"  {contrib.display_name}: {contrib.signal} "
              f"(conf: {contrib.confidence:.2f}, weight: {contrib.adjusted_weight:.3f}, "
              f"contrib: {contrib.contribution:+.3f})")
    
    print("\nLayer Aggregator initialized successfully!")

