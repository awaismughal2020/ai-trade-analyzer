"""
OpenAI Service for generating user profile descriptions and insights
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib
import json

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import openai_config, cache_config, post_trade_review_config, api_config
from core.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, sentry_fallback_warning

logger = logging.getLogger(__name__)


class OpenAIService:
    """
    Service for generating AI-powered profile descriptions and insights
    """
    
    def __init__(self):
        """Initialize OpenAI service"""
        self.config = openai_config
        self.client = None
        self.description_cache: Dict[str, Tuple[str, datetime]] = {}  # cache_key -> (description, cached_at)
        self.circuit_breaker = CircuitBreaker(
            service_name="openai",
            failure_threshold=api_config.CIRCUIT_BREAKER_THRESHOLD,
            cooldown_period=api_config.CIRCUIT_BREAKER_COOLDOWN,
        )
        
        if not self.config.is_configured():
            logger.warning("OpenAI API key not configured. Profile descriptions will be disabled.")
            return
        
        if OpenAI is None:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
            return
        
        try:
            self.client = OpenAI(api_key=self.config.API_KEY)
            logger.info(f"OpenAI service initialized with model: {self.config.MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def _get_cache_key(self, profile_data: Dict[str, Any]) -> str:
        """Generate cache key from profile data"""
        # Create hash from key profile metrics
        cache_data = {
            'wallet': profile_data.get('wallet_address', ''),
            'total_trades': profile_data.get('total_trades', 0),
            'win_rate': round(profile_data.get('overall_win_rate', 0), 2),
            'lifetime_pnl': round(profile_data.get('lifetime_pnl', 0), 2),
            'trader_type': profile_data.get('trader_type', ''),
            'best_pattern': profile_data.get('best_pattern', {}).get('pattern_type', '') if profile_data.get('best_pattern') else '',
            'worst_pattern': profile_data.get('worst_pattern', {}).get('pattern_type', '') if profile_data.get('worst_pattern') else '',
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_at: datetime) -> bool:
        """Check if cache entry is still valid"""
        if not self.config.PROFILE_CACHE_ENABLED:
            return False
        age = (datetime.now() - cached_at).total_seconds()
        return age < self.config.PROFILE_CACHE_TTL_SECONDS
    
    def generate_profile_description(self, profile_data: Dict[str, Any]) -> str:
        """
        Generate AI-powered profile description (3-4 sentences)
        
        Args:
            profile_data: Complete user profile dictionary
            
        Returns:
            Generated description or fallback message
        """
        # Check cache first
        if self.config.PROFILE_CACHE_ENABLED:
            cache_key = self._get_cache_key(profile_data)
            if cache_key in self.description_cache:
                description, cached_at = self.description_cache[cache_key]
                if self._is_cache_valid(cached_at):
                    logger.debug(f"Using cached profile description for {profile_data.get('wallet_address', 'unknown')[:8]}...")
                    return description
                else:
                    # Remove expired cache
                    del self.description_cache[cache_key]
        
        # Check if AI is enabled and configured
        if not self.config.PROFILE_DESCRIPTION_ENABLED:
            return None
        
        if not self.client:
            logger.debug("OpenAI not configured, returning default description")
            sentry_fallback_warning("openai", "OpenAI client not configured — returning static profile description")
            return "Profile description unavailable."
        
        try:
            self.circuit_breaker.check()
        except CircuitBreakerOpen:
            sentry_fallback_warning("openai", "Circuit breaker OPEN — returning static profile description")
            return "Profile description unavailable."
        
        win_rate = profile_data.get('overall_win_rate', 0) * 100
        trader_type = profile_data.get('trader_type', 'UNKNOWN')
        total_trades = profile_data.get('total_trades', 0)
        lifetime_pnl = profile_data.get('lifetime_pnl', 0)
        avg_hold_time = profile_data.get('avg_hold_time_hours', 0)
        entry_phase = profile_data.get('typical_entry_phase', 'UNKNOWN')
        fomo_raw = profile_data.get('fomo_tendency')
        panic_raw = profile_data.get('panic_sell_tendency')
        fomo_tendency_pct = (fomo_raw * 100) if fomo_raw is not None else None
        panic_tendency_pct = (panic_raw * 100) if panic_raw is not None else None

        best_pattern = profile_data.get('best_pattern', {})
        worst_pattern = profile_data.get('worst_pattern', {})

        best_pattern_name = best_pattern.get('pattern_type', 'None') if best_pattern else 'None'
        best_win_rate = best_pattern.get('win_rate', 0) * 100 if best_pattern else 0
        best_occurrences = best_pattern.get('occurrences', 0) if best_pattern else 0

        worst_pattern_name = worst_pattern.get('pattern_type', 'None') if worst_pattern else 'None'
        worst_win_rate = worst_pattern.get('win_rate', 0) * 100 if worst_pattern else 0
        worst_occurrences = worst_pattern.get('occurrences', 0) if worst_pattern else 0

        prompt = f"""Based on this trader's analytics, write a 3-4 sentence profile description:

- Win Rate: {win_rate:.1f}%
- Trader Type: {trader_type}
- Total Trades: {total_trades:,}
- Lifetime PnL: ${lifetime_pnl:,.2f}
- Best Pattern: {best_pattern_name} ({best_win_rate:.1f}% win rate, {best_occurrences} occurrences)
- Worst Pattern: {worst_pattern_name} ({worst_win_rate:.1f}% win rate, {worst_occurrences} occurrences)
- Average Hold Time: {avg_hold_time:.1f} hours
- Typical Entry: {entry_phase} tokens
- FOMO Tendency: {f"{fomo_tendency_pct:.1f}%" if fomo_tendency_pct is not None else "Not computed"}
- Panic Sell Tendency: {f"{panic_tendency_pct:.1f}%" if panic_tendency_pct is not None else "Not computed"}

Write a brief, direct personality summary. Use "You are..." format. 
Mention their trading style, main strength, and biggest weakness.
Keep it to 3-4 sentences only. Be specific about numbers and patterns."""

        last_err = None
        for attempt in range(self.config.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.MODEL,
                    messages=[
                        {"role": "system", "content": "You are a trading analyst providing concise, actionable trader profiles."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.MAX_TOKENS,
                    temperature=self.config.TEMPERATURE,
                    timeout=self.config.TIMEOUT,
                )
                self.circuit_breaker.record_success()
                description = response.choices[0].message.content.strip()

                if self.config.PROFILE_CACHE_ENABLED:
                    cache_key = self._get_cache_key(profile_data)
                    self.description_cache[cache_key] = (description, datetime.now())
                    if len(self.description_cache) > 100:
                        sorted_cache = sorted(self.description_cache.items(), key=lambda x: x[1][1])
                        for key, _ in sorted_cache[:-100]:
                            del self.description_cache[key]

                logger.debug(f"Generated profile description for {profile_data.get('wallet_address', 'unknown')[:8]}...")
                return description
            except Exception as e:
                last_err = e
                if attempt < self.config.MAX_RETRIES:
                    backoff = 2 ** attempt
                    logger.warning(f"OpenAI profile description attempt {attempt + 1} failed, retrying in {backoff}s: {e}")
                    time.sleep(backoff)

        self.circuit_breaker.record_failure(last_err)
        logger.error(f"Error generating profile description after {self.config.MAX_RETRIES + 1} attempts: {last_err}", exc_info=True)
        sentry_fallback_warning(
            "openai",
            f"All {self.config.MAX_RETRIES + 1} attempts failed — returning static profile description",
            {"last_error": str(last_err)},
        )
        return "Profile description unavailable."
    
    def generate_risk_rating(self, profile_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate risk rating (GREEN/YELLOW/RED) with reason
        
        Rule-based logic (no AI needed):
        - GREEN: win_rate >= 50% AND worst_pattern occurrences < 10% of total trades
        - RED: win_rate < 30% OR panic_sell_tendency > 30% OR worst_pattern occurrences > 30% of total trades
        - YELLOW: otherwise
        
        Args:
            profile_data: Complete user profile dictionary
            
        Returns:
            Dictionary with 'rating' and 'reason'
        """
        win_rate = profile_data.get('overall_win_rate', 0)
        total_trades = profile_data.get('total_trades', 0)
        panic_tendency = profile_data.get('panic_sell_tendency') or 0
        worst_pattern = profile_data.get('worst_pattern', {})
        
        worst_occurrences = worst_pattern.get('occurrences', 0) if worst_pattern else 0
        worst_pattern_name = worst_pattern.get('pattern_type', 'None') if worst_pattern else 'None'
        worst_pct = (worst_occurrences / total_trades * 100) if total_trades > 0 else 0
        
        # Determine rating
        if win_rate >= 0.50 and worst_pct < 10:
            rating = "GREEN"
            reason = f"Strong win rate ({win_rate*100:.1f}%) and low worst pattern occurrence ({worst_pct:.1f}%)"
        elif win_rate < 0.30 or panic_tendency > 0.30 or worst_pct > 30:
            reasons = []
            if win_rate < 0.30:
                reasons.append(f"win rate below 30% ({win_rate*100:.1f}%)")
            if panic_tendency > 0.30:
                reasons.append(f"high panic sell tendency ({panic_tendency*100:.1f}%)")
            if worst_pct > 30:
                reasons.append(f"high worst pattern occurrence ({worst_pct:.1f}% - {worst_pattern_name})")
            rating = "RED"
            reason = " and ".join(reasons)
        else:
            rating = "YELLOW"
            reason = f"Moderate performance (win rate: {win_rate*100:.1f}%, worst pattern: {worst_pct:.1f}%)"
        
        return {
            "rating": rating,
            "reason": reason
        }
    
    def generate_recommendations(self, profile_data: Dict[str, Any]) -> List[str]:
        """
        Generate 2-3 actionable, conversational recommendations based on profile
        
        Rule-based logic (no AI needed)
        Recommendations are written in natural, non-technical language
        
        Args:
            profile_data: Complete user profile dictionary
            
        Returns:
            List of recommendation strings (conversational, actionable)
        """
        recommendations = []
        
        panic_tendency = profile_data.get('panic_sell_tendency') or 0
        fomo_tendency = profile_data.get('fomo_tendency') or 0
        dip_buy_tendency = profile_data.get('dip_buy_tendency') or 0
        entry_phase = profile_data.get('typical_entry_phase', 'UNKNOWN')
        is_flipper = profile_data.get('is_flipper', False)
        avg_hold_time = profile_data.get('avg_hold_time_hours', 0)
        worst_pattern = profile_data.get('worst_pattern', {})
        best_pattern = profile_data.get('best_pattern', {})
        win_rate = profile_data.get('overall_win_rate', 0)
        total_trades = profile_data.get('total_trades', 0)
        patterns = profile_data.get('patterns', [])
        
        # Check patterns array for specific patterns
        panic_sell_pattern = next((p for p in patterns if p.get('pattern_type') == 'PANIC_SELL'), None)
        fomo_pattern = next((p for p in patterns if p.get('pattern_type') == 'FOMO'), None)
        dip_buy_pattern = next((p for p in patterns if p.get('pattern_type') == 'DIP_BUY'), None)
        early_entry_pattern = next((p for p in patterns if p.get('pattern_type') == 'EARLY_ENTRY'), None)
        quick_flip_pattern = next((p for p in patterns if p.get('pattern_type') == 'QUICK_FLIP'), None)
        
        # Recommendation 1: Panic selling (highest priority)
        if panic_sell_pattern and panic_sell_pattern.get('occurrences', 0) > total_trades * 0.05:
            recommendations.append(
                "Avoid panic selling during dips — set stop-losses in advance instead of emotional exits."
            )
        
        # Recommendation 2: Best performing pattern (use the one from profile, only if it exists)
        # Only recommend if best_pattern is set (user_profiler already validated it has >= 10 occurrences and > 30% win rate)
        if best_pattern and best_pattern.get('pattern_type') and best_pattern.get('pattern_type') != 'UNKNOWN':
            best_pattern_name = best_pattern.get('pattern_type', '')
            
            pattern_messages = {
                'DIP_BUY': "Your best performance is buying dips. Focus on this strategy — it's working for you.",
                'FOMO': "You perform well when entering after momentum. Be selective — not all pumps are sustainable.",
                'EARLY_ENTRY': "You excel at early entries. Continue focusing on tokens in their early stages.",
                'QUICK_FLIP': "Your quick flips are profitable. Keep your exit strategy tight and take profits early.",
                'LONG_HOLD': "Your long holds show strong results. Patience pays off for you."
            }
            
            if best_pattern_name in pattern_messages:
                recommendations.append(pattern_messages[best_pattern_name])
            else:
                recommendations.append(
                    f"Your best performance comes from {best_pattern_name.replace('_', ' ').lower()} setups. Focus on these opportunities."
                )
        
        # Recommendation 3: Worst pattern (avoid it)
        if worst_pattern and worst_pattern.get('pattern_type') != 'UNKNOWN':
            worst_pattern_name = worst_pattern.get('pattern_type', '')
            # Get PnL from patterns array
            worst_pattern_data = next((p for p in patterns if p.get('pattern_type') == worst_pattern_name), None)
            worst_pnl = worst_pattern_data.get('avg_pnl_percent', 0) if worst_pattern_data else 0
            
            if worst_pattern_name == 'PANIC_SELL':
                recommendations.append(
                    "Avoid panic selling — it's your biggest weakness. Set stop-losses in advance instead of emotional exits."
                )
            elif worst_pattern_name == 'FOMO':
                recommendations.append(
                    "Avoid chasing pumps — wait for pullbacks instead of buying at the top."
                )
            elif worst_pnl < -20:
                pattern_display = worst_pattern_name.replace('_', ' ').lower()
                recommendations.append(
                    f"Avoid {pattern_display} — it's consistently losing you money."
                )
        
        # Recommendation 4: Early entry risk (P1 phase)
        if entry_phase == "P1" and early_entry_pattern and early_entry_pattern.get('win_rate', 0) < 0.30:
            recommendations.append(
                "Consider waiting for tokens to mature slightly — your early entries carry higher risk."
            )
        
        # Recommendation 5: Quick flips (if not working)
        if is_flipper and avg_hold_time < 2.0 and quick_flip_pattern:
            if quick_flip_pattern.get('win_rate', 0) < 0.30:
                recommendations.append(
                    "Your quick flips aren't working well. Try holding positions longer to let winners run."
                )
        
        # Recommendation 7: Overtrading
        if total_trades > 5000 and win_rate < 0.30 and len(recommendations) < 3:
            recommendations.append(
                "You're overtrading. Focus on quality setups over quantity — be more selective."
            )
        
        # Recommendation 8: Low win rate general advice
        if win_rate < 0.30 and len(recommendations) < 3:
            recommendations.append(
                "Focus on quality over quantity — wait for higher-probability setups before entering."
            )
        
        # Remove duplicates (keep first occurrence)
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            rec_lower = rec.lower()
            # Check if similar recommendation already exists
            is_duplicate = False
            for seen_rec in seen:
                # Check if they're talking about the same thing (e.g., both about panic selling)
                if any(keyword in rec_lower and keyword in seen_rec for keyword in ['panic', 'dips', 'pumps', 'early', 'quick flip', 'overtrading']):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_recommendations.append(rec)
                seen.add(rec_lower)
        
        recommendations = unique_recommendations
        
        # Ensure we have exactly 3 recommendations
        if len(recommendations) == 0:
            recommendations.append("Continue tracking your patterns to identify what works best for you.")
            recommendations.append("Focus on setups that match your strengths and avoid your weaknesses.")
            recommendations.append("Be patient and wait for higher-probability setups before entering.")
        elif len(recommendations) == 1:
            recommendations.append("Focus on setups that match your strengths and avoid your weaknesses.")
            recommendations.append("Be patient and wait for higher-probability setups before entering.")
        elif len(recommendations) == 2:
            recommendations.append("Be patient and wait for higher-probability setups before entering.")
        elif len(recommendations) > 3:
            # Prioritize: panic sell > best pattern > entry phase > FOMO > general
            priority_keywords = ['panic', 'best performance', 'dips', 'mature', 'chasing', 'overtrading', 'quality']
            recommendations_sorted = sorted(
                recommendations,
                key=lambda r: min([i for i, kw in enumerate(priority_keywords) if kw.lower() in r.lower()] or [999])
            )
            recommendations = recommendations_sorted[:3]
        
        return recommendations
    
    # ==================== POST-TRADE REVIEW (FEATURE 4) ====================
    
    def generate_post_trade_review(
        self,
        analysis_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[str]]:
        """
        Generate AI-powered post-trade coaching message and recommendations
        
        Args:
            analysis_data: PostTradeAnalysis data as dictionary containing:
                - trade_type: BUY or SELL
                - symbol: Token symbol
                - outcome: OutcomeComparison data
                - mistakes: List of identified mistakes
                - missed_opportunity_pct: Percentage of missed gains/better entry
                - monthly_impact_estimate_pct: Extrapolated monthly impact
            user_context: Optional user profile data for personalization
            
        Returns:
            Tuple of (coaching_message, recommendations_list)
        """
        ptr_config = post_trade_review_config
        
        # Check if coaching is enabled
        if not ptr_config.COACHING_ENABLED:
            return self._generate_rule_based_review(analysis_data, user_context)
        
        if not self.client:
            logger.debug("OpenAI not configured, using rule-based review")
            sentry_fallback_warning("openai", "OpenAI client not configured — using rule-based post-trade review")
            return self._generate_rule_based_review(analysis_data, user_context)
        
        try:
            self.circuit_breaker.check()
        except CircuitBreakerOpen:
            sentry_fallback_warning("openai", "Circuit breaker OPEN — using rule-based post-trade review")
            return self._generate_rule_based_review(analysis_data, user_context)
        
        trade_type = analysis_data.get('trade_type', 'UNKNOWN')
        symbol = analysis_data.get('symbol', 'Unknown')
        outcome = analysis_data.get('outcome', {})
        mistakes = analysis_data.get('mistakes', [])
        missed_pct = analysis_data.get('missed_opportunity_pct', 0)
        monthly_impact = analysis_data.get('monthly_impact_estimate_pct', 0)

        primary_mistake = mistakes[0] if mistakes else None

        outcome_text = self._build_outcome_text(trade_type, outcome)
        mistakes_text = self._build_mistakes_text(mistakes)

        user_text = ""
        if user_context:
            fomo_ctx = user_context.get('fomo_tendency')
            panic_ctx = user_context.get('panic_sell_tendency')
            fomo_str = f"{fomo_ctx * 100:.1f}%" if fomo_ctx is not None else "Not computed"
            panic_str = f"{panic_ctx * 100:.1f}%" if panic_ctx is not None else "Not computed"
            user_text = f"""
User Profile:
- Trader Type: {user_context.get('trader_type', 'Unknown')}
- Win Rate: {user_context.get('win_rate', 0)*100:.1f}%
- Avg Hold Time: {user_context.get('avg_hold_time_hours', 0):.1f} hours
- FOMO Tendency: {fomo_str}
- Panic Sell Tendency: {panic_str}"""

        prompt = f"""Analyze this completed {trade_type} trade and provide coaching feedback:

Trade: {trade_type} {symbol}
{outcome_text}

Mistakes Identified:
{mistakes_text}

Missed Opportunity: {missed_pct:.1f}%
Estimated Monthly Impact: ~{monthly_impact:.1f}%
{user_text}

Write a 2-3 sentence coaching message that:
1. Directly addresses the primary mistake ({primary_mistake.get('type', 'Unknown') if primary_mistake else 'None'})
2. Includes specific numbers from the analysis
3. Provides an actionable fix

Use direct "You" language. Be specific about the impact.
Example format: "You closed too early. After your exit, price ran 12% more. This behavior costs you ~14% monthly."

Then list 2-3 specific recommendations to fix this pattern."""

        last_err = None
        for attempt in range(self.config.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a trading coach providing direct, actionable feedback on completed trades. Be specific with numbers and focus on improvement."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=ptr_config.REVIEW_MAX_TOKENS,
                    temperature=ptr_config.REVIEW_TEMPERATURE,
                    timeout=self.config.TIMEOUT,
                )
                self.circuit_breaker.record_success()
                full_response = response.choices[0].message.content.strip()
                coaching_message, recommendations = self._parse_coaching_response(full_response)
                logger.debug(f"Generated post-trade review for {symbol} ({trade_type})")
                return coaching_message, recommendations
            except Exception as e:
                last_err = e
                if attempt < self.config.MAX_RETRIES:
                    backoff = 2 ** attempt
                    logger.warning(f"OpenAI post-trade review attempt {attempt + 1} failed, retrying in {backoff}s: {e}")
                    time.sleep(backoff)

        self.circuit_breaker.record_failure(last_err)
        logger.error(f"Error generating post-trade review after {self.config.MAX_RETRIES + 1} attempts: {last_err}", exc_info=True)
        sentry_fallback_warning(
            "openai",
            f"All {self.config.MAX_RETRIES + 1} attempts failed — using rule-based post-trade review",
            {"last_error": str(last_err), "symbol": analysis_data.get("symbol", "unknown")},
        )
        return self._generate_rule_based_review(analysis_data, user_context)
    
    def _build_outcome_text(self, trade_type: str, outcome: Dict[str, Any]) -> str:
        """Build outcome text for prompt"""
        lines = [f"Entry/Exit Price: ${outcome.get('price_at_trade', 0):.8f}"]
        
        if trade_type == "SELL":
            if outcome.get('price_after_1h'):
                pct_1h = ((outcome['price_after_1h'] - outcome['price_at_trade']) / outcome['price_at_trade']) * 100
                lines.append(f"Price 1h after exit: {pct_1h:+.1f}%")
            if outcome.get('price_after_4h'):
                pct_4h = ((outcome['price_after_4h'] - outcome['price_at_trade']) / outcome['price_at_trade']) * 100
                lines.append(f"Price 4h after exit: {pct_4h:+.1f}%")
            if outcome.get('max_price_after_exit'):
                max_pct = ((outcome['max_price_after_exit'] - outcome['price_at_trade']) / outcome['price_at_trade']) * 100
                lines.append(f"Max price after exit: {max_pct:+.1f}% (reached in {outcome.get('time_to_max', 'N/A')})")
            if outcome.get('missed_gain_pct'):
                lines.append(f"Missed gain: {outcome['missed_gain_pct']:.1f}%")
        else:  # BUY
            if outcome.get('min_price_before_entry'):
                better_pct = ((outcome['price_at_trade'] - outcome['min_price_before_entry']) / outcome['price_at_trade']) * 100
                lines.append(f"Better entry available: {better_pct:.1f}% lower in prior window")
            if outcome.get('max_price_before_entry'):
                pump_pct = ((outcome['max_price_before_entry'] - outcome.get('min_price_before_entry', outcome['max_price_before_entry'])) / outcome.get('min_price_before_entry', 1)) * 100
                if pump_pct > 10:
                    lines.append(f"Recent pump before entry: +{pump_pct:.1f}%")
        
        return "\n".join(lines)
    
    def _build_mistakes_text(self, mistakes: List[Dict[str, Any]]) -> str:
        """Build mistakes text for prompt"""
        if not mistakes:
            return "No significant mistakes identified."
        
        lines = []
        for i, m in enumerate(mistakes[:3], 1):  # Limit to top 3 mistakes
            mistake_type = m.get('type', 'Unknown')
            severity = m.get('severity', 'Unknown')
            impact = m.get('impact_pct', 0)
            description = m.get('description', '')
            
            lines.append(f"{i}. {mistake_type} ({severity}): {description}")
            if impact > 0:
                lines.append(f"   Impact: {impact:.1f}%")
        
        return "\n".join(lines)
    
    def _parse_coaching_response(self, response: str) -> Tuple[str, List[str]]:
        """Parse OpenAI response into coaching message and recommendations"""
        lines = response.strip().split('\n')
        
        coaching_lines = []
        recommendations = []
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if we've hit recommendations section
            if any(marker in line.lower() for marker in ['recommendation', 'tips:', 'suggestions:', 'to fix:', 'action items:']):
                in_recommendations = True
                continue
            
            # Check if line starts with a number or bullet (likely a recommendation)
            if line[0].isdigit() or line.startswith('-') or line.startswith('•'):
                in_recommendations = True
                # Clean up the line
                clean_line = line.lstrip('0123456789.-•) ').strip()
                if clean_line:
                    recommendations.append(clean_line)
            elif not in_recommendations:
                coaching_lines.append(line)
        
        coaching_message = ' '.join(coaching_lines).strip()
        
        # Ensure we have at least some content
        if not coaching_message:
            coaching_message = "Trade analysis complete. Review the identified mistakes above."
        
        if not recommendations:
            recommendations = ["Review your exit strategy for similar setups."]
        
        return coaching_message, recommendations[:3]  # Limit to 3 recommendations
    
    def _generate_rule_based_review(
        self,
        analysis_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[str]]:
        """
        Generate rule-based coaching when OpenAI is not available
        
        This provides sensible feedback based on the identified mistakes
        without requiring AI.
        """
        trade_type = analysis_data.get('trade_type', 'UNKNOWN')
        symbol = analysis_data.get('symbol', 'Unknown')
        outcome = analysis_data.get('outcome', {})
        mistakes = analysis_data.get('mistakes', [])
        missed_pct = analysis_data.get('missed_opportunity_pct', 0)
        monthly_impact = analysis_data.get('monthly_impact_estimate_pct', 0)
        
        coaching_parts = []
        recommendations = []
        
        if not mistakes:
            coaching_parts.append(f"Your {trade_type.lower()} on {symbol} appears well-executed.")
            recommendations.append("Continue following your current strategy.")
            return ' '.join(coaching_parts), recommendations
        
        # Get primary mistake
        primary = mistakes[0]
        mistake_type = primary.get('type', '')
        impact = primary.get('impact_pct', 0)
        
        # Generate coaching based on mistake type
        if mistake_type == 'EARLY_EXIT':
            max_price = outcome.get('max_price_after_exit', 0)
            time_to_max = outcome.get('time_to_max', 'a few hours')
            coaching_parts.append(f"You closed too early.")
            if impact > 0:
                coaching_parts.append(f"After your exit, price ran {impact:.1f}% more.")
            if monthly_impact > 0:
                coaching_parts.append(f"This pattern costs you approximately {monthly_impact:.1f}% monthly.")
            recommendations.append("Consider using trailing stop-losses instead of manual exits.")
            recommendations.append("Wait for momentum indicators (RSI, MACD) to confirm exit timing.")
            if user_context and user_context.get('avg_hold_time_hours', 0) > 0:
                avg_hold = user_context['avg_hold_time_hours']
                recommendations.append(f"Your average profitable hold is {avg_hold:.1f}h - consider waiting longer.")
        
        elif mistake_type == 'LATE_ENTRY':
            evidence = primary.get('evidence', {})
            dip_pct = evidence.get('dip_pct', impact)
            time_to_dip = evidence.get('time_to_dip', 'a few hours')
            coaching_parts.append(f"You entered too early — the price dipped {dip_pct:.1f}% within {time_to_dip} after your buy.")
            coaching_parts.append(f"A better entry was available by waiting.")
            if monthly_impact > 0:
                coaching_parts.append(f"This pattern costs you approximately {monthly_impact:.1f}% monthly.")
            recommendations.append("Set limit orders at support levels instead of market orders.")
            recommendations.append("Wait for pullbacks after pumps before entering.")
            recommendations.append("Use RSI oversold signals to time entries.")
        
        elif mistake_type == 'FOMO_ENTRY':
            coaching_parts.append(f"You bought after a significant pump.")
            evidence = primary.get('evidence', {})
            pump_pct = evidence.get('price_change_before_entry', impact)
            coaching_parts.append(f"The token pumped {pump_pct:.1f}% before your entry.")
            recommendations.append("Avoid chasing green candles - wait for consolidation.")
            recommendations.append("Set price alerts at your target entry instead of reacting to pumps.")
            recommendations.append("If a token has pumped >20%, wait for a 10% pullback before entering.")
        
        elif mistake_type == 'PANIC_SELL':
            loss_pct = abs(primary.get('impact_pct', 0))
            coaching_parts.append(f"You panic sold at a {loss_pct:.1f}% loss.")
            coaching_parts.append("Emotional exits often lock in losses that would have recovered.")
            recommendations.append("Set stop-losses in advance instead of manual panic exits.")
            recommendations.append("Define your maximum acceptable loss before entering a trade.")
            recommendations.append("Take a break after a loss before making your next trade.")
        
        elif mistake_type == 'REVENGE_TRADE':
            coaching_parts.append("This trade was made shortly after a loss.")
            coaching_parts.append("Revenge trading typically leads to poor decision-making.")
            recommendations.append("Wait at least 4 hours after a significant loss before trading again.")
            recommendations.append("Review your trading journal before making emotionally-driven trades.")
            recommendations.append("Set a daily loss limit and stop trading when reached.")
        
        elif mistake_type == 'OVER_TRADING':
            evidence = primary.get('evidence', {})
            trade_count = evidence.get('trades_in_window', 0)
            coaching_parts.append(f"You made {trade_count} trades on {symbol} in a short window.")
            coaching_parts.append("Over-trading increases fees and often reduces returns.")
            recommendations.append("Limit yourself to 1-2 trades per token per day.")
            recommendations.append("Each trade should have a clear thesis - avoid impulsive entries.")
            recommendations.append("Transaction fees add up - factor them into your expected returns.")
        
        elif mistake_type == 'BAD_RISK_REWARD':
            coaching_parts.append("This trade had a poor risk-reward ratio.")
            if user_context:
                avg_pnl = user_context.get('avg_pnl_per_trade', 0) * 100
                coaching_parts.append(f"Your average trade returns {avg_pnl:.1f}%, this was significantly worse.")
            recommendations.append("Define your target profit and stop-loss before entering.")
            recommendations.append("Aim for at least 2:1 reward-to-risk ratio on each trade.")
            recommendations.append("Skip setups where the risk exceeds potential reward.")
        
        else:
            coaching_parts.append(f"Review your {trade_type.lower()} on {symbol}.")
            if impact > 0:
                coaching_parts.append(f"You may have missed {impact:.1f}% in potential gains.")
            recommendations.append("Keep a trading journal to identify patterns in your decisions.")
        
        coaching_message = ' '.join(coaching_parts)
        
        # Ensure we have recommendations
        if not recommendations:
            recommendations = [
                "Review this trade type in your history.",
                "Look for similar setups to identify patterns.",
                "Consider adjusting your entry/exit criteria."
            ]
        
        return coaching_message, recommendations[:3]

