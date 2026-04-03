"""
LLM Analyzer Service for AI-powered user profile behavioral analysis.

Uses Anthropic Claude as primary provider with OpenAI GPT-4o as fallback.
Replaces internal scoring logic with LLM-generated behavioral assessment
for the /user/ai-profile endpoint.
"""

import json
import logging
import time
from typing import Dict, Any, Optional

try:
    import anthropic as anthropic_sdk
except ImportError:
    anthropic_sdk = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import llm_analyzer_config, api_config
from core.circuit_breaker import CircuitBreaker, CircuitBreakerOpen

logger = logging.getLogger(__name__)

_MAX_RECOMMENDATIONS = 15


class LLMAnalyzerService:
    """
    Sends raw trading data to an LLM and receives structured behavioral
    analysis (patterns, tendencies, risk rating, recommendations).

    Provider priority:  config.PROVIDER  ->  fallback to the other.
    Both providers share a single circuit breaker (tripped = service-wide
    unavailability; callers should handle the 502).
    """

    def __init__(self):
        self.config = llm_analyzer_config
        self._anthropic_client = None
        self._openai_client = None

        self.circuit_breaker = CircuitBreaker(
            service_name="llm_analyzer",
            failure_threshold=api_config.CIRCUIT_BREAKER_THRESHOLD,
            cooldown_period=api_config.CIRCUIT_BREAKER_COOLDOWN,
        )

        if anthropic_sdk and self.config.is_anthropic_configured():
            try:
                self._anthropic_client = anthropic_sdk.Anthropic(
                    api_key=self.config.ANTHROPIC_API_KEY,
                    timeout=self.config.ANTHROPIC_TIMEOUT,
                )
                logger.info(f"LLM Analyzer: Anthropic client ready (model: {self.config.ANTHROPIC_MODEL})")
            except Exception as e:
                logger.error(f"LLM Analyzer: failed to init Anthropic client: {e}")

        if OpenAI and self.config.is_openai_configured():
            try:
                self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
                logger.info(f"LLM Analyzer: OpenAI fallback ready (model: {self.config.OPENAI_FALLBACK_MODEL})")
            except Exception as e:
                logger.error(f"LLM Analyzer: failed to init OpenAI client: {e}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, raw_data: Dict[str, Any], token_type: str) -> Dict[str, Any]:
        """
        Analyse raw trading data and return LLM-generated behavioral fields.

        Returns a dict with keys consumed by UserProfileResponse:
            trader_type, is_sniper, is_holder, is_flipper,
            fomo_tendency, panic_sell_tendency, dip_buy_tendency,
            whale_follow_tendency, patterns, worst_pattern, best_pattern,
            risk_rating, recommendations, profile_description.
        """
        is_valid = raw_data.get("is_valid", False)
        if not is_valid:
            return self._insufficient_data_response(token_type)

        system_prompt = self._build_system_prompt(token_type)
        user_prompt = self._build_user_prompt(raw_data, token_type)

        self.circuit_breaker.check()

        providers = self._ordered_providers()
        last_error: Optional[Exception] = None

        for provider_name, call_fn in providers:
            for attempt in range(1, self.config.MAX_RETRIES + 1):
                try:
                    logger.info(
                        f"LLM Analyzer: calling {provider_name} "
                        f"(attempt {attempt}/{self.config.MAX_RETRIES})"
                    )
                    start = time.perf_counter()
                    raw_text = call_fn(system_prompt, user_prompt)
                    elapsed = time.perf_counter() - start
                    logger.info(f"LLM Analyzer: {provider_name} responded in {elapsed:.2f}s")

                    parsed = self._parse_json(raw_text)
                    validated = self._validate(parsed, token_type)
                    self.circuit_breaker.record_success()
                    return validated

                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as parse_err:
                    logger.warning(
                        f"LLM Analyzer: {provider_name} returned unparseable response "
                        f"(attempt {attempt}): {parse_err}"
                    )
                    last_error = parse_err
                except CircuitBreakerOpen:
                    raise
                except Exception as e:
                    logger.warning(f"LLM Analyzer: {provider_name} error (attempt {attempt}): {e}")
                    last_error = e

        self.circuit_breaker.record_failure(last_error)
        raise RuntimeError(f"All LLM providers failed: {last_error}")

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    def _ordered_providers(self):
        primary = self.config.PROVIDER.lower()
        providers = []
        if primary == "anthropic" and self._anthropic_client:
            providers.append(("anthropic", self._call_anthropic))
            if self._openai_client:
                providers.append(("openai", self._call_openai))
        elif primary == "openai" and self._openai_client:
            providers.append(("openai", self._call_openai))
            if self._anthropic_client:
                providers.append(("anthropic", self._call_anthropic))
        else:
            if self._anthropic_client:
                providers.append(("anthropic", self._call_anthropic))
            if self._openai_client:
                providers.append(("openai", self._call_openai))
        return providers

    def _call_anthropic(self, system: str, user: str) -> str:
        response = self._anthropic_client.messages.create(
            model=self.config.ANTHROPIC_MODEL,
            max_tokens=self.config.ANTHROPIC_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        block = next((b for b in response.content if b.type == "text"), None)
        if not block:
            raise ValueError("Anthropic returned no text block")
        return block.text

    def _call_openai(self, system: str, user: str) -> str:
        response = self._openai_client.chat.completions.create(
            model=self.config.OPENAI_FALLBACK_MODEL,
            max_tokens=self.config.OPENAI_FALLBACK_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content
        if not text:
            raise ValueError("OpenAI returned empty content")
        return text

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, token_type: str) -> str:
        schema = self._meme_schema() if token_type == "meme" else self._perps_schema()
        analysis_framework = self._meme_analysis_framework() if token_type == "meme" else self._perps_analysis_framework()

        return f"""You are an elite crypto trading behavioral analyst with deep expertise in on-chain trading forensics, behavioral finance, and risk management. Your analysis directly impacts a trader's financial decisions — accuracy, specificity, and honesty are paramount.

You will receive:
1. Aggregate metrics for a wallet (total trades, PnL, win rate, volume, drawdown, hold times, entry phases)
2. Individual trade-level data (every buy/sell with timestamps, prices, PnL, token phases, hold durations)
3. Current market context (Fear & Greed index, BTC dominance, market regime, altcoin index)

Your job: Forensically analyze every trade, detect behavioral patterns, score tendencies, assess risk, and produce actionable recommendations.

═══════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════

Return ONLY valid JSON matching this exact structure. No markdown fences. No explanation. No preamble. No trailing text.

{schema}

═══════════════════════════════════════════════
ANALYSIS METHODOLOGY — FOLLOW THIS STEP BY STEP
═══════════════════════════════════════════════

Before generating the JSON output, you MUST perform this internal analysis (do NOT output it — it's your reasoning process):

{analysis_framework}

═══════════════════════════════════════════════
PATTERN DETECTION RULES
═══════════════════════════════════════════════

Classify EVERY trade into exactly one pattern. A trade that matches multiple patterns gets assigned to the MOST SPECIFIC one (PANIC_SELL > QUICK_FLIP > EARLY_ENTRY > UNKNOWN).

PATTERN DEFINITIONS WITH PRECISE THRESHOLDS:

1. EARLY_ENTRY
   - Token was in P1 (0-5 min after launch) or P2 (5-30 min) phase at time of entry
   - The trader bought BEFORE the token hit its first major peak
   - Win condition: sold for any profit
   - This pattern indicates sniper behavior — fast discovery, early positioning

2. QUICK_FLIP
   - Hold duration < 24 hours (1440 minutes)
   - Entered and exited within a single trading session
   - Win condition: sold for any profit
   - This pattern indicates short-term momentum capture

3. PANIC_SELL
   - Sold at a loss of >= 40% from entry price
   - Hold duration < 120 minutes (2 hours) OR sold during a sharp price drop (> 25% in < 30 min)
   - Win condition: NEVER (by definition, panic sells are always losses)
   - CRITICAL: count these precisely — every trade matching this is a panic sell

4. DIP_BUY
   - Bought when token price was > 20% below its recent high (last 24h)
   - Indicates contrarian behavior
   - Win condition: sold for any profit after the dip entry

5. MOMENTUM_CHASE (FOMO)
   - Bought AFTER a > 50% price increase in the last 2 hours
   - The trader entered late into an already-pumping token
   - Win condition: sold for any profit (but typically low win rate)

6. DIAMOND_HANDS
   - Held through a drawdown of > 30% from peak unrealized gains
   - Eventually sold (at profit or loss)
   - Win condition: sold for profit after weathering the drawdown

7. UNKNOWN
   - Trades that genuinely don't match any pattern above
   - This should be a SMALL category — most trades should be classifiable
   - If > 20% of trades are UNKNOWN, re-examine your classification

IMPORTANT:
- One TRADE = one buy-sell pair (or buy-hold for open positions)
- One TOKEN POSITION may include multiple trades (DCA in, partial exits)
- Pattern occurrences count TOKEN POSITIONS, not individual transactions
- A single position should be assigned ONE primary pattern

═══════════════════════════════════════════════
TENDENCY SCORING — PRECISE CALCULATION
═══════════════════════════════════════════════

Each tendency is a float 0.0 to 1.0 representing the proportion of relevant trades.

fomo_tendency:
  numerator = count of MOMENTUM_CHASE entries
  denominator = total entry events
  score = numerator / denominator
  → 0.0 means the trader never chases pumps
  → > 0.3 is high FOMO
  → If no MOMENTUM_CHASE trades found, score = 0.0

panic_sell_tendency:
  numerator = count of PANIC_SELL exits
  denominator = total exit events (closed positions)
  score = numerator / denominator
  → 0.0 means the trader never panic sells
  → > 0.25 is concerning
  → > 0.5 is critical — the trader panic sells more often than not

dip_buy_tendency:
  numerator = count of DIP_BUY entries
  denominator = total entry events
  score = numerator / denominator
  → 0.0 means the trader never buys dips
  → > 0.3 indicates disciplined contrarian behavior

whale_follow_tendency:
  - This is harder to calculate from trade data alone
  - Estimate based on: clustering of entries right after large volume spikes on the same token
  - If trade data doesn't include whale activity markers, estimate conservatively (0.0 - 0.15)
  - Be honest if data is insufficient — default to 0.0 rather than guessing

═══════════════════════════════════════════════
TRADER TYPE CLASSIFICATION
═══════════════════════════════════════════════

Assign 1-3 labels from this list. Each has strict criteria:

SNIPER:
  - > 40% of entries are EARLY_ENTRY (P1/P2 phase)
  - Average entry is within first 30 minutes of token lifecycle
  - Indicates fast discovery pipeline and decisive execution

FLIPPER:
  - > 50% of positions held < 24 hours
  - High trade frequency relative to observation window
  - Takes quick profits/losses, rarely holds

HOLDER:
  - > 40% of positions held > 48 hours
  - Willing to sit through volatility
  - Lower trade frequency

DEGEN:
  - Win rate < 20% but continues trading aggressively
  - OR max drawdown > 200% with no improvement over time
  - OR frequently enters rug-pulled / scam tokens
  - This is a WARNING label — use it honestly when warranted

SCALPER:
  - > 60% of positions held < 2 hours
  - Very high frequency, very small per-trade PnL targets
  - Usually tight stop-losses

DAY_TRADER:
  - Positions typically opened and closed within same day
  - Mix of holding durations from minutes to hours
  - More deliberate than scalper, less patient than swing trader

SWING_TRADER:
  - Typical hold time 2-14 days
  - Enters on technical setups, exits on targets or stops
  - Lower frequency than day trader

POSITION_TRADER:
  - Holds > 2 weeks on average
  - Conviction-based entries
  - Very low frequency

RULES:
- Assign is_sniper, is_flipper, is_holder based on whether those specific labels are in trader_type
- A trader CAN be both SNIPER and FLIPPER (enters early, exits fast)
- A trader CANNOT be both HOLDER and FLIPPER (contradictory)
- If data is insufficient to classify, use an empty array []

═══════════════════════════════════════════════
RISK RATING — MULTI-FACTOR ASSESSMENT
═══════════════════════════════════════════════

Evaluate ALL of these factors and combine:

RED (High Risk) — ANY of these triggers RED:
  - Win rate < 30% with > 30 trades (statistically significant sample)
  - Max drawdown worse than -80%
  - Lifetime PnL deeply negative (lost > 50% of estimated capital deployed)
  - Panic sell tendency > 0.4 (selling in panic more than 40% of the time)
  - Worst pattern has 0% win rate with > 15 occurrences (persistent self-destructive behavior)
  - No improvement trend — recent trades are as bad or worse than early trades

YELLOW (Moderate Risk) — ANY of these triggers YELLOW (if not already RED):
  - Win rate 30-50% with > 20 trades
  - Max drawdown between -50% and -80%
  - Positive PnL but driven by 1-2 outsized wins (fragile profitability)
  - Panic sell tendency 0.15-0.40
  - Mixed patterns — some good, some destructive

GREEN (Low Risk) — ALL of these must be true:
  - Win rate > 50%
  - Max drawdown better than -50%
  - Positive lifetime PnL with consistent profits (not just lucky outliers)
  - No destructive patterns with > 10 occurrences
  - Evidence of risk management (stops, position sizing)

REASON FORMAT:
  Combine the triggering factors into a concise string.
  Examples:
  - "win rate below 30% (11.4%); severe drawdown (-432.9%); persistent panic selling (26 occurrences at 0% win rate)"
  - "healthy win rate (62.5%); controlled drawdown (-18.3%); consistent profits across observation window"
  - "marginal win rate (38.2%); profitability depends on 2 outsized wins; moderate panic selling tendency"

═══════════════════════════════════════════════
RECOMMENDATION GENERATION — QUALITY STANDARDS
═══════════════════════════════════════════════

Generate 10-15 recommendations. EVERY recommendation must:
  - Reference specific numbers from the data (not vague)
  - Be actionable (tell the trader WHAT to do, not just what's wrong)
  - Consider current market context
  - Be unique (no two recommendations saying the same thing differently)

STRUCTURE (follow this order):

1. CRITICAL ISSUES (1-3 items)
   - If worst_pattern has 0% win rate: "Critical: your worst pattern [NAME] has a 0% win rate across [N] occurrences — [description]. [specific action]."
   - If panic_sell_tendency > 0.25: address panic selling with specific mitigation
   - If win rate < 20%: address fundamental edge problem

2. MARKET-CONTEXT ADJUSTMENTS (2-3 items)
   - Reference Fear & Greed index value and market_regime
   - How does the current market affect THIS trader's specific tendencies?
   - If F&G is extreme (< 15 or > 85), always generate at least one recommendation about it

3. BEHAVIORAL INSIGHTS (2-3 items)
   - If trader_type has conflicting styles, discuss the tension
   - Positive reinforcement for any metric that's above average
   - If profitable despite low win rate, explain the few-big-wins dynamic

4. RISK MANAGEMENT (2-3 items)
   - If max_drawdown > -50%: recommend stop-losses and position sizing
   - Specific suggestions based on their trading style

5. SUMMARY / META (1-2 items)
   - Overall account health assessment
   - Methodology note about win rate calculation (per-trade vs per-token)

ANTI-PATTERNS (never do these):
  - "Consider diversifying your portfolio" — too generic
  - "Do your own research" — useless
  - "Be careful with leverage" — only relevant for perps
  - Recommendations that contradict each other
  - Recommendations that don't reference the trader's actual data
  - More than 2 recommendations about the same issue

═══════════════════════════════════════════════
PROFILE DESCRIPTION
═══════════════════════════════════════════════

Write a 2-4 sentence behavioral summary that:
  - Opens with the trader_type classification in natural language
  - Mentions their strongest and weakest behavioral pattern
  - References their profitability status and risk level
  - Reads like an analyst's assessment, not a generic horoscope

Good example:
  "This trader operates as a sniper-flipper hybrid, consistently entering meme tokens within their first 5 minutes of launch and exiting within hours. Despite a critically low 11.4% win rate, they've generated $2,625 in profit — driven by a handful of outsized early entries that more than compensate for frequent small losses. Their Achilles heel is panic selling: 26 positions were closed at significant losses during sharp drops, a pattern with a 0% win rate that actively erodes their edge. In the current extreme-fear market (F&G: 8), this panic selling tendency is especially dangerous."

Bad example:
  "This trader is active in meme coins and has a mixed track record. They tend to enter early and exit quickly. Their win rate could be improved."

═══════════════════════════════════════════════
CRITICAL REMINDERS
═══════════════════════════════════════════════

1. ANALYZE EVERY TRADE — don't skim. The trade_details array is your primary evidence.
2. NUMBERS MUST ADD UP — pattern occurrences should sum to approximately total_mints_traded (some overlap is acceptable due to multi-trade positions).
3. CONSISTENCY — if you classify someone as having high panic_sell_tendency, there MUST be PANIC_SELL patterns with meaningful occurrences.
4. HONESTY OVER FLATTERY — if a trader is losing money consistently, say so. Don't sugarcoat.
5. MARKET CONTEXT MATTERS — a 40% win rate in a bear market (F&G < 20) is different from 40% in a bull market (F&G > 70).
6. VALIDATE YOUR OWN OUTPUT — before returning, mentally check: do the tendencies match the patterns? Does the risk rating match the metrics? Do recommendations reference real numbers?"""

    def _meme_analysis_framework(self) -> str:
        return """
STEP 1: TRADE-BY-TRADE SCAN
  - For each token position in trade_details:
    a) What phase did they enter? (P1 = very early, P2 = early, P3+ = late)
    b) How long did they hold? (minutes -> classify as scalp/flip/hold)
    c) What was the exit PnL? (profit or loss, and magnitude)
    d) Was the exit forced by a sharp drop? (panic sell indicator)
    e) Was the entry after a pump? (FOMO indicator)
  - Tally: how many positions match each pattern?

STEP 2: PATTERN AGGREGATION
  - Group positions by assigned pattern
  - For each pattern: count occurrences, calculate win rate, calculate avg PnL %
  - Identify best pattern (highest win rate with meaningful occurrences)
  - Identify worst pattern (lowest win rate OR 0% win rate with most occurrences)
  - Flag any patterns with 0% win rate — these are critical issues

STEP 3: TENDENCY CALCULATION
  - Count MOMENTUM_CHASE entries -> fomo_tendency
  - Count PANIC_SELL exits -> panic_sell_tendency
  - Count DIP_BUY entries -> dip_buy_tendency
  - Estimate whale_follow_tendency (conservative if data is limited)
  - Cross-validate: tendencies should be consistent with pattern counts

STEP 4: TRADER TYPE CLASSIFICATION
  - Check entry phases -> if mostly P1/P2, likely SNIPER
  - Check hold durations -> if mostly < 24h, likely FLIPPER; if > 48h, likely HOLDER
  - Check frequency and PnL distribution -> DEGEN if losing consistently but trading aggressively
  - Assign 1-3 labels and set boolean flags

STEP 5: RISK ASSESSMENT
  - Apply RED/YELLOW/GREEN criteria against: win rate, drawdown, PnL, tendencies, pattern quality
  - Write a specific reason string citing the triggering factors with exact numbers

STEP 6: RECOMMENDATION GENERATION
  - Follow the 5-section structure (Critical, Market, Behavioral, Risk Mgmt, Summary)
  - Each recommendation references specific data points
  - 10-15 total, no duplicates

STEP 7: PROFILE DESCRIPTION
  - Synthesize steps 1-6 into a 2-4 sentence narrative
  - Lead with trader type, mention best/worst patterns, state profitability and risk level

STEP 8: SELF-VALIDATION
  - Do pattern occurrences roughly sum to total positions?
  - Do tendencies match pattern counts?
  - Does risk rating align with the metrics?
  - Are all recommendations referencing real numbers?
  - Is the profile description consistent with everything else?"""

    def _perps_analysis_framework(self) -> str:
        return """
STEP 1: TRADE-BY-TRADE SCAN
  - For each position in trade_details:
    a) What leverage was used? (1x-5x = low, 5x-20x = moderate, 20x+ = high)
    b) Was the position liquidated?
    c) How long was it held?
    d) What was the PnL relative to margin?
    e) Was there a stop-loss? (inferred from exit price vs liquidation price)
    f) Was the entry during high volatility? (potential FOMO)
    g) Was the exit during a sharp adverse move? (panic close)
  - Tally: how many positions match each pattern?

STEP 2: LEVERAGE ANALYSIS
  - Average leverage across all trades
  - Max leverage used
  - Correlation between leverage and outcomes (do high-leverage trades lose more?)
  - Leverage consistency (always same leverage = disciplined, wildly varying = erratic)

STEP 3: LIQUIDATION ANALYSIS
  - Total liquidations vs total positions = liquidation rate
  - Average loss on liquidated positions
  - Were liquidations clustered in time? (suggests market event or tilt behavior)

STEP 4: PATTERN AGGREGATION
  - Same as meme but with perps-specific patterns:
    - HIGH_LEVERAGE_GAMBLE: > 20x leverage on a position that got liquidated
    - REVENGE_TRADE: opening a new position within 30 minutes of a liquidation, often with higher leverage
    - DISCIPLINED_EXIT: closing at a small loss before liquidation (evidence of stop-loss)
    - TREND_FOLLOW: positions aligned with broader market direction
  - Include standard patterns too (QUICK_FLIP, PANIC_SELL, etc.)

STEP 5-8: Same as meme framework (tendency calc, type classification, risk assessment, recommendations, profile description, self-validation)

ADDITIONAL PERPS-SPECIFIC RISK FACTORS:
  - Liquidation rate > 10%: automatic RED flag
  - Average leverage > 20x: YELLOW or RED depending on win rate
  - Funding rate costs eating into profits: mention in recommendations
  - Revenge trading pattern: CRITICAL issue to flag"""

    def _meme_schema(self) -> str:
        return """{
  "trader_type": ["SNIPER", "FLIPPER"],
  "is_sniper": true,
  "is_holder": false,
  "is_flipper": true,
  "fomo_tendency": 0.05,
  "panic_sell_tendency": 0.30,
  "dip_buy_tendency": 0.02,
  "whale_follow_tendency": 0.0,
  "patterns": [
    {
      "pattern_type": "EARLY_ENTRY",
      "occurrences": 74,
      "win_rate": 32.43,
      "avg_pnl_percent": -15.41
    },
    {
      "pattern_type": "QUICK_FLIP",
      "occurrences": 59,
      "win_rate": 40.68,
      "avg_pnl_percent": 6.09
    },
    {
      "pattern_type": "PANIC_SELL",
      "occurrences": 26,
      "win_rate": 0.0,
      "avg_pnl_percent": -78.40
    },
    {
      "pattern_type": "UNKNOWN",
      "occurrences": 11,
      "win_rate": 0.0,
      "avg_pnl_percent": 0.0
    }
  ],
  "best_pattern": {
    "pattern_type": "QUICK_FLIP",
    "win_rate": 40.68,
    "occurrences": 59
  },
  "worst_pattern": {
    "pattern_type": "PANIC_SELL",
    "win_rate": 0.0,
    "occurrences": 26
  },
  "risk_rating": {
    "rating": "RED",
    "reason": "win rate below 30% (11.4%); severe drawdown (-432.9%); persistent panic selling (26 occurrences at 0% win rate)"
  },
  "recommendations": [
    "Critical: your worst pattern PANIC_SELL has a 0% win rate across 26 occurrences — you tend to sell at significant losses during sharp drops. Set hard stop-losses at -30% BEFORE entering any position to prevent emotional exits.",
    "Your panic sell tendency (29.5% of exits) is your single biggest edge destroyer. Pre-commit to exit rules: either a -25% stop-loss or a 2-hour time stop, decided before entry.",
    "You don't chase pumps (fomo_tendency: 0.0) — this discipline is rare and valuable. Maintain it especially in the current extreme-fear market (F&G: 8) where relief rallies can bait FOMO entries.",
    "Extreme fear (F&G: 8) amplifies your panic selling tendency — sharp drops feel existential in this environment. Reduce position sizes by 50% until F&G recovers above 25.",
    "Bear market (F&G: 8) makes your early-entry strategy (P1 phase) higher risk — fewer tokens sustain momentum. Be more selective: only enter P1 tokens showing > 3 SOL initial liquidity.",
    "Your sniper + flipper combination works when entries are selective, but you're entering 74 early positions and only winning 32.4% — tighten your filter. Aim for fewer, higher-conviction entries.",
    "Account is net profitable ($2,625.10) despite an 11.4% win rate — your edge is in the magnitude of wins, not frequency. Protect this by never letting panic sells erase the big winners.",
    "Positive PnL on a sub-12% win rate means your R-multiple distribution is heavily right-skewed: a few 10x+ winners carry everything. This is fragile — one missed winner significantly impacts total returns.",
    "Maximum drawdown of -432.9% is extreme and unsustainable. Implement a daily loss limit: if down > 15% of portfolio in a day, stop trading for 24 hours.",
    "Your QUICK_FLIP pattern is your best performer (40.7% win rate, +6.09% avg PnL) — consider leaning more into this style and reducing EARLY_ENTRY positions which average -15.4% PnL.",
    "Win rate of 11.4% is critically low but masked by positive total PnL. Track your win rate weekly — if it drops below 8% for 2 consecutive weeks, pause and review your entry criteria.",
    "Dominant pattern: EARLY_ENTRY (74 tokens, 32.4% win rate) — you have strong discovery skills but inconsistent execution. Focus on exit strategy rather than finding more tokens.",
    "Note: overall win rate (11.4%) is per-trade, while pattern win rates are per-token-position — a single token may involve multiple buy/sell transactions. This explains the apparent discrepancy.",
    "In the current bear regime, consider allocating 70% of your activity to QUICK_FLIP (your best pattern) and only 30% to EARLY_ENTRY until market conditions improve."
  ],
  "profile_description": "This trader operates as a sniper-flipper hybrid, consistently entering meme tokens in their earliest lifecycle phase (P1) and typically exiting within 2.5 hours. Despite a critically low 11.4% win rate, they have generated $2,625 in realized profit — a testament to outsized wins on a small fraction of trades. Their most damaging behavior is panic selling: 26 positions were closed at an average loss of -78.4% with a 0% win rate, systematically eroding the gains from successful entries. In the current extreme-fear bear market (F&G: 8), this panic tendency poses an acute risk to continued profitability."
}"""

    def _perps_schema(self) -> str:
        return """{
  "trader_type": ["DAY_TRADER"],
  "is_sniper": false,
  "is_holder": false,
  "is_flipper": false,
  "fomo_tendency": 0.10,
  "panic_sell_tendency": 0.20,
  "dip_buy_tendency": 0.05,
  "whale_follow_tendency": 0.0,
  "patterns": [
    {
      "pattern_type": "QUICK_FLIP",
      "occurrences": 30,
      "win_rate": 60.0,
      "avg_pnl_percent": 5.2
    },
    {
      "pattern_type": "DISCIPLINED_EXIT",
      "occurrences": 12,
      "win_rate": 0.0,
      "avg_pnl_percent": -8.5
    }
  ],
  "best_pattern": {
    "pattern_type": "QUICK_FLIP",
    "win_rate": 60.0,
    "occurrences": 30
  },
  "worst_pattern": {
    "pattern_type": "DISCIPLINED_EXIT",
    "win_rate": 0.0,
    "occurrences": 12
  },
  "risk_rating": {
    "rating": "GREEN",
    "reason": "healthy win rate (62.5%); controlled leverage (avg 5.2x); positive PnL with consistent profit distribution"
  },
  "recommendations": [
    "Your disciplined exit pattern shows effective risk management — you're cutting losses at -8.5% avg rather than riding to liquidation.",
    "Consider maintaining a detailed trade journal to identify what differentiates your 60% winning quick flips from the 40% losers."
  ],
  "profile_description": "A disciplined day trader in perpetual futures with a healthy 62.5% win rate and controlled leverage. Primarily operates through quick-flip scalps with consistent small gains, while cutting losses early through disciplined exits — a hallmark of sustainable perps trading."
}"""

    def _build_user_prompt(self, raw_data: Dict[str, Any], token_type: str) -> str:
        wallet = raw_data.get("wallet_address", "unknown")
        obs = raw_data.get("observation_window", {})
        from_d = obs.get("from_date", "?")
        to_d = obs.get("to_date", "?")

        sections = [
            f"## Wallet: {wallet}",
            f"## Token Type: {token_type}",
            f"## Observation Window: {from_d} to {to_d}",
        ]

        mkt = raw_data.get("market_context")
        if mkt:
            sections.append(f"## Market Context:\n{json.dumps(mkt, indent=2, default=str)}")

        metrics = raw_data.get("raw_metrics")
        if metrics:
            sections.append(f"## Aggregate Metrics:\n{json.dumps(metrics, indent=2, default=str)}")

        trades = raw_data.get("trade_details", [])
        sections.append(
            f"## Individual Trade Data ({len(trades)} trades):\n"
            f"{json.dumps(trades, indent=1, default=str)}"
        )

        positions = raw_data.get("open_positions")
        if positions:
            sections.append(
                f"## Open Positions ({len(positions)}):\n"
                f"{json.dumps(positions, indent=2, default=str)}"
            )

        sections.append("Return ONLY the JSON object. No other text.")
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Parsing & validation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        return json.loads(cleaned)

    @staticmethod
    def _validate(parsed: Dict[str, Any], token_type: str) -> Dict[str, Any]:
        clamp01 = lambda v: max(0.0, min(1.0, float(v) if v is not None else 0.0))
        parsed["fomo_tendency"] = clamp01(parsed.get("fomo_tendency"))
        parsed["panic_sell_tendency"] = clamp01(parsed.get("panic_sell_tendency"))
        parsed["dip_buy_tendency"] = clamp01(parsed.get("dip_buy_tendency"))
        parsed["whale_follow_tendency"] = clamp01(parsed.get("whale_follow_tendency"))

        if not isinstance(parsed.get("patterns"), list):
            parsed["patterns"] = []
        for p in parsed["patterns"]:
            if "win_rate" in p:
                p["win_rate"] = round(max(0.0, min(100.0, float(p["win_rate"]))), 2)
            if "occurrences" in p:
                p["occurrences"] = max(0, int(p["occurrences"]))

        if not isinstance(parsed.get("recommendations"), list):
            parsed["recommendations"] = []
        parsed["recommendations"] = parsed["recommendations"][:_MAX_RECOMMENDATIONS]

        rating = parsed.get("risk_rating")
        if isinstance(rating, dict):
            valid = {"RED", "YELLOW", "GREEN"}
            if rating.get("rating") not in valid:
                rating["rating"] = "YELLOW"
        else:
            parsed["risk_rating"] = None

        if not isinstance(parsed.get("trader_type"), list):
            tt = parsed.get("trader_type")
            parsed["trader_type"] = [tt] if isinstance(tt, str) and tt else ["UNKNOWN"]

        for bool_key in ("is_sniper", "is_holder", "is_flipper"):
            parsed[bool_key] = bool(parsed.get(bool_key, False))

        if not isinstance(parsed.get("profile_description"), str):
            parsed["profile_description"] = None

        if parsed.get("worst_pattern") is not None and not isinstance(parsed["worst_pattern"], dict):
            parsed["worst_pattern"] = None
        if parsed.get("best_pattern") is not None and not isinstance(parsed["best_pattern"], dict):
            parsed["best_pattern"] = None

        return parsed

    # ------------------------------------------------------------------
    # Insufficient data fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _insufficient_data_response(token_type: str) -> Dict[str, Any]:
        return {
            "trader_type": ["UNKNOWN"],
            "is_sniper": False,
            "is_holder": False,
            "is_flipper": False,
            "fomo_tendency": 0.0,
            "panic_sell_tendency": 0.0,
            "dip_buy_tendency": 0.0,
            "whale_follow_tendency": 0.0,
            "patterns": [],
            "best_pattern": None,
            "worst_pattern": None,
            "risk_rating": None,
            "recommendations": [
                f"Insufficient {token_type} trading history to generate AI-powered recommendations."
            ],
            "profile_description": None,
        }

    # ------------------------------------------------------------------
    # Hybrid analysis (internal engine output + LLM refinement)
    # ------------------------------------------------------------------

    def analyze_hybrid(
        self,
        raw_data: Dict[str, Any],
        engine_output: Dict[str, Any],
        token_type: str,
    ) -> Dict[str, Any]:
        """
        Refine the internal engine's behavioral analysis using the LLM.

        Receives both the raw trading data AND the engine's computed output
        (patterns, tendencies, risk rating, recommendations). The LLM acts
        as a reviewer that validates, corrects, and enriches the rule-based
        analysis while staying anchored to real calculations.

        Returns the same dict shape as ``analyze()`` — consumed by
        ``UserProfileResponse``.
        """
        is_valid = raw_data.get("is_valid", False)
        if not is_valid:
            return self._insufficient_data_response(token_type)

        system_prompt = self._build_hybrid_system_prompt(token_type)
        user_prompt = self._build_hybrid_user_prompt(raw_data, engine_output, token_type)

        self.circuit_breaker.check()

        providers = self._ordered_providers()
        last_error: Optional[Exception] = None

        for provider_name, call_fn in providers:
            for attempt in range(1, self.config.MAX_RETRIES + 1):
                try:
                    logger.info(
                        f"LLM Hybrid Analyzer: calling {provider_name} "
                        f"(attempt {attempt}/{self.config.MAX_RETRIES})"
                    )
                    start = time.perf_counter()
                    raw_text = call_fn(system_prompt, user_prompt)
                    elapsed = time.perf_counter() - start
                    logger.info(f"LLM Hybrid Analyzer: {provider_name} responded in {elapsed:.2f}s")

                    parsed = self._parse_json(raw_text)
                    validated = self._validate(parsed, token_type)
                    self.circuit_breaker.record_success()
                    return validated

                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as parse_err:
                    logger.warning(
                        f"LLM Hybrid Analyzer: {provider_name} returned unparseable response "
                        f"(attempt {attempt}): {parse_err}"
                    )
                    last_error = parse_err
                except CircuitBreakerOpen:
                    raise
                except Exception as e:
                    logger.warning(f"LLM Hybrid Analyzer: {provider_name} error (attempt {attempt}): {e}")
                    last_error = e

        self.circuit_breaker.record_failure(last_error)
        raise RuntimeError(f"All LLM providers failed for hybrid analysis: {last_error}")

    # ------------------------------------------------------------------
    # Hybrid prompt construction
    # ------------------------------------------------------------------

    def _build_hybrid_system_prompt(self, token_type: str) -> str:
        schema = self._meme_schema() if token_type == "meme" else self._perps_schema()
        analysis_framework = self._meme_analysis_framework() if token_type == "meme" else self._perps_analysis_framework()

        return f"""You are an elite crypto trading behavioral analyst performing a HYBRID ANALYSIS. You have a unique advantage: you receive BOTH raw trading data AND an automated system's pre-computed behavioral analysis. Your job is to REVIEW, VALIDATE, CORRECT, and ENHANCE the system's output.

You will receive:
1. Aggregate metrics for a wallet (total trades, PnL, win rate, volume, drawdown, hold times, entry phases)
2. Individual trade-level data (every buy/sell with timestamps, prices, PnL, token phases, hold durations)
3. Current market context (Fear & Greed index, BTC dominance, market regime, altcoin index)
4. **SYSTEM-COMPUTED ANALYSIS** — the automated engine's pattern detection, tendency scores, risk rating, trader type classification, and rule-based recommendations

Your job: Use the raw trade data as ground truth to VALIDATE the system's analysis, CORRECT any misclassifications or shallow assessments, and PRODUCE a refined behavioral profile that combines the system's computational accuracy with your deeper analytical capabilities.

═══════════════════════════════════════════════
HYBRID ANALYSIS PROTOCOL
═══════════════════════════════════════════════

You MUST follow this protocol when reviewing the system's output:

1. PATTERN VALIDATION
   - Cross-check each pattern's occurrences against the raw trade data
   - If the system missed a pattern (e.g., didn't detect PANIC_SELL when trades show >=40% loss exits in <2 hours), ADD it
   - If the system's pattern counts seem wrong, CORRECT them based on your trade-by-trade analysis
   - If the system classified trades incorrectly, RECLASSIFY them
   - You MAY add patterns the system didn't detect (e.g., DIAMOND_HANDS, DIP_BUY, MOMENTUM_CHASE)

2. TENDENCY REFINEMENT
   - The system's tendency scores are calculated mechanically (count of pattern trades / total trades)
   - VALIDATE these against the raw trade data
   - You may ADJUST scores if the system's mechanical calculation missed nuance (e.g., clustering of panic sells during specific market events vs spread-out occurrences)
   - Keep scores within 0.0-1.0

3. TRADER TYPE REVIEW
   - The system assigns trader types based on simple thresholds (hold time, entry phase)
   - REVIEW whether the classification captures the full picture
   - ADD missing types if warranted (e.g., system said SNIPER only, but data shows SNIPER+FLIPPER)
   - REMOVE incorrect types if the data doesn't support them

4. RISK RATING ENHANCEMENT
   - The system uses basic win-rate + drawdown thresholds
   - You MUST apply the full multi-factor assessment (tendency severity, pattern destructiveness, PnL trajectory, market context)
   - UPGRADE or DOWNGRADE the rating if the system's simple rules missed important factors
   - ALWAYS rewrite the reason string with richer, multi-factor detail

5. RECOMMENDATION OVERHAUL
   - The system generates rule-based recommendations that are often generic or repetitive
   - You MUST generate 10-15 NEW recommendations that are:
     * More specific (reference exact numbers, pattern names, trade counts)
     * Market-context aware (reference Fear & Greed, market regime)
     * Actionable (tell the trader WHAT to do, not just what's wrong)
     * Structured (Critical issues -> Market context -> Behavioral -> Risk management -> Summary)
   - You may KEEP system recommendations that are already high-quality, but REWRITE weak ones
   - The system's recommendations serve as a STARTING POINT — improve, don't just copy

6. PROFILE DESCRIPTION (NEW)
   - The system typically does NOT generate a profile description for meme tokens
   - You MUST write a 2-4 sentence behavioral summary (see quality standards below)

═══════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════

Return ONLY valid JSON matching this exact structure. No markdown fences. No explanation. No preamble. No trailing text.

{schema}

═══════════════════════════════════════════════
ANALYSIS METHODOLOGY — FOLLOW THIS STEP BY STEP
═══════════════════════════════════════════════

Before generating the JSON output, perform this internal analysis (do NOT output it):

{analysis_framework}

ADDITIONAL HYBRID STEP (after Step 8):

STEP 9: SYSTEM OUTPUT COMPARISON
  - Compare your analysis from Steps 1-8 with the system's pre-computed output
  - Where they agree: keep your version (it's more detailed)
  - Where they disagree: use the raw trade data as the tiebreaker
  - Note any patterns the system missed that you found
  - Note any system classifications you corrected

═══════════════════════════════════════════════
PATTERN DETECTION RULES
═══════════════════════════════════════════════

Classify EVERY trade into exactly one pattern. A trade that matches multiple patterns gets assigned to the MOST SPECIFIC one (PANIC_SELL > QUICK_FLIP > EARLY_ENTRY > UNKNOWN).

PATTERN DEFINITIONS WITH PRECISE THRESHOLDS:

1. EARLY_ENTRY — Token in P1/P2 phase at entry, bought before first major peak. Win = sold for profit.
2. QUICK_FLIP — Hold < 24 hours. Win = sold for profit.
3. PANIC_SELL — Sold at >= 40% loss, hold < 120 min OR sold during > 25% drop in < 30 min. Win = NEVER.
4. DIP_BUY — Bought > 20% below recent 24h high. Win = sold for profit after dip entry.
5. MOMENTUM_CHASE (FOMO) — Bought after > 50% price increase in last 2 hours. Win = sold for profit.
6. DIAMOND_HANDS — Held through > 30% drawdown from peak unrealized. Win = sold for profit after weathering drawdown.
7. UNKNOWN — Genuinely unclassifiable. Should be < 20% of trades.

Pattern occurrences count TOKEN POSITIONS, not individual transactions.

═══════════════════════════════════════════════
TENDENCY SCORING, TRADER TYPE, RISK RATING, RECOMMENDATIONS, PROFILE DESCRIPTION
═══════════════════════════════════════════════

Apply the same rules as a full analysis (tendency calculation formulas, trader type criteria, multi-factor risk assessment, 5-section recommendation structure, profile description quality standards).

Key difference from pure AI analysis: you have the system's output as a REFERENCE. Use it to catch things you might miss, but ALWAYS validate against the raw data. The system's mechanical calculations are reliable for counts and ratios; your advantage is in interpretation, context, and nuance.

═══════════════════════════════════════════════
CRITICAL REMINDERS
═══════════════════════════════════════════════

1. The system's pattern counts are a STARTING POINT — validate and correct them against trade data.
2. The system's recommendations are often too generic — make yours specific and actionable.
3. ALWAYS generate a profile_description (the system usually returns null for meme).
4. NUMBERS MUST ADD UP — pattern occurrences should sum to approximately total_mints_traded.
5. CONSISTENCY — tendencies must match patterns, risk rating must match metrics.
6. HONESTY OVER FLATTERY — if a trader is losing money, say so clearly.
7. MARKET CONTEXT MATTERS — factor in Fear & Greed and market regime throughout.
8. VALIDATE YOUR OUTPUT — before returning, mentally verify internal consistency."""

    def _build_hybrid_user_prompt(
        self,
        raw_data: Dict[str, Any],
        engine_output: Dict[str, Any],
        token_type: str,
    ) -> str:
        wallet = raw_data.get("wallet_address", "unknown")
        obs = raw_data.get("observation_window", {})
        from_d = obs.get("from_date", "?")
        to_d = obs.get("to_date", "?")

        sections = [
            f"## Wallet: {wallet}",
            f"## Token Type: {token_type}",
            f"## Observation Window: {from_d} to {to_d}",
        ]

        mkt = raw_data.get("market_context")
        if mkt:
            sections.append(f"## Market Context:\n{json.dumps(mkt, indent=2, default=str)}")

        metrics = raw_data.get("raw_metrics")
        if metrics:
            sections.append(f"## Aggregate Metrics:\n{json.dumps(metrics, indent=2, default=str)}")

        trades = raw_data.get("trade_details", [])
        sections.append(
            f"## Individual Trade Data ({len(trades)} trades):\n"
            f"{json.dumps(trades, indent=1, default=str)}"
        )

        positions = raw_data.get("open_positions")
        if positions:
            sections.append(
                f"## Open Positions ({len(positions)}):\n"
                f"{json.dumps(positions, indent=2, default=str)}"
            )

        sections.append(
            f"## System-Computed Analysis (automated engine output — review and refine):\n"
            f"{json.dumps(engine_output, indent=2, default=str)}"
        )

        sections.append("Return ONLY the JSON object. No other text.")
        return "\n\n".join(sections)
