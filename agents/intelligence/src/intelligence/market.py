"""Market data and technical analysis for Sovereign Edge Intelligence Squad.

Indicators ported from the Manna Trading System (manna-trading/):
  - RSI:             manna/services/trading/mathematicalTradingSystem.ts:calculateRSI
  - Bollinger Bands: manna/services/trading/mathematicalTradingSystem.ts:calculateBollingerPosition
  - ATR:             manna/lib/atr.ts
  - Market Regime:   manna/services/trading/mathematicalTradingSystem.ts:detectMarketRegime
  - Volume Ratio:    manna/services/trading/marketScannerService.ts
  - Signal:          manna/services/trading/mathematicalTradingSystem.ts (mean reversion + momentum)

Data source: yFinance (unlimited, no API key required).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import structlog
from core.config import get_settings

logger = structlog.get_logger(__name__)

# ── Signal / regime labels ────────────────────────────────────────────────────

_SIGNAL_LABELS: dict[str, str] = {
    "MEAN_REVERSION_BUY": "OVERSOLD BOUNCE",
    "MEAN_REVERSION_SELL": "OVERBOUGHT PULLBACK",
    "MOMENTUM_BUY": "MOMENTUM LONG",
    "MOMENTUM_SELL": "MOMENTUM SHORT",
    "NEUTRAL": "",
}

_REGIME_LABELS: dict[str, str] = {
    "TRENDING_UP": "TREND↑",
    "TRENDING_DOWN": "TREND↓",
    "RANGING": "RANGING",
    "VOLATILE": "VOLATILE",
    "UNKNOWN": "",
}


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Quote:
    # Core price data
    symbol: str
    price: float
    change_pct: float  # daily % change vs previous close
    volume: int
    timestamp: str

    # Technical indicators (populated from 2-month daily history)
    rsi: float = 50.0  # 14-period RSI; <30 oversold, >70 overbought
    bb_position: float = 0.0  # Bollinger position [-1, 1]; ±1 = ±2σ from 20-day mean
    atr_pct: float = 0.0  # 14-period ATR as % of price; measures volatility
    volume_ratio: float = 1.0  # today's vol / 20-day avg; >1.5 = unusual activity
    regime: str = field(default="UNKNOWN")  # TRENDING_UP/DOWN, RANGING, VOLATILE
    signal: str = field(default="NEUTRAL")  # mean reversion or momentum signal
    signal_confidence: float = 0.0  # 0–1; only set when signal != NEUTRAL
    earnings_context: str = ""  # populated by enrich_quotes_with_earnings()


# ── Pure indicator functions (ported from manna) ──────────────────────────────


def _compute_rsi(closes: list[float], period: int = 14) -> float:
    """14-period RSI.  Returns 50.0 when insufficient data.

    Source: mathematicalTradingSystem.ts:calculateRSI
    """
    if len(closes) < period + 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(len(closes) - period, len(closes)):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change  # keep positive
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)


def _compute_bb_position(closes: list[float], period: int = 20) -> float:
    """Bollinger Band position normalised to [-1, 1].

    0 = at the 20-day mean.  ±1 = ±2 standard deviations.
    Source: mathematicalTradingSystem.ts:calculateBollingerPosition
    """
    if len(closes) < period:
        return 0.0
    recent = closes[-period:]
    mean = sum(recent) / period
    variance = sum((p - mean) ** 2 for p in recent) / period
    std = variance**0.5
    if std == 0:
        return 0.0
    z_score = (closes[-1] - mean) / std
    # Normalise: divide by 2 so ±2σ maps to ±1
    return round(max(-1.0, min(1.0, z_score / 2)), 3)


def _compute_atr_pct(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> float:
    """14-period Average True Range expressed as % of current price.

    True Range = max(H-L, |H-prev_C|, |L-prev_C|).
    Source: manna/lib/atr.ts
    """
    if len(closes) < period + 1:
        return 0.0
    true_ranges: list[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)
    atr = sum(true_ranges[-period:]) / period
    last = closes[-1]
    return round((atr / last) * 100, 2) if last else 0.0


def _compute_volume_ratio(volumes: list[int], period: int = 20) -> float:
    """Current volume divided by trailing average.

    volumes[-1] is treated as today's volume; the prior `period` values are
    the baseline.  Source: manna/services/trading/marketScannerService.ts
    """
    if len(volumes) < 2:
        return 1.0
    current = volumes[-1]
    baseline = volumes[max(0, len(volumes) - period - 1) : -1]
    if not baseline:
        return 1.0
    avg = sum(baseline) / len(baseline)
    return round(current / avg, 2) if avg else 1.0


def _detect_regime(
    closes: list[float],
    highs: list[float],
    lows: list[float],
) -> str:
    """Classify current market regime for a symbol.

    Returns one of: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, UNKNOWN.
    Logic: 5-day vs 20-day MA crossover + ATR% threshold.
    Source: mathematicalTradingSystem.ts:detectMarketRegime
    """
    if len(closes) < 20:
        return "UNKNOWN"
    short_ma = sum(closes[-5:]) / 5
    long_ma = sum(closes[-20:]) / 20
    atr_pct = _compute_atr_pct(highs, lows, closes)
    if atr_pct > 5.0:
        return "VOLATILE"
    trend_strength = ((short_ma - long_ma) / long_ma * 100) if long_ma else 0.0
    if trend_strength > 1.0:
        return "TRENDING_UP"
    if trend_strength < -1.0:
        return "TRENDING_DOWN"
    return "RANGING"


def _compute_signal(
    rsi: float,
    bb_position: float,
    volume_ratio: float,
    regime: str,
) -> tuple[str, float]:
    """Classify the technical signal and return (signal_name, confidence).

    Two strategies, in priority order:
      1. Mean Reversion: extreme Bollinger + RSI confirmation
      2. Momentum: trend regime + healthy RSI range + volume spike

    Confidence formula mirrors manna's opportunityRanker thresholds.
    Source: mathematicalTradingSystem.ts (lines 249-296)
    """
    # ── Mean reversion (higher confidence when more extreme) ──────────────────
    if bb_position <= -0.8 and rsi < 30:
        confidence = min(0.60 + abs(bb_position) * 0.20, 0.95)
        return "MEAN_REVERSION_BUY", round(confidence, 2)
    if bb_position >= 0.8 and rsi > 70:
        confidence = min(0.60 + abs(bb_position) * 0.20, 0.95)
        return "MEAN_REVERSION_SELL", round(confidence, 2)

    # ── Momentum (requires trend regime + volume confirmation) ────────────────
    if regime == "TRENDING_UP" and 50 < rsi < 70 and volume_ratio > 1.2:
        confidence = 0.55 + (0.15 if volume_ratio > 1.5 else 0.05)
        return "MOMENTUM_BUY", round(confidence, 2)
    if regime == "TRENDING_DOWN" and 30 < rsi < 50 and volume_ratio > 1.2:
        confidence = 0.55 + (0.15 if volume_ratio > 1.5 else 0.05)
        return "MOMENTUM_SELL", round(confidence, 2)

    return "NEUTRAL", 0.0


# ── Data fetching ─────────────────────────────────────────────────────────────


def _fetch_quotes_sync(symbols: list[str]) -> list[Quote]:
    """Fetch price + compute all indicators for `symbols`.

    Runs synchronously; call via asyncio.to_thread to avoid blocking the loop.
    Uses 2 months of daily history for indicator computation.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        logger.warning("intelligence.market.yfinance_not_installed")
        return []

    results: list[Quote] = []
    for symbol in symbols:
        try:
            t = yf.Ticker(symbol)

            # 2-month daily history — enough for RSI-14, BB-20, ATR-14, regime-20
            hist = t.history(period="2mo")
            if hist.empty or len(hist) < 2:
                logger.warning("intelligence.market.insufficient_history", symbol=symbol)
                continue

            closes: list[float] = hist["Close"].tolist()
            highs: list[float] = hist["High"].tolist()
            lows: list[float] = hist["Low"].tolist()
            volumes: list[int] = [int(v) for v in hist["Volume"].tolist()]

            # Real-time price (fast_info) is more current than last daily close
            fi = t.fast_info
            price = float(fi.last_price or closes[-1])
            prev = float(fi.previous_close or (closes[-2] if len(closes) >= 2 else price))
            change_pct = round(((price - prev) / prev * 100.0) if prev else 0.0, 2)
            volume = int(getattr(fi, "last_volume", None) or volumes[-1])

            try:
                timestamp = str(hist.index[-1].date())
            except Exception:
                timestamp = ""

            # Append today's real-time volume to the history list for ratio
            vol_series = volumes + [volume]

            rsi = _compute_rsi(closes)
            bb_pos = _compute_bb_position(closes)
            atr_pct = _compute_atr_pct(highs, lows, closes)
            vol_ratio = _compute_volume_ratio(vol_series)
            regime = _detect_regime(closes, highs, lows)
            signal, signal_conf = _compute_signal(rsi, bb_pos, vol_ratio, regime)

            results.append(
                Quote(
                    symbol=symbol,
                    price=price,
                    change_pct=change_pct,
                    volume=volume,
                    timestamp=timestamp,
                    rsi=rsi,
                    bb_position=bb_pos,
                    atr_pct=atr_pct,
                    volume_ratio=vol_ratio,
                    regime=regime,
                    signal=signal,
                    signal_confidence=signal_conf,
                )
            )
            logger.debug(
                "intelligence.market.quote_fetched",
                symbol=symbol,
                price=price,
                change_pct=change_pct,
                rsi=rsi,
                signal=signal,
            )
        except Exception:
            logger.error("intelligence.market.quote_failed", symbol=symbol, exc_info=True)

    logger.info(
        "intelligence.market.batch_done",
        fetched=len(results),
        failed=len(symbols) - len(results),
    )
    return results


async def get_quotes(symbols: list[str]) -> list[Quote]:
    """Fetch real-time quotes with full technical analysis.

    yFinance is synchronous — runs in a thread pool to avoid blocking the loop.
    """
    if not symbols:
        return []
    return await asyncio.to_thread(_fetch_quotes_sync, symbols)


async def get_watchlist_alerts() -> list[Quote]:
    """Return quotes with significant price moves OR actionable technical signals.

    A symbol is included if:
      - |change_pct| >= settings.market_alert_threshold (default 2%), OR
      - signal != NEUTRAL (e.g. RSI 28 + lower Bollinger Band)

    The second condition catches high-quality setups regardless of daily % move.
    """
    settings = get_settings()
    quotes = await get_quotes(settings.watchlist)
    threshold = settings.market_alert_threshold * 100
    return [q for q in quotes if abs(q.change_pct) >= threshold or q.signal != "NEUTRAL"]


# ── Formatting ────────────────────────────────────────────────────────────────


def format_market_summary(quotes: list[Quote]) -> str:
    """Return a Telegram-ready Markdown summary with technical context."""
    if not quotes:
        return "No market data available."

    lines = ["**Market Summary**\n"]
    for q in quotes:
        arrow = "▲" if q.change_pct >= 0 else "▼"
        line = f"{q.symbol}: ${q.price:.2f} {arrow}{abs(q.change_pct):.2f}%"

        # Technical context — only show non-default / interesting values
        tech: list[str] = []

        rsi_tag = " ↓" if q.rsi < 35 else (" ↑" if q.rsi > 65 else "")
        tech.append(f"RSI {q.rsi:.0f}{rsi_tag}")

        if abs(q.bb_position) >= 0.3:
            tech.append(f"BB {q.bb_position:+.2f}")

        if q.volume_ratio >= 1.3:
            tech.append(f"Vol {q.volume_ratio:.1f}x")

        if q.atr_pct > 0:
            tech.append(f"ATR {q.atr_pct:.1f}%")

        regime_label = _REGIME_LABELS.get(q.regime, "")
        if regime_label:
            tech.append(regime_label)

        line += " | " + " · ".join(tech)

        if q.signal != "NEUTRAL":
            label = _SIGNAL_LABELS.get(q.signal, q.signal)
            line += f"\n  ⚡ *{label}* ({q.signal_confidence:.0%} conf)"

        if q.earnings_context:
            line += f"\n  Earnings: {q.earnings_context}"

        lines.append(line)

    return "\n".join(lines)
