"""Market data intelligence — quotes, technical indicators, regime detection.

Uses yFinance for free, real-time market data.  All indicators are computed
locally — no paid API keys required.

Layer 1 (free, always):  price, change%, volume, RSI, Bollinger Bands, ATR,
                          volume ratio, regime classification, trade signal.
Layer 2 (optional):      earnings_context populated by earnings.py.

Usage:
    quotes = await get_quotes(["NVDA", "MSFT", "AAPL"])
    alerts = await get_watchlist_alerts()
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

from core.config import get_settings
from observability.logging import get_logger

logger = get_logger(__name__, component="intelligence")

# ── Constants ─────────────────────────────────────────────────────────────────

_RSI_PERIOD = 14
_BB_PERIOD = 20
_ATR_PERIOD = 14
_SHORT_MA_PERIOD = 5
_LONG_MA_PERIOD = 20
_MIN_REGIME_BARS = max(_LONG_MA_PERIOD, _ATR_PERIOD)  # need at least 20 bars

_VOLATILE_ATR_THRESHOLD = 5.0  # ATR% above this → VOLATILE regime
_TREND_THRESHOLD = 0.01  # ±1% short/long MA divergence → trending


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Quote:
    """Snapshot of a single equity with pre-computed technical indicators."""

    symbol: str
    price: float
    change_pct: float
    volume: int
    timestamp: str
    rsi: float = 50.0
    bb_position: float = 0.0  # Bollinger Band position [-1, 1]
    signal: str = "NEUTRAL"  # NEUTRAL | MEAN_REVERSION_BUY | MEAN_REVERSION_SELL
    # | MOMENTUM_BUY | MOMENTUM_SELL
    signal_confidence: float = 0.0  # 0.0 to 0.95
    regime: str = "UNKNOWN"  # UNKNOWN | TRENDING_UP | TRENDING_DOWN
    # | RANGING | VOLATILE
    volume_ratio: float = 1.0  # current vs 20-day average
    earnings_context: str = ""  # populated by earnings.enrich_quotes_with_earnings()
    _indicator_data: dict = field(default_factory=dict, repr=False, compare=False)


# ── Technical indicators (pure functions) ─────────────────────────────────────


def _compute_rsi(closes: list[float]) -> float:
    """Wilder RSI-14.  Returns 50.0 when fewer than 15 data points."""
    if len(closes) < _RSI_PERIOD + 1:
        return 50.0

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas[-_RSI_PERIOD:]]
    losses = [abs(min(d, 0.0)) for d in deltas[-_RSI_PERIOD:]]

    avg_gain = sum(gains) / _RSI_PERIOD
    avg_loss = sum(losses) / _RSI_PERIOD

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - 100.0 / (1.0 + rs), 2)


def _compute_bb_position(closes: list[float]) -> float:
    """Bollinger Band position of the last close relative to the ±2 std-dev band.

    Returns a value in [-1.0, 1.0]:
      +1.0 = at or above upper band (+2 std dev)
      -1.0 = at or below lower band (-2 std dev)
       0.0 = at the 20-period mean (or zero std dev)
    """
    if len(closes) < _BB_PERIOD:
        return 0.0

    window = closes[-_BB_PERIOD:]
    mean = sum(window) / _BB_PERIOD
    variance = sum((x - mean) ** 2 for x in window) / _BB_PERIOD
    std = math.sqrt(variance)

    if std == 0:
        return 0.0

    last = closes[-1]
    pos = (last - mean) / (2.0 * std)
    return max(-1.0, min(1.0, round(pos, 4)))


def _compute_atr_pct(highs: list[float], lows: list[float], closes: list[float]) -> float:
    """Average True Range as a percentage of the last close price.

    Returns 0.0 when fewer than 2 data points per series.
    """
    n = min(len(highs), len(lows), len(closes))
    if n < 2:
        return 0.0

    trs: list[float] = []
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        trs.append(max(hl, hc, lc))

    period = min(_ATR_PERIOD, len(trs))
    atr = sum(trs[-period:]) / period
    last_close = closes[-1]
    if last_close == 0:
        return 0.0
    return round(atr / last_close * 100.0, 2)


def _compute_volume_ratio(volumes: list[int]) -> float:
    """Ratio of today's volume to the 20-day average.

    Returns 1.0 when fewer than 2 data points (no baseline to compare).
    """
    if len(volumes) < 2:
        return 1.0
    baseline = volumes[:-1]
    avg = sum(baseline) / len(baseline)
    if avg == 0:
        return 1.0
    return round(volumes[-1] / avg, 2)


def _detect_regime(closes: list[float], highs: list[float], lows: list[float]) -> str:
    """Classify the current market regime.

    Returns one of: UNKNOWN | VOLATILE | TRENDING_UP | TRENDING_DOWN | RANGING
    """
    n = min(len(closes), len(highs), len(lows))
    if n < _MIN_REGIME_BARS:
        return "UNKNOWN"

    # Volatility check first — supersedes trend signals
    atr_pct = _compute_atr_pct(highs[-n:], lows[-n:], closes[-n:])
    if atr_pct > _VOLATILE_ATR_THRESHOLD:
        return "VOLATILE"

    # Trend via short/long moving average divergence
    short_ma = sum(closes[-_SHORT_MA_PERIOD:]) / _SHORT_MA_PERIOD
    long_ma = sum(closes[-_LONG_MA_PERIOD:]) / _LONG_MA_PERIOD

    if long_ma == 0:
        return "UNKNOWN"

    ratio = short_ma / long_ma
    if ratio > 1.0 + _TREND_THRESHOLD:
        return "TRENDING_UP"
    if ratio < 1.0 - _TREND_THRESHOLD:
        return "TRENDING_DOWN"
    return "RANGING"


def _compute_signal(
    *,
    rsi: float,
    bb_position: float,
    volume_ratio: float,
    regime: str,
) -> tuple[str, float]:
    """Derive a trade signal from technical indicators.

    Returns (signal_name, confidence) where confidence is in [0.0, 0.95].
    """
    # Mean-reversion: oversold/overbought + Bollinger Band confirmation
    if rsi < 30 and bb_position < -0.8:
        confidence = min(0.6 + 0.35 * abs(bb_position), 0.95)
        return "MEAN_REVERSION_BUY", round(confidence, 2)

    if rsi > 70 and bb_position > 0.8:
        confidence = min(0.6 + 0.35 * abs(bb_position), 0.95)
        return "MEAN_REVERSION_SELL", round(confidence, 2)

    # Momentum: trend confirmed by volume surge
    if regime == "TRENDING_UP" and volume_ratio > 1.5:
        confidence = min(0.5 + 0.05 * volume_ratio, 0.85)
        return "MOMENTUM_BUY", round(confidence, 2)

    if regime == "TRENDING_DOWN" and volume_ratio > 1.5:
        confidence = min(0.5 + 0.05 * volume_ratio, 0.85)
        return "MOMENTUM_SELL", round(confidence, 2)

    return "NEUTRAL", 0.0


# ── yFinance data fetch ───────────────────────────────────────────────────────


def _fetch_quotes_sync(symbols: list[str]) -> list[Quote]:
    """Fetch OHLCV history and compute indicators for each symbol.

    Runs synchronously — call via asyncio.to_thread.
    Returns an empty list when yfinance is not installed.
    """
    if not symbols:
        return []

    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        logger.warning("intelligence.market.yfinance_not_installed")
        return []

    quotes: list[Quote] = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            if hist.empty:
                logger.debug("intelligence.market.no_history", symbol=symbol)
                continue

            closes = hist["Close"].tolist()
            highs = hist["High"].tolist()
            lows = hist["Low"].tolist()
            volumes = [int(v) for v in hist["Volume"].tolist()]

            last_close = closes[-1]
            prev_close = closes[-2] if len(closes) > 1 else last_close
            change_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0.0
            timestamp = str(hist.index[-1].date())

            rsi = _compute_rsi(closes)
            bb_pos = _compute_bb_position(closes)
            vol_ratio = _compute_volume_ratio(volumes)
            regime = _detect_regime(closes, highs, lows)
            signal, confidence = _compute_signal(
                rsi=rsi,
                bb_position=bb_pos,
                volume_ratio=vol_ratio,
                regime=regime,
            )

            quotes.append(
                Quote(
                    symbol=symbol,
                    price=round(last_close, 2),
                    change_pct=round(change_pct, 2),
                    volume=volumes[-1],
                    timestamp=timestamp,
                    rsi=rsi,
                    bb_position=bb_pos,
                    signal=signal,
                    signal_confidence=confidence,
                    regime=regime,
                    volume_ratio=vol_ratio,
                )
            )
            logger.debug(
                "intelligence.market.quote_computed",
                symbol=symbol,
                price=last_close,
                rsi=rsi,
                regime=regime,
                signal=signal,
            )
        except Exception:
            logger.warning("intelligence.market.fetch_failed", symbol=symbol, exc_info=True)

    return quotes


# ── Public async API ──────────────────────────────────────────────────────────


async def get_quotes(symbols: list[str]) -> list[Quote]:
    """Async wrapper — runs yFinance in a thread pool."""
    if not symbols:
        return []
    return await asyncio.to_thread(_fetch_quotes_sync, symbols)


async def get_watchlist_alerts() -> list[Quote]:
    """Return quotes from the configured watchlist that exceed the alert threshold.

    A quote is included when:
      - |change_pct| >= market_alert_threshold * 100 (e.g. 2% for threshold=0.02), OR
      - signal is not NEUTRAL (RSI/BB signal regardless of move size)
    """
    settings = get_settings()
    if not settings.watchlist:
        return []

    threshold_pct = settings.market_alert_threshold * 100
    quotes = await get_quotes(list(settings.watchlist))
    return [q for q in quotes if abs(q.change_pct) >= threshold_pct or q.signal != "NEUTRAL"]


# ── Formatting ────────────────────────────────────────────────────────────────

_SIGNAL_LABELS: dict[str, str] = {
    "MEAN_REVERSION_BUY": "OVERSOLD BOUNCE",
    "MEAN_REVERSION_SELL": "OVERBOUGHT FADE",
    "MOMENTUM_BUY": "MOMENTUM LONG",
    "MOMENTUM_SELL": "MOMENTUM SHORT",
}

_REGIME_LABELS: dict[str, str] = {
    "TRENDING_UP": "TREND↑",
    "TRENDING_DOWN": "TREND↓",
    "RANGING": "RANGE",
    "VOLATILE": "VOLATILE",
}


def format_market_summary(quotes: list[Quote]) -> str:
    """Format a list of quotes as a compact Telegram-ready market brief."""
    if not quotes:
        return "No market data available."

    lines: list[str] = []
    for q in quotes:
        arrow = "▲" if q.change_pct >= 0 else "▼"
        line = f"{q.symbol}: ${q.price:.2f} {arrow}{abs(q.change_pct):.1f}%"

        extras: list[str] = []
        if q.rsi != 50.0:
            extras.append(f"RSI {q.rsi:.0f}")
        if q.volume_ratio > 1.5:
            extras.append(f"Vol {q.volume_ratio:.1f}x")
        if q.regime in _REGIME_LABELS:
            extras.append(_REGIME_LABELS[q.regime])
        if q.signal != "NEUTRAL" and q.signal in _SIGNAL_LABELS:
            extras.append(f"{_SIGNAL_LABELS[q.signal]} {q.signal_confidence:.0%}")
        if q.earnings_context:
            extras.append(q.earnings_context[:60])

        if extras:
            line += " | " + " | ".join(extras)
        lines.append(line)

    return "\n".join(lines)
