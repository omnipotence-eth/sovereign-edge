from __future__ import annotations

from dataclasses import dataclass

import httpx
import structlog
from core.config import get_settings

logger = structlog.get_logger(__name__)

_BASE = "https://www.alphavantage.co/query"


@dataclass
class Quote:
    symbol: str
    price: float
    change_pct: float
    volume: int
    timestamp: str


async def get_quotes(symbols: list[str]) -> list[Quote]:
    """Fetch real-time quotes from Alpha Vantage for a list of symbols."""
    settings = get_settings()
    if not settings.alpha_vantage_api_key:
        logger.warning("intelligence.market.no_api_key")
        return []

    quotes: list[Quote] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for symbol in symbols:
            try:
                resp = await client.get(
                    _BASE,
                    params={
                        "function": "GLOBAL_QUOTE",
                        "symbol": symbol,
                        "apikey": settings.alpha_vantage_api_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json().get("Global Quote", {})
                if not data:
                    continue
                quotes.append(
                    Quote(
                        symbol=symbol,
                        price=float(data.get("05. price", 0)),
                        change_pct=float(data.get("10. change percent", "0%").rstrip("%")),
                        volume=int(data.get("06. volume", 0)),
                        timestamp=data.get("07. latest trading day", ""),
                    )
                )
            except Exception:
                logger.error("intelligence.market.quote_failed", symbol=symbol, exc_info=True)

    return quotes


async def get_watchlist_alerts() -> list[Quote]:
    """Return quotes that moved more than the configured threshold."""
    settings = get_settings()
    quotes = await get_quotes(settings.watchlist)
    threshold = settings.market_alert_threshold * 100  # pct stored as float e.g. 2.0
    return [q for q in quotes if abs(q.change_pct) >= threshold]


def format_market_summary(quotes: list[Quote]) -> str:
    if not quotes:
        return "No market data available."
    lines = ["**Market Summary**\n"]
    for q in quotes:
        arrow = "▲" if q.change_pct >= 0 else "▼"
        lines.append(f"{q.symbol}: ${q.price:.2f} {arrow}{abs(q.change_pct):.2f}%")
    return "\n".join(lines)
