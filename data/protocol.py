"""Data protocol — types, schemas, and the abstract DataSource base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional
import pandas as pd
import re

# ---------------------------------------------------------------------------
# Canonical column layout — every source MUST return a DataFrame with these
# columns, indexed by date (pd.DatetimeIndex).
# ---------------------------------------------------------------------------

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Sources are tried in this order for each market
SOURCE_PRIORITY = {
    "us": ["sina_us", "tencent", "yahoo_chart"],
    "cn": ["sina", "tencent", "akshare"],
    "vix": ["cboe"],
    "default": ["tencent"],
}


@dataclass(frozen=True)
class SymbolInfo:
    """Internal symbol descriptor — enough to route to the right source."""

    symbol: str
    market: str  # 'us' | 'cn'
    exchange: str  # 'NASDAQ' | 'NYSE' | 'SSE' | 'SZSE' | 'OTC' | ''
    name: str = ""

    @property
    def display(self) -> str:
        return f"{self.symbol} ({self.market.upper()})"


# ---------------------------------------------------------------------------
# Data-source interface — every adapter must implement these two methods
# ---------------------------------------------------------------------------


class DataSource(ABC):
    """Abstract data source.

    Subclass responsibilities:
      - `supports(symbol)` — quick routing check
      - `name` — short string identifier (e.g. "yfinance", "sina")
      - `fetch(symbol, start, end)` — return pd.DataFrame with OHLCV columns
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this source — used in cache metadata."""
        ...

    @abstractmethod
    def supports(self, symbol: str) -> bool:
        """Return True if this source can handle *symbol*."""
        ...

    @abstractmethod
    def fetch(
        self, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        """Download daily OHLCV data for *symbol*.

        Returns a DataFrame with columns ['Open','High','Low','Close','Volume']
        and a pd.DatetimeIndex sorted ascending.
        """
        ...

    # ------------------------------------------------------------------
    # Validation helpers that subclasses can call
    # ------------------------------------------------------------------

    @staticmethod
    def validate(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Sanitise and validate a raw OHLCV DataFrame."""
        if df is None or df.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # Normalise column names
        df.columns = [c.strip().title() for c in df.columns]
        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                raise KeyError(f"[{symbol}] missing column '{col}'")

        df = df[OHLCV_COLUMNS]

        # Convert to numeric, coerce errors
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)

        # Drop rows with NaN prices
        before = len(df)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        dropped = before - len(df)
        if dropped:
            pass  # silently drop — caller can log if desired

        # Ensure sorted index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove timezone for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()
        return df


# ---------------------------------------------------------------------------
# Market classification helpers — used by symbol routing
# ---------------------------------------------------------------------------

# Well-known China symbols (A-shares, ETFs listed on SSE/SZSE)
CN_SYMBOLS = {
    "510300": "sh510300",
    "510500": "sh510500",
    "159919": "sz159919",
    "510050": "sh510050",
    "159915": "sz159915",
}

_US_CLASS_SHARE_RE = re.compile(r"^[A-Z]{1,5}[.-][A-Z]$")


def classify_symbol(symbol: str) -> str:
    """Return market tag: 'us', 'cn', 'vix', or 'global'."""
    s = symbol.upper().strip()
    # Resolve CN_SYMBOLS alias first
    if s in CN_SYMBOLS:
        s = CN_SYMBOLS[s].upper()
    # VIX index — routes to CBOE official source
    if s.lstrip("^") == "VIX":
        return "vix"
    # Chinese A-share symbols are 6 digits or start with sh/sz
    if s.isdigit() and len(s) == 6:
        return "cn"
    if s[:2] in ("SH", "SZ") and s[2:].isdigit() and len(s[2:]) == 6:
        return "cn"
    # US stocks are 1-5 uppercase letters
    # Class-share tickers such as BRK.B / BRK-B are routed to Yahoo.
    if (s.isalpha() and 1 <= len(s) <= 5) or _US_CLASS_SHARE_RE.fullmatch(s):
        return "us"
    return "global"
