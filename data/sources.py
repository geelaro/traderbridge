"""Data-source adapters — each wraps a specific external data provider.

Every source implements the DataSource protocol from .protocol.
"""

import json
import logging
import re
from abc import ABC
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .protocol import DataSource, OHLCV_COLUMNS, CN_SYMBOLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


_ET = ZoneInfo("America/New_York")

_YAHOO_CLASS_SHARE_RE = re.compile(r"^[A-Z]{1,5}\.[A-Z]$")
_US_CLASS_SHARE_RE = re.compile(r"^[A-Z]{1,5}[.-][A-Z]$")


def _et_today_naive() -> pd.Timestamp:
    """Current US/Eastern date as a naive Timestamp at midnight.

    Exists as a seam for tests — monkeypatch this instead of pd.Timestamp.now.
    """
    return pd.Timestamp.now(tz=_ET).normalize().tz_localize(None)


_YAHOO_SESSION: Optional[requests.Session] = None


def _yahoo_session() -> requests.Session:
    """Return a requests.Session with Yahoo cookie set.

    Yahoo Finance requires a cookie from fc.yahoo.com before serving
    chart endpoints.  The session is cached per process.

    Honours ``HTTPS_PROXY`` / ``HTTP_PROXY`` env vars so users behind a
    local proxy (Clash / V2Ray) can reach Yahoo.  Set ``trust_env=True``
    is the default for requests, but we make it explicit here because
    some earlier version of this code disabled it.
    """
    global _YAHOO_SESSION
    if _YAHOO_SESSION is not None:
        return _YAHOO_SESSION
    s = requests.Session()
    s.trust_env = True
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    })
    try:
        s.get("https://fc.yahoo.com", timeout=10)
    except requests.RequestException:
        pass
    _YAHOO_SESSION = s
    return s


# ---------------------------------------------------------------------------
# Tencent — US stock daily K-line (free, no key required)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tencent — primary US-stock source (free, no key required)
# ---------------------------------------------------------------------------

_TENCENT_CODE_MAP = {
    "AAPL": "usAAPL.OQ", "MSFT": "usMSFT.OQ", "GOOGL": "usGOOGL.OQ",
    "AMZN": "usAMZN.OQ", "TSLA": "usTSLA.OQ", "NVDA": "usNVDA.OQ",
    "META": "usMETA.OQ", "QQQ": "usQQQ.OQ", "SPY": "usSPY.AM",
    "MU": "usMU.OQ", "INTC": "usINTC.OQ", "ORCL": "usORCL.N",
    # SPDR Select Sector ETFs — NYSE Arca (.AM suffix like SPY).
    # Without explicit mapping the fallback "usXLK" returns a single-bar
    # bogus quote that pollutes the cache.
    "XLK": "usXLK.AM", "XLF": "usXLF.AM", "XLY": "usXLY.AM",
    "XLV": "usXLV.AM", "XLE": "usXLE.AM", "XLI": "usXLI.AM",
    "XLU": "usXLU.AM", "XLB": "usXLB.AM", "XLRE": "usXLRE.AM",
    "XLC": "usXLC.AM", "XLP": "usXLP.AM",
}

# Manual split adjustments — applied to ALL US sources (Tencent, SinaUS,
# YahooChart).  Tencent's qfq parameter is a no-op for US stocks; Sina
# returns raw unadjusted prices; Yahoo's chart v8 ``quote`` field is also
# unadjusted (``adjclose`` exists but we ignore it for source uniformity).
# Prices before split_date are DIVIDED by ratio, volume MULTIPLIED.
# Edit data/splits.json to add/correct entries.
def _load_splits() -> dict[str, list[tuple[str, int]]]:
    import json
    from pathlib import Path
    path = Path(__file__).parent / "splits.json"
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {sym: [(d, int(r)) for d, r in entries] for sym, entries in raw.items()}
    return {}

_US_SPLITS: dict[str, list[tuple[str, int]]] = _load_splits()


def apply_us_splits(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Apply ``splits.json`` corrections to a US-equity OHLCV frame in place.

    No-op for symbols not in the config.  Must be called by every US-equity
    source (TencentSource / SinaUSSource / YahooChartSource) so the cache
    sees a single consistent post-split price scale across sources — mixing
    adjusted Tencent bars with unadjusted Sina bars produced 10× cliffs in
    NVDA history (2023-12-26 case).
    """
    sym = symbol.upper()
    if sym not in _US_SPLITS or df.empty:
        return df
    for split_date, ratio in _US_SPLITS[sym]:
        split_dt = pd.Timestamp(split_date)
        pre = df.index < split_dt
        if not pre.any():
            continue
        for col in ("Open", "High", "Low", "Close"):
            if col in df.columns:
                df.loc[pre, col] = df.loc[pre, col] / ratio
        if "Volume" in df.columns:
            df.loc[pre, "Volume"] = df.loc[pre, "Volume"] * ratio
    return df


class TencentSource(DataSource):
    """Tencent Finance (ifzq.gtimg.cn) — US stocks only.

    Notes
    -----
    - Tencent does NOT adjust prices for splits.  We apply manual corrections.
    - Max ~2000 bars per request.
    """

    URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    @property
    def name(self) -> str:
        return "tencent"

    def supports(self, symbol: str) -> bool:
        sym = symbol.upper()
        if sym in _TENCENT_CODE_MAP:
            return True
        # Fallback: any US-style ticker (1-5 letters) maps to us{sym}.OQ
        return sym.isalpha() and 1 <= len(sym) <= 5

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        sym = symbol.upper()
        code = _TENCENT_CODE_MAP.get(sym, f"us{sym}")

        params = {"param": f"{code},day,{start},,2000,qfq"}
        try:
            s = _make_session()
            r = s.get(self.URL, params=params, timeout=30)
            data = r.json()
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.warning("Tencent fetch error for %s: %s", sym, e)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if data.get("code") != 0 or not isinstance(data.get("data"), dict):
            logger.debug("Tencent API error for %s: %s", sym, data.get("msg", "unknown"))
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        stock_data = data["data"].get(code)
        if not isinstance(stock_data, dict):
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        raw_rows = stock_data.get("day", [])
        if not raw_rows:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        rows = []
        for row in raw_rows:
            if len(row) < 6:
                continue
            rows.append({
                "date":  pd.Timestamp(row[0]),
                "Open":  float(row[1]),
                "Close": float(row[2]),
                "High":  float(row[3]),
                "Low":   float(row[4]),
                "Volume": float(row[5]),
            })

        df = pd.DataFrame(rows).set_index("date").sort_index()
        df = apply_us_splits(df, sym)
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        # Tencent's day endpoint live-updates the last bar during US market
        # hours — OHLC is a mid-session snapshot, not EOD. Drop the ET-today
        # row so we never cache a partial bar; sina_us / yahoo_chart will
        # fill it once the close prints.
        df = df[df.index < _et_today_naive()]
        return self.validate(df, symbol)


# ---------------------------------------------------------------------------
# Sina — Chinese A-shares & ETFs
# ---------------------------------------------------------------------------

class SinaSource(DataSource):
    """Sina Finance — SSE / SZSE equities and ETFs."""

    URL = (
        "https://money.finance.sina.com.cn/quotes_service/api/"
        "json_v2.php/CN_MarketData.getKLineData"
    )

    @property
    def name(self) -> str:
        return "sina"

    def supports(self, symbol: str) -> bool:
        sid = symbol.lower()
        return sid.startswith(("sh", "sz")) or (
            sid.isdigit() and len(sid) == 6
        )

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        code = self._normalise_symbol(symbol)
        params = {"symbol": code, "scale": "240", "ma": "no", "datalen": "2000"}
        try:
            s = _make_session()
            r = s.get(self.URL, params=params, timeout=30)
            data = json.loads(r.text)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            logger.warning("Sina fetch error for %s: %s", symbol, e)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if not data or not isinstance(data, list):
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        rows = []
        for row in data:
            rows.append({
                "date":  pd.Timestamp(row["day"]),
                "Open":  float(row["open"]),
                "High":  float(row["high"]),
                "Low":   float(row["low"]),
                "Close": float(row["close"]),
                "Volume": float(row["volume"]),
            })

        df = pd.DataFrame(rows).set_index("date").sort_index()
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return self.validate(df, symbol)

    @staticmethod
    def _normalise_symbol(symbol: str) -> str:
        """Convert '510300' → 'sh510300', etc."""
        s = symbol.lower().strip()
        if s in CN_SYMBOLS:
            return CN_SYMBOLS[s]
        if s.startswith(("sh", "sz")):
            return s
        # Guess exchange — most 6-digit ETF codes starting with 5 are SSE
        prefix = "sh" if s[0] in ("5", "6", "9") else "sz"
        return prefix + s


# ---------------------------------------------------------------------------
# AKShare — modern A-share source (primary for CN stocks)
# ---------------------------------------------------------------------------

_A_KSHARE_MAP = {
    "510050": "sh510050",
    "510300": "sh510300",
    "510500": "sh510500",
    "159915": "sz159915",
    "159919": "sz159919",
}


class AKShareSource(DataSource):
    """AKShare — free A-share / index / ETF data.

    Requires: pip install akshare
    """

    @property
    def name(self) -> str:
        return "akshare"

    def supports(self, symbol: str) -> bool:
        sid = symbol.lower().strip()
        if sid in _A_KSHARE_MAP:
            return True
        return sid.startswith(("sh", "sz"))

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare not installed — skipping")
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        code = self._normalise_symbol(symbol)
        market = "sh" if code.startswith("sh") else "sz"

        try:
            # ETF / index data
            if code[2:] in ("510050", "510300", "510500", "159915", "159919"):
                df = ak.fund_etf_hist_em(
                    symbol=code[2:], period="daily",
                    start_date=start.replace("-", ""),
                    end_date=end.replace("-", ""), adjust="qfq",
                )
                if df is not None and not df.empty:
                    return self._transform_akshare(df, market, start, end)

            # Try index data for common indices
            index_map = {
                "000300": "sh000300", "000905": "sh000905",
                "000016": "sh000016", "399006": "sz399006",
            }
            if code[2:] in index_map:
                df = ak.stock_zh_index_daily(symbol=index_map[code[2:]])
                if df is not None and not df.empty:
                    return self._transform_akshare(df, market, start, end)

        except Exception:
            logger.exception("akshare fetch failed for %s", symbol)

        return pd.DataFrame(columns=OHLCV_COLUMNS)

    @staticmethod
    def _transform_akshare(
        df: pd.DataFrame, market: str, start: str, end: str
    ) -> pd.DataFrame:
        """Normalise akshare output to OHLCV_COLUMNS."""
        col_map = {
            "日期": "date",     "开盘": "Open",
            "最高": "High",     "最低": "Low",
            "收盘": "Close",    "成交量": "Volume",
            "open": "Open",     "high": "High",
            "low": "Low",       "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns=col_map)
        if "date" not in df.columns and df.index.name != "date":
            # Try to find date column
            for c in df.columns:
                if "日期" in c or "date" in c.lower():
                    df["date"] = df[c]
                    break
            else:
                if "date" not in df.columns:
                    df["date"] = df.index

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = np.nan
        df = df[OHLCV_COLUMNS]
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return df

    @staticmethod
    def _normalise_symbol(symbol: str) -> str:
        s = symbol.lower().strip()
        if s in _A_KSHARE_MAP:
            return _A_KSHARE_MAP[s]
        if s in CN_SYMBOLS:
            return CN_SYMBOLS[s]
        if s.startswith(("sh", "sz")):
            return s
        # Guess exchange — most 6-digit ETF codes starting with 5 are SSE
        prefix = "sh" if s[0] in ("5", "6", "9") else "sz"
        return prefix + s


# ---------------------------------------------------------------------------
# Sina US stock daily K-line — primary US source (back to 1984)
# ---------------------------------------------------------------------------

class SinaUSSource(DataSource):
    """Sina Finance US stock daily K-line API.

    Longer history than Tencent (back to 1984), zero authentication.
    """

    URL = (
        "https://stock.finance.sina.com.cn/usstock/api/"
        "jsonp.php/var/US_MinKService.getDailyK"
    )

    @property
    def name(self) -> str:
        return "sina_us"

    def supports(self, symbol: str) -> bool:
        sym = symbol.upper().strip()
        return (
            sym.isalpha() and 1 <= len(sym) <= 5
        ) or _US_CLASS_SHARE_RE.fullmatch(sym) is not None

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        sym = symbol.upper()
        params = {"symbol": sym, "num": 5000}
        try:
            s = _make_session()
            r = s.get(self.URL, params=params,
                      headers={"Referer": "https://finance.sina.com.cn/"}, timeout=30)
        except requests.RequestException as e:
            logger.warning("SinaUS fetch error for %s: %s", sym, e)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        m = re.search(r"\((\[.+\])\)", r.text)
        if not m:
            logger.debug("SinaUS empty/unparsable response for %s", sym)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        try:
            items = json.loads(m.group(1))
        except json.JSONDecodeError:
            logger.debug("SinaUS JSON decode failed for %s", sym)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if not items:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        rows = []
        for item in items:
            rows.append({
                "date": pd.Timestamp(item.get("d")),
                "Open": float(item.get("o", 0)),
                "High": float(item.get("h", 0)),
                "Low": float(item.get("l", 0)),
                "Close": float(item.get("c", 0)),
                "Volume": float(item.get("v", 0)),
            })

        df = pd.DataFrame(rows).set_index("date").sort_index()
        df = apply_us_splits(df, sym)
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return self.validate(df, symbol)



# ---------------------------------------------------------------------------
# Yahoo Finance chart v8 — fallback for US stocks
# ---------------------------------------------------------------------------

class YahooChartSource(DataSource):
    """Yahoo Finance chart v8 API.

    Uses period1/period2 Unix timestamps for precise date range control.
    Covers US equities, ETFs, and global markets.
    Requires a Yahoo cookie to avoid 403.
    """

    URL = "https://query2.finance.yahoo.com/v8/finance/chart"

    @property
    def name(self) -> str:
        return "yahoo_chart"

    def supports(self, symbol: str) -> bool:
        sym = symbol.upper().strip()
        if sym[:2] in ("SH", "SZ") or (sym.isdigit() and len(sym) == 6):
            return False
        return True

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        sym = symbol.upper()
        yahoo_sym = sym.replace(".", "-") if _YAHOO_CLASS_SHARE_RE.fullmatch(sym) else sym
        try:
            t1 = int(pd.Timestamp(start).timestamp())
            t2 = int(pd.Timestamp(end).timestamp())
        except Exception:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        params = {
            "interval": "1d",
            "period1": t1,
            "period2": t2,
        }
        try:
            s = _yahoo_session()
            r = s.get(f"{self.URL}/{yahoo_sym}", params=params, timeout=30)
            r.raise_for_status()
        except (requests.RequestException, ValueError) as e:
            logger.warning("YahooChart fetch error for %s: %s", sym, e)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        data = r.json()
        chart = data.get("chart", {}).get("result", [{}])
        if not chart or chart[0] is None:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        result = chart[0]
        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]

        rows = []
        for i, ts in enumerate(timestamps):
            o = quote.get("open", [None])[i]
            h = quote.get("high", [None])[i]
            lv = quote.get("low", [None])[i]
            c = quote.get("close", [None])[i]
            v = quote.get("volume", [None])[i]
            if any(x is None for x in (o, h, lv, c, v)):
                continue
            rows.append({
                "date": pd.Timestamp.fromtimestamp(ts).normalize(),
                "Open": round(float(o), 4),
                "High": round(float(h), 4),
                "Low": round(float(lv), 4),
                "Close": round(float(c), 4),
                "Volume": float(v),
            })

        df = pd.DataFrame(rows).set_index("date").sort_index()
        df = apply_us_splits(df, sym)
        return self.validate(df, symbol)


# ---------------------------------------------------------------------------
# CBOE — official VIX daily history (free, public, no auth, no API key)
# ---------------------------------------------------------------------------


class CBOEVixSource(DataSource):
    """CBOE official VIX daily history.

    Serves a single CSV with full history back to 1990. The whole file is
    pulled on each fetch (~470 KB / 9000 rows), then clipped to the requested
    range — slower than incremental APIs but the data source is rock-solid.
    Since VIX values are slow-moving and the cache layer dedupes, refetching
    the full CSV every few days is acceptable.
    """

    URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

    @property
    def name(self) -> str:
        return "cboe"

    def supports(self, symbol: str) -> bool:
        # Accept both '^VIX' (standard ticker) and 'VIX'
        return symbol.upper().lstrip("^") == "VIX"

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import io
        try:
            s = _make_session()
            r = s.get(self.URL, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            logger.warning("CBOE fetch error for %s: %s", symbol, e)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        try:
            df = pd.read_csv(io.StringIO(r.text))
        except Exception as e:
            logger.warning("CBOE parse error for %s: %s", symbol, e)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if not {"DATE", "OPEN", "HIGH", "LOW", "CLOSE"}.issubset(df.columns):
            logger.warning("CBOE: unexpected columns %s", list(df.columns))
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df["date"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        df = df.rename(columns={"OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close"})
        # VIX is an index, not tradable — no volume. Fill 0 so validate() passes.
        df["Volume"] = 0
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return self.validate(df, symbol)
