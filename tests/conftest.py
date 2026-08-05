"""Shared fixtures for all tests."""

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root on path before bootstrap import.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Runtime bootstrap — must run before any business import that touches
# matplotlib, .env, or Windows-specific encoding paths.
from utils.bootstrap import setup_runtime
setup_runtime()

import numpy as np
import pandas as pd
import pytest

# Isolate test database from production (per-session temp file)
import tempfile as _tempfile
_test_db_dir = _tempfile.mkdtemp(prefix="traderbridge_test_")
os.environ["TRADERBRIDGE_DB"] = os.path.join(_test_db_dir, "test.db")


# ---------------------------------------------------------------------------
# Synthetic OHLCV data — 300 bars, up-trend then sideways
# ---------------------------------------------------------------------------


def make_ohlcv(n_bars: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic daily OHLCV data.

    The series has a clear uptrend (first 200 bars) then enters a
    sideways/choppy regime (last 100 bars).  This tests whether a
    strategy can stay in a trend and avoid whipsaws in a range.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_bars)

    # Log-returns: mild drift + noise
    drift = 0.0005
    noise = rng.normal(0, 0.015, n_bars)
    # After bar 200, reduce drift — creates a sideways regime
    drift_arr = np.full(n_bars, drift)
    drift_arr[200:] = 0.0

    log_ret = drift_arr + noise
    close = 100 * np.exp(np.cumsum(log_ret))

    # Build OHLCV
    data = []
    for i, c in enumerate(close):
        daily_range = c * abs(rng.normal(0.015, 0.005))
        h = c + daily_range * abs(rng.uniform(0.3, 1.0))
        l = c - daily_range * abs(rng.uniform(0.3, 1.0))
        o = l + rng.uniform(0, h - l)
        v = int(rng.uniform(5_000_000, 20_000_000))
        data.append({
            "Date": dates[i], "Open": round(o, 2), "High": round(h, 2),
            "Low": round(l, 2), "Close": round(c, 2), "Volume": v,
        })

    df = pd.DataFrame(data).set_index("Date")
    df.index.name = "date"
    return df


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """300-bar synthetic OHLCV DataFrame."""
    return make_ohlcv(300)


@pytest.fixture
def ohlcv_short() -> pd.DataFrame:
    """60-bar synthetic OHLCV — enough for strategy min_bars but not much more."""
    return make_ohlcv(60)


# ---------------------------------------------------------------------------
# Temp SQLite cache
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_cache():
    """CacheManager pointed at a temporary SQLite file."""
    from data.cache import CacheManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name

    cache = CacheManager(db_path=path)
    yield cache
    cache.close()
    # init_schema enables WAL — the -wal/-shm sidecar files survive the
    # main db unlink on Windows and leak into the temp dir otherwise.
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(path + suffix)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Mock broker with pre-set prices
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_broker():
    """MockBroker with $100k cash and preset last prices."""
    from broker import MockBroker

    broker = MockBroker(initial_cash=100_000)
    broker.last_prices = {
        "AAPL": 195.0, "NVDA": 850.0, "TSLA": 240.0,
        "QQQ": 450.0, "SPY": 500.0,
    }
    return broker
