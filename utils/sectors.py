"""Default sector / industry classification for common symbols."""

DEFAULT_SECTORS: dict[str, str] = {
    # Technology — Semiconductors
    "NVDA": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "MU": "Technology",
    "TSM": "Technology",
    "AVGO": "Technology",
    "QCOM": "Technology",
    # Technology — Hardware / Consumer
    "AAPL": "Technology",
    # Technology — Internet / Software
    "GOOGL": "Technology",
    "GOOG": "Technology",
    "META": "Technology",
    "NFLX": "Technology",
    "MSFT": "Technology",
    "AMZN": "Technology",
    "ORCL": "Technology",
    "CRM": "Technology",
    "ADBE": "Technology",
    # Technology — Semiconductor ETFs
    "SMH": "Technology",
    "DRAM": "Technology",
    # Automotive / EV
    "TSLA": "Automotive",
    "F": "Automotive",
    "GM": "Automotive",
    "RIVN": "Automotive",
    # Financial
    "JPM": "Financial",
    "BAC": "Financial",
    "GS": "Financial",
    "V": "Financial",
    "MA": "Financial",
    "BRK.B": "Financial",
    "BRK-A": "Financial",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    # Healthcare
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "UNH": "Healthcare",
    "ABBV": "Healthcare",
    "LLY": "Healthcare",
    # Consumer
    "WMT": "Consumer",
    "KO": "Consumer",
    "PEP": "Consumer",
    "PG": "Consumer",
    "COST": "Consumer",
    "HD": "Consumer",
    "MCD": "Consumer",
    "NKE": "Consumer",
    "SBUX": "Consumer",
    # Broad-market ETFs
    "SPY": "Broad Market ETF",
    "QQQ": "Broad Market ETF",
    "IWM": "Broad Market ETF",
    "DIA": "Broad Market ETF",
    "VTI": "Broad Market ETF",
    # China
    "510300": "China Equity",
    "510500": "China Equity",
    "510050": "China Equity",
    "159919": "China Equity",
}


def get_sector(symbol: str) -> str:
    """Return sector label for *symbol*, or 'Unknown'."""
    return DEFAULT_SECTORS.get(symbol.upper(), "Unknown")
