"""Formatting helpers for per-symbol sentiment labels in dashboard views."""

from html import escape


_SENTIMENT_COLORS = {
    "极度超卖": "#dc3545",
    "超卖": "#dc3545",
    "中性": "#d39e00",
    "超买": "#198754",
    "极度超买": "#198754",
}


def sentiment_text(sentiment: dict) -> str:
    """Format the sentiment label and score without a redundant denominator."""
    return f"{sentiment['label']} {sentiment['score']:.0f}"


def sentiment_markup(sentiment: dict) -> str:
    """Return a colored label with a bold score for Streamlit Markdown."""
    label = str(sentiment["label"])
    color = _SENTIMENT_COLORS.get(label, "#6c757d")
    score = f"{float(sentiment['score']):.0f}"
    return (
        f'<span style="color:{color}; font-weight:600">{escape(label)} '
        f'<strong style="font-weight:700">{score}</strong></span>'
    )


def sentiment_cell_style(value: str) -> str:
    """Pandas Styler rule matching the same colors as sentiment_markup."""
    label = str(value).split(" ", 1)[0]
    color = _SENTIMENT_COLORS.get(label, "#6c757d")
    return f"color: {color}; font-weight: 700;"
