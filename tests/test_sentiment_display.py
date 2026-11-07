from dashboard.sentiment_display import (
    sentiment_cell_style,
    sentiment_markup,
    sentiment_text,
)


def test_sentiment_display_omits_denominator_and_bolds_score():
    sentiment = {"label": "超买", "score": 72.6}

    assert sentiment_text(sentiment) == "超买 73"
    markup = sentiment_markup(sentiment)
    assert "超买" in markup
    assert "<strong" in markup and ">73</strong>" in markup
    assert "/100" not in markup


def test_sentiment_colors_follow_existing_market_sentiment_convention():
    assert "#dc3545" in sentiment_markup({"label": "超卖", "score": 20})
    assert "#d39e00" in sentiment_markup({"label": "中性", "score": 50})
    assert "#198754" in sentiment_markup({"label": "超买", "score": 80})


def test_sentiment_table_cell_uses_matching_color_and_bold_weight():
    style = sentiment_cell_style("极度超卖 12 · 2026-11-07")
    assert "#dc3545" in style
    assert "font-weight: 700" in style
