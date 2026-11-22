"""Rendering checks for the portfolio backtest dashboard."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa


def test_drawdown_recovery_column_is_arrow_serializable(monkeypatch):
    import analysis.drawdown_attribution as drawdowns
    import dashboard.portfolio_backtest as dashboard

    episodes = [
        {
            "peak_date": pd.Timestamp("2026-01-01"),
            "trough_date": pd.Timestamp("2026-01-02"),
            "depth_pct": -10.0,
            "depth_usd": -100.0,
            "duration_to_trough": 1,
            "duration_to_recovery": recovery,
            "n_trades_in_window": 0,
            "by_symbol": [],
        }
        for recovery in (3, None)
    ]
    streamlit = MagicMock()
    streamlit.slider.side_effect = [5, -1.0]
    streamlit.columns.return_value = [MagicMock() for _ in range(4)]
    monkeypatch.setattr(dashboard, "st", streamlit)
    monkeypatch.setattr(
        drawdowns, "historical_drawdown_attribution", lambda *args, **kwargs: episodes,
    )
    result = SimpleNamespace(
        equity_curve=pd.Series([100.0, 90.0], index=pd.date_range("2026-01-01", periods=2)),
        closed_trades=[],
    )

    dashboard._render_drawdown_attribution(result)

    call = streamlit.dataframe.call_args_list[0]
    frame = call.args[0]
    assert frame["恢复天数"].tolist() == ["3", "仍在 DD"]
    pa.Table.from_pandas(frame)
    assert call.kwargs["width"] == "stretch"
