from dashboard.config_editor import _has_duplicate_symbol


def test_duplicate_symbol_check_normalizes_and_excludes_current_item():
    watchlist = [{"symbol": "aapl"}, {"symbol": "MSFT"}]

    assert _has_duplicate_symbol(watchlist, " Aapl ")
    assert not _has_duplicate_symbol(watchlist, "aapl", exclude_index=0)
    assert _has_duplicate_symbol(watchlist, " msft ", exclude_index=0)
    assert not _has_duplicate_symbol(watchlist, " ")
