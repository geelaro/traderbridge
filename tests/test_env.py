import pytest

from utils.env import load_toml, save_toml


def test_save_toml_round_trips_special_strings(tmp_path):
    path = tmp_path / "watchlist.toml"
    config = {
        "watchlist": [{
            "symbol": 'BRK"B',
            "name": 'Line 1\\part "quoted"\n沪深',
            "active": "weekly_macd",
            "monitor": ["spy_ma_breakout"],
        }],
        "strategy": {"weekly_macd": {"macd_fast": 12}},
    }

    save_toml(path, config)

    assert load_toml(path) == config


def test_save_toml_replace_failure_preserves_existing_file(tmp_path, monkeypatch):
    from utils import env

    path = tmp_path / "watchlist.toml"
    path.write_text("existing config\n", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(env.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        save_toml(path, {"watchlist": []})

    assert path.read_text(encoding="utf-8") == "existing config\n"
    assert list(tmp_path.iterdir()) == [path]
