"""DataSource interface-contract tests — no network calls.

Complementary to the existing per-source TestXXXSourceFetch classes
(which mock HTTP and verify fetch *behavior*): this file only checks
that every concrete DataSource satisfies the shape DataProvider assumes
when it default-constructs its source list (data/provider.py:89-96).
A source whose __init__ gained a required argument, or whose .name/
.supports() signature drifted, would silently break DataProvider() at
import time — this is the test that catches it.
"""

import pytest

from data.protocol import DataSource
from data.sources import (
    AKShareSource,
    CBOEVixSource,
    SinaSource,
    SinaUSSource,
    TencentSource,
    YahooChartSource,
)

ALL_SOURCE_CLASSES = [
    TencentSource, SinaSource, AKShareSource,
    SinaUSSource, YahooChartSource, CBOEVixSource,
]

SAMPLE_SYMBOLS = ["AAPL", "600519", "^VIX", "not-a-real-symbol-!!!", ""]


@pytest.mark.parametrize("cls", ALL_SOURCE_CLASSES)
class TestSourceContract:
    def test_constructs_with_no_arguments(self, cls):
        """DataProvider.__init__'s default source list constructs every
        class with zero arguments — this must keep working."""
        instance = cls()
        assert isinstance(instance, DataSource)

    def test_name_is_nonempty_string(self, cls):
        instance = cls()
        assert isinstance(instance.name, str)
        assert instance.name.strip() != ""

    @pytest.mark.parametrize("symbol", SAMPLE_SYMBOLS)
    def test_supports_returns_bool_without_raising(self, cls, symbol):
        instance = cls()
        result = instance.supports(symbol)
        assert isinstance(result, bool)


class TestSourceNamesMatchPriorityRouting:
    """DataProvider._find_source_by_name looks sources up *by name* against
    SOURCE_PRIORITY (data/protocol.py) — a source whose .name doesn't match
    any priority-list entry is silently unreachable. Every concrete source
    must be routable from at least one market bucket."""

    def test_every_source_name_appears_in_some_priority_list(self):
        from data.protocol import SOURCE_PRIORITY

        all_priority_names = {n for names in SOURCE_PRIORITY.values() for n in names}
        for cls in ALL_SOURCE_CLASSES:
            name = cls().name
            assert name in all_priority_names, (
                f"{cls.__name__}.name={name!r} does not appear in any "
                f"SOURCE_PRIORITY list — DataProvider can never select it"
            )
