"""Layer-boundary enforcement — encodes AGENTS.md "Layer-import enforcement"
as executable checks instead of manually-run grep commands (see
AGENTS.md "Module boundaries (layer contract)").

Uses ``ast`` rather than regex/grep so docstring/comment mentions of e.g.
"broker.submit_order" don't produce false positives — AGENTS.md's own grep
commands have exactly this limitation in practice: `live/decision_logger.py`
and `broker/mock.py` both mention `.submit_order` only inside a docstring
(a bullet point and a usage example respectively), which a plain grep
would flag but isn't a real violation. ast.walk() only sees real
Import/ImportFrom/Attribute nodes, not string literals.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Mirrors .coveragerc's omit list — these dirs aren't part of the runtime
# layer contract (tests/scripts intentionally cross layers for fixtures;
# archive/htmlcov aren't live code).
_EXCLUDED_TOP_DIRS = {"tests", "scripts", "archive", "htmlcov", ".venv", ".git"}

_BROKER_VALUE_TYPES = {"Order", "OrderSide", "OrderStatus", "OrderType"}

# AGENTS.md: broker.submit_order calls should only appear in the daemon
# order path, the emergency Kill Switch path, and the RetryingBroker
# passthrough (broker/base.py is the ABC declaring the method signature).
_SUBMIT_ORDER_ALLOWED_FILES = {
    "live/order_manager.py",
    "live/kill_switch.py",
    "broker/middleware.py",
    "broker/base.py",
}


def _iter_py_files(*dirs: str):
    for d in dirs:
        base = ROOT / d
        if base.exists():
            yield from base.rglob("*.py")


def _iter_all_py_files():
    for path in ROOT.rglob("*.py"):
        rel_parts = path.relative_to(ROOT).parts
        if rel_parts[0] in _EXCLUDED_TOP_DIRS:
            continue
        yield path


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_type_checking_guard(node: ast.AST) -> bool:
    """True if *node* is an `if TYPE_CHECKING:` guard — its body is type-only
    (annotations, Protocol stubs), not a runtime dependency."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True  # typing.TYPE_CHECKING
    if isinstance(test, ast.BoolOp) and all(
        isinstance(v, ast.Name) and v.id == "TYPE_CHECKING" for v in test.values
    ):
        return True
    return False


def _iter_code_nodes(tree: ast.Module):
    """Yield every AST node in *tree* except those inside an ``if
    TYPE_CHECKING:`` guard — type-only imports/annotations are not runtime
    layer dependencies, so a ``from live import ...`` that only feeds a
    type hint must not trip the boundary check."""
    def _walk(node):
        if _is_type_checking_guard(node):
            return
        yield node
        for child in ast.iter_child_nodes(node):
            yield from _walk(child)

    yield from _walk(tree)


def _top_level_import_modules(tree: ast.Module):
    """Yield (root_module_name, lineno) for every import in *tree*.

    'from a.b import c' → ('a', lineno); 'import a.b' → ('a', lineno).
    """
    for node in _iter_code_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".")[0], node.lineno


class TestAnalysisDoesNotImportLive:
    """analysis/ MUST NOT import live/ — analysis is pure-compute and must
    stay usable without a broker/daemon in scope (AGENTS.md "analysis/ is
    pure compute")."""

    def test_no_live_imports(self):
        violations = []
        for path in _iter_py_files("analysis"):
            for mod, lineno in _top_level_import_modules(_parse(path)):
                if mod == "live":
                    violations.append(f"{path.relative_to(ROOT).as_posix()}:{lineno}")
        assert not violations, f"analysis/ must not import live/: {violations}"


class TestLowLayersDoNotImportLiveOrBroker:
    """strategy/, data/, utils/ are below live/ and broker/ in the layer
    stack — depending on either would invert the dependency direction."""

    def test_no_live_or_broker_imports(self):
        violations = []
        for d in ("strategy", "data", "utils"):
            for path in _iter_py_files(d):
                for mod, lineno in _top_level_import_modules(_parse(path)):
                    if mod in ("live", "broker"):
                        rel = path.relative_to(ROOT).as_posix()
                        violations.append(f"{rel}:{lineno} imports {mod}")
        assert not violations, (
            f"strategy/, data/, utils/ must not import live/ or broker/: {violations}"
        )


class TestEngineOnlyImportsBrokerValueTypes:
    """engine/ may reference broker VALUE TYPES (Order/OrderSide/OrderStatus
    /OrderType — plain dataclasses/enums shared as vocabulary) but never a
    broker IMPLEMENTATION (Broker/MockBroker/FutuBroker/RetryingBroker) —
    the backtest engine must never be able to place a real order."""

    def test_only_value_types_imported(self):
        violations = []
        for path in _iter_py_files("engine"):
            tree = _parse(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "broker":
                    for alias in node.names:
                        if alias.name not in _BROKER_VALUE_TYPES:
                            rel = path.relative_to(ROOT).as_posix()
                            violations.append(f"{rel}:{node.lineno} imports broker.{alias.name}")
        assert not violations, (
            f"engine/ may only import broker value types {_BROKER_VALUE_TYPES}: {violations}"
        )


class TestTypeCheckingGuardsIgnored:
    """Type-only imports (`if TYPE_CHECKING:`) are not runtime layer
    dependencies — a ``from live import ...`` that only feeds an annotation
    must not trip the boundary check."""

    def test_bare_guard_skipped(self):
        src = (
            "if TYPE_CHECKING:\n"
            "    from live.order_manager import OrderManager\n"
            "from data import provider\n"
        )
        mods = [m for m, _ in _top_level_import_modules(ast.parse(src))]
        assert mods == ["data"]  # the live import is type-only

    def test_typing_qualified_guard_skipped(self):
        src = (
            "if typing.TYPE_CHECKING:\n"
            "    from broker import Broker\n"
        )
        mods = [m for m, _ in _top_level_import_modules(ast.parse(src))]
        assert mods == []


class TestSubmitOrderCallSites:
    """.submit_order() is the one call that actually places a live/paper
    order — restrict where it can be invoked from to the daemon order
    path, the Kill Switch emergency path, and the RetryingBroker
    passthrough (tests/ excluded entirely — fixtures legitimately call
    it against MockBroker)."""

    def test_only_called_from_allowed_files(self):
        violations = []
        for path in _iter_all_py_files():
            rel = path.relative_to(ROOT).as_posix()
            tree = _parse(path)
            has_call = any(
                isinstance(node, ast.Attribute) and node.attr == "submit_order"
                for node in _iter_code_nodes(tree)
            )
            if has_call and rel not in _SUBMIT_ORDER_ALLOWED_FILES:
                violations.append(rel)
        assert not violations, (
            f".submit_order may only be referenced from {sorted(_SUBMIT_ORDER_ALLOWED_FILES)}: "
            f"{violations}"
        )
