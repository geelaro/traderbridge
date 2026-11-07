"""Tests for utils/bootstrap.py — explicit runtime initialisation."""

import ast
import os
from pathlib import Path

import pytest

from utils import bootstrap


_RUNTIME_ENTRYPOINTS = (
    "scripts/check_golden_drift.py",
    "scripts/compare_factor_attribution.py",
    "scripts/diagnose_turtle.py",
    "scripts/exit_experiment.py",
    "scripts/portfolio_diversification_audit.py",
    "scripts/strategy_fit_audit.py",
    "scripts/weekly_risk_report.py",
    "analysis/cost_sensitivity.py",
    "analysis/monte_carlo.py",
    "analysis/param_robustness.py",
    "analysis/rolling_alpha.py",
    "engine/optimize.py",
    "engine/portfolio.py",
    "engine/trader.py",
)
_RUNTIME_IMPORTS = {"analysis", "data", "engine", "matplotlib", "numpy",
                    "pandas", "strategy"}


def _imports_runtime_dependency(node):
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return (module.split(".")[0] in _RUNTIME_IMPORTS
                or module == "utils.font")
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] in _RUNTIME_IMPORTS
                   for alias in node.names)
    return False


def test_idempotent_call():
    """Multiple invocations must not error or repeat side effects."""
    bootstrap.setup_runtime()
    bootstrap.setup_runtime()
    assert bootstrap.is_setup_done() is True


def test_matplotlib_backend_env_set():
    """setup_runtime must export MPLBACKEND so any subsequent matplotlib
    import defaults to the non-interactive Agg backend.  Without this
    headless deployments (Docker / Streamlit) would crash on chart code."""
    bootstrap.setup_runtime()
    assert os.environ.get("MPLBACKEND") == "Agg"


def test_setup_runtime_exported_from_utils():
    """Public API check — `from utils import setup_runtime` works."""
    from utils import setup_runtime
    assert setup_runtime is bootstrap.setup_runtime


@pytest.mark.parametrize("relative_path", _RUNTIME_ENTRYPOINTS)
def test_script_bootstraps_before_business_imports(relative_path):
    script_path = Path(__file__).parent.parent / relative_path
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    setup_calls = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "setup_runtime"
    ]
    runtime_imports = [
        node.lineno for node in ast.walk(tree)
        if _imports_runtime_dependency(node)
    ]

    assert setup_calls, f"{relative_path} must call setup_runtime()"
    assert runtime_imports, f"{relative_path} has no runtime imports to validate"
    assert min(setup_calls) < min(runtime_imports)
