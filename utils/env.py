"""TOML helpers — load/save watchlist.toml and friends.

Runtime bootstrap (matplotlib backend / Windows encoding / dotenv) has
moved to :mod:`utils.bootstrap`.  Entry-point scripts now call
``setup_runtime()`` explicitly instead of relying on import-time side
effects.  See ``utils/bootstrap.py`` for the rationale.
"""

import json
import os
import stat
import tempfile


def load_toml(path: str) -> dict:
    """Load a TOML config file. Works on Python 3.10+."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


def save_toml(path: str, data: dict):
    """Write a dict as TOML (hand-rolled, no external dependency)."""
    lines = []
    for section, content in data.items():
        if isinstance(content, list):
            for item in content:
                lines.append(f"\n[[{section}]]")
                for k, v in item.items():
                    lines.append(f"{k} = {_toml_value(v)}")
        elif isinstance(content, dict):
            first = True
            for k, v in content.items():
                if isinstance(v, dict):
                    lines.append(f"\n[{section}.{k}]")
                    for sk, sv in v.items():
                        lines.append(f"{sk} = {_toml_value(sv)}")
                else:
                    if first:
                        lines.append(f"\n[{section}]")
                        first = False
                    lines.append(f"{k} = {_toml_value(v)}")
        else:
            lines.append(f"{section} = {_toml_value(content)}")
    content = "\n".join(lines).lstrip() + "\n"
    target = os.path.abspath(os.fspath(path))
    try:
        existing_mode = stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        existing_mode = None
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp",
        dir=os.path.dirname(target),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if existing_mode is not None:
            os.chmod(temp_path, existing_mode)
        os.replace(temp_path, target)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        items = ", ".join(_toml_value(i) for i in v)
        return f"[{items}]"
    return f'"{v}"'


