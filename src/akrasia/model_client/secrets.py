"""Loads API keys from a local ``.secrets`` file into ``os.environ``.

This does not change where keys come from at read time -- ``client.py`` still
reads only ``os.environ`` (CLAUDE.md §6). It just saves having to ``export``
each key in the shell before running anything. Existing environment
variables always win: a key already exported takes precedence over the same
key in ``.secrets``.

``.secrets`` format: one ``KEY=VALUE`` pair per line. Blank lines and lines
starting with ``#`` are ignored. Surrounding quotes on the value are
stripped.
"""

from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def _find_secrets_file(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / ".secrets"
        if candidate.is_file():
            return candidate
    return None


def load_secrets() -> None:
    """Populate ``os.environ`` from the nearest ``.secrets`` file, if any.

    Idempotent and safe to call from every :class:`ModelClient` construction.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True

    path = _find_secrets_file(Path.cwd())
    if path is None:
        return

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
