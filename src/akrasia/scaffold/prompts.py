"""Loads the frozen scaffold directives directly from DESIGN.md §II.2a/§II.3a.

DESIGN.md is the **sole home** of every agent-under-test-facing prompt/scaffold
string (CLAUDE.md §2 rule 1). Rather than hand-copying the directive text into
dev code — which risks silent drift from the authoritative copy — this module
parses the fenced ```text blocks under the §II.2a (C1), §II.2b (C2), §II.2c
(C3b), and §II.3a (control) headers straight out of DESIGN.md at import time,
exactly as ``agent/prompts.py`` does for the §I.4a base prompt.

The two loaders anchor on disjoint headers (``agent/prompts.py`` on the
``#### Variant A/B/C`` headers inside §I.4a; this module on the ``### II.2a`` /
``### II.2b`` / ``### II.2c`` / ``### II.3a`` headers far below), so neither can
ever cross-match the other's blocks.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_MD_PATH = _REPO_ROOT / "DESIGN.md"

_C1_HEADER = "### II.2a — C1 belief-state directive (frozen scaffold artifact)"
_C2_HEADER = "### II.2b — C2 re-evaluation-loop directive (frozen scaffold artifact)"
_CONTROL_HEADER = "### II.3a — Token-matched control CoT prompt (frozen scaffold artifact)"
_C3B_HEADER = "### II.2c — C3 sufficient-statistic observation directive (frozen scaffold artifact)"


def _extract_block(design_md_text: str, header: str) -> str:
    try:
        after_header = design_md_text[design_md_text.index(header) :]
        fence_start = after_header.index("```text") + len("```text")
        fence_end = after_header.index("```", fence_start)
    except ValueError as exc:
        raise RuntimeError(
            f"could not find a ```text block under {header!r} in DESIGN.md — "
            "has §II been restructured? scaffold/prompts.py's parser must be "
            "updated to match."
        ) from exc
    return after_header[fence_start:fence_end].strip("\n")


def _load() -> tuple[str, str, str, str]:
    if not _DESIGN_MD_PATH.exists():
        raise FileNotFoundError(
            f"DESIGN.md not found at {_DESIGN_MD_PATH} — the scaffold directives "
            "have no other source (CLAUDE.md §2 firewall)."
        )
    text = _DESIGN_MD_PATH.read_text(encoding="utf-8")
    return (
        _extract_block(text, _C1_HEADER),
        _extract_block(text, _C2_HEADER),
        _extract_block(text, _CONTROL_HEADER),
        _extract_block(text, _C3B_HEADER),
    )


C1_DIRECTIVE, C2_DIRECTIVE, CONTROL_COT_DIRECTIVE, C3B_DIRECTIVE = _load()
