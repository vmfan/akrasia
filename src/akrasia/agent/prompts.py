"""Loads the frozen base-prompt variants directly from DESIGN.md §I.4a.

DESIGN.md is the **sole home** of the agent-under-test's base prompt
(CLAUDE.md §2 firewall; DESIGN.md §I.4a). Rather than hand-copying the text
into a second file — which risks silent drift between the "authoritative"
copy and what actually gets sent to the model — this module parses the three
fenced ```text blocks under the "#### Variant A/B/C" headers straight out of
DESIGN.md at import time. There is exactly one copy of the prompt text; this
is a view onto it, not a mirror that could go stale.

Variant A is primary (used in every §II.4 matrix row); B and C exist only for
the §II.6 prompt-sensitivity check.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_MD_PATH = _REPO_ROOT / "DESIGN.md"

_VARIANT_HEADERS: dict[str, str] = {
    "A": "#### Variant A — primary (headed spec)",
    "B": "#### Variant B — sensitivity (continuous prose)",
    "C": "#### Variant C — sensitivity (numbered rules)",
}


def _extract_variant(design_md_text: str, header: str) -> str:
    try:
        after_header = design_md_text[design_md_text.index(header) :]
        fence_start = after_header.index("```text") + len("```text")
        fence_end = after_header.index("```", fence_start)
    except ValueError as exc:
        raise RuntimeError(
            f"could not find a ```text block under {header!r} in DESIGN.md — "
            "has §I.4a been restructured? agent/prompts.py's parser must be "
            "updated to match."
        ) from exc
    return after_header[fence_start:fence_end].strip("\n")


def _load_variants() -> dict[str, str]:
    if not _DESIGN_MD_PATH.exists():
        raise FileNotFoundError(
            f"DESIGN.md not found at {_DESIGN_MD_PATH} — the base prompt has "
            "no other source (CLAUDE.md §2 firewall)."
        )
    text = _DESIGN_MD_PATH.read_text(encoding="utf-8")
    return {key: _extract_variant(text, header) for key, header in _VARIANT_HEADERS.items()}


_variants = _load_variants()

PROMPT_VARIANT_A: str = _variants["A"]
PROMPT_VARIANT_B: str = _variants["B"]
PROMPT_VARIANT_C: str = _variants["C"]

# Variant A is the primary base prompt used in every §II.4 matrix row.
BASE_PROMPT: str = PROMPT_VARIANT_A
