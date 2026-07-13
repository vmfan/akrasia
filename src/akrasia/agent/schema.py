"""Same-pass elicitation schema and response validation (DESIGN.md §I.2, §I.4).

The agent must output, in one forward pass: a stated posterior over ``V`` and
over counterparty-type mix, and an action ``(spread, size)`` over the 9-action
grid. ``model_client`` only guarantees the response is well-formed JSON
matching this schema's *shape*; this module adds the semantic checks JSON
Schema can't express (probabilities in range and summing to ~1, action in the
committed grid) — a `ModelResponse.valid=True` can still fail here, and that
is logged rather than coerced (CLAUDE.md §6).
"""

from __future__ import annotations

from dataclasses import dataclass

from akrasia.actions import Action, SIZES, SPREAD_NAMES
from akrasia.agent.game_structure import V_SET

_BELIEF_KEYS = tuple(f"v{v}" for v in V_SET)  # ("v10", "v15", "v20")

RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "belief_v": {
            "type": "object",
            "properties": {key: {"type": "number"} for key in _BELIEF_KEYS},
            "required": list(_BELIEF_KEYS),
            "additionalProperties": False,
        },
        "belief_informed_share": {
            "type": "number",
            "description": "Stated belief about the share of arriving "
            "counterparties that are informed (not the true rate).",
        },
        "action": {
            "type": "object",
            "properties": {
                "spread": {"type": "string", "enum": list(SPREAD_NAMES)},
                "size": {"type": "integer", "enum": list(SIZES)},
            },
            "required": ["spread", "size"],
            "additionalProperties": False,
        },
    },
    "required": ["belief_v", "belief_informed_share", "action"],
    "additionalProperties": False,
}

# 1e-6 is too tight against real model output: a live HKUST/Llama-3.3-70B
# call (format-sanity check, DESIGN.md §I.6 #3) returned belief_v values each
# independently rounded to 6 decimal places (e.g. three copies of 0.333333
# for a uniform-ish belief), whose sum misses 1.0 by ~1.0000000000287557e-06
# — over the old bound purely from per-value decimal rounding, not a real
# reasoning error. 1e-3 comfortably absorbs realistic decimal rounding at
# any precision models commonly use, while still catching genuine
# mis-summing beliefs (which are wrong by orders of magnitude more).
_SUM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class AgentResponse:
    belief_v: tuple[float, float, float]  # aligned to V_SET order
    belief_informed_share: float
    action: Action


def parse_response(parsed: dict) -> tuple[AgentResponse | None, str | None]:
    """Semantically validate an already-JSON-parsed response.

    Returns ``(response, None)`` on success or ``(None, error)`` on failure —
    never raises, mirroring ``ModelResponse``'s non-raising contract.
    """
    try:
        belief_v_raw = parsed["belief_v"]
        belief_v = tuple(float(belief_v_raw[key]) for key in _BELIEF_KEYS)
        informed_share = float(parsed["belief_informed_share"])
        action_raw = parsed["action"]
        spread = action_raw["spread"]
        size = action_raw["size"]
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"missing or malformed field: {exc}"

    if spread not in SPREAD_NAMES:
        return None, f"spread {spread!r} not in {SPREAD_NAMES}"
    if size not in SIZES:
        return None, f"size {size!r} not in {SIZES}"

    if any(p < 0 or p > 1 for p in belief_v):
        return None, f"belief_v probabilities out of [0, 1]: {belief_v}"
    total = sum(belief_v)
    if abs(total - 1.0) > _SUM_TOLERANCE:
        return None, f"belief_v does not sum to 1 (sum={total})"

    if not (0.0 <= informed_share <= 1.0):
        return None, f"belief_informed_share out of [0, 1]: {informed_share}"

    return (
        AgentResponse(
            belief_v=belief_v,
            belief_informed_share=informed_share,
            action=Action(spread=spread, size=size),
        ),
        None,
    )
