"""Output schema for the token-matched control arm (DESIGN.md §II.3, §II.3a).

The control must emit a *visible* chain-of-thought: the frozen matrix runs
``reasoning_effort='none'`` (§I.6, §II.5a), so there is no hidden reasoning
trace to make the token budget visible and bounded (§II.3's whole point). The
frozen ``agent/schema.py::RESPONSE_SCHEMA`` cannot change, so this module defines
a **superset** schema — the frozen belief/action fields plus a ``reasoning``
string — and projects it back down to a plain
:class:`akrasia.agent.schema.AgentResponse` for the runner/metrics, delegating
semantic validation to the frozen ``parse_response`` rather than forking it.

Only the control uses this. The C1 arm reuses the frozen ``RESPONSE_SCHEMA``
unchanged (no free-text field — that would confound C1's structure with the
control's chain-of-thought, DESIGN.md §II.3).
"""

from __future__ import annotations

from copy import deepcopy

from akrasia.agent.schema import RESPONSE_SCHEMA, AgentResponse, parse_response

# Superset of the frozen RESPONSE_SCHEMA: same belief/action fields, plus a
# leading free-text reasoning field. Built from a deepcopy so the frozen dict is
# never mutated.
CONTROL_SCHEMA: dict = deepcopy(RESPONSE_SCHEMA)
CONTROL_SCHEMA["properties"] = {
    "reasoning": {
        "type": "string",
        "description": "Step-by-step reasoning, stated before the beliefs and action.",
    },
    **CONTROL_SCHEMA["properties"],
}
CONTROL_SCHEMA["required"] = ["reasoning", *CONTROL_SCHEMA["required"]]


def parse_control_response(
    parsed: dict,
) -> tuple[AgentResponse | None, str | None, str | None]:
    """Validate a control response.

    Returns ``(response, reasoning, None)`` on success or ``(None, reasoning,
    error)`` on failure. The belief/action validation is delegated verbatim to
    the frozen :func:`akrasia.agent.schema.parse_response` (which ignores the
    extra ``reasoning`` key), so the control's semantic checks are byte-for-byte
    the baseline's. ``reasoning`` is returned for logging, then discarded by the
    caller — it never reaches the runner or the metrics.
    """
    reasoning = parsed.get("reasoning") if isinstance(parsed, dict) else None
    response, error = parse_response(parsed)
    return response, reasoning, error


def parse_frozen_response(
    parsed: dict,
) -> tuple[AgentResponse | None, str | None, str | None]:
    """``ParseFn`` adapter for arms that keep the frozen ``RESPONSE_SCHEMA`` (no
    ``reasoning`` field) but must route through ``scaffold_elicit_with_retry`` to
    supply a custom user-prompt renderer — the C3a/C3b sufficient-statistic
    observation arms (DESIGN.md §II.2c). Delegates verbatim to the frozen
    :func:`akrasia.agent.schema.parse_response`; there is no reasoning to
    return, so the middle element is always ``None``."""
    response, error = parse_response(parsed)
    return response, None, error
