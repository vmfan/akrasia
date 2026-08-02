"""Pure system-prompt composition for the scaffold arms (DESIGN.md §II.2a, §II.3a).

Each composer layers a scaffold directive (the frozen prose from DESIGN.md §II,
loaded by ``scaffold/prompts.py``) *after* the verbatim frozen base prompt
(``agent/prompts.py::BASE_PROMPT``). The base prompt appears unchanged as a
prefix — the scaffold is a layer *around* it, never an edit to it (freeze gate,
DESIGN.md §I.6). Dynamic carried state rides in the system message (not the user
message) so the C1 arm can reuse the frozen ``harness.elicit_with_retry``
unchanged — that ladder builds the user turn internally from
``observation.to_prompt_text()``.

These functions take the directive strings as arguments (rather than importing
them) so they stay pure and trivially testable.
"""

from __future__ import annotations

from akrasia.scaffold.belief_state import CarriedBeliefState
from akrasia.scaffold.tentative import TentativeRead

# First-round plumbing string, parallel to observation.py's "No prior rounds
# observed yet." — a dynamic-state renderer, not frozen directive text.
_NO_PRIOR_BELIEF = (
    "No belief has been recorded yet; this is the first round of the episode."
)

# First-pass plumbing string for the C2 within-round loop (DESIGN.md §II.2b):
# on pass 1 there is no tentative read to revise yet. A dynamic-state renderer,
# not frozen directive text.
_NO_TENTATIVE_READ = (
    "This is your first read of the round; there is no tentative read to "
    "re-evaluate yet. Produce your tentative beliefs and action."
)

# First-pass plumbing string for the pass-matched control loop (DESIGN.md §II.3a):
# on pass 1 there is no earlier reasoning to revisit. Parallel to
# _NO_TENTATIVE_READ — a dynamic-state renderer, not frozen directive text.
_NO_PRIOR_REASONING = (
    "This is your first pass this round; there is no earlier reasoning to "
    "revisit yet. Produce your step-by-step reasoning, beliefs, and action."
)


def _render_prior_reasoning(reasoning: str) -> str:
    """Re-show the control's own prior-pass reasoning verbatim for its next pass.

    Deterministic structured->text plumbing (same category as
    :func:`akrasia.scaffold.tentative.TentativeRead.render`), NOT frozen directive
    text: it presents the prior read but issues no belief object and no re-evaluation
    instruction — the pass-matched control shares the full scaffold's *pass budget*,
    not its structure (DESIGN.md §II.3, §II.3a)."""
    return (
        "The step-by-step reasoning you produced a moment ago for this same "
        "round, to reconsider before you commit:\n"
        f"{reasoning}"
    )


def compose_c1_system(
    base_prompt: str, c1_directive: str, carried: CarriedBeliefState | None
) -> str:
    """System prompt for the C1 arm: frozen base + C1 directive + the carried
    belief-state (or the first-round placeholder)."""
    recorded = _NO_PRIOR_BELIEF if carried is None else carried.render()
    return "\n\n".join((base_prompt, c1_directive, recorded))


def compose_c3b_system(
    base_prompt: str, c3b_directive: str, carried: CarriedBeliefState | None
) -> str:
    """System prompt for the C3b arm (DESIGN.md §II.2c): frozen base + C3b
    incremental/sufficient-statistic directive + the carried belief-state (or
    the first-round placeholder).

    Structurally identical to :func:`compose_c1_system`; the difference from C1
    is (a) the directive (incremental — "revise the recorded belief on the new
    flow shown", vs C1's "keep consistent with everything observed so far") and
    (b) that C3b's *user turn* is rendered increment-only
    (``observation_render.render_increment_only``), so the recorded belief and
    the observation never re-present the same evidence — which is what makes the
    directive's "you are shown only the new flow" description truthful."""
    recorded = _NO_PRIOR_BELIEF if carried is None else carried.render()
    return "\n\n".join((base_prompt, c3b_directive, recorded))


def compose_c2_system(
    base_prompt: str, c2_directive: str, tentative: TentativeRead | None
) -> str:
    """System prompt for one pass of the C2 arm: frozen base + C2 directive +
    the tentative read to re-evaluate (or the first-pass placeholder).

    ``tentative is None`` builds pass 1's prompt (produce a tentative read); a
    populated ``tentative`` builds a re-evaluation pass's prompt (DESIGN.md
    §II.2b)."""
    read = _NO_TENTATIVE_READ if tentative is None else tentative.render()
    return "\n\n".join((base_prompt, c2_directive, read))


def compose_c1_c2_system(
    base_prompt: str,
    c1_directive: str,
    c2_directive: str,
    carried: CarriedBeliefState | None,
    tentative: TentativeRead | None,
) -> str:
    """System prompt for one pass of the full-scaffold (+C1+C2) arm: frozen base
    + C1 directive + C2 directive + the cross-round carried belief-state + the
    within-round tentative read (each with its first-round / first-pass
    placeholder). C1 supplies the cross-round carry, C2 the within-round loop
    (DESIGN.md §II.2b)."""
    recorded = _NO_PRIOR_BELIEF if carried is None else carried.render()
    read = _NO_TENTATIVE_READ if tentative is None else tentative.render()
    return "\n\n".join((base_prompt, c1_directive, c2_directive, recorded, read))


def compose_control_pass_system(
    base_prompt: str, cot_directive: str, prior_reasoning: str | None
) -> str:
    """System prompt for one pass of the pass-matched control: frozen base +
    frozen §II.3a CoT directive + the control's own prior-pass reasoning to
    reconsider (or the first-pass placeholder).

    The control runs the same fixed ``P`` passes as the full scaffold so its token
    budget is matched by construction (DESIGN.md §II.3, §II.3a). ``prior_reasoning
    is None`` builds pass 1's prompt (produce a first read); a populated string
    builds a reconsideration pass's prompt. The frozen CoT directive is reused
    verbatim every pass — only the re-shown reasoning is dynamic plumbing, so the
    control adds *budget*, never structure (no belief object, no re-eval directive)."""
    read = _NO_PRIOR_REASONING if prior_reasoning is None else _render_prior_reasoning(prior_reasoning)
    return "\n\n".join((base_prompt, cot_directive, read))
