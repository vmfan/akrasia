"""Verify the base-prompt loader (DESIGN.md §I.4a) parses successfully and
stays consistent with the elicitation schema (§I.4a F15 vs schema.py).
"""

from __future__ import annotations

from akrasia.agent.prompts import (
    BASE_PROMPT,
    PROMPT_VARIANT_A,
    PROMPT_VARIANT_B,
    PROMPT_VARIANT_C,
)
from akrasia.agent.schema import RESPONSE_SCHEMA


def test_variants_are_nonempty_and_distinct():
    variants = [PROMPT_VARIANT_A, PROMPT_VARIANT_B, PROMPT_VARIANT_C]
    assert all(v.strip() for v in variants)
    assert len(set(variants)) == 3


def test_base_prompt_is_variant_a():
    assert BASE_PROMPT == PROMPT_VARIANT_A


def test_variants_disclose_no_arrival_mix_value():
    # Firewall audit (DESIGN.md §I.4a): P(Informed)=0.35 must never appear.
    for variant in (PROMPT_VARIANT_A, PROMPT_VARIANT_B, PROMPT_VARIANT_C):
        assert "0.35" not in variant
        assert "35%" not in variant


def test_variants_are_ascii_only():
    for variant in (PROMPT_VARIANT_A, PROMPT_VARIANT_B, PROMPT_VARIANT_C):
        variant.encode("ascii")  # raises UnicodeEncodeError if not


def test_variants_reference_schema_field_names_in_order():
    schema_fields = list(RESPONSE_SCHEMA["properties"].keys())
    assert schema_fields == ["belief_v", "belief_informed_share", "action"]
    for variant in (PROMPT_VARIANT_A, PROMPT_VARIANT_B, PROMPT_VARIANT_C):
        idx_belief_v = variant.index('"belief_v"')
        idx_informed = variant.index('"belief_informed_share"')
        idx_action = variant.index('"action"')
        assert idx_belief_v < idx_informed < idx_action

    belief_v_keys = list(RESPONSE_SCHEMA["properties"]["belief_v"]["properties"].keys())
    assert belief_v_keys == ["v10", "v15", "v20"]

    action_props = RESPONSE_SCHEMA["properties"]["action"]["properties"]
    assert action_props["spread"]["enum"] == ["tight", "medium", "wide"]
    assert action_props["size"]["enum"] == [1, 2, 3]
