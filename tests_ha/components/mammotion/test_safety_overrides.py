"""Pins the deliberate safety-gate override mechanism (2026-08-19).

The operator asked for a toggle on every blocker, so a restriction can be
lifted ON PURPOSE instead of being worked around by editing a constant and
redeploying. This is a bespoke tool for one yard (standing decision 1), every
real run is supervised, and an override is recorded in the run JSON -- which is
strictly better than the two options it replaces: edit-and-redeploy, or don't
run.

The properties that make it safe rather than merely permissive:

* **Off by default.** A caller that omits the parameter dispatches exactly as
  before. This is the constraint the whole hardware-accepted profile rests on.
* **Fail closed on typos.** An unrecognised gate name is refused by schema
  validation, never silently ignored -- a typo must not read as a granted
  override.
* **The refusal is honest.** An overridden gate keeps ``original_passed: False``
  and gains ``overridden: True``. A run that proceeded past a gate can never
  present itself as a run where the gate passed.
* **Nothing else is weakened.** Overriding one gate leaves every other gate
  enforcing.

⚠️ Four gates are deliberately absent from the registry, and NOT as a safety
veto -- the operator asked for everything and got it. They are INCOHERENT to
override: ``stop_primitive_available`` is ``hasattr(coordinator,
"async_stop_manual_motion")`` and an override does not create the method, it
just dispatches motion with no stop path; ``turn_mode_valid`` means there is no
code path to run; and the two ``operator_confirmed_*`` gates ARE the operator's
deliberate act, already exposed as card checkboxes.
"""

from __future__ import annotations

import pytest
import voluptuous as vol

from custom_components.mammotion.services import (
    _NON_OVERRIDABLE_GATES,
    _OVERRIDABLE_GATES,
    RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA,
    RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA,
    _apply_safety_overrides,
)

_BASE = {
    "entity_id": "lawn_mower.test",
    "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}],
}


def _gates(*names: str) -> list[dict[str, object]]:
    return [
        {"name": name, "passed": False, "detail": f"{name} fired"} for name in names
    ]


def test_off_by_default_changes_nothing() -> None:
    """The most important test in the file."""
    gates = _gates("segment_too_long", "mower_ready")

    for overrides in (None, [], ()):
        summary = _apply_safety_overrides(gates, overrides)
        assert summary["any_applied"] is False
        assert summary["applied_names"] == []
        assert all(gate["passed"] is False for gate in gates)
        assert all("overridden" not in gate for gate in gates)

    validated = RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA(dict(_BASE))
    assert validated["safety_overrides"] == []


def test_an_override_clears_only_its_own_gate() -> None:
    """Lifting one gate must leave every other gate enforcing."""
    gates = _gates("segment_too_long", "mower_ready", "ble_link_live")

    _apply_safety_overrides(gates, ["segment_too_long"])
    blockers = [gate["name"] for gate in gates if not gate["passed"]]

    assert blockers == ["mower_ready", "ble_link_live"]


def test_an_overridden_gate_cannot_masquerade_as_a_clean_one() -> None:
    """The run JSON must show both that the gate FIRED and that it was lifted.

    Silently flipping ``passed`` would make an overridden run indistinguishable
    from a clean one -- the exact class of dishonesty this project has been
    bitten by repeatedly.
    """
    gates = _gates("mower_reports_blades_off")

    summary = _apply_safety_overrides(gates, ["mower_reports_blades_off"])
    gate = gates[0]

    assert gate["passed"] is True
    assert gate["original_passed"] is False
    assert gate["overridden"] is True
    # And the rationale travels with it, so the record says what was risked.
    assert summary["applied"][0]["name"] == "mower_reports_blades_off"
    assert "BLADES" in summary["applied"][0]["why"].upper()


def test_a_typo_is_refused_by_the_schema_not_silently_ignored() -> None:
    """Fail closed: a misspelled gate must never read as a granted override."""
    for schema in (
        RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA,
        RAW_PYMAMMOTION_EXECUTE_VECTOR_SEGMENT_SCHEMA,
    ):
        with pytest.raises(vol.Invalid):
            schema({**_BASE, "safety_overrides": ["segment_to_long"]})


def test_the_incoherent_gates_are_refused_at_both_layers() -> None:
    """Not a safety veto -- these cannot be meaningfully overridden."""
    assert {
        "stop_primitive_available",
        "turn_mode_valid",
        "operator_confirmed_blades_off",
        "operator_confirmed_clear_area",
    } == _NON_OVERRIDABLE_GATES
    assert not (_NON_OVERRIDABLE_GATES & set(_OVERRIDABLE_GATES))

    for name in sorted(_NON_OVERRIDABLE_GATES):
        with pytest.raises(vol.Invalid):
            RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA(
                {**_BASE, "safety_overrides": [name]}
            )
        # And the helper refuses it even if it somehow bypassed the schema.
        gates = _gates(name)
        summary = _apply_safety_overrides(gates, [name])
        assert summary["refused"] == [name]
        assert gates[0]["passed"] is False


def test_the_arming_gate_is_not_overridable() -> None:
    """`experimental_motion_disabled` is the ARMING control, not a blocker.

    Overriding it would mean motion without arming, which is the one thing the
    whole gate exists to prevent.
    """
    assert "experimental_motion_disabled" not in _OVERRIDABLE_GATES
    with pytest.raises(vol.Invalid):
        RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA(
            {**_BASE, "safety_overrides": ["experimental_motion_disabled"]}
        )


def test_requesting_an_override_for_a_passing_gate_claims_nothing() -> None:
    """Recorded as `unused` so the run JSON does not imply it did something."""
    gates = [{"name": "segment_too_long", "passed": True, "detail": "ok"}]

    summary = _apply_safety_overrides(gates, ["segment_too_long"])

    assert summary["applied_names"] == []
    assert summary["unused"] == ["segment_too_long"]
    assert summary["any_applied"] is False
    assert "overridden" not in gates[0]


def test_every_registry_entry_explains_what_the_gate_protected() -> None:
    """A gate's NAME never says what it was for, and the card renders `why`."""
    for name, meta in _OVERRIDABLE_GATES.items():
        assert meta["tier"] in {"cap", "night", "sensing", "link", "physical"}, name
        assert len(meta["why"]) > 30, name


def test_the_schema_accepts_every_registered_gate() -> None:
    """The card can offer any registered gate; none may be un-sendable."""
    for name in sorted(_OVERRIDABLE_GATES):
        validated = RAW_PYMAMMOTION_EXECUTE_MULTI_SEGMENT_SCHEMA(
            {**_BASE, "safety_overrides": [name]}
        )
        assert validated["safety_overrides"] == [name]
