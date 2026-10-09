"""The companion gate helper must arm both flags or neither, and prove it.

Every HTTP call is answered by an in-memory fake, so nothing is sent anywhere.
The fake models the companion: one options flow whose schema is exactly the two
gate fields, and an export whose ``experimental_motion`` echoes the stored
options.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
COMPANION = "01M47EMDWW2JJYSFPNB70VKJ89"


def _load() -> ModuleType:
    """Import the helper by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "ha_set_experimental_motion",
        REPO / "scripts" / "ha_set_experimental_motion.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeHA:
    """Just enough of the HA REST API for the helper."""

    def __init__(self) -> None:
        """Start with one base entry, one loaded companion, both flags off."""
        self.entries: list[dict[str, Any]] = [
            {"entry_id": "BASE", "domain": "mammotion", "state": "loaded"},
            {"entry_id": COMPANION, "domain": "mammotion_motion", "state": "loaded"},
        ]
        self.options = {
            "enable_experimental_motion": False,
            "enable_supervised_qualification": False,
        }
        self.schema = [{"name": n} for n in self.options]
        self.calls: list[tuple[str, str, Any]] = []
        self.submissions: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        #: Fault injection: drop this field from the next applied submission.
        self.ignore_field: str | None = None

    def __call__(self, path: str, payload: Any = None, *, method: str | None = None):
        """Answer one REST call."""
        method = method or ("GET" if payload is None else "POST")
        self.calls.append((method, path, payload))
        if path == "/api/config/config_entries/entry":
            return self.entries
        if path.startswith("/api/services/"):
            assert path.startswith("/api/services/mammotion_motion/"), path
            return {
                "service_response": {
                    "experimental_motion": {
                        "enabled": self.options["enable_experimental_motion"],
                        "supervised_qualification": self.options[
                            "enable_supervised_qualification"
                        ],
                        "real_motion_allowed": False,
                        "blockers": [],
                        "active_session": None,
                    }
                }
            }
        if path == "/api/config/config_entries/options/flow":
            assert payload == {"handler": COMPANION}
            return {"type": "form", "flow_id": "F1", "data_schema": self.schema}
        if path.startswith("/api/config/config_entries/options/flow/"):
            if method == "DELETE":
                self.deleted.append(path)
                return {}
            self.submissions.append(dict(payload))
            for key, value in payload.items():
                if key != self.ignore_field:
                    self.options[key] = value
            return {"type": "create_entry"}
        raise AssertionError(f"unexpected call {method} {path}")


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, FakeHA]:
    """Load the helper with its transport replaced by the fake."""
    module = _load()
    ha = FakeHA()
    monkeypatch.setattr(module, "_api", ha)
    return module, ha


def _run(module: ModuleType, monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr("sys.argv", ["ha_set_experimental_motion.py", *argv])
    return module.main()


def test_on_arms_both_flags_with_exactly_two_fields(
    fake: tuple[ModuleType, FakeHA], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The submission carries only the two gate fields, both true."""
    module, ha = fake

    assert _run(module, monkeypatch, "on", "--yes") == 0

    assert ha.submissions == [
        {"enable_experimental_motion": True, "enable_supervised_qualification": True}
    ]


def test_off_disarms_both_flags(
    fake: tuple[ModuleType, FakeHA], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off clears both, including a qualification flag left on alone."""
    module, ha = fake
    ha.options["enable_supervised_qualification"] = True

    assert _run(module, monkeypatch, "off") == 0

    assert ha.options == {
        "enable_experimental_motion": False,
        "enable_supervised_qualification": False,
    }


def test_a_partial_arm_fails_and_is_disarmed_before_exit(
    fake: tuple[ModuleType, FakeHA], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A readback with one flag on is refused, and OFF is submitted at once."""
    module, ha = fake
    ha.ignore_field = "enable_supervised_qualification"

    with pytest.raises(SystemExit) as err:
        _run(module, monkeypatch, "on", "--yes")

    assert err.value.code not in (0, None)
    assert ha.submissions[-1] == {
        "enable_experimental_motion": False,
        "enable_supervised_qualification": False,
    }
    assert ha.options["enable_experimental_motion"] is False


def test_an_unexpected_flow_schema_is_aborted_without_submitting(
    fake: tuple[ModuleType, FakeHA], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema this script does not know is deleted, never guessed at."""
    module, ha = fake
    ha.schema = [*ha.schema, {"name": "prefer_ble_over_wifi"}]

    with pytest.raises(SystemExit):
        _run(module, monkeypatch, "on", "--yes")

    assert ha.submissions == []
    assert ha.deleted  # both the arm and the fail-closed disarm flows aborted
    assert ha.options["enable_experimental_motion"] is False


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([{"entry_id": "BASE", "domain": "mammotion", "state": "loaded"}], "No "),
        (
            [
                {"entry_id": "A", "domain": "mammotion_motion", "state": "loaded"},
                {"entry_id": "B", "domain": "mammotion_motion", "state": "loaded"},
            ],
            "Multiple",
        ),
        (
            [
                {
                    "entry_id": COMPANION,
                    "domain": "mammotion_motion",
                    "state": "setup_retry",
                }
            ],
            "not loaded",
        ),
    ],
)
def test_entry_resolution_fails_closed(
    fake: tuple[ModuleType, FakeHA],
    monkeypatch: pytest.MonkeyPatch,
    entries: list[dict[str, Any]],
    message: str,
) -> None:
    """Wrong domain, a duplicate or an unloaded companion never reaches a flow."""
    module, ha = fake
    ha.entries = entries

    with pytest.raises(SystemExit, match=message):
        _run(module, monkeypatch, "on", "--yes")

    assert ha.submissions == []


def test_status_and_already_in_state_never_open_a_flow(
    fake: tuple[ModuleType, FakeHA], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-only actions stay read-only."""
    module, ha = fake

    assert _run(module, monkeypatch, "status") == 0
    assert _run(module, monkeypatch, "off") == 0

    assert not any("options/flow" in path for _m, path, _p in ha.calls)


def test_a_missing_qualification_report_is_not_treated_as_false(
    fake: tuple[ModuleType, FakeHA], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An export without the field is unproved state, not 'disarmed'."""
    module, _ha = fake
    monkeypatch.setattr(
        module,
        "_api",
        lambda path, payload=None, *, method=None: {
            "service_response": {"experimental_motion": {"enabled": False}}
        },
    )

    with pytest.raises(SystemExit, match="supervised_qualification"):
        _run(module, monkeypatch, "status")


def test_runners_route_retained_calls_to_the_companion() -> None:
    """No runner still sends a retained service to the upstream domain."""
    helpers = (REPO / "scripts" / "mammotion_ha_helpers.py").read_text()
    assert 'MOTION_DOMAIN = "mammotion_motion"' in helpers
    for name in ("phase1_leg_runner.py", "rtk_square_return.py"):
        source = (REPO / "scripts" / name).read_text()
        assert '"mammotion"' not in source, name
        assert "MOTION_DOMAIN" in source, name
