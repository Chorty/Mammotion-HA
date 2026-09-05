"""The empty `mcu: , ` string that cost a session hours on 2026-09-04.

The mower failed to dock twice. The vendor app said why, five times, in one
line: `Robot orientation unavailable (1309)`, with the manufacturer's own
remedy. `sensor.<mower>_last_error` in Home Assistant read `"mcu: , "` with an
hour-stale timestamp. An operator watching this integration had strictly less
information than one watching the app.

Findings section 6.6 asked whether the code had been available all along and
merely dropped. It had been, twice over:

* The live push carries the code AND its timestamp. `_async_update_event_message`
  parsed it, logged it at DEBUG, and threw it away, keeping only a re-poll of
  the error log -- which is refreshed at setup and on a `sys_status` transition,
  and so was an hour stale.
* `get_error_message` formatted `f"{module}: {implication}, {solution}"` from
  the cloud table. The numeric code appears nowhere in that string, so a row
  with blank localised text renders as punctuation and nothing else.
"""

from __future__ import annotations

import datetime
from functools import partial
from types import SimpleNamespace
from typing import Any

from custom_components.mammotion.coordinator import (
    MammotionDeviceErrorUpdateCoordinator as ErrorCoordinator,
)
from custom_components.mammotion.error_codes import (
    describe_error_code,
    error_code_table,
    lookup_error_code,
)

#: The code the app showed while Home Assistant showed nothing.
ORIENTATION_UNAVAILABLE = 1309


def _blank_cloud_row(module: str = "mcu") -> SimpleNamespace:
    """Return a cloud table row whose localised text is empty -- the observed case."""
    return SimpleNamespace(
        module=module,
        en_implication="",
        en_solution="",
    )


def _error_coordinator(
    *,
    err_code_list: list[int] | None = None,
    err_code_list_time: list[int] | None = None,
    error_codes: dict[str, Any] | None = None,
    notifications: list[dict[str, Any]] | None = None,
    language: str = "en",
) -> Any:
    """Build the minimum surface the error accessors actually touch.

    The accessors are called unbound against this namespace, so the helpers
    they call each other through are bound onto it explicitly -- that keeps the
    test exercising the shipped methods rather than a re-implementation.
    """
    coordinator = SimpleNamespace(
        device_name="Luba-TEST",
        hass=SimpleNamespace(config=SimpleNamespace(language=language)),
        data=SimpleNamespace(
            errors=SimpleNamespace(
                err_code_list=err_code_list or [],
                err_code_list_time=err_code_list_time or [],
                error_codes=error_codes or {},
            )
        ),
        _notification_codes=notifications or [],
    )
    for name in ("_latest_fault", "_cloud_error_text"):
        setattr(
            coordinator,
            name,
            partial(getattr(ErrorCoordinator, name), coordinator),
        )
    return coordinator


# ---------------------------------------------------------------------------
# The bundled offline table
# ---------------------------------------------------------------------------


def test_the_code_the_app_showed_is_decodable_offline() -> None:
    """1309 appeared nowhere in the integration or the pinned pymammotion."""
    entry = lookup_error_code(ORIENTATION_UNAVAILABLE)

    assert entry is not None
    assert "Heading calibration failed" in entry["implication"]
    assert "task area" in entry["solution"]


def test_the_bundled_table_loads_and_is_not_trivially_small() -> None:
    """A silently empty table would reintroduce the failure it prevents."""
    table = error_code_table()

    assert len(table) > 400
    assert all(row["implication"] for row in table.values())


# ---------------------------------------------------------------------------
# The message never hides the number again
# ---------------------------------------------------------------------------


def test_a_blank_cloud_row_no_longer_renders_as_punctuation() -> None:
    """🚨 The exact regression: module "mcu", both text fields empty."""
    message = describe_error_code(
        ORIENTATION_UNAVAILABLE, implication="", solution="", module="mcu"
    )

    assert message != "mcu: , "
    assert "1309" in message
    assert "Heading calibration failed" in message


def test_an_entirely_unknown_code_still_reaches_the_operator() -> None:
    """A full mapping table was the bonus; the number was the requirement."""
    message = describe_error_code(987654, module="")

    assert message.startswith("987654")
    assert "no description available" in message


def test_cloud_text_wins_over_the_bundle_when_it_has_any() -> None:
    """The cloud table is localised and current; the bundle is the fallback."""
    message = describe_error_code(
        ORIENTATION_UNAVAILABLE,
        implication="Ausrichtung nicht verfügbar",
        solution="In eine Zone fahren",
        module="mcu",
    )

    assert "Ausrichtung nicht verfügbar" in message
    assert "Heading calibration failed" not in message


def test_the_code_leads_every_message() -> None:
    """Whatever the text, an operator can always look the number up."""
    for code in (1309, 131, 987654):
        assert describe_error_code(code).startswith(str(code))


# ---------------------------------------------------------------------------
# The live push is kept instead of discarded
# ---------------------------------------------------------------------------


def test_a_warning_code_push_is_recorded() -> None:
    """`[{"c":-2801,"ct":1,"ft":...}]` -- the shape in the shipped comment."""
    coordinator = _error_coordinator()

    recorded = ErrorCoordinator._record_notification_codes(
        coordinator,
        [
            {"c": -2801, "ct": 1, "ft": 1731493734000},
            {"c": -1008, "ct": 1, "ft": 1731493734000},
        ],
    )

    assert recorded == 2
    assert [entry["code"] for entry in coordinator._notification_codes] == [2801, 1008]
    assert coordinator._notification_codes[0]["timestamp"] == 1731493734.0


def test_a_notification_push_is_recorded() -> None:
    """`{"localTime":..., "code":"1002"}` -- the other shape in the wild."""
    coordinator = _error_coordinator()

    recorded = ErrorCoordinator._record_notification_codes(
        coordinator, {"localTime": 1725159492000, "code": "1309"}
    )

    assert recorded == 1
    assert coordinator._notification_codes[0]["code"] == ORIENTATION_UNAVAILABLE


def test_a_push_beats_an_hour_stale_error_log() -> None:
    """The whole 6.6 failure, end to end.

    The polled log holds an old fault with blank cloud text; the live push holds
    1309. The operator must see 1309, with words.
    """
    stale = datetime.datetime(2026, 9, 5, 0, 24, tzinfo=datetime.UTC).timestamp()
    fresh = datetime.datetime(2026, 9, 5, 1, 24, tzinfo=datetime.UTC).timestamp()
    coordinator = _error_coordinator(
        err_code_list=[2801],
        err_code_list_time=[int(stale)],
        error_codes={"2801": _blank_cloud_row(), "1309": _blank_cloud_row()},
        notifications=[{"code": ORIENTATION_UNAVAILABLE, "timestamp": fresh}],
    )

    assert ErrorCoordinator.get_error_code(coordinator, 1) == ORIENTATION_UNAVAILABLE
    message = ErrorCoordinator.get_error_message(coordinator, 1)
    assert "1309" in message
    assert "Heading calibration failed" in message
    reported = ErrorCoordinator.get_error_time(coordinator, 1)
    assert reported is not None
    assert reported.timestamp() == fresh


def test_the_polled_log_is_used_when_no_push_arrived() -> None:
    """A BLE-primary mower is not guaranteed to receive a cloud push at all."""
    coordinator = _error_coordinator(
        err_code_list=[-1309, 0, 0],
        err_code_list_time=[1725159492, 0, 0],
        error_codes={},
    )

    assert ErrorCoordinator.get_error_code(coordinator, 1) == ORIENTATION_UNAVAILABLE
    assert "Heading calibration failed" in ErrorCoordinator.get_error_message(
        coordinator, 1
    )


def test_zero_slots_in_the_log_are_not_faults() -> None:
    """The device reports ten slots and pads with zeros."""
    coordinator = _error_coordinator(
        err_code_list=[0, 0, 0], err_code_list_time=[0] * 3
    )

    assert ErrorCoordinator.get_error_code(coordinator, 1) == 0
    assert ErrorCoordinator.get_error_message(coordinator, 1) == "No Error"
    assert ErrorCoordinator.get_error_time(coordinator, 1) is None


def test_every_slot_is_exposed_not_just_the_first() -> None:
    """Only `err_code_list[0]` was ever surfaced; a burst hid behind it."""
    coordinator = _error_coordinator(
        err_code_list=[1309, 131, 0, 0],
        err_code_list_time=[1725159492, 1725159400, 0, 0],
        notifications=[{"code": 1309, "timestamp": 1725159492.0}],
    )

    snapshot = ErrorCoordinator.error_log_snapshot(coordinator)

    assert [entry["code"] for entry in snapshot["logged_faults"]] == [1309, 131]
    assert [entry["code"] for entry in snapshot["pushed_faults"]] == [1309]
    assert "wheel hub motor" in snapshot["logged_faults"][1]["message"]
    assert snapshot["cloud_error_table_loaded"] is False


def test_a_malformed_push_records_nothing_rather_than_a_zero() -> None:
    """A junk payload must not manufacture a fault or crash the update."""
    coordinator = _error_coordinator()

    assert (
        ErrorCoordinator._record_notification_codes(coordinator, [{"c": None}, "junk"])
        == 0
    )
    assert coordinator._notification_codes == []
