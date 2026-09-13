"""Pins for the Phase 1 leg runner's daylight/VIO halt."""  # noqa: INP001

from __future__ import annotations

import datetime

from scripts.phase1_leg_runner import (
    VIO_REQUIRED_STATUS,
    VIO_WINDOW_SECONDS,
    daylight_vio_verdict,
    solar_elevation_degrees,
    window_values,
)

# Real HA history, 2026-09-12. SETUP1 ran in full daylight and landed 5.1 cm
# from target. SETUP2 dispatched 23:55:07Z, ~6 min after sunset, and halted on
# vio_realign_incomplete as VIO collapsed.
SETUP1_DISPATCH = "2026-09-12T17:06:54+00:00"
SETUP1_FEATURES = [
    ("2026-09-12T17:05:00+00:00", "80"),
    ("2026-09-12T17:07:02+00:00", "76"),
    ("2026-09-12T17:07:03+00:00", "80"),
    ("2026-09-12T17:07:06+00:00", "76"),
    ("2026-09-12T17:07:07+00:00", "75"),
    ("2026-09-12T17:07:07+00:00", "77"),
    ("2026-09-12T17:07:08+00:00", "80"),
    ("2026-09-12T17:07:10+00:00", "76"),
    ("2026-09-12T17:07:11+00:00", "80"),
    ("2026-09-12T17:07:11+00:00", "76"),
    ("2026-09-12T17:07:12+00:00", "80"),
    ("2026-09-12T17:07:14+00:00", "77"),
    ("2026-09-12T17:07:15+00:00", "75"),
    ("2026-09-12T17:07:16+00:00", "80"),
    ("2026-09-12T17:07:19+00:00", "77"),
    ("2026-09-12T17:07:20+00:00", "75"),
    ("2026-09-12T17:07:21+00:00", "74"),
    ("2026-09-12T17:07:22+00:00", "80"),
    ("2026-09-12T17:07:24+00:00", "78"),
    ("2026-09-12T17:07:25+00:00", "77"),
    ("2026-09-12T17:07:26+00:00", "73"),
    ("2026-09-12T17:07:27+00:00", "80"),
    ("2026-09-12T17:07:29+00:00", "78"),
    ("2026-09-12T17:07:30+00:00", "77"),
    ("2026-09-12T17:07:31+00:00", "80"),
    ("2026-09-12T17:07:34+00:00", "75"),
    ("2026-09-12T17:07:35+00:00", "73"),
    ("2026-09-12T17:07:36+00:00", "80"),
    ("2026-09-12T17:07:39+00:00", "79"),
    ("2026-09-12T17:07:39+00:00", "76"),
    ("2026-09-12T17:07:40+00:00", "80"),
]
SETUP1_STATUS = [("2026-09-12T17:05:00+00:00", "signal_good")]
SETUP2_DISPATCH = "2026-09-12T23:55:07+00:00"
SETUP2_FEATURES = [
    ("2026-09-12T23:54:00+00:00", "62"),
    ("2026-09-12T23:54:02+00:00", "22"),
    ("2026-09-12T23:54:03+00:00", "17"),
    ("2026-09-12T23:54:04+00:00", "20"),
    ("2026-09-12T23:54:05+00:00", "21"),
    ("2026-09-12T23:54:06+00:00", "34"),
    ("2026-09-12T23:54:07+00:00", "23"),
    ("2026-09-12T23:54:08+00:00", "22"),
    ("2026-09-12T23:54:09+00:00", "27"),
    ("2026-09-12T23:54:10+00:00", "28"),
    ("2026-09-12T23:54:11+00:00", "26"),
    ("2026-09-12T23:54:12+00:00", "20"),
    ("2026-09-12T23:54:13+00:00", "22"),
    ("2026-09-12T23:54:14+00:00", "33"),
    ("2026-09-12T23:54:17+00:00", "35"),
    ("2026-09-12T23:54:17+00:00", "30"),
    ("2026-09-12T23:54:17+00:00", "32"),
    ("2026-09-12T23:54:18+00:00", "31"),
    ("2026-09-12T23:54:18+00:00", "26"),
    ("2026-09-12T23:54:19+00:00", "18"),
    ("2026-09-12T23:54:20+00:00", "22"),
    ("2026-09-12T23:54:21+00:00", "17"),
    ("2026-09-12T23:54:22+00:00", "14"),
    ("2026-09-12T23:54:23+00:00", "17"),
    ("2026-09-12T23:54:24+00:00", "24"),
    ("2026-09-12T23:54:25+00:00", "15"),
    ("2026-09-12T23:54:26+00:00", "22"),
    ("2026-09-12T23:54:27+00:00", "27"),
    ("2026-09-12T23:54:28+00:00", "45"),
    ("2026-09-12T23:54:31+00:00", "47"),
    ("2026-09-12T23:54:34+00:00", "55"),
    ("2026-09-12T23:54:34+00:00", "57"),
    ("2026-09-12T23:55:09+00:00", "46"),
    ("2026-09-12T23:55:11+00:00", "44"),
    ("2026-09-12T23:55:12+00:00", "40"),
    ("2026-09-12T23:55:16+00:00", "42"),
    ("2026-09-12T23:55:17+00:00", "48"),
    ("2026-09-12T23:55:19+00:00", "41"),
    ("2026-09-12T23:55:20+00:00", "47"),
    ("2026-09-12T23:55:22+00:00", "33"),
    ("2026-09-12T23:55:23+00:00", "42"),
    ("2026-09-12T23:55:24+00:00", "40"),
    ("2026-09-12T23:55:25+00:00", "47"),
    ("2026-09-12T23:55:27+00:00", "31"),
    ("2026-09-12T23:55:28+00:00", "32"),
    ("2026-09-12T23:55:29+00:00", "42"),
    ("2026-09-12T23:55:29+00:00", "38"),
    ("2026-09-12T23:55:30+00:00", "43"),
    ("2026-09-12T23:55:31+00:00", "48"),
    ("2026-09-12T23:55:32+00:00", "45"),
    ("2026-09-12T23:55:33+00:00", "27"),
    ("2026-09-12T23:55:34+00:00", "34"),
    ("2026-09-12T23:55:35+00:00", "43"),
    ("2026-09-12T23:55:35+00:00", "48"),
    ("2026-09-12T23:55:36+00:00", "47"),
    ("2026-09-12T23:55:36+00:00", "39"),
    ("2026-09-12T23:55:37+00:00", "27"),
    ("2026-09-12T23:55:38+00:00", "31"),
    ("2026-09-12T23:55:39+00:00", "39"),
    ("2026-09-12T23:55:39+00:00", "36"),
    ("2026-09-12T23:55:40+00:00", "35"),
    ("2026-09-12T23:55:41+00:00", "43"),
    ("2026-09-12T23:55:42+00:00", "46"),
    ("2026-09-12T23:55:43+00:00", "32"),
    ("2026-09-12T23:55:45+00:00", "38"),
    ("2026-09-12T23:55:45+00:00", "43"),
    ("2026-09-12T23:55:46+00:00", "49"),
    ("2026-09-12T23:55:47+00:00", "44"),
    ("2026-09-12T23:55:48+00:00", "46"),
    ("2026-09-12T23:55:49+00:00", "49"),
    ("2026-09-12T23:55:50+00:00", "47"),
    ("2026-09-12T23:55:53+00:00", "41"),
    ("2026-09-12T23:55:53+00:00", "36"),
    ("2026-09-12T23:55:54+00:00", "43"),
    ("2026-09-12T23:55:55+00:00", "46"),
    ("2026-09-12T23:55:56+00:00", "22"),
    ("2026-09-12T23:55:57+00:00", "27"),
    ("2026-09-12T23:55:58+00:00", "32"),
    ("2026-09-12T23:55:59+00:00", "26"),
    ("2026-09-12T23:56:00+00:00", "20"),
    ("2026-09-12T23:56:00+00:00", "18"),
    ("2026-09-12T23:56:01+00:00", "39"),
    ("2026-09-12T23:56:02+00:00", "37"),
    ("2026-09-12T23:56:03+00:00", "18"),
    ("2026-09-12T23:56:04+00:00", "11"),
    ("2026-09-12T23:56:05+00:00", "39"),
    ("2026-09-12T23:56:06+00:00", "36"),
    ("2026-09-12T23:56:07+00:00", "34"),
    ("2026-09-12T23:56:08+00:00", "35"),
]
SETUP2_STATUS = [
    ("2026-09-12T23:54:00+00:00", "signal_good"),
    ("2026-09-12T23:55:12+00:00", "signal_bad"),
    ("2026-09-12T23:55:16+00:00", "signal_good"),
    ("2026-09-12T23:55:50+00:00", "signal_bad"),
    ("2026-09-12T23:55:53+00:00", "signal_none"),
    ("2026-09-12T23:55:54+00:00", "signal_good"),
]


def _series(raw: list[tuple[str, str]]) -> list[tuple[datetime.datetime, str]]:
    return [(datetime.datetime.fromisoformat(ts), v) for ts, v in raw]


def _verdict(dispatch: str, features: list, status: list) -> list[str]:
    when = datetime.datetime.fromisoformat(dispatch)
    return daylight_vio_verdict(
        sun_elevation=solar_elevation_degrees(when),
        features_window=window_values(_series(features), when, VIO_WINDOW_SECONDS),
        status_window=window_values(_series(status), when, VIO_WINDOW_SECONDS),
    )


def test_solar_elevation_matches_home_assistant_sun() -> None:
    """HA's sun.sun read 57.36 deg at 16:41:30Z; sunset was 23:49:30Z (~-0.83)."""
    ha = datetime.datetime(2026, 9, 12, 16, 41, 30, tzinfo=datetime.UTC)
    sunset = datetime.datetime(2026, 9, 12, 23, 49, 30, tzinfo=datetime.UTC)
    assert abs(solar_elevation_degrees(ha) - 57.36) < 0.5
    assert abs(solar_elevation_degrees(sunset) - (-0.83)) < 0.5


def test_daylight_leg_passes() -> None:
    """The leg that ran cleanly in daylight must not be halted."""
    assert _verdict(SETUP1_DISPATCH, SETUP1_FEATURES, SETUP1_STATUS) == []


def test_dusk_leg_halts_on_sun_and_features() -> None:
    """The dusk leg must halt, on both the sun and the feature-minimum clauses."""
    reasons = _verdict(SETUP2_DISPATCH, SETUP2_FEATURES, SETUP2_STATUS)
    assert any("sun elevation" in r for r in reasons)
    assert any("vio_tracked_features" in r for r in reasons)


def test_status_alone_would_not_have_caught_the_dusk_leg() -> None:
    """Pins WHY the feature and sun clauses exist.

    visual_positioning_status read signal_good across the whole look-back window
    before the dusk dispatch; it only flipped after the leg started.
    """
    when = datetime.datetime.fromisoformat(SETUP2_DISPATCH)
    status = window_values(_series(SETUP2_STATUS), when, VIO_WINDOW_SECONDS)
    assert set(status) == {VIO_REQUIRED_STATUS}


def test_window_values_keeps_the_state_in_force_at_window_start() -> None:
    """An unchanged state is still the reading for the whole window."""
    base = datetime.datetime(2026, 9, 12, 12, 0, 0, tzinfo=datetime.UTC)
    series = [(base, "80"), (base + datetime.timedelta(seconds=100), "40")]
    assert window_values(series, base + datetime.timedelta(seconds=90), 60) == ["80"]
    assert window_values(series, base + datetime.timedelta(seconds=110), 60) == [
        "80",
        "40",
    ]
