"""Pins for scripts/s2_leg.py: offline scoring and the run's safety ordering.

`score` is driven against S1's real evidence (private docs/fixtures/s2/s1, see its README)
and against synthetic attempt directories. `run` is driven with every network
and subprocess call replaced by fakes; nothing here contacts Home Assistant.
"""  # noqa: INP001

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import s2_leg
from scripts.s2_leg import (
    EXIT_DISARM_UNPROVED,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_PRE_DISPATCH,
    EXIT_REFUSED,
    EXIT_RUNNER_NONZERO,
    HELPER,
    RUNNER,
    S2Run,
    admission,
    build_parser,
    normalize_rows,
    parse_ts,
    pass_condition,
    score_dir,
)

#: S1's real evidence is private: it lives in the local-only docs/ repo, and the
#: tests that read it are skipped when docs/ is absent (tests_ha/conftest.py).
S1_DIR = Path(__file__).resolve().parents[2] / "docs" / "fixtures" / "s2" / "s1"
UTC = dt.UTC


# ------------------------------------------------------------------ S1 (real)


def test_s1_scores_criteria_1_2_4_pass_and_3_unevaluable_for_missing_brackets() -> None:
    """S1 scores criteria 1 2 4 pass and 3 unevaluable for missing brackets."""
    verdict = score_dir(S1_DIR)
    assert verdict["criterion_1"]["status"] == "PASS"
    assert verdict["criterion_1"]["landing_error_m"] == pytest.approx(0.1421, abs=1e-4)
    assert verdict["criterion_2"]["status"] == "PASS"
    assert verdict["criterion_2"]["stops_confirmed"] == 6
    assert verdict["criterion_2"]["timing_samples_in_window"] == 43
    assert verdict["criterion_4"]["status"] == "PASS"
    assert verdict["criterion_4"]["raw_after_reference_s"] == pytest.approx(
        19.418, abs=1e-3
    )
    c3 = verdict["criterion_3"]
    assert c3["status"] == "UNEVALUABLE"
    assert any("brackets" in reason for reason in c3["unevaluable"])
    # It reports, never infers: no admission or pass condition was computed.
    assert "admission" not in c3
    assert "pass_condition" not in c3
    assert verdict["outcome"] == "INCONCLUSIVE"
    assert verdict["precedence_path"][-1]["step"] == "s5.1 criterion 3 unevaluable"


def test_s1_window_matches_the_recorded_criterion3_window() -> None:
    """S1 window matches the recorded criterion3 window."""
    leg = json.loads((S1_DIR / "runner" / "leg_S1.json").read_text())
    window, why = s2_leg.criterion3_window(leg)
    assert why is None
    assert window is not None
    assert window["start"] == parse_ts("2026-10-06T21:01:24.589714+00:00")
    assert window["end"] == parse_ts("2026-10-06T21:01:56.582290+00:00")


def test_s1_admission_given_the_inferred_postleg_before() -> None:
    """With the post-leg 'before' INFERRED in this test (S1 never recorded it).

    S1's README says the span rule would read 37 against 37 with the export time
    inferred from receipt_age_s. Under Amendment 5 s2 (lower bound
    command_results[0].sent_at_utc, exclusive; advance from the warm-up's fresh
    sequence 414) that is 36 rows against an advance of 36: admitted. The
    driving-window pass condition then misses on the 7.02 s gap.
    """
    leg = json.loads((S1_DIR / "runner" / "leg_S1.json").read_text())
    post = json.loads((S1_DIR / "postleg.json").read_text())["attempts"][0]["export"]
    stamps = json.loads((S1_DIR / "tracker_history.json").read_text())["response"]
    rows = normalize_rows(
        [
            [
                {
                    "state": "not_home",
                    "attributes": {"latitude": 34.0, "longitude": -84.7},
                    "last_updated": stamp,
                }
                for stamp in stamps
            ]
        ]
    )
    inferred_before = parse_ts("2026-10-06T21:02:01.400000+00:00")
    adm = admission(leg, inferred_before, post, rows)
    assert adm["unevaluable"] == []
    assert adm["advance"] == 36
    assert adm["rows"] == 36
    assert adm["admitted"] is True
    window, _ = s2_leg.criterion3_window(leg)
    result = pass_condition(
        [r["last_updated"] for r in adm["counted_rows"]], window["start"], window["end"]
    )
    assert result["pass"] is False
    assert result["payloads"] == 29
    assert result["max_interval_s"] == pytest.approx(7.019352, abs=1e-6)
    assert result["first_payload_after_start_s"] == pytest.approx(1.003649, abs=1e-6)
    assert result["end_after_last_payload_s"] == pytest.approx(0.677597, abs=1e-6)


# ------------------------------------------------------------ synthetic dirs

T0 = dt.datetime(2026, 10, 7, 15, 0, 0, tzinfo=UTC)


def at(seconds: float) -> str:
    """Return the ISO time ``seconds`` after T0."""
    return (T0 + dt.timedelta(seconds=seconds)).isoformat()


def export(*, seq: int, epoch: int, age: float, armed: bool = False) -> dict[str, Any]:
    """Build a minimal export_runtime_state payload."""
    return {
        "position_pipeline": {
            "latest_sequence": seq,
            "latest_epoch": epoch,
            "receipt_age_s": age,
            "presentation_stream_replacements": 0,
        },
        "experimental_motion": {
            "enabled": armed,
            "supervised_qualification": armed,
            "real_motion_allowed": armed,
            "blockers": [] if armed else ["experimental_motion_disabled"],
        },
    }


def command(sent: float, *, stop_ok: bool = True) -> dict[str, Any]:
    """Build one motion command record."""
    return {
        "index": 1,
        "phase": "linear_forward_to_target",
        "command": "send_movement",
        "kwargs": {"linear_speed": 400, "angular_speed": 0, "t": sent},
        "sent_at_utc": at(sent),
        "ok": True,
        "motion_refresh": {"elapsed_ms": 1300.0},
        "stop_result": {"attempted": True, "ok": stop_ok, "duration_ms": 150.0},
    }


def tracker_row(seconds: float, state: str = "not_home") -> dict[str, Any]:
    """Build one recorder history row."""
    attrs = (
        {}
        if state in ("unavailable", "unknown")
        else {"latitude": 34.02, "longitude": -84.77}
    )
    return {"state": state, "attributes": attrs, "last_updated": at(seconds)}


#: Driving window (60.0, 65.45]: commands at 60.0 and 64.0, 1300 ms + 150 ms.
DEFAULT_ROWS = [60.5 + i for i in range(11)]  # 60.5 .. 70.5


def make_dir(  # noqa: PLR0913
    root: Path,
    *,
    rows: list[float] | None = None,
    extra_rows: list[dict[str, Any]] | None = None,
    post_seq: int | None = None,
    post_epoch: int = 3,
    warm_ok: bool = True,
    stop_reason: str = "target_reached",
    commands_sent: int = 2,
    final_xy: tuple[float, float] = (1.05, 1.0),
    raw_at: float = 95.0,
    timing_samples: list[dict[str, Any]] | None = None,
    operator_stop: bool | None = None,
    postleg_attempts: list[dict[str, Any]] | None = None,
) -> Path:
    """Write one attempt directory in the run's layout."""
    rows = DEFAULT_ROWS if rows is None else rows
    fresh_seq = 100
    root.mkdir(parents=True, exist_ok=True)
    (root / "runner").mkdir(exist_ok=True)
    leg = {
        "stop_reason": stop_reason,
        "commands_sent": commands_sent,
        "target": {"x": 1.0, "y": 1.0},
        "final_telemetry": {
            "position": {"x": final_xy[0], "y": final_xy[1], "rtk_status_label": "Fix"}
        },
        "samples": [],
        "command_results": [command(60.0), command(64.0)] if commands_sent else [],
        "position_feed_warmup": {
            "ok": warm_ok,
            "fresh_sequence": fresh_seq,
            "fresh_epoch": 3,
        },
    }
    write = lambda name, obj: (root / name).write_text(json.dumps(obj))  # noqa: E731
    write("runner/leg_S2.json", leg)
    history_rows = [
        tracker_row(-120.0),  # HA's synthetic start-time row: never counted
        tracker_row(59.6),  # the warm-up receipt: before the exclusive lower bound
        *[tracker_row(t) for t in rows],
        *(extra_rows or []),
    ]
    write(
        "tracker_history.json",
        {
            "query": {
                "start_time": at(-120.0),
                "end_time": at(100.0),
                "significant_changes_only": 0,
            },
            "response": [history_rows],
        },
    )
    seq = fresh_seq + len(rows) if post_seq is None else post_seq
    write(
        "prearm.json",
        {
            "attempts": [
                {
                    "before_utc": at(0.0),
                    "after_utc": at(0.5),
                    "export": export(seq=fresh_seq - 1, epoch=3, age=200.0),
                }
            ],
            "selected_index": 0,
        },
    )
    attempts = postleg_attempts or [
        {
            "before_utc": at(75.0),
            "after_utc": at(75.5),
            "export": export(seq=seq, epoch=post_epoch, age=4.5),
        }
    ]
    write("postleg.json", {"attempts": attempts})
    write("arm.json", {"before_utc": at(1.0), "after_utc": at(3.0), "rc": 0})
    write(
        "disarm.json",
        {
            "attempts": [{"before_utc": at(72.0), "after_utc": at(74.0), "rc": 0}],
            "proved": True,
        },
    )
    write(
        "raw_gate.json",
        {
            "before_utc": at(raw_at),
            "entries": [
                {"domain": "mammotion", "flags": {"enable_experimental_motion": False}},
                {
                    "domain": "mammotion_motion",
                    "flags": {
                        "enable_experimental_motion": False,
                        "enable_supervised_qualification": False,
                    },
                },
            ],
        },
    )
    write(
        "live_final.json",
        {
            "before_utc": at(raw_at + 1),
            "after_utc": at(raw_at + 2),
            "export": export(seq=seq, epoch=post_epoch, age=20.0),
        },
    )
    write("timing_post.json", {"samples": timing_samples or []})
    write(
        "summary.json",
        {
            "arm": {"rc": 0},
            "readback": {"ok": True},
            "runner": {"rc": 0},
            "disarm": {"proved": True},
        },
    )
    if operator_stop is not None:
        write("operator_record.json", {"operator_stop_mid_leg": operator_stop})
    return root


def test_matched_counts_and_good_intervals_pass(tmp_path: Path) -> None:
    """Matched counts and good intervals pass."""
    verdict = score_dir(make_dir(tmp_path, operator_stop=False))
    c3 = verdict["criterion_3"]
    assert c3["admission"]["rows"] == 11 == c3["admission"]["advance"]
    assert c3["synthetic_start_rows_excluded"] == 1
    assert c3["pass_condition"]["payloads"] == 5
    assert c3["status"] == "PASS"
    assert verdict["outcome"] == "PASS", verdict["reason"]


def test_matched_counts_with_a_gap_is_a_miss_and_warmup_ok_makes_it_fail(
    tmp_path: Path,
) -> None:
    """Matched counts with a gap is a miss and warmup ok makes it fail."""
    rows = [r for r in DEFAULT_ROWS if r not in (62.5, 63.5)]  # 61.5 -> 64.5 = 3.0 s
    verdict = score_dir(make_dir(tmp_path, rows=rows, operator_stop=False))
    c3 = verdict["criterion_3"]
    assert c3["admission"]["admitted"] is True
    assert c3["pass_condition"]["max_interval_s"] == pytest.approx(3.0)
    assert c3["status"] == "MISS"
    assert verdict["outcome"] == "FAIL"
    assert (
        verdict["precedence_path"][-1]["step"]
        == "s5.3 criterion 3 miss with warm-up ok"
    )


def test_miss_without_warmup_ok_is_not_scored(tmp_path: Path) -> None:
    """Miss without warmup ok is not scored."""
    rows = [r for r in DEFAULT_ROWS if r not in (62.5, 63.5)]
    verdict = score_dir(
        make_dir(tmp_path, rows=rows, warm_ok=False, operator_stop=False)
    )
    assert verdict["outcome"] == "BUILD_MISMATCH_NOT_SCORED"


def test_unavailable_row_in_span_is_unevaluable(tmp_path: Path) -> None:
    """Unavailable row in span is unevaluable."""
    verdict = score_dir(
        make_dir(
            tmp_path, extra_rows=[tracker_row(66.0, "unavailable")], operator_stop=False
        )
    )
    c3 = verdict["criterion_3"]
    assert c3["status"] == "UNEVALUABLE"
    assert any("unavailable/unknown" in r for r in c3["unevaluable"])
    assert verdict["outcome"] == "INCONCLUSIVE"


def test_epoch_change_is_unevaluable(tmp_path: Path) -> None:
    """Epoch change is unevaluable."""
    verdict = score_dir(make_dir(tmp_path, post_epoch=4, operator_stop=False))
    assert verdict["criterion_3"]["status"] == "UNEVALUABLE"
    assert any("epoch changed" in r for r in verdict["criterion_3"]["unevaluable"])
    assert verdict["outcome"] == "INCONCLUSIVE"


def test_count_mismatch_is_unevaluable(tmp_path: Path) -> None:
    """Count mismatch is unevaluable."""
    verdict = score_dir(make_dir(tmp_path, post_seq=113, operator_stop=False))
    assert any("count mismatch" in r for r in verdict["criterion_3"]["unevaluable"])
    assert verdict["outcome"] == "INCONCLUSIVE"


def test_operator_stop_precedes_a_criterion3_miss(tmp_path: Path) -> None:
    """Operator stop precedes a criterion3 miss."""
    rows = [r for r in DEFAULT_ROWS if r not in (62.5, 63.5)]
    verdict = score_dir(make_dir(tmp_path, rows=rows, operator_stop=True))
    assert verdict["criterion_3"]["status"] == "MISS"
    assert verdict["outcome"] == "INCONCLUSIVE"
    assert verdict["precedence_path"][-1]["step"] == "s5.2 operator stop mid-leg"


def test_unevaluable_precedes_operator_stop(tmp_path: Path) -> None:
    """Unevaluable precedes operator stop."""
    verdict = score_dir(make_dir(tmp_path, post_epoch=4, operator_stop=True))
    assert verdict["precedence_path"][-1]["step"] == "s5.1 criterion 3 unevaluable"


def test_unrecorded_operator_stop_is_undetermined_never_assumed(tmp_path: Path) -> None:
    """Unrecorded operator stop is undetermined never assumed."""
    verdict = score_dir(make_dir(tmp_path))
    assert verdict["outcome"] == "UNDETERMINED"
    assert score_dir(tmp_path, operator_stop=False)["outcome"] == "PASS"


def test_closing_export_skips_a_bracket_with_a_row_inside(tmp_path: Path) -> None:
    """Closing export skips a bracket with a row inside."""
    seq = 100 + len(DEFAULT_ROWS) + 1
    attempts = [
        {
            "before_utc": at(73.0),
            "after_utc": at(73.5),
            "export": export(seq=seq, epoch=3, age=3.2),
        },
        {
            "before_utc": at(75.0),
            "after_utc": at(75.5),
            "export": export(seq=seq, epoch=3, age=4.5),
        },
    ]
    root = make_dir(
        tmp_path,
        extra_rows=[tracker_row(73.2)],
        postleg_attempts=attempts,
        operator_stop=False,
    )
    c3 = score_dir(root)["criterion_3"]
    assert c3["postleg_checks"][0]["rows_inside_bracket"] == [at(73.2)]
    assert c3["closing_index"] == 1
    assert c3["admission"]["rows"] == 12 == c3["admission"]["advance"]


def test_low_receipt_age_never_closes_the_span(tmp_path: Path) -> None:
    """Low receipt age never closes the span."""
    attempts = [
        {
            "before_utc": at(75.0),
            "after_utc": at(75.5),
            "export": export(seq=111, epoch=3, age=2.9),
        }
    ]
    verdict = score_dir(
        make_dir(tmp_path, postleg_attempts=attempts, operator_stop=False)
    )
    assert verdict["criterion_3"]["status"] == "UNEVALUABLE"


def test_queue_start_timeout_counts_only_inside_arm_disarm(tmp_path: Path) -> None:
    """Queue start timeout counts only inside arm disarm."""
    outside = {
        "recorded_at_utc": at(-30.0),
        "outcome": "queue_start_timeout",
        "command": "send_movement",
    }
    inside = {
        "recorded_at_utc": at(61.0),
        "outcome": "queue_start_timeout",
        "command": "send_movement",
    }
    assert (
        score_dir(
            make_dir(tmp_path / "a", timing_samples=[outside], operator_stop=False)
        )["outcome"]
        == "PASS"
    )
    verdict = score_dir(
        make_dir(tmp_path / "b", timing_samples=[outside, inside], operator_stop=False)
    )
    assert verdict["criterion_2"]["status"] == "FAIL"
    assert verdict["outcome"] == "FAIL"


def test_raw_read_under_15_s_after_disarm_is_not_proved(tmp_path: Path) -> None:
    """Raw read under 15 s after disarm is not proved."""
    verdict = score_dir(make_dir(tmp_path, raw_at=80.0, operator_stop=False))
    assert verdict["criterion_4"]["status"] == "FAIL"
    assert verdict["outcome"] == "FAIL"


def test_nothing_sent_classes(tmp_path: Path) -> None:
    """Nothing sent classes."""
    feed = score_dir(
        make_dir(tmp_path / "a", commands_sent=0, stop_reason="position_feed_not_live")
    )
    assert feed["outcome"] == "INCONCLUSIVE"
    other = score_dir(
        make_dir(tmp_path / "b", commands_sent=0, stop_reason="safety_gates_failed")
    )
    assert other["outcome"] == "FAIL"


def test_pre_dispatch_attempt_still_needs_a_proved_raw_disarm(tmp_path: Path) -> None:
    """A readback timeout is INCONCLUSIVE only if the disarm is proved in RAW."""
    root = make_dir(tmp_path)
    (root / "runner" / "leg_S2.json").unlink()
    (root / "summary.json").write_text(
        json.dumps(
            {
                "arm": {"rc": 0},
                "readback": {"ok": False},
                "runner": {"rc": None},
                "disarm": {"proved": True},
            }
        )
    )
    assert score_dir(root)["outcome"] == "INCONCLUSIVE"
    raw = json.loads((root / "raw_gate.json").read_text())
    raw["entries"][1]["flags"]["enable_supervised_qualification"] = True
    (root / "raw_gate.json").write_text(json.dumps(raw))
    verdict = score_dir(root)
    assert verdict["outcome"] == "FAIL"
    assert verdict["precedence_path"][-1]["step"] == "disarm not proved"


def test_landing_outside_tolerance_fails(tmp_path: Path) -> None:
    """Landing outside tolerance fails."""
    verdict = score_dir(make_dir(tmp_path, final_xy=(1.2, 1.0), operator_stop=False))
    assert verdict["criterion_1"]["status"] == "FAIL"
    assert verdict["outcome"] == "FAIL"


def test_debug_only_cause_is_inconclusive(tmp_path: Path) -> None:
    """Amendment 6 s3: a debug-only-caused criterion-3 miss is INCONCLUSIVE (step 2a)."""
    rows = [r for r in DEFAULT_ROWS if r not in (62.5, 63.5)]
    root = make_dir(tmp_path, rows=rows, operator_stop=False)
    verdict = score_dir(root, debug_only_cause=True)
    assert verdict["outcome"] == "INCONCLUSIVE"


def test_ble_prefixed_stop_after_send_is_inconclusive(tmp_path: Path) -> None:
    """Amendment 6 s11: any ble_* stop after a send, stops confirmed, is INCONCLUSIVE."""
    verdict = score_dir(
        make_dir(tmp_path, stop_reason="ble_send_stalled", operator_stop=False)
    )
    assert verdict["outcome"] == "INCONCLUSIVE"


def test_ble_prefixed_stop_with_nothing_sent_is_inconclusive(tmp_path: Path) -> None:
    """Amendment 6 s11: a ble_* refusal with nothing sent is pre-dispatch INCONCLUSIVE."""
    verdict = score_dir(
        make_dir(tmp_path, commands_sent=0, stop_reason="ble_transport_not_usable")
    )
    assert verdict["outcome"] == "INCONCLUSIVE"


def test_unproved_disarm_outranks_an_unevaluable_criterion_3(tmp_path: Path) -> None:
    """Amendment 6 s4: an unproved disarm is checked first and is a FAIL."""
    verdict = score_dir(
        make_dir(tmp_path, raw_at=80.0, post_epoch=4, operator_stop=False)
    )
    assert verdict["criterion_3"]["status"] == "UNEVALUABLE"
    assert verdict["outcome"] == "FAIL"


# --------------------------------------------------------------- run (fakes)


class FakeClock:
    """Deterministic clock."""

    def __init__(self, start: dt.datetime) -> None:
        """Set up the fake."""
        self.start = start
        self.t = 0.0

    def monotonic(self) -> float:
        """Return fake monotonic seconds."""
        return self.t

    def sleep(self, seconds: float) -> None:
        """Advance the fake clock."""
        if seconds > 0:
            self.t += seconds

    def utcnow(self) -> dt.datetime:
        """Return the fake UTC time."""
        return self.start + dt.timedelta(seconds=self.t)


class FakeHost:
    """HA stand-in that records every call."""

    def __init__(
        self, clock: FakeClock, log: list[str], *, readback_never: bool = False
    ) -> None:
        """Set up the fake."""
        self.clock = clock
        self.log = log
        self.armed = False
        self.readback_never = readback_never
        self.syslog: list[dict[str, Any]] = []

    def host_utc(self) -> str:
        """Return the fake host UTC."""
        self.clock.t += 0.01
        return self.clock.utcnow().isoformat()

    def export(self) -> dict[str, Any]:
        """Return an export reflecting the fake gate."""
        self.log.append("export")
        self.clock.t += 0.2
        exp = export(seq=50, epoch=1, age=200.0, armed=self.armed)
        if self.armed and self.readback_never:
            exp["experimental_motion"]["real_motion_allowed"] = False
        return exp

    def service(self, domain: str, service: str, data: dict[str, Any], **_: Any) -> Any:
        """Record a service call."""
        self.log.append(
            f"service:{domain}.{service}:{json.dumps(data, sort_keys=True)}"
        )
        return {}

    def timing_report(self) -> dict[str, Any]:
        """Return an empty timing report."""
        return {"samples": []}

    def history(self, *_: Any) -> Any:
        """Return empty history."""
        self.log.append("history")
        return [[]]

    def system_log(self) -> Any:
        """Return the configured system log."""
        return self.syslog

    def logger_info(self) -> Any:
        """Return no logger levels."""
        return []

    def raw_gate_flags(self) -> list[dict[str, Any]]:
        """Record a RAW read."""
        self.log.append("raw")
        return []

    def core_log(self, lines: int = 0) -> str:
        """Return a stub log."""
        return "log"


class FakeProc:
    """Subprocess stand-in for the gate helper and the runner."""

    def __init__(
        self,
        host: FakeHost,
        log: list[str],
        *,
        on_rc: int = 0,
        off_rc: int = 0,
        runner: Any = None,
    ) -> None:
        """Set up the fake."""
        self.host = host
        self.log = log
        self.on_rc = on_rc
        self.off_rc = off_rc
        self.runner = runner or (lambda: 0)
        self.calls: list[tuple[list[str], bool]] = []

    def run(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: float,
        new_session: bool = False,
    ) -> tuple[int, str, str]:
        """Record a subprocess call and emulate it."""
        self.calls.append((argv, new_session))
        if argv[1].endswith("ha_set_experimental_motion.py"):
            action = argv[2]
            self.log.append(f"helper:{action}")
            if action == "on":
                if self.on_rc == 0:
                    self.host.armed = True
                return self.on_rc, "", ""
            if self.off_rc == 0:
                self.host.armed = False
            return self.off_rc, "", ""
        self.log.append("runner")
        return self.runner(), "runner out\n", ""


def make_run(
    tmp_path: Path,
    *,
    extra: list[str] | None = None,
    readback_never: bool = False,
    **proc_kw: Any,
) -> tuple[S2Run, FakeHost, FakeProc, list[str]]:
    """Build an S2Run wired to fakes."""
    out = tmp_path / "out"
    out.mkdir()
    argv = [
        "run",
        "--target-x",
        "4.5",
        "--target-y",
        "-5",
        "--out",
        str(out),
        "--operator-go",
        "go S2",
        *(["--root-level", "warning"] if extra is None else extra),
    ]
    args = build_parser().parse_args(argv)
    args.python = Path("/venv/python")
    clock = FakeClock(dt.datetime(2026, 10, 7, 17, 0, 0, tzinfo=UTC))  # sun ~45 deg
    log: list[str] = []
    host = FakeHost(clock, log, readback_never=readback_never)
    proc = FakeProc(host, log, **proc_kw)
    run = S2Run(args, host=host, proc=proc, clock=clock, repo_root=tmp_path, env={})
    return run, host, proc, log


def _summary(run: S2Run) -> dict[str, Any]:
    """Read the run's summary.json."""
    return json.loads((run.out / "summary.json").read_text())


def test_happy_path_order_and_pinned_argv(tmp_path: Path) -> None:
    """Happy path order and pinned argv."""
    run, host, proc, log = make_run(tmp_path)
    assert run.execute() == EXIT_OK
    assert log.index("helper:on") < log.index("runner") < log.index("helper:off")
    on_argv, on_new_session = proc.calls[0]
    assert on_argv == ["/venv/python", str(tmp_path / HELPER), "on", "--yes"]
    assert on_new_session is False
    runner_argv = proc.calls[1][0]
    assert runner_argv[1] == str(tmp_path / RUNNER)
    assert runner_argv[2:7] == ["S2", "4.5", "-5.0", "scored", "--out"]
    off_argv, off_new_session = proc.calls[2]
    assert off_argv[2:] == ["off"] and off_new_session is True
    assert host.armed is False
    for name in (
        "prearm.json",
        "arm.json",
        "readback.json",
        "runner.json",
        "disarm.json",
        "postleg.json",
        "raw_gate.json",
        "live_final.json",
        "tracker_history.json",
        "SHA256SUMS",
    ):
        assert (run.out / name).is_file(), name
    assert _summary(run)["operator_go"] == "go S2"


def test_profiler_off_and_level_restore_come_after_the_postleg_export(
    tmp_path: Path,
) -> None:
    """Profiler off and level restore come after the postleg export."""
    run, _host, _proc, log = make_run(tmp_path)
    run.execute()
    events = run.events
    last_postleg = max(i for i, e in enumerate(events) if e == "postleg_export")
    assert events.index("profiler_on") < events.index("prearm")
    assert (
        events.index("disarm")
        < last_postleg
        < events.index("profiler_off")
        < events.index("restore_level")
    )
    off_call = 'service:profiler.set_asyncio_debug:{"enabled": false}'
    level_call = 'service:logger.set_default_level:{"level": "warning"}'
    last_export = max(
        i for i, e in enumerate(log) if e == "export" and i < log.index(off_call)
    )
    assert (
        log.index("helper:off")
        < last_export
        < log.index(off_call)
        < log.index(level_call)
    )
    assert log.index(level_call) < log.index("raw")
    assert run.clock.t >= 300  # the soak ran before the pre-arm export


def test_unknown_root_level_skips_restore_with_a_warning(tmp_path: Path) -> None:
    """Unknown root level skips restore with a warning."""
    run, _host, _proc, log = make_run(tmp_path, extra=[])
    run.execute()
    assert not any(e.startswith("service:logger.") for e in log)
    assert any("root logger level unknown" in w for w in _summary(run)["warnings"])


def test_soak_finding_turns_the_profiler_off_before_arming(tmp_path: Path) -> None:
    """Soak finding turns the profiler off before arming."""
    run, host, _proc, log = make_run(tmp_path)
    host.syslog = [
        {
            "name": "homeassistant.components.mammotion",
            "message": ["boom"],
            "exception": "RuntimeError: Non-thread-safe operation invoked",
            "level": "ERROR",
            "timestamp": (host.clock.start + dt.timedelta(seconds=10)).timestamp(),
        }
    ]
    run.execute()
    off_call = 'service:profiler.set_asyncio_debug:{"enabled": false}'
    assert log.index(off_call) < log.index("helper:on")
    assert _summary(run)["profiler"]["used"] is False


def test_arm_failure_still_disarms_and_never_launches_the_runner(
    tmp_path: Path,
) -> None:
    """Arm failure still disarms and never launches the runner."""
    run, _host, _proc, log = make_run(tmp_path, on_rc=1)
    assert run.execute() == EXIT_PRE_DISPATCH
    assert "runner" not in log
    assert log.index("helper:on") < log.index("helper:off")
    assert _summary(run)["disarm"]["proved"] is True


def test_runner_nonzero_still_disarms(tmp_path: Path) -> None:
    """Runner nonzero still disarms."""
    run, _host, _proc, log = make_run(tmp_path, runner=lambda: 2)
    assert run.execute() == EXIT_RUNNER_NONZERO
    assert log.index("runner") < log.index("helper:off")
    assert _summary(run)["runner"]["rc"] == 2


def test_keyboard_interrupt_still_disarms(tmp_path: Path) -> None:
    """Keyboard interrupt still disarms."""

    def interrupted() -> int:
        """Raise KeyboardInterrupt like a Ctrl-C."""
        raise KeyboardInterrupt

    run, host, _proc, log = make_run(tmp_path, runner=interrupted)
    assert run.execute() == EXIT_INTERRUPTED
    assert log.index("runner") < log.index("helper:off")
    assert host.armed is False
    assert _summary(run)["interrupted"] is True


def test_readback_timeout_disarms_without_launching_the_runner(tmp_path: Path) -> None:
    """Readback timeout disarms without launching the runner."""
    run, _host, _proc, log = make_run(tmp_path, readback_never=True)
    assert run.execute() == EXIT_PRE_DISPATCH
    assert "runner" not in log
    assert "helper:off" in log
    summary = _summary(run)
    assert summary["readback"]["ok"] is False
    assert summary["readback"]["polls"] >= 50  # ~180 s at 3 s


def test_disarm_failure_is_retried_and_reported(tmp_path: Path) -> None:
    """Disarm failure is retried and reported."""
    run, _host, _proc, log = make_run(tmp_path, off_rc=1)
    assert run.execute() == EXIT_DISARM_UNPROVED
    assert log.count("helper:off") == 3
    assert _summary(run)["disarm"]["proved"] is False


def test_raw_read_waits_at_least_15_s_after_disarm(tmp_path: Path) -> None:
    """Raw read waits at least 15 s after disarm."""
    run, _host, _proc, _log = make_run(tmp_path)
    run.execute()
    raw = json.loads((run.out / "raw_gate.json").read_text())
    gap = parse_ts(raw["before_utc"]) - parse_ts(raw["disarm_after_utc"])
    assert gap.total_seconds() >= 15


def test_low_sun_stops_before_arming(tmp_path: Path) -> None:
    """Low sun stops before arming."""
    run, _host, _proc, log = make_run(tmp_path)
    run.clock.start = dt.datetime(2026, 10, 7, 4, 0, 0, tzinfo=UTC)  # night
    assert run.execute() == EXIT_PRE_DISPATCH
    assert not any(e.startswith("helper:") for e in log)


def test_missing_operator_go_refuses_before_any_host_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing operator go refuses before any host contact."""

    def boom(*_: Any, **__: Any) -> None:
        """Fail if called."""
        raise AssertionError("host contacted")

    monkeypatch.setattr(s2_leg, "HAHost", boom)
    monkeypatch.setattr(s2_leg, "S2Run", boom)
    base = [
        "run",
        "--target-x",
        "1",
        "--target-y",
        "2",
        "--out",
        str(tmp_path / "o"),
        "--repo-root",
        str(tmp_path),
    ]
    assert s2_leg.main(base) == EXIT_REFUSED
    assert s2_leg.main([*base, "--operator-go", "   "]) == EXIT_REFUSED
    assert not (tmp_path / "o").exists()


def test_pinned_file_mismatch_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinned file mismatch refuses."""

    def boom(*_: Any, **__: Any) -> None:
        """Fail if called."""
        raise AssertionError("host contacted")

    monkeypatch.setattr(s2_leg, "HAHost", boom)
    argv = [
        "run",
        "--target-x",
        "1",
        "--target-y",
        "2",
        "--out",
        str(tmp_path / "o"),
        "--repo-root",
        str(tmp_path),
        "--operator-go",
        "go",
    ]
    assert s2_leg.main(argv) == EXIT_REFUSED


def test_score_cli_prints_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Score cli prints json."""
    assert s2_leg.main(["score", str(S1_DIR)]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "INCONCLUSIVE"


def test_fixture_has_no_emails() -> None:
    """Fixture has no emails."""
    for path in S1_DIR.rglob("*"):
        if path.is_file():
            assert not s2_leg.EMAIL.search(path.read_text()), path


def test_redactor_strips_secrets_and_emails() -> None:
    """Redactor strips secrets and emails."""
    redact = s2_leg.make_redactor({"HA_TOKEN": "abcdefgh12345", "HA_URL": "http://x"})
    assert redact("t=abcdefgh12345 a@b.com") == "t=<HA_TOKEN> <email>"


def test_score_does_not_mutate_inputs(tmp_path: Path) -> None:
    """Scoring reads the artifacts and never rewrites them."""
    root = make_dir(tmp_path, operator_stop=False)
    before = {p.name: p.read_bytes() for p in root.rglob("*.json")}
    score_dir(root)
    assert before == {p.name: p.read_bytes() for p in root.rglob("*.json")}
