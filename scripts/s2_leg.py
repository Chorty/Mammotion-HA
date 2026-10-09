#!/usr/bin/env python3
"""Orchestrate and score the supervised repeat leg S2 on the split stack.

Rules: the private ``docs/predeclared-split-stack-clicktogo-leg-20261006.md``
section 1-3 plus Amendments 1-5 (Amendment 5 replaces the matching Amendment 4
text), and ``docs/predeclared-backend-096-clicktogo-leg-20260928.md`` section 2
plus Amendment 6 section 5 for the criterion-3 window and pass condition.

Two subcommands:

``run``
    One attempt, on an explicit operator go. It never chooses the target and
    never infers a go. Order (Amendments 2-5):

    1. Local refusals: ``--operator-go`` must be given and non-blank, ``--out``
       must be new or empty, and the pinned runner, gate helper, profile and
       siting files must match their predeclared sha256. Nothing touches the
       host until these pass.
    2. Record the root logger level. Home Assistant has no REST or WebSocket
       read of the ROOT level (``logger/log_info`` lists per-integration
       levels only), so it is ``unknown`` unless the operator passes
       ``--root-level``. Unknown means the restore is skipped, with a warning.
    3. Profiler soak (``--profiler``): ``profiler.set_asyncio_debug`` on, wait
       300 s, scan the system log for a ``Non-thread-safe operation`` error or
       a new exception from mammotion / pymammotion / bluetooth / esphome. A
       finding turns the profiler off; the leg then runs without it.
    4. Pre-arm export, bracketed by HA-host UTC reads (``POST /api/template``).
       It must show the gate off and ``receipt_age_s >= 5``. Repeated >= 1 s
       apart for up to 60 s; otherwise nothing is armed.
    5. Sun >= 20 deg whenever the VIO window will read cache (Amendment 3 s1).
    6. Arm with the pinned helper ``on --yes``; its exit code is read directly.
    7. Readback poll, <= 180 s: ``enabled``, ``supervised_qualification`` and
       ``real_motion_allowed`` all true, ``blockers == []``.
    8. The pinned runner ``LABEL TX TY scored --out DIR/runner``, exit code read
       directly (no shell, no PIPESTATUS).
    9. ALWAYS (finally, including KeyboardInterrupt): disarm with the pinned
       helper ``off``, SIGINT-shielded and in its own session, up to 3 tries.
   10. Post-leg exports >= 1 s apart for up to 60 s, until one shows
       ``receipt_age_s >= 3`` and no tracker row inside its bracket.
   11. Profiler off, AFTER that export; then ``logger.set_default_level`` back
       to the recorded root level.
   12. RAW ``core.config_entries`` gate flags (SSH) and a final live export,
       both >= 15 s after the disarm (default 20 s).
   13. Tracker history (``significant_changes_only=0``, start >= 60 s before
       every bound), timing report, system log and core log tail.

    Every artifact lands in ``--out``. ``summary.json`` is a record, not a
    verdict: run ``score`` for that.

``score DIR``
    Pure offline scoring from the saved artifacts. Criteria 1-4 of section 2,
    the Amendment 5 s2 counting rule, and the Amendment 5 s5 precedence. It
    never infers a missing timestamp or bracket: anything absent makes the
    affected criterion UNEVALUABLE (criterion 3) or not proven (criterion 4).
    The operator stop control is invisible in telemetry, so an operator stop
    is read only from ``--operator-stop`` or ``operator_record.json``; without
    it a verdict that depends on it is UNDETERMINED.

Exit codes for ``run``: 0 runner exit 0 and disarm proved; 2 refused before
any host contact; 3 stopped before dispatch; 4 runner non-zero; 5 disarm NOT
proved (dominates); 130 interrupted (disarm proved).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_leg_runner import solar_elevation_degrees  # noqa: E402

MOTION_DOMAIN = "mammotion_motion"
BASE_DOMAIN = "mammotion"
ENTITY = "lawn_mower.back_yard_clip_skywalker"
TRACKER = "device_tracker.back_yard_clip_skywalker_luba_vsplv397"

#: Predeclared identities (Amendment 1 s1-s3, Amendment 2 s2 and s13). Paths
#: are relative to the root checkout, which is also the runner's working
#: directory so its relative siting and profile paths resolve.
RUNNER = ".worktrees/runners-companion/scripts/phase1_leg_runner.py"
HELPER = ".worktrees/runners-companion/scripts/ha_set_experimental_motion.py"
PINNED_SHA256 = {
    RUNNER: "9bbcf4f7a035b66a78ad217e86097a65ee153b6a6ec327754bbdf6bb052fd5b9",
    HELPER: "51a23b25fe663c283f570460998c08e4396fe5bf079ad30643ce11e454bef2c8",
    ".worktrees/runners-companion/scripts/mammotion_ha_helpers.py": (
        "2d8e2d9c18bcf8dcfb22cd9260d2e597b991560f1819d829fce5b97a7c66fe63"
    ),
    "scripts/accepted-profile.json": (
        "ef529356ddec10603536db371fa55ef339825aa27d7b64f196bc3d6a3f3593af"
    ),
    "docs/evidence-phase1-siting-20260912.json": (
        "6c08c56d797f5ef0004cc34db9d37346ea11cd28af90262dfb6fd33e145ae826"
    ),
}

SOAK_SECONDS = 300.0
PREARM_MIN_RECEIPT_AGE_S = 5.0
PREARM_RETRY_WINDOW_S = 60.0
READBACK_TIMEOUT_S = 180.0
READBACK_POLL_S = 3.0
POSTLEG_MIN_RECEIPT_AGE_S = 3.0
POSTLEG_SPACING_S = 1.0
POSTLEG_WINDOW_S = 60.0
#: The recorder commits in batches; wait this long before asking history
#: whether a row landed inside a post-leg export's bracket.
RECORDER_SETTLE_S = 6.0
RAW_MIN_DELAY_S = 15.0
RAW_DEFAULT_DELAY_S = 20.0
HISTORY_MIN_LEAD_S = 60.0
HISTORY_LEAD_S = 120.0
VIO_FRESH_RECEIPT_AGE_S = 5.0
SUN_MIN_CACHED_VIO_DEG = 20.0
RUNNER_TIMEOUT_S = 900.0
DISARM_ATTEMPTS = 3

C3_MAX_GAP_S = 2.0
LANDING_TOLERANCE_M = 0.15
RAW_AFTER_DISARM_S = 15.0
NOTHING_SENT_INCONCLUSIVE = {
    "ble_client_not_connected",
    "position_feed_not_live",
    # Amendment 9 s2: the companion's own pin re-read refused before the claim.
    "travel_speed_not_pinned",
    "travel_speed_unreadable",
}
#: Superseded by Amendment 6 s11 (any `ble_*` stop reason); kept for reference.
BLE_LOSS_AFTER_SEND = {"ble_transport_lost", "ble_client_not_connected"}

SOAK_LOGGERS = re.compile(r"mammotion|pymammotion|bluetooth|esphome", re.IGNORECASE)
NON_THREAD_SAFE = "Non-thread-safe operation"
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ROOT_LEVELS = ("critical", "fatal", "error", "warning", "warn", "info", "debug")

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_PRE_DISPATCH = 3
EXIT_RUNNER_NONZERO = 4
EXIT_DISARM_UNPROVED = 5
EXIT_INTERRUPTED = 130


class Refused(Exception):
    """A local precondition failed; nothing was sent to the host."""


class Abort(Exception):
    """A host-side precondition failed before dispatch."""


# --------------------------------------------------------------------- shared


def parse_ts(value: Any) -> dt.datetime | None:
    """Parse an aware ISO-8601 timestamp to UTC; anything else is None."""
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(dt.UTC)


def _iso(value: dt.datetime | None) -> str | None:
    return value.isoformat() if value else None


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(float(value)) else None


def sha256_file(path: Path) -> str:
    """Return the sha256 of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_env(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE lines. Values are never printed."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def make_redactor(env: dict[str, str]) -> Callable[[str], str]:
    """Return a function that strips emails and secret values from text."""
    secrets = [
        (key, value)
        for key, value in env.items()
        if len(value) >= 8
        and key.endswith(("PASS", "PASSWORD", "TOKEN", "KEY", "PASSPHRASE"))
    ]

    def redact(text: str) -> str:
        text = EMAIL.sub("<email>", text)
        for key, value in secrets:
            text = text.replace(value, f"<{key}>")
        return text

    return redact


# ---------------------------------------------------------------- HA transport


class _WS:
    """Minimal stdlib Home Assistant WebSocket client (read-only calls)."""

    def __init__(self, url: str, token: str) -> None:
        parsed = urllib.parse.urlparse(url)
        self.sock = socket.create_connection(
            (parsed.hostname, parsed.port or 80), timeout=60
        )
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            (
                f"GET /api/websocket HTTP/1.1\r\nHost: {parsed.hostname}:{parsed.port}"
                f"\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("websocket closed during upgrade")
            buf += chunk
        head, self.rest = buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise ConnectionError("websocket upgrade refused")
        self._recv()
        self._send({"type": "auth", "access_token": token})
        if self._recv().get("type") != "auth_ok":
            raise ConnectionError("websocket auth failed")
        self.n = 0

    def _read(self, n: int) -> bytes:
        while len(self.rest) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("websocket closed")
            self.rest += chunk
        out, self.rest = self.rest[:n], self.rest[n:]
        return out

    def _send(self, obj: dict[str, Any]) -> None:
        data = json.dumps(obj).encode()
        mask = os.urandom(4)
        header = bytes([0x81])
        if len(data) < 126:
            header += bytes([0x80 | len(data)])
        elif len(data) < 65536:
            header += bytes([0x80 | 126]) + struct.pack(">H", len(data))
        else:
            header += bytes([0x80 | 127]) + struct.pack(">Q", len(data))
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(header + mask + masked)

    def _recv(self) -> dict[str, Any]:
        message = b""
        while True:
            b1, b2 = self._read(2)
            length = b2 & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(length)
            if (b1 & 0x0F) in (0, 1):
                message += payload
                if b1 & 0x80:
                    return json.loads(message)

    def call(self, kind: str, **kw: Any) -> Any:
        self.n += 1
        self._send({"id": self.n, "type": kind, **kw})
        while True:
            reply = self._recv()
            if reply.get("id") == self.n:
                if not reply.get("success"):
                    raise RuntimeError(f"{kind}: {reply.get('error')}")
                return reply.get("result")

    def close(self) -> None:
        self.sock.close()


class HAHost:
    """The only object in ``run`` that talks to Home Assistant."""

    def __init__(self, url: str, token: str, ssh_exp: Path, env: dict[str, str]):
        """Hold the HA URL, token, SSH transport and .env values."""
        self.url = url.rstrip("/")
        self.token = token
        self.ssh_exp = ssh_exp
        self.env = env

    def _request(self, path: str, body: Any = None, timeout: float = 90) -> bytes:
        request = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode(),
            method="GET" if body is None else "POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.read()
        except urllib.error.HTTPError as err:
            detail = err.read()[:300].decode(errors="replace")
            raise RuntimeError(f"HTTP {err.code} on {path}: {detail}") from err

    def host_utc(self) -> str:
        """HA-host UTC, via the template API (Amendment 5 s1)."""
        text = self._request(
            "/api/template", {"template": "{{ utcnow().isoformat() }}"}, timeout=20
        ).decode()
        if parse_ts(text) is None:
            raise RuntimeError(f"host UTC template returned {text[:80]!r}")
        return text.strip()

    def service(
        self,
        domain: str,
        service: str,
        data: dict[str, Any],
        *,
        response: bool = True,
        timeout: float = 90,
    ) -> Any:
        """Call one HA service; return its response data when asked."""
        path = f"/api/services/{domain}/{service}"
        if response:
            path += "?return_response"
        raw = self._request(path, data, timeout=timeout)
        if not response:
            return None
        return json.loads(raw or b"{}").get("service_response", {})

    def export(self) -> dict[str, Any]:
        """Return the companion's ``export_runtime_state`` (read-only)."""
        return self.service(
            MOTION_DOMAIN, "export_runtime_state", {"entity_id": ENTITY}
        )

    def timing_report(self) -> dict[str, Any]:
        """Return the companion's ``motion_dispatch_timing_report`` (read-only)."""
        return self.service(
            MOTION_DOMAIN, "motion_dispatch_timing_report", {"entity_id": ENTITY}
        )

    def history(self, entity_id: str, start_iso: str, end_iso: str) -> Any:
        """Raw recorder history, every row (``significant_changes_only=0``)."""
        query = urllib.parse.urlencode(
            {
                "filter_entity_id": entity_id,
                "end_time": end_iso,
                "significant_changes_only": "0",
            }
        )
        start = urllib.parse.quote(start_iso, safe="")
        return json.loads(self._request(f"/api/history/period/{start}?{query}"))

    def _ws(self, kind: str) -> Any:
        ws = _WS(self.url, self.token)
        try:
            return ws.call(kind)
        finally:
            ws.close()

    def system_log(self) -> Any:
        """WebSocket ``system_log/list`` (read-only)."""
        return self._ws("system_log/list")

    def logger_info(self) -> Any:
        """WebSocket ``logger/log_info``: per-integration levels, never root."""
        return self._ws("logger/log_info")

    def _ssh(self, command: str, timeout: float = 180) -> str:
        env = {
            **os.environ,
            **{k: v for k, v in self.env.items() if k.startswith("HA_SSH")},
        }
        result = subprocess.run(  # noqa: S603
            ["expect", str(self.ssh_exp), command],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ssh exited {result.returncode}")
        return result.stdout

    def raw_gate_flags(self) -> list[dict[str, Any]]:
        """RAW ``.storage/core.config_entries`` gate options, both domains."""
        code = (
            "import json; d=json.load(open('/homeassistant/.storage/core.config_entries'));"
            "print(json.dumps([{'domain':e['domain'],'entry_id':e['entry_id'],"
            "'disabled_by':e.get('disabled_by'),'flags':{k:v for k,v in "
            "e['options'].items() if k.startswith('enable_')}} for e in "
            "d['data']['entries'] if e['domain'] in ('mammotion','mammotion_motion')]))"
        )
        out = self._ssh(f"python3 -c {shlex.quote(code)}")
        for line in reversed(out.strip().splitlines()):
            with contextlib.suppress(ValueError):
                parsed = json.loads(line)
                if isinstance(parsed, list):
                    return parsed
        raise RuntimeError("RAW read returned no JSON list")

    def core_log(self, lines: int = 6000) -> str:
        """Tail of the HA core log (timestamps are host-local, not UTC)."""
        return self._ssh(f"ha core logs -n {int(lines)} 2>&1")


class Proc:
    """Subprocess launcher; exit codes are returned, never piped."""

    def run(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: float,
        new_session: bool = False,
    ) -> tuple[int, str, str]:
        """Run argv; return (rc, stdout, stderr). Timeout returns rc 124."""
        try:
            result = subprocess.run(  # noqa: S603
                argv,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                start_new_session=new_session,
                check=False,
            )
        except subprocess.TimeoutExpired as err:
            out = err.stdout.decode() if isinstance(err.stdout, bytes) else err.stdout
            errs = err.stderr.decode() if isinstance(err.stderr, bytes) else err.stderr
            return 124, out or "", (errs or "") + f"\nTIMEOUT after {timeout} s"
        return result.returncode, result.stdout, result.stderr


class Clock:
    """Wall and monotonic time, injectable for tests."""

    def monotonic(self) -> float:
        """Monotonic seconds."""
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        """Sleep, ignoring non-positive requests."""
        if seconds > 0:
            time.sleep(seconds)

    def utcnow(self) -> dt.datetime:
        """Local UTC (used only for the sun and as a labelled fallback)."""
        return dt.datetime.now(dt.UTC)


@contextlib.contextmanager
def sigint_shielded() -> Iterator[None]:
    """Ignore SIGINT for the block (disarm and cleanup must not be cut short)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


# ------------------------------------------------------------------------- run


def _receipt_age(export: Any) -> float | None:
    if not isinstance(export, dict):
        return None
    return _num((export.get("position_pipeline") or {}).get("receipt_age_s"))


def _gate(export: Any) -> dict[str, Any]:
    motion = (
        (export or {}).get("experimental_motion") if isinstance(export, dict) else None
    )
    return motion if isinstance(motion, dict) else {}


class S2Run:
    """One orchestrated attempt. All host I/O goes through ``host``/``proc``."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        host: Any,
        proc: Any,
        clock: Any,
        repo_root: Path,
        env: dict[str, str],
        redact: Callable[[str], str] = lambda s: s,
    ) -> None:
        """Bind one attempt to its injected host, process and clock."""
        self.args = args
        self.host = host
        self.proc = proc
        self.clock = clock
        self.repo_root = repo_root
        self.env = env
        self.redact = redact
        self.out: Path = args.out
        self.events: list[str] = []
        self.summary: dict[str, Any] = {
            "script": "scripts/s2_leg.py",
            "label": args.label,
            "target": [args.target_x, args.target_y],
            "operator_go": args.operator_go,
            "profiler_requested": bool(args.profiler),
            "arm": {"attempted": False, "rc": None},
            "readback": {"ok": None},
            "runner": {"launched": False, "rc": None},
            "disarm": {"attempted": False, "proved": None, "rcs": []},
            "postleg": {"closing_index": None},
            "interrupted": False,
            "abort_reason": None,
            "warnings": [],
        }
        self.profiler_maybe_on = False
        self.profiler_touched = False
        self.disarm_needed = False
        self.disarm_done_mono: float | None = None
        self.disarm_done_utc: str | None = None

    # -- recording

    def save(self, name: str, obj: Any) -> None:
        """Write one artifact (JSON or text), redacted."""
        text = obj if isinstance(obj, str) else json.dumps(obj, indent=1, default=str)
        (self.out / name).write_text(self.redact(text))

    def warn(self, message: str) -> None:
        """Record and print a warning."""
        self.summary["warnings"].append(message)
        print("WARNING:", message, file=sys.stderr)

    def say(self, message: str) -> None:
        """Print one progress line."""
        print(f"[s2] {message}", flush=True)

    def host_utc_or_none(self) -> str | None:
        """Host UTC; None (recorded by the caller) if the read fails."""
        try:
            return self.host.host_utc()
        except Exception as err:  # noqa: BLE001
            self.warn(f"host UTC read failed: {err}")
            return None

    def bracketed_export(self, label: str) -> dict[str, Any]:
        """Export bracketed by host UTC reads immediately before and after."""
        record: dict[str, Any] = {"label": label}
        record["before_utc"] = self.host.host_utc()
        try:
            record["export"] = self.host.export()
        finally:
            record["after_utc"] = self.host.host_utc()
        record["receipt_age_s"] = _receipt_age(record.get("export"))
        return record

    def helper(self, action: str) -> tuple[int, str, str]:
        """Run the pinned gate helper with the predeclared arguments."""
        argv = [str(self.args.python), str(self.repo_root / HELPER), action]
        if action == "on":
            argv.append("--yes")
        return self.proc.run(
            argv,
            cwd=self.repo_root,
            env=self.env,
            timeout=180,
            new_session=action == "off",
        )

    # -- steps

    def record_root_level(self) -> None:
        """Amendment 5 s3: record the root logger level first."""
        record: dict[str, Any] = {"level": "unknown", "source": None}
        if self.args.root_level:
            record = {"level": self.args.root_level, "source": "operator --root-level"}
        else:
            record["note"] = (
                "HA exposes no REST/WS read of the root logger level; restore will "
                "be skipped."
            )
        try:
            record["logger_log_info"] = self.host.logger_info()
        except Exception as err:  # noqa: BLE001
            record["logger_log_info_error"] = str(err)
        self.summary["root_level"] = record
        self.save("root_level.json", record)
        if record["level"] == "unknown" and self.args.profiler:
            self.warn(
                "root logger level unknown: after the profiler turns off, the root "
                "level will NOT be restored (Amendment 5 s3 cannot be completed)."
            )

    def _syslog_findings(
        self, entries: Any, since: dt.datetime
    ) -> list[dict[str, Any]]:
        findings = []
        for entry in entries if isinstance(entries, list) else []:
            stamp = _num(entry.get("timestamp"))
            if stamp is None or dt.datetime.fromtimestamp(stamp, dt.UTC) <= since:
                continue
            text = " ".join(
                str(part)
                for part in (
                    entry.get("name"),
                    entry.get("message"),
                    entry.get("exception"),
                    entry.get("source"),
                )
            )
            level = str(entry.get("level", "")).upper()
            if NON_THREAD_SAFE in text:
                findings.append({"why": "non_thread_safe", "entry": entry})
            elif SOAK_LOGGERS.search(text) and (
                entry.get("exception") or level in ("ERROR", "CRITICAL")
            ):
                findings.append({"why": "new_exception", "entry": entry})
        return findings

    def profiler_soak(self) -> None:
        """Amendment 5 s3: profiler on >= 5 min before the pre-arm export."""
        record: dict[str, Any] = {"requested": True}
        self.summary["profiler"] = record
        start = self.host_utc_or_none()
        record["on_before_utc"] = start
        self.profiler_maybe_on = True
        self.profiler_touched = True
        try:
            self.host.service(
                "profiler", "set_asyncio_debug", {"enabled": True}, response=False
            )
        except Exception as err:  # noqa: BLE001
            record["on_error"] = str(err)
            record["used"] = False
            self.warn(f"profiler unavailable, running without it: {err}")
            self.save("profiler.json", record)
            return
        self.events.append("profiler_on")
        record["on_after_utc"] = self.host_utc_or_none()
        self.say(f"profiler on; soaking {SOAK_SECONDS:.0f} s")
        self.clock.sleep(SOAK_SECONDS)
        since = parse_ts(start) or self.clock.utcnow()
        try:
            entries = self.host.system_log()
            self.save("syslog_after_soak.json", entries)
            findings = self._syslog_findings(entries, since)
        except Exception as err:  # noqa: BLE001
            record["scan_error"] = str(err)
            findings = []
            self.warn(f"soak scan failed ({err}); profiler stays on as predeclared")
        record["soak_end_utc"] = self.host_utc_or_none()
        record["findings"] = findings
        if findings:
            self.warn(f"soak found {len(findings)} issue(s); profiler off, no profiler")
            self.profiler_off("soak_finding")
            record["used"] = False
        else:
            record["used"] = True
        self.save("profiler.json", record)

    def profiler_off(self, why: str) -> None:
        """Turn asyncio debug off (idempotent at our level)."""
        record = self.summary.setdefault("profiler", {})
        try:
            self.host.service(
                "profiler", "set_asyncio_debug", {"enabled": False}, response=False
            )
            record.setdefault("off", []).append(
                {"why": why, "after_utc": self.host_utc_or_none()}
            )
        except Exception as err:  # noqa: BLE001
            record.setdefault("off", []).append({"why": why, "error": str(err)})
            self.warn(f"profiler off failed: {err}")
        self.profiler_maybe_on = False
        self.events.append("profiler_off")

    def restore_root_level(self) -> None:
        """Amendment 5 s3: restore the recorded root level after the profiler."""
        level = (self.summary.get("root_level") or {}).get("level", "unknown")
        if level == "unknown":
            self.warn("root logger level unknown: NOT restored")
            self.summary["root_level_restore"] = {"skipped": "unknown level"}
            return
        try:
            self.host.service(
                "logger", "set_default_level", {"level": level}, response=False
            )
            self.summary["root_level_restore"] = {
                "level": level,
                "after_utc": self.host_utc_or_none(),
            }
            self.events.append("restore_level")
        except Exception as err:  # noqa: BLE001
            self.summary["root_level_restore"] = {"level": level, "error": str(err)}
            self.warn(f"root level restore failed: {err}")

    def check_sun(self, why: str) -> None:
        """Amendment 3 s1 / Amendment 4 s4: cached VIO needs the sun >= 20 deg."""
        elevation = solar_elevation_degrees(self.clock.utcnow())
        self.summary.setdefault("sun_checks", []).append(
            {"why": why, "elevation_deg": round(elevation, 2)}
        )
        floor = float(getattr(self.args, "min_cached_vio_sun", SUN_MIN_CACHED_VIO_DEG))
        if elevation < floor:
            raise Abort(
                f"sun {elevation:.1f} deg < {floor} with a cached VIO window ({why})"
            )

    def check_travel_speed(self) -> None:
        """Amendment 9 s2: the speed register reads pinned before arming."""
        record = self.host.service(
            MOTION_DOMAIN, "read_travel_speed", {"entity_id": ENTITY}
        )
        self.save("travel_speed_prearm.json", record)
        self.summary["travel_speed_prearm"] = record
        if not isinstance(record, dict) or record.get("pinned") is not True:
            raise Abort(
                "travel speed not pinned before arming: "
                f"{(record or {}).get('speed_mps')!r} m/s, "
                f"blocker {(record or {}).get('blocker')!r}"
            )

    def travel_speed_postleg(self) -> None:
        """Amendment 9 s4: read-only register read after the disarm proof."""
        record = self.host.service(
            MOTION_DOMAIN, "read_travel_speed", {"entity_id": ENTITY}
        )
        self.save("travel_speed_postleg.json", record)
        self.summary["travel_speed_postleg"] = record

    def prearm(self) -> dict[str, Any]:
        """Amendment 5 s1: bracketed pre-arm export with receipt_age_s >= 5."""
        attempts: list[dict[str, Any]] = []
        start = self.clock.monotonic()
        chosen = None
        while True:
            record = self.bracketed_export(f"prearm_{len(attempts)}")
            attempts.append(record)
            gate = _gate(record.get("export"))
            if (
                gate.get("enabled") is not False
                or gate.get("supervised_qualification") is not False
            ):
                self.disarm_needed = True
                self.save("prearm.json", {"attempts": attempts, "selected_index": None})
                raise Abort(f"gate not off before arming: {gate.get('enabled')!r}")
            age = record["receipt_age_s"]
            if age is not None and age >= PREARM_MIN_RECEIPT_AGE_S:
                chosen = len(attempts) - 1
                break
            if (
                self.clock.monotonic() - start + POSTLEG_SPACING_S
                > PREARM_RETRY_WINDOW_S
            ):
                break
            self.clock.sleep(POSTLEG_SPACING_S)
        self.save("prearm.json", {"attempts": attempts, "selected_index": chosen})
        if chosen is None:
            raise Abort(
                f"no pre-arm export with receipt_age_s >= {PREARM_MIN_RECEIPT_AGE_S} "
                f"in {PREARM_RETRY_WINDOW_S:.0f} s"
            )
        self.events.append("prearm")
        return attempts[chosen]

    def arm(self) -> None:
        """Arm with the pinned helper; the exit code decides."""
        self.disarm_needed = True
        self.summary["arm"]["attempted"] = True
        before = self.host_utc_or_none()
        self.events.append("arm")
        rc, out, err = self.helper("on")
        after = self.host_utc_or_none()
        self.summary["arm"]["rc"] = rc
        self.save(
            "arm.json",
            {
                "before_utc": before,
                "after_utc": after,
                "rc": rc,
                "stdout": out,
                "stderr": err,
            },
        )
        if rc != 0:
            raise Abort(f"arm helper exited {rc}")

    def readback(self) -> dict[str, Any]:
        """Amendment 3 s3: poll <= 180 s for the full armed readback."""
        polls: list[dict[str, Any]] = []
        start = self.clock.monotonic()
        ok_record = None
        while True:
            record = self.bracketed_export(f"readback_{len(polls)}")
            polls.append(record)
            gate = _gate(record.get("export"))
            if (
                gate.get("enabled") is True
                and gate.get("supervised_qualification") is True
                and gate.get("real_motion_allowed") is True
                and gate.get("blockers") == []
            ):
                ok_record = record
                break
            if self.clock.monotonic() - start + READBACK_POLL_S > READBACK_TIMEOUT_S:
                break
            self.clock.sleep(READBACK_POLL_S)
        self.summary["readback"] = {"ok": ok_record is not None, "polls": len(polls)}
        self.save("readback.json", {"polls": polls, "ok": ok_record is not None})
        self.events.append("readback")
        if ok_record is None:
            raise Abort(f"armed readback not seen within {READBACK_TIMEOUT_S:.0f} s")
        return ok_record

    def launch_runner(self) -> int:
        """Run the pinned runner once; read its exit code directly."""
        runner_out = self.out / "runner"
        runner_out.mkdir(exist_ok=True)
        argv = [
            str(self.args.python),
            str(self.repo_root / RUNNER),
            self.args.label,
            repr(float(self.args.target_x)),
            repr(float(self.args.target_y)),
            "scored",
            "--out",
            str(runner_out),
        ]
        self.summary["runner"]["launched"] = True
        self.summary["runner"]["argv"] = argv
        before = self.host_utc_or_none()
        self.events.append("runner")
        self.say("launching runner")
        try:
            rc, out, err = self.proc.run(
                argv, cwd=self.repo_root, env=self.env, timeout=RUNNER_TIMEOUT_S
            )
        except KeyboardInterrupt:
            self.save("runner.json", {"before_utc": before, "interrupted": True})
            raise
        after = self.host_utc_or_none()
        self.summary["runner"]["rc"] = rc
        self.save(
            "runner.json",
            {
                "before_utc": before,
                "after_utc": after,
                "rc": rc,
                "stdout": out,
                "stderr": err,
            },
        )
        print(self.redact(out), end="")
        if err:
            print(self.redact(err), end="", file=sys.stderr)
        return rc

    def disarm(self) -> None:
        """Unconditional disarm, SIGINT-shielded, retried, each rc checked."""
        self.summary["disarm"]["attempted"] = True
        attempts = []
        with sigint_shielded():
            for _ in range(DISARM_ATTEMPTS):
                before = self.host_utc_or_none()
                self.events.append("disarm")
                try:
                    rc, out, err = self.helper("off")
                except Exception as exc:  # noqa: BLE001
                    rc, out, err = -1, "", f"launch failed: {exc}"
                after = self.host_utc_or_none()
                attempts.append(
                    {
                        "before_utc": before,
                        "after_utc": after,
                        "rc": rc,
                        "stdout": out,
                        "stderr": err,
                    }
                )
                self.summary["disarm"]["rcs"].append(rc)
                if rc == 0:
                    break
        proved = bool(attempts) and attempts[-1]["rc"] == 0
        self.summary["disarm"]["proved"] = proved
        self.disarm_done_mono = self.clock.monotonic()
        self.disarm_done_utc = attempts[-1]["after_utc"] if attempts else None
        self.save("disarm.json", {"attempts": attempts, "proved": proved})
        if not proved:
            self.warn("DISARM NOT PROVED -- check the gate by hand NOW; session ends")

    def postleg(self) -> None:
        """Amendment 5 s1: repeat until receipt_age_s >= 3 and an empty bracket."""
        attempts: list[dict[str, Any]] = []
        start = self.clock.monotonic()
        closing = None
        last = None
        while True:
            if last is not None:
                self.clock.sleep(last + POSTLEG_SPACING_S - self.clock.monotonic())
            last = self.clock.monotonic()
            try:
                record = self.bracketed_export(f"postleg_{len(attempts)}")
            except Exception as err:  # noqa: BLE001
                record = {"label": f"postleg_{len(attempts)}", "error": str(err)}
            attempts.append(record)
            self.events.append("postleg_export")
            age = record.get("receipt_age_s")
            if age is not None and age >= POSTLEG_MIN_RECEIPT_AGE_S:
                self.clock.sleep(RECORDER_SETTLE_S)
                inside = self._rows_inside(record)
                record["rows_inside_bracket_online"] = inside
                if inside == []:
                    closing = len(attempts) - 1
                    break
            if self.clock.monotonic() - start + POSTLEG_SPACING_S > POSTLEG_WINDOW_S:
                break
        self.summary["postleg"] = {"closing_index": closing, "attempts": len(attempts)}
        self.save("postleg.json", {"attempts": attempts, "closing_index": closing})
        if closing is None:
            self.warn("no post-leg export closed the span: criterion 3 unevaluable")

    def _rows_inside(self, record: dict[str, Any]) -> list[str] | None:
        before, after = (
            parse_ts(record.get("before_utc")),
            parse_ts(record.get("after_utc")),
        )
        if before is None or after is None:
            return None
        try:
            now = self.host.host_utc()
            response = self.host.history(
                TRACKER, _iso(before - dt.timedelta(seconds=10)) or "", now
            )
        except Exception as err:  # noqa: BLE001
            record["rows_inside_error"] = str(err)
            return None
        return [
            row["last_updated_raw"]
            for row in normalize_rows(response)
            if row["last_updated"] and before <= row["last_updated"] <= after
        ]

    def raw_and_live(self) -> None:
        """Criterion 4 evidence, at least 15 s after the disarm."""
        delay = max(RAW_MIN_DELAY_S, float(self.args.raw_delay))
        if self.disarm_done_mono is not None:
            self.clock.sleep(self.disarm_done_mono + delay - self.clock.monotonic())
        ref = parse_ts(self.disarm_done_utc)
        for _ in range(3):
            now = parse_ts(self.host_utc_or_none())
            if ref is None or now is None:
                break
            short = RAW_AFTER_DISARM_S - (now - ref).total_seconds()
            if short <= 0:
                break
            self.clock.sleep(short + 1)
        record: dict[str, Any] = {"disarm_after_utc": self.disarm_done_utc}
        record["before_utc"] = self.host_utc_or_none()
        try:
            record["entries"] = self.host.raw_gate_flags()
        except Exception as err:  # noqa: BLE001
            record["error"] = str(err)
            self.warn(f"RAW gate read failed: {err}")
        record["after_utc"] = self.host_utc_or_none()
        self.events.append("raw")
        self.save("raw_gate.json", record)
        try:
            self.save("live_final.json", self.bracketed_export("live_final"))
        except Exception as err:  # noqa: BLE001
            self.warn(f"final live export failed: {err}")

    def collect(self, prearm_before: str | None) -> None:
        """Tracker history, timing report, system log, core log tail."""
        first = parse_ts(prearm_before) or self.clock.utcnow()
        start = first - dt.timedelta(seconds=HISTORY_LEAD_S)
        try:
            end = self.host.host_utc()
            response = self.host.history(TRACKER, start.isoformat(), end)
            self.save(
                "tracker_history.json",
                {
                    "query": {
                        "entity_id": TRACKER,
                        "start_time": start.isoformat(),
                        "end_time": end,
                        "significant_changes_only": 0,
                    },
                    "response": response,
                },
            )
        except Exception as err:  # noqa: BLE001
            self.warn(f"tracker history fetch failed: {err}")
        for name, call in (
            ("timing_post.json", self.host.timing_report),
            ("syslog_post.json", self.host.system_log),
        ):
            try:
                self.save(name, call())
            except Exception as err:  # noqa: BLE001
                self.warn(f"{name} failed: {err}")
        try:
            self.save("ha_core_log.txt", self.host.core_log())
        except Exception as err:  # noqa: BLE001
            self.warn(f"core log fetch failed: {err}")

    # -- orchestration

    def execute(self) -> int:  # noqa: C901, PLR0912
        """Run the attempt; cleanup always runs; returns the exit code."""
        runner_rc: int | None = None
        stage = "start"
        prearm_before = None
        try:
            try:
                self.record_root_level()
                if self.args.profiler:
                    stage = "soak"
                    self.profiler_soak()
                else:
                    self.summary["profiler"] = {"requested": False}
                stage = "prearm"
                self.check_sun("pre-arm (pre-arm receipt_age_s >= 5 means cached VIO)")
                self.check_travel_speed()
                prearm = self.prearm()
                prearm_before = prearm["before_utc"]
                stage = "arm"
                self.arm()
                stage = "readback"
                ready = self.readback()
                age = ready.get("receipt_age_s")
                if age is None or age > VIO_FRESH_RECEIPT_AGE_S:
                    self.check_sun("pre-runner (VIO window reads cache)")
                stage = "runner"
                runner_rc = self.launch_runner()
                stage = "done"
            except KeyboardInterrupt:
                self.summary["interrupted"] = True
                self.summary["abort_reason"] = f"KeyboardInterrupt during {stage}"
                print("\nInterrupted: disarming.", file=sys.stderr)
            except Abort as err:
                self.summary["abort_reason"] = f"{stage}: {err}"
                print(f"STOP before dispatch: {err}", file=sys.stderr)
            except Exception as err:  # noqa: BLE001
                self.summary["abort_reason"] = f"{stage}: unexpected {err!r}"
                print(f"STOP ({stage}): {err!r}", file=sys.stderr)
            finally:
                with sigint_shielded():
                    if self.disarm_needed:
                        self.disarm()
                    if self.summary["arm"]["attempted"]:
                        self._guarded("postleg", self.postleg)
        finally:
            with sigint_shielded():
                if self.profiler_maybe_on:
                    self.profiler_off("after_postleg_export")
                if self.profiler_touched:
                    self.restore_root_level()
                if self.summary["disarm"]["attempted"]:
                    self._guarded("raw_and_live", self.raw_and_live)
                if self.summary["arm"]["attempted"]:
                    self._guarded("travel_speed_postleg", self.travel_speed_postleg)
                if self.summary["arm"]["attempted"]:
                    self._guarded("collect", lambda: self.collect(prearm_before))
                self.save("summary.json", self.summary)
                self._write_sums()
        if self.summary["disarm"]["attempted"] and not self.summary["disarm"]["proved"]:
            return EXIT_DISARM_UNPROVED
        if self.summary["interrupted"]:
            return EXIT_INTERRUPTED
        if runner_rc is None:
            return EXIT_PRE_DISPATCH
        return EXIT_OK if runner_rc == 0 else EXIT_RUNNER_NONZERO

    def _guarded(self, name: str, step: Callable[[], None]) -> None:
        """Run one cleanup step; a failure is recorded, never fatal to cleanup."""
        try:
            step()
        except Exception as err:  # noqa: BLE001
            self.warn(f"cleanup step {name} failed: {err!r}")

    def _write_sums(self) -> None:
        lines = [
            f"{sha256_file(path)}  {path.relative_to(self.out)}"
            for path in sorted(self.out.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        ]
        (self.out / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def default_repo_root() -> Path:
    """Return the root checkout: the parent of the shared git directory."""
    here = Path(__file__).resolve().parent
    result = subprocess.run(  # noqa: S603
        [
            "git",
            "-C",
            str(here),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise Refused("cannot locate the root checkout; pass --repo-root")
    return Path(result.stdout.strip()).parent


def preflight_local(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    """Refusals that need no host contact."""
    go = args.operator_go
    if go is None or not str(go).strip():
        raise Refused(
            "no --operator-go: this script never infers a go. Pass the operator's "
            "exact go text, given immediately before arming."
        )
    for value in (args.target_x, args.target_y):
        if not math.isfinite(value):
            raise Refused("target coordinates must be finite")
    if args.out.exists() and any(args.out.iterdir()):
        raise Refused(f"--out {args.out} is not empty; evidence is never overwritten")
    identities = {}
    for rel, want in PINNED_SHA256.items():
        path = repo_root / rel
        got = sha256_file(path) if path.is_file() else None
        identities[rel] = got
        if got != want:
            raise Refused(f"pinned file mismatch: {rel} sha256 {got} != {want}")
    if not Path(args.python).is_file():
        raise Refused(f"python not found: {args.python}")
    return identities


def cmd_run(args: argparse.Namespace) -> int:
    """Entry point for ``run``."""
    try:
        repo_root = args.repo_root or default_repo_root()
        if args.python is None:
            args.python = repo_root / ".venv" / "bin" / "python"
        identities = preflight_local(args, repo_root)
        env_file = load_env(repo_root / ".env")
        if not env_file.get("HA_URL") or not env_file.get("HA_TOKEN"):
            raise Refused("HA_URL / HA_TOKEN missing from .env")
    except Refused as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return EXIT_REFUSED
    args.out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **env_file}
    redact = make_redactor(env_file)
    ssh_exp = args.ssh_exp or Path(__file__).resolve().parent / "ha_ssh.exp"
    host = HAHost(env_file["HA_URL"], env_file["HA_TOKEN"], ssh_exp, env_file)
    run = S2Run(
        args,
        host=host,
        proc=Proc(),
        clock=Clock(),
        repo_root=repo_root,
        env=env,
        redact=redact,
    )
    run.save(
        "operator_go.json",
        {
            "operator_go": args.operator_go,
            "recorded_local_utc": Clock().utcnow().isoformat(),
        },
    )
    run.save(
        "preflight.json", {"pinned_sha256": identities, "repo_root": str(repo_root)}
    )
    return run.execute()


# ----------------------------------------------------------------------- score


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def normalize_rows(response: Any) -> list[dict[str, Any]]:
    """Flatten an HA history response (or a list of rows) to dicts.

    A bare timestamp string becomes a row with no state, so it can be shown
    but never counted as a payload.
    """
    rows: list[Any] = []
    if isinstance(response, list):
        for item in response:
            if isinstance(item, list):
                rows.extend(item)
            else:
                rows.append(item)
    out = []
    for row in rows:
        if isinstance(row, str):
            out.append(
                {
                    "last_updated_raw": row,
                    "last_updated": parse_ts(row),
                    "state": None,
                    "has_state": False,
                    "numeric": False,
                }
            )
            continue
        if not isinstance(row, dict):
            continue
        attrs = row.get("attributes") or {}
        raw = row.get("last_updated") or row.get("lu")
        out.append(
            {
                "last_updated_raw": raw,
                "last_updated": parse_ts(raw),
                "state": row.get("state"),
                "has_state": "state" in row,
                "numeric": _num(attrs.get("latitude")) is not None
                and _num(attrs.get("longitude")) is not None,
            }
        )
    return sorted(
        out, key=lambda r: r["last_updated"] or dt.datetime.min.replace(tzinfo=dt.UTC)
    )


def collect_commands(leg: Any) -> list[dict[str, Any]]:
    """Every motion command record in the leg (any dict with sent_at_utc)."""
    found: dict[tuple[Any, ...], dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "sent_at_utc" in node and "command" in node:
                key = (
                    node.get("sent_at_utc"),
                    node.get("command"),
                    node.get("phase"),
                    json.dumps(node.get("kwargs"), sort_keys=True),
                )
                found.setdefault(key, node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(leg)
    return sorted(
        found.values(),
        key=lambda c: (
            parse_ts(c.get("sent_at_utc")) or dt.datetime.max.replace(tzinfo=dt.UTC)
        ),
    )


def score_criterion1(leg: dict[str, Any]) -> dict[str, Any]:
    """Section 2 criterion 1: target_reached and landing <= 0.15 m."""
    final = ((leg.get("final_telemetry") or {}).get("position")) or {}
    target = leg.get("target") or {}
    fx, fy, tx, ty = (
        _num(final.get("x")),
        _num(final.get("y")),
        _num(target.get("x")),
        _num(target.get("y")),
    )
    out: dict[str, Any] = {"stop_reason": leg.get("stop_reason")}
    if None in (fx, fy, tx, ty):
        out.update(status="FAIL", reason="final position or target missing")
        return out
    landing = math.hypot(fx - tx, fy - ty)  # type: ignore[operator]
    out.update(
        final=[fx, fy],
        target=[tx, ty],
        landing_error_m=round(landing, 4),
        tolerance_m=LANDING_TOLERANCE_M,
    )
    ok = leg.get("stop_reason") == "target_reached" and landing <= LANDING_TOLERANCE_M
    out["status"] = "PASS" if ok else "FAIL"
    return out


def score_criterion2(
    leg: dict[str, Any],
    samples: list[dict[str, Any]],
    arm_utc: dt.datetime | None,
    disarm_utc: dt.datetime | None,
) -> dict[str, Any]:
    """Section 2 criterion 2 (+ Amendment 2 s10 sample window)."""
    commands = collect_commands(leg)
    unconfirmed = []
    for cmd in commands:
        stop = cmd.get("stop_result")
        if (
            not isinstance(stop, dict)
            or stop.get("attempted") is not True
            or stop.get("ok") is not True
        ):
            unconfirmed.append(
                {
                    "sent_at_utc": cmd.get("sent_at_utc"),
                    "phase": cmd.get("phase"),
                    "stop_result": stop,
                }
            )
    reason = str(leg.get("stop_reason") or "")
    bad_reason = reason.startswith("stop_failed") or reason == "command_failed"
    notes = []
    if arm_utc and disarm_utc:
        in_window = [
            s
            for s in samples
            if (t := parse_ts(s.get("recorded_at_utc"))) and arm_utc <= t <= disarm_utc
        ]
        window = [_iso(arm_utc), _iso(disarm_utc)]
    else:
        in_window = list(samples)
        window = None
        notes.append(
            "no [arm, disarm] host timestamps: every timing sample evaluated "
            "(a superset of the attempt's window)"
        )
    timeouts = [s for s in in_window if s.get("outcome") == "queue_start_timeout"]
    outcomes: dict[str, int] = {}
    for s in in_window:
        outcomes[str(s.get("outcome"))] = outcomes.get(str(s.get("outcome")), 0) + 1
    ok = commands and not unconfirmed and not bad_reason and not timeouts
    return {
        "status": "PASS" if ok else "FAIL",
        "commands": len(commands),
        "stops_confirmed": len(commands) - len(unconfirmed),
        "unconfirmed_stops": unconfirmed,
        "stop_reason_is_stop_failed_or_command_failed": bad_reason,
        "timing_window": window,
        "timing_samples_in_window": len(in_window),
        "timing_outcomes": outcomes,
        "queue_start_timeouts": timeouts,
        "notes": notes if commands else [*notes, "no motion command records"],
    }


def criterion3_window(leg: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Amendment 6 s5 window: first send -> last send + pulse + stop."""
    commands = collect_commands(leg)
    if not commands:
        return None, "no motion command records"
    first, last = commands[0], commands[-1]
    start, last_sent = (
        parse_ts(first.get("sent_at_utc")),
        parse_ts(last.get("sent_at_utc")),
    )
    if start is None or last_sent is None:
        return None, "command sent_at_utc unparseable"
    refresh = last.get("motion_refresh")
    pulse_ms = _num(refresh.get("elapsed_ms")) if isinstance(refresh, dict) else None
    pulse_source = "motion_refresh.elapsed_ms"
    if pulse_ms is None:
        for key in ("commanded_duration_ms", "pulse_duration_ms"):
            if (pulse_ms := _num(last.get(key))) is not None:
                pulse_source = key
                break
    if pulse_ms is None:
        return (
            None,
            "last command has neither motion_refresh.elapsed_ms nor a commanded duration",
        )
    stop_ms = _num((last.get("stop_result") or {}).get("duration_ms"))
    if stop_ms is None:
        return None, "last command has no stop_result.duration_ms"
    end = last_sent + dt.timedelta(milliseconds=pulse_ms + stop_ms)
    return {
        "start": start,
        "end": end,
        "last_sent": last_sent,
        "pulse_ms": pulse_ms,
        "pulse_source": pulse_source,
        "stop_ms": stop_ms,
    }, None


def _attempts(doc: Any) -> list[dict[str, Any]]:
    if isinstance(doc, dict) and isinstance(doc.get("attempts"), list):
        return [a for a in doc["attempts"] if isinstance(a, dict)]
    if isinstance(doc, dict) and "export" in doc:
        return [doc]
    return []


def select_closing(
    attempts: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> tuple[int | None, list[dict[str, Any]]]:
    """Amendment 5 s1: the first post-leg export that qualifies closes the span."""
    checks = []
    for index, attempt in enumerate(attempts):
        before, after = (
            parse_ts(attempt.get("before_utc")),
            parse_ts(attempt.get("after_utc")),
        )
        age = _receipt_age(attempt.get("export"))
        inside = (
            [
                r["last_updated_raw"]
                for r in rows
                if r["last_updated"] and before <= r["last_updated"] <= after
            ]
            if before and after
            else None
        )
        check = {
            "index": index,
            "bracketed": bool(before and after),
            "receipt_age_s": age,
            "rows_inside_bracket": inside,
        }
        check["qualifies"] = bool(
            before
            and after
            and age is not None
            and age >= POSTLEG_MIN_RECEIPT_AGE_S
            and inside == []
        )
        checks.append(check)
        if check["qualifies"]:
            return index, checks
    return None, checks


def admission(
    leg: dict[str, Any],
    closing_before: dt.datetime,
    closing_export: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Amendment 5 s2 counting rule, given the closing bracket's 'before'."""
    out: dict[str, Any] = {"unevaluable": []}
    commands = leg.get("command_results") or []
    lower = parse_ts(commands[0].get("sent_at_utc")) if commands else None
    if lower is None:
        out["unevaluable"].append("command_results[0].sent_at_utc missing")
        return out
    warm = leg.get("position_feed_warmup") or {}
    pipeline = (closing_export or {}).get("position_pipeline") or {}
    fresh_seq, fresh_epoch = _num(warm.get("fresh_sequence")), warm.get("fresh_epoch")
    post_seq, post_epoch = (
        _num(pipeline.get("latest_sequence")),
        pipeline.get("latest_epoch"),
    )
    out.update(
        lower_exclusive=_iso(lower),
        upper_inclusive=_iso(closing_before),
        fresh_sequence=fresh_seq,
        fresh_epoch=fresh_epoch,
        post_sequence=post_seq,
        post_epoch=post_epoch,
    )
    if fresh_seq is None or post_seq is None:
        out["unevaluable"].append("fresh or post-leg sequence missing")
    elif fresh_epoch is None or fresh_epoch != post_epoch:
        out["unevaluable"].append(
            f"epoch changed: fresh {fresh_epoch} != post-leg {post_epoch}"
        )
    span = [
        r
        for r in rows
        if r["last_updated"] and lower < r["last_updated"] <= closing_before
    ]
    bad = [
        r["last_updated_raw"]
        for r in span
        if str(r["state"]).lower() in ("unavailable", "unknown")
    ]
    if bad:
        out["unevaluable"].append(f"unavailable/unknown tracker rows in span: {bad}")
    stateless = [r["last_updated_raw"] for r in span if not r["has_state"]]
    if stateless:
        out["unevaluable"].append(
            "tracker rows carry no state/attributes (timestamps only)"
        )
    counted = [r for r in span if r["numeric"]]
    out["rows"] = len(counted)
    out["rows_in_span_total"] = len(span)
    if fresh_seq is not None and post_seq is not None:
        out["advance"] = int(post_seq - fresh_seq)
        if not out["unevaluable"] and len(counted) != out["advance"]:
            out["unevaluable"].append(
                f"count mismatch: rows {len(counted)} != advance {out['advance']}"
            )
    out["admitted"] = not out["unevaluable"]
    out["counted_rows"] = counted
    return out


def pass_condition(
    times: list[dt.datetime], start: dt.datetime, end: dt.datetime
) -> dict[str, Any]:
    """Amendment 6 s5 pass condition over the driving-window payloads."""
    inside = sorted(t for t in times if start < t <= end)
    if not inside:
        return {"pass": False, "payloads": 0, "reason": "no payload inside the window"}
    first = (inside[0] - start).total_seconds()
    gaps = [(b - a).total_seconds() for a, b in zip(inside, inside[1:], strict=False)]
    tail = (end - inside[-1]).total_seconds()
    max_gap = max(gaps) if gaps else 0.0
    worst = None
    if gaps:
        i = gaps.index(max_gap)
        worst = [_iso(inside[i]), _iso(inside[i + 1])]
    return {
        "pass": first <= C3_MAX_GAP_S
        and max_gap <= C3_MAX_GAP_S
        and tail <= C3_MAX_GAP_S,
        "payloads": len(inside),
        "first_payload_after_start_s": round(first, 6),
        "max_interval_s": round(max_gap, 6),
        "max_interval_between": worst,
        "end_after_last_payload_s": round(tail, 6),
        "bound_s": C3_MAX_GAP_S,
    }


def score_criterion3(  # noqa: C901
    leg: dict[str, Any], prearm: Any, postleg: Any, history: Any
) -> dict[str, Any]:
    """Criterion 3 with Amendment 5 s1/s2 admission; UNEVALUABLE never infers."""
    out: dict[str, Any] = {"unevaluable": []}
    window, why = criterion3_window(leg)
    if window is None:
        out["unevaluable"].append(f"window: {why}")
    else:
        out["window"] = {
            "start": _iso(window["start"]),
            "end": _iso(window["end"]),
            "seconds": round((window["end"] - window["start"]).total_seconds(), 6),
            "last_pulse_ms": window["pulse_ms"],
            "last_pulse_source": window["pulse_source"],
            "last_stop_ms": window["stop_ms"],
        }
        first_cmd = (leg.get("command_results") or [{}])[0].get("sent_at_utc")
        if parse_ts(first_cmd) != window["start"]:
            out.setdefault("notes", []).append(
                "command_results[0] is not the earliest command record"
            )
    pre = _attempts(prearm)
    selected = prearm.get("selected_index") if isinstance(prearm, dict) else None
    pre_export = (
        pre[selected]
        if isinstance(selected, int) and selected < len(pre)
        else (pre[-1] if pre else None)
    )
    if pre_export is None:
        out["unevaluable"].append("no pre-arm export")
    else:
        bracketed = parse_ts(pre_export.get("before_utc")) and parse_ts(
            pre_export.get("after_utc")
        )
        age = _receipt_age(pre_export.get("export"))
        out["prearm"] = {"bracketed": bool(bracketed), "receipt_age_s": age}
        if not bracketed:
            out["unevaluable"].append("pre-arm export has no HA-host UTC brackets")
        if age is None or age < PREARM_MIN_RECEIPT_AGE_S:
            out["unevaluable"].append(
                f"pre-arm receipt_age_s {age} < {PREARM_MIN_RECEIPT_AGE_S}"
            )
        pipe = (pre_export.get("export") or {}).get("position_pipeline") or {}
        out["prearm"]["presentation_stream_replacements"] = pipe.get(
            "presentation_stream_replacements"
        )
    post = _attempts(postleg)
    if not post:
        out["unevaluable"].append("no post-leg export")
    elif not all(
        parse_ts(a.get("before_utc")) and parse_ts(a.get("after_utc")) for a in post
    ):
        out["unevaluable"].append("post-leg export(s) have no HA-host UTC brackets")
    query = (history or {}).get("query") if isinstance(history, dict) else None
    rows = normalize_rows(
        (history or {}).get("response") if isinstance(history, dict) else history
    )
    q_start = parse_ts((query or {}).get("start_time"))
    q_end = parse_ts((query or {}).get("end_time"))
    if q_start is not None:
        synthetic = [
            r for r in rows if r["last_updated"] and r["last_updated"] <= q_start
        ]
        out["synthetic_start_rows_excluded"] = len(synthetic)
        rows = [r for r in rows if r["last_updated"] and r["last_updated"] > q_start]
    if out["unevaluable"]:
        out["status"] = "UNEVALUABLE"
        return out
    closing_index, checks = select_closing(post, rows)
    out["postleg_checks"] = checks
    if closing_index is None:
        out["unevaluable"].append(
            "no post-leg export with receipt_age_s >= 3 and an empty bracket"
        )
        out["status"] = "UNEVALUABLE"
        return out
    closing = post[closing_index]
    closing_before = parse_ts(closing["before_utc"])
    lower = parse_ts((leg.get("command_results") or [{}])[0].get("sent_at_utc"))
    if q_start is None or q_end is None:
        out["unevaluable"].append("history query start/end not recorded")
    else:
        if lower is None or (lower - q_start).total_seconds() < HISTORY_MIN_LEAD_S:
            out["unevaluable"].append(
                "history start_time is not >= 60 s before the first bound"
            )
        if q_end < parse_ts(closing["after_utc"]):  # type: ignore[operator]
            out["unevaluable"].append("history end_time precedes the closing bracket")
    adm = admission(leg, closing_before, closing.get("export") or {}, rows)  # type: ignore[arg-type]
    out["unevaluable"].extend(adm["unevaluable"])
    counted = adm.pop("counted_rows")
    out["admission"] = adm
    post_pipe = (closing.get("export") or {}).get("position_pipeline") or {}
    out["postleg_presentation_stream_replacements"] = post_pipe.get(
        "presentation_stream_replacements"
    )
    out["closing_index"] = closing_index
    if out["unevaluable"]:
        out["status"] = "UNEVALUABLE"
        return out
    result = pass_condition(
        [r["last_updated"] for r in counted], window["start"], window["end"]
    )  # type: ignore[index]
    out["pass_condition"] = result
    out["status"] = "PASS" if result["pass"] else "MISS"
    return out


def score_criterion4(  # noqa: C901
    raw: Any,
    live: Any,
    postleg: Any,
    ref: dt.datetime | None,
    ref_source: str,
    legacy: bool,
) -> dict[str, Any]:
    """Criterion 4: disarmed in RAW and live, >= 15 s after the leg/disarm."""
    out: dict[str, Any] = {
        "reference": _iso(ref),
        "reference_source": ref_source,
        "problems": [],
    }
    entries = (raw or {}).get("entries") if isinstance(raw, dict) else None
    read_at = parse_ts((raw or {}).get("before_utc")) if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        out["problems"].append("no RAW gate read")
        entries = []
    motion = [e for e in entries if e.get("domain") == MOTION_DOMAIN]
    base = [e for e in entries if e.get("domain") == BASE_DOMAIN]
    if len(motion) != 1:
        out["problems"].append(
            f"expected one {MOTION_DOMAIN} entry, found {len(motion)}"
        )
    else:
        flags = motion[0].get("flags") or {}
        for key in ("enable_experimental_motion", "enable_supervised_qualification"):
            if flags.get(key) is not False:
                out["problems"].append(
                    f"RAW {MOTION_DOMAIN}.{key} = {flags.get(key)!r}"
                )
    if (
        len(base) != 1
        or (base[0].get("flags") or {}).get("enable_experimental_motion") is not False
    ):
        out["problems"].append(
            "RAW base legacy enable_experimental_motion not proved false"
        )
    if ref is None or read_at is None:
        out["problems"].append("RAW read time or reference time not recorded")
    else:
        out["raw_after_reference_s"] = round((read_at - ref).total_seconds(), 3)
        if out["raw_after_reference_s"] < RAW_AFTER_DISARM_S:
            out["problems"].append(
                f"RAW read {out['raw_after_reference_s']} s < 15 s after"
            )
    live_export, live_note = None, None
    if isinstance(live, dict) and "export" in live:
        live_export = live.get("export")
        live_at = parse_ts(live.get("before_utc"))
        if (
            ref is None
            or live_at is None
            or (live_at - ref).total_seconds() < RAW_AFTER_DISARM_S
        ):
            out["problems"].append("final live read not proven >= 15 s after")
        else:
            out["live_after_reference_s"] = round((live_at - ref).total_seconds(), 3)
    elif legacy and _attempts(postleg):
        live_export = _attempts(postleg)[-1].get("export")
        live_note = (
            "legacy evidence: live state from the post-leg export, time not recorded"
        )
    else:
        out["problems"].append("no final live read")
    if live_export is not None:
        gate = _gate(live_export)
        for key in ("enabled", "supervised_qualification", "real_motion_allowed"):
            if gate.get(key) is not False:
                out["problems"].append(f"live {key} = {gate.get(key)!r}")
    if live_note:
        out["notes"] = [live_note]
    out["status"] = "PASS" if not out["problems"] else "FAIL"
    return out


def _rtk_left_fix(leg: dict[str, Any]) -> list[str]:
    labels = []
    for sample in leg.get("samples") or []:
        pos = ((sample or {}).get("telemetry") or {}).get("position") or {}
        labels.append(pos.get("rtk_status_label"))
    labels.append(
        (((leg.get("final_telemetry") or {}).get("position")) or {}).get(
            "rtk_status_label"
        )
    )
    return [str(x) for x in labels if x is not None and x != "Fix"]


def effective_stop_reason(leg: dict[str, Any]) -> Any:
    """Amendment 9 s3a: a calibration failure carries its real cause inside."""
    reason = leg.get("stop_reason")
    if reason == "vio_calibration_failed":
        inner = ((leg.get("vio") or {}).get("calibration") or {}).get("reason")
        if inner:
            return inner
    return reason


def ble_lost_after_send(leg: dict[str, Any], post_epoch: Any) -> bool | None:
    """Amendment 9 s3a: True on a ble_* cause or an epoch change; None if unproved.

    ``post_epoch`` is ``latest_epoch`` from the earliest post-leg export, taken
    after the disarm, so a reconnect after the leg also counts (accepted bias).
    """
    if str(effective_stop_reason(leg)).startswith("ble_"):
        return True
    fresh = (leg.get("position_feed_warmup") or {}).get("fresh_epoch")
    if fresh is None or post_epoch is None:
        return None
    return fresh != post_epoch


def first_postleg_epoch(postleg: Any) -> Any:
    """Return ``latest_epoch`` from the earliest post-leg export attempt."""
    for attempt in _attempts(postleg):
        export = attempt.get("export")
        if isinstance(export, dict):
            return (export.get("position_pipeline") or {}).get("latest_epoch")
    return None


#: Amendment 9 s3c: a card abort or a missing reason after arming.
OPERATOR_OR_MISSING_REASONS = {None, "operator_stop"}


def decide(ctx: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:  # noqa: C901, PLR0911, PLR0912
    """Outcome: pre-dispatch classes, then Amendment 5 s5 precedence."""
    path: list[dict[str, Any]] = []

    def step(name: str, applies: bool, detail: str = "") -> bool:
        path.append({"step": name, "applies": applies, "detail": detail})
        return applies

    disarm_unproved = ctx["disarm_proved"] is False
    if step(
        "arm helper failed (arm not proved)",
        ctx["arm_rc"] not in (None, 0),
        f"rc {ctx['arm_rc']}",
    ):
        if ctx["disarm_proved"] is True and ctx["leg"] is None:
            return (
                "INCONCLUSIVE",
                path,
                "arm helper failed but both flags are proved false: pre-dispatch (Amendment 6 s5)",
            )
        return "FAIL", path, "arm cannot be proved; the session ends"
    if ctx["leg"] is None:
        c4_pre = ctx.get("c4_pre")
        pre_problems = c4_pre["problems"] if c4_pre else []
        if step(
            "disarm not proved",
            disarm_unproved or bool(pre_problems),
            "; ".join(pre_problems),
        ):
            return "FAIL", path, "disarm cannot be proved; the session ends"
        if step("armed readback not seen", ctx["readback_ok"] is False):
            return "INCONCLUSIVE", path, "pre-dispatch (Amendment 3 s3)"
        if step("runner exited before a leg record", ctx["runner_rc"] not in (None, 0)):
            if ctx["session_after_arm"]:
                return (
                    "INCONCLUSIVE",
                    path,
                    (
                        "a session started after arming but no leg record: real-call "
                        "response lost (Amendment 6 s12: INCONCLUSIVE, no hand-scoring)"
                    ),
                )
            return (
                "INCONCLUSIVE",
                path,
                "runner halt before the real call (pre-dispatch)",
            )
        step("no leg record", True)
        return "INCONCLUSIVE", path, "no dispatch record (pre-dispatch or not run)"
    leg = ctx["leg"]
    sent = _num(leg.get("commands_sent")) or 0
    reason = leg.get("stop_reason")
    if step("nothing sent", sent == 0, f"stop_reason {reason}"):
        if disarm_unproved:
            return "FAIL", path, "disarm cannot be proved"
        if (
            reason in NOTHING_SENT_INCONCLUSIVE
            or reason in OPERATOR_OR_MISSING_REASONS
            or str(reason).startswith("ble_")
        ):
            return (
                "INCONCLUSIVE",
                path,
                f"{reason} with nothing sent (retry after >= 60 s)",
            )
        return (
            "FAIL",
            path,
            f"real-call refusal {reason} with commands_sent 0 (Amendment 2 s3)",
        )
    c1, c2, c3, c4 = ctx["c1"], ctx["c2"], ctx["c3"], ctx["c4"]
    if step(
        "s5.0 disarm/arm not proved (criterion 4, Amendment 6 s4)",
        disarm_unproved or c4["status"] != "PASS",
        "; ".join(c4["problems"]),
    ):
        return "FAIL", path, "disarm cannot be proved"
    if ctx["debug_only_cause"] and (c3["status"] == "MISS" or c2["status"] == "FAIL"):
        step("debug-only exception recorded as a criterion 2/3 cause", True)
        return (
            "INCONCLUSIVE",
            path,
            "debug-only exception caused the criterion-2/3 miss (Amendment 6 s3, step 2a)",
        )
    op = ctx["operator_stop"]
    named_refusal = reason != "target_reached"
    # Amendment 9 s3c and s3a run for every criterion-3 status, ahead of s5.1-s5.4.
    if step(
        "A9 s3c operator_stop or missing stop_reason after a send",
        reason in OPERATOR_OR_MISSING_REASONS,
        str(reason),
    ):
        return (
            "INCONCLUSIVE",
            path,
            f"stop_reason {reason!r} after arming (Amendment 9 s3c)",
        )
    if named_refusal:
        lost = ble_lost_after_send(leg, ctx.get("post_epoch_first"))
        if step(
            "A9 s3a BLE loss after a send",
            lost is not False,
            f"{effective_stop_reason(leg)}; lost={lost}",
        ):
            if c2["unconfirmed_stops"]:
                return "FAIL", path, "BLE loss after a send with an unconfirmed stop"
            if lost is None:
                return (
                    "INCONCLUSIVE",
                    path,
                    "BLE liveness not proved: an epoch is missing (Amendment 9 s3a)",
                )
            return (
                "INCONCLUSIVE",
                path,
                "BLE loss after the first send, all stops confirmed (Amendment 2 s8, 9 s3a)",
            )
    if named_refusal and c3["status"] == "UNEVALUABLE":
        # Amendment 9 s3: a named refusal after a send fails on criterion 1
        # alone, so an unevaluable criterion 3 no longer masks it. The
        # operator-stop and RTK exceptions keep their precedence.
        rtk_left = _rtk_left_fix(leg)
        if op is None:
            step("A9 named refusal after send", False, "operator stop NOT RECORDED")
            return (
                "UNDETERMINED",
                path,
                "operator stop mid-leg not recorded; rescore with --operator-stop yes|no",
            )
        if step("A9 named refusal after send: operator stop", op is True):
            return (
                "INCONCLUSIVE",
                path,
                "operator stop mid-leg (Amendment 5 s5.2), no retry",
            )
        if step(
            "A9 named refusal after send: RTK left Fix", bool(rtk_left), str(rtk_left)
        ):
            return "INCONCLUSIVE", path, "RTK left Fix mid-leg"
        step("A9 named refusal after send", True, str(effective_stop_reason(leg)))
        return (
            "FAIL",
            path,
            f"named stop reason {reason} after a send; criterion 3 not needed (Amendment 9 s3)",
        )
    if step(
        "s5.1 criterion 3 unevaluable",
        c3["status"] == "UNEVALUABLE",
        "; ".join(c3["unevaluable"]),
    ):
        return "INCONCLUSIVE", path, "criterion 3 unevaluable (Amendment 5 s5.1)"
    if op is None:
        step("s5.2 operator stop mid-leg", False, "NOT RECORDED")
        return (
            "UNDETERMINED",
            path,
            (
                "operator stop mid-leg not recorded (invisible in telemetry); rescore with "
                "--operator-stop yes|no"
            ),
        )
    if step("s5.2 operator stop mid-leg", op is True):
        return (
            "INCONCLUSIVE",
            path,
            "operator stop mid-leg (Amendment 5 s5.2), no retry",
        )
    warm_ok = (leg.get("position_feed_warmup") or {}).get("ok") is True
    if step(
        "s5.3 criterion 3 miss with warm-up ok", c3["status"] == "MISS" and warm_ok
    ):
        return (
            "FAIL",
            path,
            "criterion 3 miss with position_feed_warmup.ok (Amendment 5 s5.3)",
        )
    rtk = _rtk_left_fix(leg)
    if step("s5.4 RTK left Fix mid-leg", bool(rtk), str(rtk)):
        return "INCONCLUSIVE", path, "RTK left Fix mid-leg"
    if step(
        "s5.4 stop reason other than target_reached",
        reason != "target_reached",
        str(reason),
    ):
        if str(reason).startswith("ble_") and not c2["unconfirmed_stops"]:
            return (
                "INCONCLUSIVE",
                path,
                "BLE drop after the first send, all stops confirmed",
            )
        return "FAIL", path, f"named stop reason {reason}"
    if step("s5.4 criterion 2", c2["status"] != "PASS"):
        return "FAIL", path, "unconfirmed stop / command_failed / queue_start_timeout"
    if step("s5.4 criterion 1", c1["status"] != "PASS", str(c1.get("landing_error_m"))):
        return "FAIL", path, "landing outside 0.15 m"
    if step("s5.4 criterion 3 miss without warm-up", c3["status"] == "MISS"):
        return (
            "BUILD_MISMATCH_NOT_SCORED",
            path,
            (
                "criterion 3 missed without position_feed_warmup.ok: Amendment 6 s5 says stop"
            ),
        )
    step("all four criteria", True)
    return "PASS", path, "criteria 1-4 met"


def register_falsifier(root: Path, leg: dict[str, Any] | None) -> dict[str, Any]:
    """Amendment 9 s4: say whether this attempt tests the speed-register hypothesis."""
    pre = _load(root / "travel_speed_prearm.json") or {}
    post = _load(root / "travel_speed_postleg.json") or {}
    both_pinned = pre.get("pinned") is True and post.get("pinned") is True
    cause = effective_stop_reason(leg) if leg else None
    if not both_pinned:
        status = "NOT_TESTED: register not proved pinned before and after"
    elif cause == "insufficient_calibration_distance":
        status = "FIRED: pinned register, calibration still short"
    elif ((leg or {}).get("vio") or {}).get("calibration", {}).get("passed") is True:
        status = (
            "SUPPORTED: pinned register, calibration distance normal "
            "(support, not proof)"
        )
    else:
        status = f"NOT_TESTED: other cause {cause!r}"
    return {
        "status": status,
        "prearm_pinned": pre.get("pinned"),
        "prearm_speed_mps": pre.get("speed_mps"),
        "postleg_pinned": post.get("pinned"),
        "postleg_speed_mps": post.get("speed_mps"),
        "effective_stop_reason": cause,
    }


def score_dir(
    root: Path,
    *,
    operator_stop: bool | None = None,
    debug_only_cause: bool | None = None,
) -> dict[str, Any]:
    """Score one attempt directory; pure, offline."""
    summary = _load(root / "summary.json")
    record = _load(root / "operator_record.json") or {}
    if operator_stop is None:
        # Amendment 9 s7: the canonical key, then S2's spelling.
        for key in ("operator_stop_mid_leg", "operator_stop"):
            if isinstance(record.get(key), bool):
                operator_stop = record[key]
                break
    if debug_only_cause is None:
        debug_only_cause = bool(record.get("debug_only_exception_cause"))
    legs = sorted((root / "runner").glob("leg_*.json")) + sorted(
        root.glob("leg_*.json")
    )
    if len(legs) > 1:
        raise SystemExit(f"more than one leg record in {root}: {legs}")
    leg = _load(legs[0]) if legs else None
    arm = _load(root / "arm.json") or {}
    disarm = _load(root / "disarm.json") or {}
    disarm_attempts = disarm.get("attempts") or []
    arm_utc = parse_ts(arm.get("before_utc"))
    disarm_utc = (
        parse_ts(disarm_attempts[-1].get("after_utc")) if disarm_attempts else None
    )
    samples: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in [
        *sorted((root / "runner").glob("timing_after_*.json")),
        *sorted(root.glob("timing_after_*.json")),
        root / "timing_post.json",
    ]:
        for s in (_load(path) or {}).get("samples") or []:
            samples.setdefault(
                (s.get("recorded_at_utc"), s.get("command"), s.get("is_stop")), s
            )
    prearm, postleg = _load(root / "prearm.json"), _load(root / "postleg.json")
    post_epoch_first = first_postleg_epoch(postleg)
    history = _load(root / "tracker_history.json")
    verdict: dict[str, Any] = {
        "dir": str(root),
        "leg_file": str(legs[0]) if legs else None,
    }
    ctx: dict[str, Any] = {
        "leg": leg,
        "arm_rc": ((summary or {}).get("arm") or {}).get("rc"),
        "readback_ok": ((summary or {}).get("readback") or {}).get("ok"),
        "runner_rc": ((summary or {}).get("runner") or {}).get("rc"),
        "disarm_proved": ((summary or {}).get("disarm") or {}).get("proved"),
        "operator_stop": operator_stop,
        "debug_only_cause": debug_only_cause,
        "post_epoch_first": post_epoch_first,
        "session_after_arm": False,
    }
    if leg is None:
        if disarm_attempts:
            verdict["criterion_4"] = ctx["c4_pre"] = score_criterion4(
                _load(root / "raw_gate.json"),
                _load(root / "live_final.json"),
                postleg,
                disarm_utc,
                "disarm helper return (host UTC)",
                legacy=False,
            )
        last = (
            _gate((_attempts(postleg) or [{}])[-1].get("export")).get("last_session")
            or {}
        )
        started = parse_ts(last.get("started_at"))
        ctx["session_after_arm"] = bool(started and arm_utc and started > arm_utc)
    else:
        verdict["criterion_1"] = ctx["c1"] = score_criterion1(leg)
        verdict["criterion_2"] = ctx["c2"] = score_criterion2(
            leg, list(samples.values()), arm_utc, disarm_utc
        )
        verdict["criterion_3"] = ctx["c3"] = score_criterion3(
            leg, prearm, postleg, history
        )
        window, _ = criterion3_window(leg)
        ref, source = (
            (disarm_utc, "disarm helper return (host UTC)")
            if disarm_utc
            else (
                (window["end"], "criterion-3 window end (leg end)")
                if window
                else (None, "none")
            )
        )
        verdict["criterion_4"] = ctx["c4"] = score_criterion4(
            _load(root / "raw_gate.json"),
            _load(root / "live_final.json"),
            postleg,
            ref,
            source,
            legacy=summary is None,
        )
        verdict["position_feed_warmup"] = leg.get("position_feed_warmup")
    outcome, path, why = decide(ctx)
    verdict["register_falsifier"] = register_falsifier(root, leg)
    verdict.update(
        outcome=outcome,
        reason=why,
        precedence_path=path,
        inputs={
            "operator_stop_mid_leg": operator_stop,
            "debug_only_exception_cause": debug_only_cause,
            "run_summary_present": summary is not None,
        },
    )
    return verdict


def cmd_score(args: argparse.Namespace) -> int:
    """Entry point for ``score``."""
    op = {"yes": True, "no": False, None: None}[args.operator_stop]
    debug = True if args.debug_only_exception_cause else None
    verdict = score_dir(args.dir, operator_stop=op, debug_only_cause=debug)
    print(json.dumps(verdict, indent=1, default=str))
    return 0


# ------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="orchestrate one supervised attempt")
    run.add_argument("--target-x", type=float, required=True)
    run.add_argument("--target-y", type=float, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--operator-go", default=None, help="the operator's go, verbatim")
    run.add_argument("--label", default="S2")
    run.add_argument("--profiler", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument(
        "--root-level",
        choices=ROOT_LEVELS,
        default=None,
        help="operator-supplied root logger level to restore (HA cannot report it)",
    )
    run.add_argument(
        "--min-cached-vio-sun",
        type=float,
        default=SUN_MIN_CACHED_VIO_DEG,
        help="sun floor for a cached VIO window (Amendment 8: operator waiver to 10)",
    )
    run.add_argument("--raw-delay", type=float, default=RAW_DEFAULT_DELAY_S)
    run.add_argument("--repo-root", type=Path, default=None)
    run.add_argument("--python", type=Path, default=None)
    run.add_argument("--ssh-exp", type=Path, default=None)
    score = sub.add_parser("score", help="score a saved attempt directory offline")
    score.add_argument("dir", type=Path)
    score.add_argument("--operator-stop", choices=("yes", "no"), default=None)
    score.add_argument(
        "--debug-only-exception-cause",
        action="store_true",
        help="the analyst found a criterion-2/3 miss caused by a debug-only exception",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch the subcommand."""
    args = build_parser().parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
