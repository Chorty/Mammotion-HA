#!/usr/bin/env python3
"""Run one already-authorized HA motion service while retaining its evidence.

This helper deliberately does not arm experimental motion or construct a
payload.  Its caller must supply the reviewed request and own the safety
teardown.  It keeps telemetry capture in the same foreground process so a
terminal-session cleanup cannot silently discard a physical-run record.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from mammotion_ha_helpers import load_dotenv, post_service
from motion_capture import sample


def _capture(stop: threading.Event, out: Path, duration: float) -> None:
    deadline = time.monotonic() + duration
    with out.open("w") as handle:
        while time.monotonic() < deadline and not stop.is_set():
            try:
                handle.write(json.dumps(sample(), sort_keys=True) + "\n")
                handle.flush()
            except Exception as err:  # noqa: BLE001 - evidence must include collection faults
                handle.write(json.dumps({"capture_error": repr(err)}) + "\n")
                handle.flush()
            stop.wait(1)


def main() -> int:
    """Capture telemetry around one reviewed motion service call, durably."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-seconds", type=float, default=55)
    args = parser.parse_args()

    load_dotenv()
    payload: dict[str, Any] = json.loads(args.request.read_text())
    stop = threading.Event()
    worker = threading.Thread(
        target=_capture, args=(stop, args.capture, args.capture_seconds), daemon=True
    )
    worker.start()
    time.sleep(1)  # ensure evidence exists before the service can command movement
    record: dict[str, Any]
    try:
        result = post_service(
            os.environ["HA_URL"],
            os.environ["HA_TOKEN"],
            "mammotion",
            args.service,
            payload,
            int(max(60, args.capture_seconds + 30)),
        )
        record = {"request": payload, "result": result}
        code = 0
    except Exception as err:  # noqa: BLE001 - the record must retain any fault
        record = {"request": payload, "service_error": repr(err)}
        code = 1
    finally:
        # Keep post-stop telemetry for the rest of the requested observation window.
        worker.join(timeout=args.capture_seconds + 2)
        stop.set()
        worker.join(timeout=2)
        args.result.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"result": str(args.result), "capture": str(args.capture)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
