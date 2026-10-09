#!/usr/bin/env python3
"""Exercise native backend position contracts offline, without a live client.

Uses a real DeviceHandle and simulated report bytes. No handle, transport, or
queue is started, and the coordinator's only report request injects local bytes.
This demonstrates software contracts, not hardware qualification.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from pymammotion.data.model.device import MowerDevice
from pymammotion.device.handle import DeviceHandle
from pymammotion.proto import LubaMsg, MctlSys, ReportInfoData, RptDevLocation, RptRtk

# Direct CLI invocation adds scripts/ to sys.path; integration imports need root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom_components.mammotion.backend_capability import (  # noqa: E402
    async_probe_backend_capabilities,
)
from custom_components.mammotion.services import _warm_position_feed  # noqa: E402


async def probe() -> dict:
    """Check receipt sequences and warm-up on the installed backend's real handle."""
    handle = DeviceHandle(
        device_id="offline-contract-probe",
        device_name="Luba-Test",
        initial_device=MowerDevice(name="Luba-Test"),
    )
    report = bytes(
        LubaMsg(
            sys=MctlSys(
                toapp_report_data=ReportInfoData(
                    locations=[
                        RptDevLocation(
                            real_pos_x=10000,
                            real_pos_y=20000,
                            real_toward=900000,
                            pos_type=1,
                            zone_hash=123,
                        )
                    ],
                    rtk=RptRtk(status=4, pos_level=1),
                )
            )
        )
    )
    for _ in range(3):
        await handle.on_raw_message(report)
    latest = getattr(handle, "latest_position_sample", None)
    receipt_sequence = getattr(latest, "sequence", None)
    requests = 0

    async def request_reports(**_kwargs) -> None:
        nonlocal requests
        requests += 1
        await handle.on_raw_message(report)

    coordinator = SimpleNamespace(
        manager=SimpleNamespace(mower=lambda _: handle),
        device_name=handle.device_name,
        async_get_reports=request_reports,
    )
    warmup = await _warm_position_feed(
        coordinator, timeout_seconds=0.05, poll_interval_seconds=0.01
    )
    return {
        "backend": importlib.metadata.version("pymammotion"),
        "capabilities": await async_probe_backend_capabilities(force=True),
        "identical_report_receipt_sequence": receipt_sequence,
        "has_position_epoch": hasattr(handle, "position_epoch"),
        "native_warmup": warmup,
        "simulated_report_requests": requests,
        "offline": True,
    }


def main() -> None:
    """Save the probe result and refuse to report a missing native contract as a pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(probe())
    output = json.dumps(result, indent=2) + "\n"
    args.output.write_text(output)
    print(output)
    if not result["capabilities"]["verified"] or not result["native_warmup"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
