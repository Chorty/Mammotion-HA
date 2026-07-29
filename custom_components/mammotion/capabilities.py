"""Centralized tri-state device capability classification."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class CapabilityState(StrEnum):
    """A capability decision that never turns absence into support."""

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


def _read_path(value: Any, path: str) -> Any:
    """Read a dotted attribute path without trusting vendor object shape."""
    current = value
    for part in path.split("."):
        try:
            current = getattr(current, part)
        except AttributeError, TypeError:
            return None
    return current


def _family_from_identity(name: str, product_key: str) -> str:
    """Classify known product families without fuzzy substring matching."""
    identity = f"{name} {product_key}".strip().upper()
    if identity.startswith("LUBA") or " LUBA" in identity:
        return "luba"
    if identity.startswith("YUKA") or " YUKA" in identity:
        return "yuka"
    if identity.startswith("SPINO") or " SPINO" in identity:
        return "spino"
    if identity.startswith("RTK") or " RTK" in identity:
        return "rtk"
    if any(token in identity for token in ("PILE", "CHARGING STATION", "ACCESSORY")):
        return "accessory"
    return "unknown"


def capability_snapshot(coordinator: Any) -> dict[str, Any]:
    """Return identity evidence plus conservative support decisions.

    ``manual_motion`` is intentionally YES only for the LUBA family with live
    acceptance evidence in this fork. Other families remain UNKNOWN until
    equivalent hardware acceptance exists; accessory roles are explicit NO.
    """
    device = getattr(coordinator, "device", None)
    raw_name = str(
        getattr(coordinator, "device_name", None)
        or getattr(device, "device_name", "")
        or ""
    )
    product_key = str(
        getattr(device, "product_key", None)
        or _read_path(getattr(coordinator, "data", None), "product_key")
        or ""
    )
    family = _family_from_identity(raw_name, product_key)
    firmware = {
        "main_controller": _read_path(
            getattr(coordinator, "data", None),
            "device_firmwares.main_controller",
        ),
        "rtk": _read_path(
            getattr(coordinator, "data", None),
            "device_firmwares.rtk",
        ),
    }
    has_cloud = bool(getattr(coordinator, "has_cloud_account", False))
    runtime_camera = _read_path(
        getattr(coordinator, "data", None),
        "report_data.dev_net.camera_status",
    )

    accessory = family in {"rtk", "accessory"}
    manual_motion = (
        CapabilityState.NO
        if accessory or family == "spino"
        else CapabilityState.YES
        if family == "luba"
        else CapabilityState.UNKNOWN
    )
    mower = (
        CapabilityState.YES
        if family in {"luba", "yuka"}
        else CapabilityState.NO
        if accessory or family == "spino"
        else CapabilityState.UNKNOWN
    )
    camera = (
        CapabilityState.YES
        if runtime_camera not in (None, 0, False, "0")
        else CapabilityState.NO
        if family in {"spino", "rtk", "accessory"}
        else CapabilityState.UNKNOWN
    )
    pool_cleaner = (
        CapabilityState.YES
        if family == "spino"
        else CapabilityState.NO
        if family in {"luba", "yuka", "rtk", "accessory"}
        else CapabilityState.UNKNOWN
    )
    return {
        "identity": {
            "raw_device_name": raw_name,
            "product_key": product_key or None,
            "parsed_family": family,
            "firmware": firmware,
            "cloud_role": "account" if has_cloud else "local_only",
            "runtime_reports": {
                "camera_status_present": runtime_camera is not None,
            },
        },
        "capabilities": {
            "mower": mower.value,
            "manual_motion": manual_motion.value,
            "camera": camera.value,
            "pool_cleaner": pool_cleaner.value,
            "rtk_station": (
                CapabilityState.YES.value
                if family == "rtk"
                else CapabilityState.UNKNOWN.value
            ),
            "firmware_install": (
                CapabilityState.NO.value if accessory else CapabilityState.UNKNOWN.value
            ),
        },
    }
