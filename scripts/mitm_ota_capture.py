"""mitmproxy addon: log OTA/firmware-related HTTP flows and auto-download the firmware.

Run with:
    mitmdump -s scripts/mitm_ota_capture.py

Every request/response whose URL contains one of the OTA keywords below is
appended to scripts/mitm_ota_capture.jsonl as one JSON object per line, with
full request/response headers and bodies (base64 for binary bodies).

The instant a response body is found to carry a firmware download URL (a
`url`/`otaUrl`/`dataLocation`-style field alongside version/md5/size), this
addon shells out to `curl` in the background to fetch it immediately into
scripts/firmware/ — no polling, no manual step. A DOWNLOAD_STARTED/_DONE
sentinel file tracks progress for an external watcher. Nothing else is
blocked or modified — nothing besides that one firmware GET is originated by
this addon.
"""  # noqa: INP001

from __future__ import annotations

import base64
import json
import shlex
import subprocess
from pathlib import Path

from mitmproxy import http

OUT_PATH = Path(__file__).parent / "mitm_ota_capture.jsonl"
FIRMWARE_DIR = Path(__file__).parent / "firmware"
KEYWORDS = ("ota", "firmware", "upgrade", "version/check", ".bin", "oss-")
URL_FIELD_CANDIDATES = ("dataLocation", "otaUrl", "url", "downloadUrl", "fileUrl")

_download_started = False


def _matches(url: str) -> bool:
    lowered = url.lower()
    return any(keyword in lowered for keyword in KEYWORDS)


def _body_json(data: bytes | None) -> dict:
    if not data:
        return {"text": None, "base64": None}
    try:
        return {"text": data.decode("utf-8"), "base64": None}
    except UnicodeDecodeError:
        return {"text": None, "base64": base64.b64encode(data).decode("ascii")}


def _find_firmware_url(node: object) -> str | None:
    """Search a parsed JSON body for a firmware download URL field."""
    if isinstance(node, dict):
        for key in URL_FIELD_CANDIDATES:
            value = node.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        for value in node.values():
            if (found := _find_firmware_url(value)) is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            if (found := _find_firmware_url(item)) is not None:
                return found
    return None


def _start_download(url: str) -> None:
    """Fire a background curl to fetch *url* into scripts/firmware/ immediately."""
    global _download_started  # noqa: PLW0603
    if _download_started:
        return
    _download_started = True
    FIRMWARE_DIR.mkdir(parents=True, exist_ok=True)
    (FIRMWARE_DIR / "DOWNLOAD_STARTED").write_text(url)
    dest = FIRMWARE_DIR / "ota_firmware.bin"
    done_marker = FIRMWARE_DIR / "DOWNLOAD_DONE"
    print(f"[ota-capture] firmware URL found, downloading now: {url}")
    part = f"{dest!s}.part"
    shell_cmd = (
        f"curl -sSL -o {shlex.quote(part)} {shlex.quote(url)} "
        f"&& mv {shlex.quote(part)} {shlex.quote(str(dest))} "
        f"&& date -u +%Y-%m-%dT%H:%M:%SZ > {shlex.quote(str(done_marker))}"
    )
    subprocess.Popen(  # noqa: S603 - fixed argv, no shell metachar injection (shlex.quote on all interpolated values)
        ["/bin/sh", "-c", shell_cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def response(flow: http.HTTPFlow) -> None:
    """Log the flow if its URL matches an OTA keyword; auto-download on a firmware URL."""
    if not _matches(flow.request.pretty_url):
        return
    record = {
        "url": flow.request.pretty_url,
        "method": flow.request.method,
        "request_headers": dict(flow.request.headers),
        "request_body": _body_json(flow.request.content),
        "status_code": flow.response.status_code if flow.response else None,
        "response_headers": dict(flow.response.headers) if flow.response else {},
        "response_body": _body_json(flow.response.content)
        if flow.response
        else {"text": None, "base64": None},
    }
    with OUT_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(
        f"[ota-capture] logged {flow.request.method} {flow.request.pretty_url} -> {record['status_code']}"
    )

    if flow.response is not None:
        try:
            parsed = json.loads(flow.response.content or b"")
        except json.JSONDecodeError, UnicodeDecodeError:
            parsed = None
        if parsed is not None and (firmware_url := _find_firmware_url(parsed)):
            _start_download(firmware_url)
