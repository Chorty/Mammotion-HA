#!/usr/bin/env python3
"""Open the Mammotion Agora outbound-audio probe with live HA stream tokens."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE_HTML = ROOT / "agora_test.html"


def _load_dotenv(path: pathlib.Path) -> None:
    """Load simple KEY=VALUE pairs from a dotenv file."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _post_ha_service(
    ha_url: str,
    token: str,
    service: str,
    payload: dict[str, Any],
    *,
    return_response: bool = False,
) -> dict[str, Any]:
    """Call a Mammotion Home Assistant service."""
    suffix = "?return_response=true" if return_response else ""
    url = (
        ha_url.rstrip("/")
        + f"/api/services/mammotion/{urllib.parse.quote(service)}"
        + suffix
    )
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HA service {service} failed: HTTP {err.code}: {detail}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Unable to reach Home Assistant at {ha_url}: {err}") from err

    if not response_body:
        return {}
    return json.loads(response_body)


def _extract_stream_tokens(response: dict[str, Any]) -> dict[str, Any]:
    service_response = response.get("service_response", response)
    required = ("appid", "channelName", "token", "uid")
    missing = [key for key in required if not service_response.get(key)]
    if missing:
        raise RuntimeError(
            "HA get_tokens response did not include "
            + ", ".join(missing)
            + f": {json.dumps(response, indent=2)}"
        )
    return service_response


def _build_probe_url(tokens: dict[str, Any], frequency: int, duration: float) -> str:
    params = urllib.parse.urlencode(
        {
            "appid": tokens["appid"],
            "channelName": tokens["channelName"],
            "token": tokens["token"],
            "uid": tokens["uid"],
            "frequency": frequency,
            "duration": duration,
            "auto": "1",
        }
    )
    return PROBE_HTML.resolve().as_uri() + "#" + params


def main() -> int:
    """Run the Agora audio probe launcher."""
    _load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description=(
            "Fetch Mammotion camera stream credentials from Home Assistant and "
            "open agora_test.html in automated outbound-audio probe mode."
        )
    )
    parser.add_argument("entity_id", help="Mammotion camera entity_id")
    parser.add_argument("--ha-url", default=os.environ.get("HA_URL"))
    parser.add_argument("--ha-token", default=os.environ.get("HA_TOKEN"))
    parser.add_argument("--frequency", type=int, default=880)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip mammotion.refresh_stream before mammotion.get_tokens.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print a redacted generated local probe URL instead of opening a browser.",
    )
    parser.add_argument(
        "--show-token-url",
        action="store_true",
        help=(
            "With --print-only, print the full token-bearing probe URL. "
            "This exposes a live stream credential in terminal output."
        ),
    )
    args = parser.parse_args()

    if not args.ha_url or not args.ha_token:
        parser.error("Provide --ha-url/--ha-token or set HA_URL/HA_TOKEN in the environment")
    if not PROBE_HTML.exists():
        raise RuntimeError(f"Probe file not found: {PROBE_HTML}")
    if not 100 <= args.frequency <= 4000:
        parser.error("--frequency must be between 100 and 4000 Hz")
    if not 0.1 <= args.duration <= 5:
        parser.error("--duration must be between 0.1 and 5 seconds")

    payload = {"entity_id": args.entity_id}
    if not args.no_refresh:
        _post_ha_service(args.ha_url, args.ha_token, "refresh_stream", payload)

    token_response = _post_ha_service(
        args.ha_url,
        args.ha_token,
        "get_tokens",
        payload,
        return_response=True,
    )
    tokens = _extract_stream_tokens(token_response)
    probe_url = _build_probe_url(tokens, args.frequency, args.duration)
    redacted_url = probe_url.replace(urllib.parse.quote(str(tokens["token"]), safe=""), "REDACTED")

    print("Generated local Agora audio probe URL.")  # noqa: T201
    print(  # noqa: T201
        "Token-bearing URL is sensitive; full URL is not printed by default."
    )

    if not args.print_only:
        webbrowser.open(probe_url)
        print("Opened probe in the default browser.")  # noqa: T201
    elif args.show_token_url:
        print(probe_url)  # noqa: T201
    else:
        print(redacted_url)  # noqa: T201

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as err:  # noqa: BLE001
        print(f"error: {err}", file=sys.stderr)  # noqa: T201
        raise SystemExit(1) from err
