"""Fetch the Mammotion mower's OTA firmware binary before installing it.

Read-only, two-stage:

1. `http.get_device_ota_firmware()` — the same Mammotion-backend check the HA
   `update` entity polls (custom_components/mammotion/coordinator.py). This only
   confirms a version is available; it carries no download URL.
2. The Aliyun IoT gateway's `/thing/ota/info/queryByUser` API — the endpoint the
   Mammotion app itself calls (via Aliyun's "Breeze" OTA SDK, see
   `LinkOTABusiness.inquiryNewVersion_W` in the decompiled app) to fetch the
   actual `FirmwareInfo` (url/md5/size/version) it then downloads.

Neither call is `http.start_ota_upgrade()` or any device-facing OTA command —
this cannot trigger an install on the mower.

Usage:
    MAMMOTION_EMAIL=you@example.com MAMMOTION_PASSWORD=... \
        .venv/bin/python scripts/fetch_ota_firmware.py [--device-name LUBA-XXXX] [--out-dir scripts/firmware]

Credentials can also go in a .env file (MAMMOTION_EMAIL / MAMMOTION_PASSWORD),
loaded via the same convention as the other scripts in this directory.
"""  # noqa: INP001

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import aiohttp
from alibabacloud_iot_api_gateway.models import CommonParams, Config, IoTApiRequest
from alibabacloud_tea_util.models import RuntimeOptions

sys.path.insert(0, str(Path(__file__).parent))
from mammotion_ha_helpers import load_dotenv  # noqa: E402
from pymammotion.aliyun.client import Client as AliyunClient  # noqa: E402
from pymammotion.aliyun.cloud_gateway import CloudIOTGateway  # noqa: E402
from pymammotion.http.http import MammotionHTTP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("fetch_ota_firmware")


async def _download(session: aiohttp.ClientSession, url: str, dest: Path) -> str:
    """Stream *url* to *dest*, return its md5 hex digest."""
    digest = hashlib.md5()  # noqa: S324 - vendor's own integrity field, not security-sensitive
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with session.get(url) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            async for chunk in resp.content.iter_chunked(1 << 16):
                fh.write(chunk)
                digest.update(chunk)
    return digest.hexdigest()


async def _bootstrap_cloud_gateway(mammotion_http: MammotionHTTP) -> CloudIOTGateway:
    """Run the same Aliyun IoT gateway setup sequence pymammotion.client._connect_iot does."""
    login_info = mammotion_http.login_info
    if login_info is None:
        msg = "login_info is None — login() must succeed before bootstrapping the cloud gateway"
        raise RuntimeError(msg)
    country_code = login_info.userInformation.domainAbbreviation
    cloud_client = CloudIOTGateway(mammotion_http)
    await cloud_client.get_region(country_code)
    await cloud_client.connect()
    await cloud_client.login_by_oauth(country_code)
    await cloud_client.aep_handle()
    await cloud_client.session_by_auth_code()
    return cloud_client


async def _resolve_aliyun_iot_id(
    cloud_client: CloudIOTGateway, device_name: str, fallback_iot_id: str
) -> str | None:
    """Resolve the Aliyun-native iot_id for *device_name*.

    The Mammotion-backend iot_id (used elsewhere in this script) can differ
    from Aliyun's own iot_id for the same physical device; Aliyun's OTA job
    lookup is keyed by the latter, so it must be resolved from the binding
    list rather than reused from the Mammotion device list.
    """
    binding_resp = await cloud_client.list_binding_by_account()
    aliyun_devices = binding_resp.data.data if binding_resp.data else []
    aliyun_device = next(
        (d for d in aliyun_devices if d.device_name == device_name), None
    )
    if aliyun_device is None:
        return None
    if aliyun_device.iot_id != fallback_iot_id:
        LOGGER.info(
            "Aliyun iot_id differs from Mammotion-backend iot_id: %s vs %s",
            aliyun_device.iot_id,
            fallback_iot_id,
        )
    return aliyun_device.iot_id


async def _query_firmware_info(cloud_client: CloudIOTGateway, iot_id: str) -> dict:
    """Call Aliyun's `/thing/ota/info/queryByUser` — the app's own OTA-info lookup."""
    config = Config(
        app_key=cloud_client._app_key,  # noqa: SLF001
        app_secret=cloud_client._app_secret,  # noqa: SLF001
        domain=cloud_client.region_response.data.apiGatewayEndpoint,  # type: ignore[union-attr]
        protocol="https",
    )
    client = AliyunClient(config)
    request = CommonParams(
        api_ver="1.0.0",
        language="en-US",
        iot_token=cloud_client.session_by_authcode_response.data.iotToken,  # type: ignore[union-attr]
    )
    body = IoTApiRequest(
        id=str(uuid.uuid4()),
        params={"iotId": iot_id},
        request=request,
        version="1.0",
    )
    response = await client.async_do_request(
        "/thing/ota/info/queryByUser", "https", "POST", {}, body, RuntimeOptions()
    )
    return json.loads(response.body.decode("utf-8"))


async def _login_and_pick_device(
    http: MammotionHTTP, email: str, password: str, device_name: str | None
):
    """Log in and return the target device, or None (after logging the reason) on failure."""
    login_resp = await http.login(email, password)
    if login_resp.code != 0 or http.login_info is None:
        LOGGER.error("Login failed: code=%s msg=%s", login_resp.code, login_resp.msg)
        return None
    LOGGER.info("Logged in.")

    devices_resp = await http.get_user_device_list()
    devices = devices_resp.data or []
    if not devices:
        LOGGER.error("No devices returned for this account.")
        return None

    if device_name:
        devices = [d for d in devices if d.device_name == device_name] or devices
    device = devices[0]
    LOGGER.info("Using device: %s (iot_id=%s)", device.device_name, device.iot_id)
    return device


async def main() -> int:
    """Log in, check for an available firmware update, and download it if offered."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-name", help="Match a specific device name; default: first device"
    )
    parser.add_argument(
        "--out-dir",
        default="scripts/firmware",
        help="Directory to write the downloaded firmware file into",
    )
    args = parser.parse_args()

    email = os.environ.get("MAMMOTION_EMAIL")
    password = os.environ.get("MAMMOTION_PASSWORD")
    if not email or not password:
        LOGGER.error(
            "Set MAMMOTION_EMAIL and MAMMOTION_PASSWORD (env or .env) before running."
        )
        return 1

    async with aiohttp.ClientSession() as session:
        http = MammotionHTTP(session=session)
        device = await _login_and_pick_device(http, email, password, args.device_name)
        if device is None:
            return 1

        ota_resp = await http.get_device_ota_firmware([device.iot_id])
        check = next(
            (c for c in (ota_resp.data or []) if c.device_id == device.iot_id), None
        )
        if check is None or not check.upgradeable:
            LOGGER.warning(
                "Mammotion backend reports no upgrade offered for this device."
            )
            return 0
        LOGGER.info(
            "Backend confirms upgrade offered: %s -> %s",
            check.current_version,
            check.product_version_info_vo.release_version
            if check.product_version_info_vo
            else "?",
        )

        LOGGER.info("Bootstrapping Aliyun IoT gateway session...")
        cloud_client = await _bootstrap_cloud_gateway(http)

        aliyun_iot_id = await _resolve_aliyun_iot_id(
            cloud_client, device.device_name, device.iot_id
        )
        if aliyun_iot_id is None:
            LOGGER.error(
                "Device %s not found in Aliyun binding list.", device.device_name
            )
            return 1

        LOGGER.info("Querying Aliyun OTA firmware info...")
        raw = await _query_firmware_info(cloud_client, aliyun_iot_id)
        LOGGER.info("Raw firmware-info response:\n%s", json.dumps(raw, indent=2))

        if int(raw.get("code") or 0) != 200:
            LOGGER.error("Firmware-info query failed: %s", raw.get("message") or raw)
            return 1

        data = raw.get("data")
        # Aliyun sometimes double-encodes `data` as a JSON string; handle both.
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict) or not data.get("url"):
            LOGGER.warning("No firmware url in the response — nothing to download.")
            return 0

        download_url = data["url"]
        version = data.get("version", "unknown")
        out_dir = Path(args.out_dir)
        dest = out_dir / f"{device.device_name}_{version}.bin"
        LOGGER.info("Downloading %s -> %s", download_url, dest)
        md5 = await _download(session, download_url, dest)
        expected_md5 = data.get("md5")
        if expected_md5 and expected_md5.lower() != md5.lower():
            LOGGER.warning("md5 mismatch: expected %s, got %s", expected_md5, md5)
        LOGGER.info("Saved %s (%d bytes, md5=%s)", dest, dest.stat().st_size, md5)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
