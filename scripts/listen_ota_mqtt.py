"""Listen on the Aliyun IoT MQTT OTA topic for the mower's firmware push.

Read-only: logs into the Mammotion cloud, bootstraps the same Aliyun IoT
session `fetch_ota_firmware.py` does, then opens an MQTT connection using the
account's own AEP-issued app-device identity (the same identity the HA
integration already uses for live device state — not the mower's own secret)
and subscribes to the device's default status topics plus the OTA push topic
Aliyun documents for firmware delivery:

    /ota/device/upgrade/{productKey}/{deviceName}

Every message received on any subscribed topic is printed and appended to
scripts/ota_mqtt_capture.jsonl, raw — no envelope-unwrapping assumptions,
since we don't know the OTA payload's exact framing ahead of time. If a
message's payload looks like it carries a firmware download URL, a background
curl is fired immediately to scripts/firmware/ota_firmware.bin.

Usage:
    MAMMOTION_EMAIL=you@example.com MAMMOTION_PASSWORD=... \
        .venv/bin/python scripts/listen_ota_mqtt.py [--device-name LUBA-XXXX] [--timeout 300]
"""  # noqa: INP001

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

import aiohttp
import aiomqtt

sys.path.insert(0, str(Path(__file__).parent))
from mammotion_ha_helpers import load_dotenv  # noqa: E402
from pymammotion.aliyun.cloud_gateway import CloudIOTGateway  # noqa: E402
from pymammotion.http.http import MammotionHTTP  # noqa: E402
from pymammotion.transport.aliyun_mqtt import (  # noqa: E402
    AliyunMQTTConfig,
    AliyunMQTTTransport,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("listen_ota_mqtt")

OUT_PATH = Path(__file__).parent / "ota_mqtt_capture.jsonl"
FIRMWARE_DIR = Path(__file__).parent / "firmware"
URL_FIELD_CANDIDATES = ("dataLocation", "otaUrl", "url", "downloadUrl", "fileUrl")


def _find_firmware_url(node: object) -> str | None:
    """Search a parsed JSON payload for a firmware download URL field."""
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
    FIRMWARE_DIR.mkdir(parents=True, exist_ok=True)
    dest = FIRMWARE_DIR / "ota_firmware.bin"
    part = f"{dest!s}.part"
    done_marker = FIRMWARE_DIR / "DOWNLOAD_DONE"
    (FIRMWARE_DIR / "DOWNLOAD_STARTED").write_text(url)
    LOGGER.info("Firmware URL found, downloading now: %s", url)
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


def _build_mqtt_config(cloud_client: CloudIOTGateway) -> AliyunMQTTConfig:
    """Build the AliyunMQTTConfig exactly as pymammotion.client._setup_aliyun_transport does."""
    aep = cloud_client.aep_response.data  # type: ignore[union-attr]
    region_id = cloud_client.region_response.data.regionId  # type: ignore[union-attr]
    session_data = cloud_client.session_by_authcode_response.data  # type: ignore[union-attr]
    return AliyunMQTTConfig(
        host=f"{aep.productKey}.iot-as-mqtt.{region_id}.aliyuncs.com",
        client_id_base=cloud_client.client_id,
        username=f"{aep.deviceName}&{aep.productKey}",
        device_name=aep.deviceName,
        product_key=aep.productKey,
        device_secret=aep.deviceSecret,
        iot_token=session_data.iotToken,  # type: ignore[union-attr]
    )


async def main() -> int:
    """Log in, open an Aliyun MQTT session, and listen for the OTA push message."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-name", help="Match a specific device name; default: first device"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Seconds to listen before giving up"
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

        LOGGER.info("Bootstrapping Aliyun IoT gateway session...")
        cloud_client = await _bootstrap_cloud_gateway(http)
        config = _build_mqtt_config(cloud_client)

        # Reuse AliyunMQTTTransport purely for its credential-signing logic —
        # we drive aiomqtt ourselves so no topic's raw payload is silently
        # dropped by the transport's envelope-unwrapping dispatch.
        transport = AliyunMQTTTransport(config, cloud_client)
        client_id, mqtt_password = transport._build_credentials()  # noqa: SLF001 - intentional reuse of signing logic
        tls_context = await AliyunMQTTTransport.get_ssl_context()

        ota_topic = f"/ota/device/upgrade/{config.product_key}/{config.device_name}"
        topics = [*transport._default_subscribe_topics(), ota_topic]  # noqa: SLF001

        LOGGER.info("Connecting to Aliyun MQTT broker %s:8883 ...", config.host)
        async with aiomqtt.Client(
            hostname=config.host,
            port=8883,
            username=config.username,
            password=mqtt_password,
            identifier=client_id,
            keepalive=60,
            tls_context=tls_context,
            protocol=aiomqtt.ProtocolVersion.V311,
            timeout=60,
        ) as client:
            for topic in topics:
                await client.subscribe(topic, qos=1)
                LOGGER.info("Subscribed: %s", topic)

            bind_topic = (
                f"/sys/{config.product_key}/{config.device_name}/app/up/account/bind"
            )
            await client.publish(
                bind_topic,
                json.dumps(
                    {
                        "id": "msgid1",
                        "version": "1.0",
                        "request": {"clientId": config.username},
                        "params": {"iotToken": config.iot_token},
                    }
                ),
                qos=1,
            )
            LOGGER.info(
                "Bound. Listening for up to %ds — waiting on the OTA push topic...",
                args.timeout,
            )

            try:
                async with asyncio.timeout(args.timeout):
                    async for message in client.messages:
                        topic = str(message.topic)
                        raw = bytes(message.payload)
                        LOGGER.info("--- message on %s (%d bytes) ---", topic, len(raw))
                        try:
                            text = raw.decode("utf-8")
                        except UnicodeDecodeError:
                            text = None
                        record = {"topic": topic, "text": text}
                        with OUT_PATH.open("a") as fh:
                            fh.write(json.dumps(record) + "\n")
                        if text:
                            LOGGER.info(text)
                            try:
                                parsed = json.loads(text)
                            except json.JSONDecodeError:
                                parsed = None
                            if parsed is not None and (
                                firmware_url := _find_firmware_url(parsed)
                            ):
                                _start_download(firmware_url)
            except TimeoutError:
                LOGGER.warning(
                    "Timed out after %ds with no OTA push received.", args.timeout
                )
                return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
