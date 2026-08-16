"""Read-only probe: try alternate Aliyun OTA query paths.

Tries them against the account's authenticated cloud session, to see if any
returns real firmware info without ever calling the device-facing
device/upgrade trigger. Every path here is a GET/query-style OTA info
lookup, never a device command.
"""  # noqa: INP001

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import aiohttp
from alibabacloud_iot_api_gateway.models import CommonParams, Config, IoTApiRequest
from alibabacloud_tea_util.models import RuntimeOptions

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mammotion_ha_helpers import load_dotenv  # noqa: E402
from pymammotion.aliyun.client import Client as AliyunClient  # noqa: E402
from pymammotion.aliyun.cloud_gateway import CloudIOTGateway  # noqa: E402
from pymammotion.http.http import MammotionHTTP  # noqa: E402

CANDIDATE_PATHS = [
    "/thing/ota/firmware/get",
    "/thing/ota/module/get",
    "/thing/ota/task/query",
    "/thing/ota/records/query",
    "/thing/ota/module/list/get",
    "/uc/listOTAUpgradeJob",
    "/ota/device/upgrade/query",
    "/living/ota/devices/list",
    "/living/ota/firmware/file/get",
    "/living/ota/progress/get",
]


async def main() -> int:
    """Log in and try each candidate OTA-info path, printing the raw response."""
    load_dotenv()
    email = os.environ["MAMMOTION_EMAIL"]
    password = os.environ["MAMMOTION_PASSWORD"]

    async with aiohttp.ClientSession() as session:
        http = MammotionHTTP(session=session)
        login_resp = await http.login(email, password)
        if login_resp.code != 0 or http.login_info is None:
            print("login failed", login_resp.code, login_resp.msg)
            return 1
        devices_resp = await http.get_user_device_list()
        device = devices_resp.data[0]
        print(f"device: {device.device_name} iot_id={device.iot_id}")

        login_info = http.login_info
        country_code = login_info.userInformation.domainAbbreviation
        cloud_client = CloudIOTGateway(http)
        await cloud_client.get_region(country_code)
        await cloud_client.connect()
        await cloud_client.login_by_oauth(country_code)
        await cloud_client.aep_handle()
        await cloud_client.session_by_auth_code()

        config = Config(
            app_key=cloud_client._app_key,  # noqa: SLF001
            app_secret=cloud_client._app_secret,  # noqa: SLF001
            domain=cloud_client.region_response.data.apiGatewayEndpoint,
            protocol="https",
        )
        client = AliyunClient(config)
        iot_token = cloud_client.session_by_authcode_response.data.iotToken

        for path in CANDIDATE_PATHS:
            request = CommonParams(
                api_ver="1.0.0", language="en-US", iot_token=iot_token
            )
            body = IoTApiRequest(
                id=str(uuid.uuid4()),
                params={"iotId": device.iot_id, "deviceName": device.device_name},
                request=request,
                version="1.0",
            )
            print(f"\n=== {path} ===")
            try:
                response = await client.async_do_request(
                    path, "https", "POST", {}, body, RuntimeOptions()
                )
                raw = json.loads(response.body.decode("utf-8"))
                print(json.dumps(raw, indent=2)[:800])
            except Exception as exc:  # noqa: BLE001
                print(f"exception: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
