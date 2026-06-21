"""HTTP compatibility helpers for Mammotion's cloud API."""

import base64
import hashlib
import hmac
import json
import secrets
import time
from json import JSONDecodeError

from aiohttp import ClientSession
from pymammotion.const import MAMMOTION_DOMAIN
from pymammotion.http.http import MammotionHTTP
from pymammotion.http.model.http import LoginResponseData, Response
from pymammotion.http.model.response_factory import response_factory

OAUTH2_CLIENT_ID = "GxebgSt8si6pKqR"
OAUTH2_CLIENT_SECRET = "JP0508SRJFa0A90ADpzLINDBxMa4Vj"
OAUTH2_TOKEN_ENDPOINT = "/oauth2/token"


def _oauth_signature(request: dict[str, str], timestamp: int) -> str:
    """Create Mammotion's OAuth2 request signature."""
    payload = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    message = f"{OAUTH2_CLIENT_ID}{timestamp}{OAUTH2_TOKEN_ENDPOINT}{payload}"
    secret = hashlib.md5(  # noqa: S324 - protocol-defined signature
        OAUTH2_CLIENT_SECRET.encode()
    ).hexdigest()
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


class MammotionResponseError(Exception):
    """Raised when Mammotion returns an unreadable successful response."""


class MammotionHTTPCompat(MammotionHTTP):
    """Handle Mammotion JSON responses that omit the content-type header."""

    async def login(self, account: str, password: str) -> Response[LoginResponseData]:
        """Log in while accepting a JSON body without a declared media type."""
        self.account = account
        self._password = password
        timestamp = int(time.time() * 1000)
        self._headers["App-Version"] = "NOT HA,2.3.8.19"
        login_request = {
            "username": account,
            "password": base64.b64encode(password.encode()).decode(),
            "client_id": OAUTH2_CLIENT_ID,
            "grant_type": "password",
            "authType": "0",
        }
        client_id = f"{int(time.time() * 1000)}_{secrets.randbelow(10_000_000):07d}_1"
        async with (
            ClientSession(MAMMOTION_DOMAIN) as session,
            session.post(
                OAUTH2_TOKEN_ENDPOINT,
                headers={
                    **self._headers,
                    "Ma-App-Key": OAUTH2_CLIENT_ID,
                    "Ma-Signature": _oauth_signature(login_request, timestamp),
                    "Ma-Timestamp": str(timestamp),
                    "Client-Id": client_id,
                    "Client-Type": "1",
                },
                params=login_request,
            ) as response,
        ):
            if response.status != 200:
                return Response.from_dict(
                    {"code": response.status, "msg": "Login failed"}
                )
            try:
                data = await response.json(content_type=None)
            except (JSONDecodeError, TypeError) as err:
                raise MammotionResponseError from err
            if not isinstance(data, dict):
                raise MammotionResponseError

        login_response = response_factory(Response[LoginResponseData], data)
        if login_response.data is None:
            return Response.from_dict(
                {"code": login_response.code, "msg": "Login failed"}
            )

        self.login_info = login_response.data
        self._headers["Authorization"] = f"Bearer {self.login_info.access_token}"
        self.response = login_response
        self.msg = login_response.msg
        self.code = login_response.code
        return login_response
