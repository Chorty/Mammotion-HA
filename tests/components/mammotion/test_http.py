"""Tests for Mammotion HTTP compatibility behavior."""

from unittest.mock import patch

import pytest

from custom_components.mammotion.http import MammotionHTTPCompat


class FakeResponse:
    """Successful response without a JSON content-type header."""

    status = 200

    async def __aenter__(self):
        """Enter the response context."""
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        """Exit the response context."""

    async def json(self, *, content_type=None):
        """Return token data and record that media-type validation was disabled."""
        assert content_type is None
        return {
            "code": 0,
            "msg": "success",
            "data": {
                "access_token": "access-token",
                "token_type": "bearer",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "authorization_code": "authorization-code",
                "userInformation": {
                    "areaCode": "1",
                    "domainAbbreviation": "US",
                    "userId": "user-id",
                    "userAccount": "account-id",
                    "authType": "email",
                },
            },
        }


class FakeSession:
    """Minimal aiohttp session replacement."""

    async def __aenter__(self):
        """Enter the session context."""
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        """Exit the session context."""

    def post(self, *args, **kwargs) -> FakeResponse:
        """Return the fake token response."""
        return FakeResponse()


@pytest.mark.asyncio
async def test_login_accepts_missing_json_content_type() -> None:
    """Mammotion's headerless JSON token response is accepted."""
    with patch(
        "custom_components.mammotion.http.ClientSession", return_value=FakeSession()
    ):
        client = MammotionHTTPCompat()
        response = await client.login("user@example.com", "password")

    assert response.code == 0
    assert client.login_info is not None
    assert client.login_info.userInformation.userAccount == "account-id"
