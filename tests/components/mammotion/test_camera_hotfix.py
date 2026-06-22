"""Regression tests for Mammotion camera recovery."""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.mammotion.camera import (
    MammotionWebRTCCamera,
    async_setup_entry,
)
from custom_components.mammotion.coordinator import (
    DEVICE_NOT_RESPONDING_CODE,
    MammotionBaseUpdateCoordinator,
)
from custom_components.mammotion.services import _get_camera_mower


class ConcreteCoordinator(MammotionBaseUpdateCoordinator):
    """Concrete coordinator used without invoking the HA coordinator constructor."""

    def get_coordinator_data(self, device):
        """Return test data unchanged."""
        return device


def _response(code: int, *, with_data: bool):
    data = None
    if with_data:
        data = MagicMock()
        data.to_dict.return_value = {
            "appid": "app-id",
            "token": "private-stream-token",
            "channelName": "private-channel",
            "uid": "123",
        }
    return SimpleNamespace(code=code, data=data)


def _coordinator(*responses):
    coordinator = object.__new__(ConcreteCoordinator)
    coordinator.device_name = "Test mower"
    coordinator.device = SimpleNamespace(iot_id="private-iot-id")
    coordinator.manager = SimpleNamespace(
        refresh_stream_subscription=AsyncMock(side_effect=responses)
    )
    coordinator.async_send_command = AsyncMock()
    coordinator._stream_data = None
    coordinator._stream_data_fetched_at = 0.0
    coordinator._STREAM_TOKEN_TTL = 300.0
    coordinator._agora_response = None
    coordinator._ice_servers = []
    return coordinator


def _agora_context():
    response = MagicMock()
    response.get_ice_servers.return_value = []
    client = MagicMock()
    client.choose_server = AsyncMock(return_value=response)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    return context, response


@pytest.mark.asyncio
async def test_camera_setup_does_not_fetch_tokens() -> None:
    """Platform setup creates entities without contacting camera cloud APIs."""
    coordinator = SimpleNamespace(async_check_stream_expiry=AsyncMock())
    mower = SimpleNamespace(
        device=SimpleNamespace(device_name="Luba-test"),
        reporting_coordinator=coordinator,
    )
    entry = SimpleNamespace(runtime_data=SimpleNamespace(mowers=[mower]))
    add_entities = MagicMock()

    with (
        patch(
            "custom_components.mammotion.camera.DeviceType.is_luba1",
            return_value=False,
        ),
        patch("custom_components.mammotion.camera.MammotionWebRTCCamera"),
    ):
        await async_setup_entry(MagicMock(), entry, add_entities)

    coordinator.async_check_stream_expiry.assert_not_awaited()
    add_entities.assert_called_once()


@pytest.mark.asyncio
async def test_stream_retries_50504_then_succeeds(caplog) -> None:
    """A temporarily unavailable mower is joined and retried with bounded delays."""
    bad = _response(DEVICE_NOT_RESPONDING_CODE, with_data=False)
    good = _response(200, with_data=True)
    coordinator = _coordinator(bad, bad, good)
    context, agora_response = _agora_context()

    caplog.set_level(logging.DEBUG)
    with (
        patch(
            "custom_components.mammotion.coordinator.AgoraAPIClient",
            return_value=context,
        ),
        patch(
            "custom_components.mammotion.coordinator.asyncio.sleep",
            AsyncMock(),
        ) as sleep,
    ):
        stream_data, returned_agora = await coordinator.async_check_stream_expiry(
            force=True
        )

    assert stream_data is good.data
    assert returned_agora is agora_response
    assert coordinator.manager.refresh_stream_subscription.await_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [2, 4]
    assert coordinator.async_send_command.await_args_list[0].args == (
        "send_todev_ble_sync",
    )
    assert coordinator.async_send_command.await_args_list[1].args == (
        "device_agora_join_channel_with_position",
    )
    assert "private-stream-token" not in caplog.text
    assert "private-iot-id" not in caplog.text


@pytest.mark.asyncio
async def test_permanent_50504_clears_stream_cache() -> None:
    """Permanent cloud unavailability leaves no stale stream credentials."""
    bad = _response(DEVICE_NOT_RESPONDING_CODE, with_data=False)
    coordinator = _coordinator(bad, bad, bad)
    coordinator._stream_data = _response(200, with_data=True)
    coordinator._stream_data_fetched_at = time.monotonic()

    with patch("custom_components.mammotion.coordinator.asyncio.sleep", AsyncMock()):
        stream_data, agora_response = await coordinator.async_check_stream_expiry(
            force=True
        )

    assert stream_data is None
    assert agora_response is None
    assert coordinator.get_stream_data() is None
    assert coordinator._ice_servers == []


@pytest.mark.asyncio
async def test_cached_stream_is_reused() -> None:
    """A valid cached token avoids device commands and network requests."""
    coordinator = _coordinator()
    cached = _response(200, with_data=True)
    coordinator._stream_data = cached
    coordinator._stream_data_fetched_at = time.monotonic()
    coordinator._agora_response = MagicMock()

    stream_data, agora_response = await coordinator.async_check_stream_expiry()

    assert stream_data is cached.data
    assert agora_response is coordinator._agora_response
    coordinator.async_send_command.assert_not_awaited()
    coordinator.manager.refresh_stream_subscription.assert_not_awaited()


@pytest.mark.asyncio
async def test_leave_camera_clears_credentials() -> None:
    """Stopping video sends leave and clears all cached credentials."""
    coordinator = _coordinator()
    coordinator._stream_data = _response(200, with_data=True)
    coordinator._stream_data_fetched_at = time.monotonic()
    coordinator._agora_response = MagicMock()
    coordinator._ice_servers = [MagicMock()]

    await coordinator.leave_webrtc_channel()

    coordinator.async_send_command.assert_awaited_once_with(
        "device_agora_join_channel_with_position", enter_state=0
    )
    assert coordinator.get_stream_data() is None
    assert coordinator._agora_response is None
    assert coordinator._ice_servers == []


@pytest.mark.asyncio
async def test_camera_state_tracks_successful_offer() -> None:
    """Camera reports streaming only after an answer is produced."""
    camera = object.__new__(MammotionWebRTCCamera)
    camera._join_lock = asyncio.Lock()
    camera._agora_handler = SimpleNamespace(candidates=[])
    camera._attr_is_streaming = False
    camera._hass = MagicMock()
    camera.async_write_ha_state = MagicMock()
    stream_data = MagicMock()
    agora_response = MagicMock()
    camera.coordinator = SimpleNamespace(
        async_check_stream_expiry=AsyncMock(return_value=(stream_data, agora_response)),
        clear_stream_data=MagicMock(),
    )
    camera._perform_webrtc_negotiation = AsyncMock(return_value="answer-sdp")
    messages = []

    await camera.async_handle_async_webrtc_offer(
        "offer-sdp", "session", messages.append
    )

    assert camera._attr_is_streaming is True
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_camera_offer_reports_temporary_unavailability() -> None:
    """A missing cloud token produces a temporary error and remains idle."""
    camera = object.__new__(MammotionWebRTCCamera)
    camera._join_lock = asyncio.Lock()
    camera._agora_handler = SimpleNamespace(candidates=[])
    camera._attr_is_streaming = False
    camera._hass = MagicMock()
    camera.async_write_ha_state = MagicMock()
    camera.coordinator = SimpleNamespace(
        async_check_stream_expiry=AsyncMock(return_value=(None, None)),
        clear_stream_data=MagicMock(),
    )
    messages = []

    await camera.async_handle_async_webrtc_offer(
        "offer-sdp", "session", messages.append
    )

    assert camera._attr_is_streaming is False
    assert messages[0].code == "503"


def test_camera_target_resolves_across_entries() -> None:
    """Camera services route to the config entry that owns the entity."""
    first = SimpleNamespace(reporting_coordinator=SimpleNamespace(unique_name="first"))
    second = SimpleNamespace(
        reporting_coordinator=SimpleNamespace(unique_name="second")
    )
    hass = SimpleNamespace(
        config_entries=SimpleNamespace(
            async_entries=lambda domain: [
                SimpleNamespace(runtime_data=SimpleNamespace(mowers=[first])),
                SimpleNamespace(runtime_data=SimpleNamespace(mowers=[second])),
            ]
        )
    )
    registry = SimpleNamespace(
        async_get=lambda entity_id: SimpleNamespace(
            domain="camera", platform="mammotion", unique_id="second_webrtc_camera"
        )
    )

    with patch(
        "custom_components.mammotion.services.er.async_get", return_value=registry
    ):
        assert _get_camera_mower(hass, "camera.second") is second
