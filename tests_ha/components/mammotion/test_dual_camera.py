"""Tests for the separate Luba 2 vision-camera feeds."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.mammotion.agora_websocket import AgoraWebSocketHandler
from custom_components.mammotion.camera import (
    CAMERAS,
    MammotionWebRTCCamera,
    async_setup_entry,
)
from custom_components.mammotion.coordinator import MammotionBaseUpdateCoordinator


class ConcreteCoordinator(MammotionBaseUpdateCoordinator):
    """Concrete coordinator used without invoking its Home Assistant setup."""

    def get_coordinator_data(self, device):
        """Return the test device unchanged."""
        return device


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("device_name", "keys", "uids"),
    [
        ("Luba-AAAAAA", [], []),
        ("Luba-VS00CLD", ["webrtc_camera", "webrtc_camera_right"], [1, 2]),
        ("Luba-VP00CLD", ["webrtc_camera", "webrtc_camera_right"], [1, 2]),
        (
            "Yuka-000CLD",
            ["webrtc_camera", "webrtc_camera_right", "webrtc_camera_rear"],
            [1, 2, 3],
        ),
    ],
)
async def test_setup_adds_one_camera_per_vision_feed(
    device_name: str, keys: list[str], uids: list[int]
) -> None:
    """Vision mowers get left and right feeds, Yuka the rear one too."""
    mower = SimpleNamespace(
        device=SimpleNamespace(device_name=device_name),
        reporting_coordinator=MagicMock(),
    )
    entry = SimpleNamespace(runtime_data=SimpleNamespace(mowers=[mower]))
    add_entities = MagicMock()

    with patch("custom_components.mammotion.camera.MammotionWebRTCCamera") as camera:
        await async_setup_entry(MagicMock(), entry, add_entities)

    descriptions = [call.args[1] for call in camera.call_args_list]
    assert [description.key for description in descriptions] == keys
    assert [description.target_uid for description in descriptions] == uids
    assert len(add_entities.call_args.args[0]) == len(keys)


def test_agora_handler_ignores_the_other_vision_peer() -> None:
    """Online, video, and offline notifications are scoped to one camera UID."""
    handler = AgoraWebSocketHandler(MagicMock(), target_uid=2)
    handler._video_streams[1] = {"ssrcId": 10, "subscribed": True}  # noqa: SLF001

    async def notify_other_peer() -> None:
        await handler._handle_user_online({"_message": {"uid": 1}})  # noqa: SLF001
        await handler._handle_add_video_stream(  # noqa: SLF001
            {"_message": {"uid": 1, "ssrcId": 11}}
        )
        await handler._handle_user_offline({"_message": {"uid": 1}})  # noqa: SLF001

    asyncio.run(notify_other_peer())

    assert handler._online_users == set()  # noqa: SLF001
    assert handler._video_streams == {  # noqa: SLF001
        1: {"ssrcId": 10, "subscribed": True}
    }


@pytest.mark.asyncio
async def test_closing_one_camera_keeps_the_other_camera_stream_alive() -> None:
    """The mower publisher is stopped only when its final camera viewer closes."""
    coordinator = object.__new__(ConcreteCoordinator)
    coordinator._active_camera_sessions = {}  # noqa: SLF001
    coordinator._camera_session_lock = asyncio.Lock()  # noqa: SLF001
    coordinator.leave_webrtc_channel = AsyncMock()

    await coordinator.async_register_camera_session("left", "left-session")
    await coordinator.async_register_camera_session("right", "right-session")
    await coordinator.async_release_camera_session("left", "left-session")

    coordinator.leave_webrtc_channel.assert_not_awaited()
    assert coordinator.has_active_camera_sessions

    await coordinator.async_release_camera_session("right", "right-session")

    coordinator.leave_webrtc_channel.assert_awaited_once()
    assert not coordinator.has_active_camera_sessions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("device_name", "states"),
    [("Luba-VS00CLD", [1, 1, 0]), ("Yuka-000CLD", [1, 1, 1])],
)
async def test_dual_camera_token_requests_every_vision_stream(
    device_name: str, states: list[int]
) -> None:
    """The token enables one cameraStates slot per vision feed the mower has."""
    coordinator = object.__new__(ConcreteCoordinator)
    coordinator.device_name = device_name
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={"code": 200, "data": {}})
    response_context = MagicMock()
    response_context.__aenter__ = AsyncMock(return_value=response)
    response_context.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.post.return_value = response_context
    http = SimpleNamespace(
        ensure_token_valid=AsyncMock(),
        login_info=SimpleNamespace(access_token="access-token"),
        _headers={"x-device": "mower"},
    )
    coordinator.manager = SimpleNamespace(mammotion_http=http)
    coordinator.device = SimpleNamespace(iot_id="mower-id")
    coordinator.hass = MagicMock()
    parsed = MagicMock()

    with (
        patch(
            "custom_components.mammotion.coordinator.aiohttp_client.async_get_clientsession",
            return_value=session,
        ),
        patch(
            "custom_components.mammotion.coordinator.response_factory",
            return_value=parsed,
        ),
    ):
        result = await coordinator._request_dual_camera_stream()  # noqa: SLF001

    assert result is parsed
    http.ensure_token_valid.assert_awaited_once_with(caller="dual_camera_stream")
    request = session.post.call_args
    assert request.kwargs["json"] == {
        "deviceId": "mower-id",
        "mode": 0,
        "cameraStates": [{"cameraState": state} for state in states],
    }
    assert request.kwargs["headers"]["Authorization"] == "Bearer access-token"


def test_camera_names_come_from_the_translation_key() -> None:
    """A None name would label every feed with the bare device name."""
    coordinator = MagicMock()
    coordinator.device.device_name = "Luba-VS00CLD"
    coordinator.device_name = "Luba-VS00CLD"
    coordinator.unique_name = "Luba-VS00CLD"
    translations = {
        f"component.mammotion.entity.camera.{description.key}.name": description.key
        for description in CAMERAS
    }

    names = []
    for description in CAMERAS:
        camera = MammotionWebRTCCamera(coordinator, description, MagicMock())
        camera.platform_data = SimpleNamespace(
            platform_name="mammotion", domain="camera"
        )
        names.append(camera._name_internal(None, translations))  # noqa: SLF001

    assert names == ["webrtc_camera", "webrtc_camera_right", "webrtc_camera_rear"]
