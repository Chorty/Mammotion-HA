"""Regression tests for Mammotion camera recovery."""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymammotion.aliyun.exceptions import DeviceOfflineException

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
    coordinator._dual_camera_stream_available = False
    coordinator._active_camera_sessions = {}
    coordinator._camera_session_lock = asyncio.Lock()
    coordinator._camera_offer_lock = asyncio.Lock()
    coordinator._webrtc_session_controls = {}
    coordinator._camera_publisher_on = False
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
        device=SimpleNamespace(device_name="Luba-VS00CLD"),
        reporting_coordinator=coordinator,
    )
    entry = SimpleNamespace(runtime_data=SimpleNamespace(mowers=[mower]))
    add_entities = MagicMock()

    with patch("custom_components.mammotion.camera.MammotionWebRTCCamera"):
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
    camera.entity_description = SimpleNamespace(key="webrtc_camera", target_uid=1)
    camera._sessions = set()
    camera._pending_sessions = set()
    camera._cancelled_sessions = set()
    camera._offer_tasks = set()
    camera._removing = False
    camera._attr_is_streaming = False
    camera._hass = MagicMock()
    camera.async_write_ha_state = MagicMock()
    stream_data = MagicMock()
    agora_response = MagicMock()
    camera.coordinator = SimpleNamespace(
        async_check_stream_expiry=AsyncMock(return_value=(stream_data, agora_response)),
        clear_stream_data=MagicMock(),
        has_active_camera_sessions=False,
        dual_camera_stream_available=False,
        async_register_camera_session=AsyncMock(),
    )
    camera._perform_webrtc_negotiation = AsyncMock(return_value="answer-sdp")
    messages = []

    await camera.async_handle_async_webrtc_offer(
        "offer-sdp", "session", messages.append
    )

    assert camera._attr_is_streaming is True
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_second_camera_offer_mints_its_own_token() -> None:
    """Reusing a sibling's token makes Agora quit the sibling (code 2003)."""
    camera = object.__new__(MammotionWebRTCCamera)
    camera._join_lock = asyncio.Lock()
    camera._agora_handler = SimpleNamespace(candidates=[])
    camera.entity_description = SimpleNamespace(key="webrtc_camera_right", target_uid=2)
    camera._sessions = set()
    camera._pending_sessions = set()
    camera._cancelled_sessions = set()
    camera._offer_tasks = set()
    camera._removing = False
    camera._attr_is_streaming = False
    camera._hass = MagicMock()
    camera.async_write_ha_state = MagicMock()
    camera.coordinator = SimpleNamespace(
        async_check_stream_expiry=AsyncMock(return_value=(MagicMock(), MagicMock())),
        clear_stream_data=MagicMock(),
        has_active_camera_sessions=True,
        dual_camera_stream_available=True,
        async_register_camera_session=AsyncMock(),
    )
    camera._perform_webrtc_negotiation = AsyncMock(return_value="answer-sdp")

    await camera.async_handle_async_webrtc_offer("offer-sdp", "session", MagicMock())

    camera.coordinator.async_check_stream_expiry.assert_awaited_once_with(force=True)


@pytest.mark.asyncio
async def test_right_camera_offer_uses_dual_stream_availability() -> None:
    """Right camera can negotiate when the coordinator has a dual-stream token."""
    coordinator = _coordinator()
    coordinator._dual_camera_stream_available = True
    coordinator.async_check_stream_expiry = AsyncMock(
        return_value=(MagicMock(), MagicMock())
    )
    coordinator.async_register_camera_session = AsyncMock()

    camera = object.__new__(MammotionWebRTCCamera)
    camera._join_lock = asyncio.Lock()
    camera._agora_handler = SimpleNamespace(candidates=[])
    camera.entity_description = SimpleNamespace(key="right_vision_camera", target_uid=2)
    camera._sessions = set()
    camera._pending_sessions = set()
    camera._cancelled_sessions = set()
    camera._offer_tasks = set()
    camera._removing = False
    camera._attr_is_streaming = False
    camera._hass = MagicMock()
    camera.async_write_ha_state = MagicMock()
    camera.coordinator = coordinator
    camera._perform_webrtc_negotiation = AsyncMock(return_value="answer-sdp")
    messages = []

    await camera.async_handle_async_webrtc_offer(
        "offer-sdp", "right-session", messages.append
    )

    assert coordinator.dual_camera_stream_available is True
    coordinator.async_register_camera_session.assert_awaited_once_with(
        "right_vision_camera", "right-session"
    )
    assert camera._attr_is_streaming is True
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_camera_offer_reports_temporary_unavailability() -> None:
    """A missing cloud token produces a temporary error and remains idle."""
    camera = object.__new__(MammotionWebRTCCamera)
    camera._join_lock = asyncio.Lock()
    camera._agora_handler = SimpleNamespace(candidates=[])
    camera.entity_description = SimpleNamespace(key="webrtc_camera", target_uid=1)
    camera._sessions = set()
    camera._pending_sessions = set()
    camera._cancelled_sessions = set()
    camera._offer_tasks = set()
    camera._removing = False
    camera._attr_is_streaming = False
    camera._hass = MagicMock()
    camera.async_write_ha_state = MagicMock()
    camera.coordinator = SimpleNamespace(
        async_check_stream_expiry=AsyncMock(return_value=(None, None)),
        clear_stream_data=MagicMock(),
        has_active_camera_sessions=False,
        dual_camera_stream_available=False,
        async_stop_camera_publisher_if_idle=AsyncMock(),
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


@pytest.mark.asyncio
async def test_overlapping_camera_offers_wait_instead_of_returning_409() -> None:
    """Overlapping frontend offers queue behind the active negotiation."""
    camera = object.__new__(MammotionWebRTCCamera)
    camera._join_lock = asyncio.Lock()
    camera._agora_handler = SimpleNamespace(candidates=[])
    camera.entity_description = SimpleNamespace(key="webrtc_camera", target_uid=1)
    camera._sessions = set()
    camera._pending_sessions = set()
    camera._cancelled_sessions = set()
    camera._offer_tasks = set()
    camera._removing = False
    camera._attr_is_streaming = False
    camera._hass = MagicMock()
    camera.async_write_ha_state = MagicMock()
    first_started = asyncio.Event()
    release = asyncio.Event()

    async def unavailable(*, force):
        first_started.set()
        await release.wait()
        return None, None

    camera.coordinator = SimpleNamespace(
        async_check_stream_expiry=unavailable,
        has_active_camera_sessions=False,
        async_stop_camera_publisher_if_idle=AsyncMock(),
    )
    first_messages = []
    second_messages = []
    first = asyncio.create_task(
        camera.async_handle_async_webrtc_offer(
            "offer-1", "session-1", first_messages.append
        )
    )
    await first_started.wait()
    second = asyncio.create_task(
        camera.async_handle_async_webrtc_offer(
            "offer-2", "session-2", second_messages.append
        )
    )
    await asyncio.sleep(0)
    assert second_messages == []

    release.set()
    await asyncio.gather(first, second)

    assert [message.code for message in first_messages + second_messages] == [
        "503",
        "503",
    ]


@pytest.mark.asyncio
async def test_camera_availability_refreshes_on_coordinator_update() -> None:
    """A camera publishes state when the coordinator reports recovery."""
    camera = object.__new__(MammotionWebRTCCamera)
    unsubscribe = MagicMock()
    camera.coordinator = SimpleNamespace(
        register_webrtc_session_control=MagicMock(),
        async_add_listener=MagicMock(return_value=unsubscribe),
    )
    camera.entity_description = SimpleNamespace(key="webrtc_camera")
    camera._hass = MagicMock()
    camera.async_on_remove = MagicMock()
    camera.async_write_ha_state = MagicMock()
    with (
        patch(
            "custom_components.mammotion.camera.MammotionCameraBaseEntity.async_added_to_hass",
            new_callable=AsyncMock,
        ),
        patch(
            "custom_components.mammotion.camera.async_register_ice_servers",
            return_value=MagicMock(),
        ),
    ):
        await camera.async_added_to_hass()

    callback = camera.coordinator.async_add_listener.call_args.args[0]
    camera.async_on_remove.assert_any_call(unsubscribe)
    callback()
    camera.async_write_ha_state.assert_called_once()


def test_camera_remains_available_with_live_mqtt_transport() -> None:
    """A stale mower-offline report does not hide an active MQTT connection."""
    camera = object.__new__(MammotionWebRTCCamera)
    camera.coordinator = SimpleNamespace(
        data=MagicMock(),
        is_online=lambda: False,
        mqtt_transport_connected=True,
    )

    assert camera.available is True


@pytest.mark.asyncio
async def test_stream_401_renews_the_rejected_bearer_once_then_retries() -> None:
    """A 401 renews the exact bearer the request carried, then retries once."""
    rejected = _response(401, with_data=False)
    good = _response(200, with_data=True)
    coordinator = _coordinator(rejected, good)
    coordinator.manager.mammotion_http = SimpleNamespace(
        login_info=SimpleNamespace(access_token="bearer-before-401")
    )
    coordinator.manager.token_manager = SimpleNamespace(
        refresh_invoke_token=AsyncMock()
    )
    coordinator.store_cloud_credentials = MagicMock()
    context, _ = _agora_context()

    with patch(
        "custom_components.mammotion.coordinator.AgoraAPIClient",
        return_value=context,
    ):
        stream_data, _ = await coordinator.async_check_stream_expiry(force=True)

    assert stream_data is good.data
    coordinator.manager.token_manager.refresh_invoke_token.assert_awaited_once_with(
        "bearer-before-401"
    )
    coordinator.store_cloud_credentials.assert_called_once()
    assert coordinator.manager.refresh_stream_subscription.await_count == 2


START_PUBLISHER = (("device_agora_join_channel_with_position",), {"enter_state": 1})
STOP_PUBLISHER = (("device_agora_join_channel_with_position",), {"enter_state": 0})


def _publisher_commands(coordinator) -> list[tuple]:
    """Return the start/stop-publisher commands the coordinator sent, in order."""
    return [
        (call.args, call.kwargs)
        for call in coordinator.async_send_command.await_args_list
        if call.args == ("device_agora_join_channel_with_position",)
    ]


def _registered_camera(coordinator, key: str = "webrtc_camera", target_uid: int = 1):
    """Build a camera entity registered with the coordinator as a stream owner."""
    camera = object.__new__(MammotionWebRTCCamera)
    camera._join_lock = coordinator.camera_offer_lock
    camera._teardown_lock = asyncio.Lock()
    camera._agora_handler = SimpleNamespace(candidates=[], disconnect=AsyncMock())
    camera.entity_description = SimpleNamespace(key=key, target_uid=target_uid)
    camera._sessions = set()
    camera._pending_sessions = set()
    camera._cancelled_sessions = set()
    camera._offer_tasks = set()
    camera._removing = False
    camera._attr_is_streaming = False
    camera._hass = MagicMock()
    camera.async_write_ha_state = MagicMock()
    camera.coordinator = coordinator
    coordinator.register_webrtc_session_control(camera, key)
    return camera


@pytest.mark.asyncio
async def test_failed_offer_stops_the_publisher_it_started() -> None:
    """An offer that cannot get a token must not leave the mower streaming.

    The offer asks the mower to publish before it requests the token, so a
    failure after that point used to leave video running with no viewer.
    """
    coordinator = _coordinator(_response(500, with_data=False))
    camera = _registered_camera(coordinator)
    messages = []

    await camera.async_handle_async_webrtc_offer(
        "offer-sdp", "session", messages.append
    )

    assert messages[0].code == "503"
    assert _publisher_commands(coordinator) == [START_PUBLISHER, STOP_PUBLISHER]
    assert coordinator._camera_publisher_on is False
    assert camera.has_pending_offer is False


@pytest.mark.asyncio
async def test_viewer_leaving_mid_negotiation_stops_the_publisher() -> None:
    """A close that arrives while the offer negotiates is honoured afterwards.

    Home Assistant registers the close callback before the offer runs and does
    not cancel the offer, so the close used to be dropped and the session then
    registered with no viewer, keeping the mower publishing indefinitely.
    """
    coordinator = _coordinator()

    async def start_stream(*, force):
        await coordinator.join_webrtc_channel()
        return MagicMock(), MagicMock()

    coordinator.async_check_stream_expiry = start_stream
    camera = _registered_camera(coordinator)
    negotiating = asyncio.Event()
    finish = asyncio.Event()

    async def negotiate(*_args):
        negotiating.set()
        await finish.wait()
        return "answer-sdp"

    camera._perform_webrtc_negotiation = negotiate
    messages = []
    offer = asyncio.create_task(
        camera.async_handle_async_webrtc_offer("offer-sdp", "session", messages.append)
    )
    await negotiating.wait()

    camera.close_webrtc_session("session")
    finish.set()
    await offer

    assert messages == []
    assert not coordinator.has_active_camera_sessions
    assert camera._sessions == set()
    assert camera._attr_is_streaming is False
    assert _publisher_commands(coordinator) == [START_PUBLISHER, STOP_PUBLISHER]
    camera._agora_handler.disconnect.assert_awaited()


@pytest.mark.asyncio
async def test_last_viewer_closing_keeps_publisher_for_a_negotiating_sibling() -> None:
    """Closing the last live view must not stop a sibling camera mid-offer.

    The sibling's offer stops the publisher itself if it then fails.
    """
    coordinator = _coordinator()
    coordinator._camera_publisher_on = True
    left = _registered_camera(coordinator, "webrtc_camera", 1)
    right = _registered_camera(coordinator, "webrtc_camera_right", 2)
    await coordinator.async_register_camera_session("webrtc_camera", "left-session")
    left._sessions.add("left-session")
    right._pending_sessions.add("right-session")

    await left.async_close_webrtc_session("left-session")

    assert _publisher_commands(coordinator) == []

    right._pending_sessions.discard("right-session")
    await coordinator.async_stop_camera_publisher_if_idle()

    assert _publisher_commands(coordinator) == [STOP_PUBLISHER]


@pytest.mark.asyncio
async def test_last_viewer_close_preserves_same_camera_pending_offer() -> None:
    """Closing an old viewer cannot disconnect a queued offer's Agora handler."""
    coordinator = _coordinator()
    coordinator._camera_publisher_on = True
    camera = _registered_camera(coordinator)
    await coordinator.async_register_camera_session("webrtc_camera", "old-session")
    camera._sessions.add("old-session")
    camera._pending_sessions.add("new-session")

    await camera.async_close_webrtc_session("old-session")

    camera._agora_handler.disconnect.assert_not_awaited()
    assert _publisher_commands(coordinator) == []
    assert camera.has_pending_offer

    camera._pending_sessions.discard("new-session")
    await coordinator.async_stop_camera_publisher_if_idle()
    assert _publisher_commands(coordinator) == [STOP_PUBLISHER]


@pytest.mark.asyncio
async def test_recovery_does_not_restart_publisher_without_a_viewer() -> None:
    """A delayed peer-recovery task cannot revive an idle mower stream."""
    coordinator = _coordinator()
    coordinator.async_check_stream_expiry = AsyncMock()
    camera = _registered_camera(coordinator)

    await camera._recover_stream()

    coordinator.async_check_stream_expiry.assert_not_awaited()
    assert _publisher_commands(coordinator) == []


@pytest.mark.asyncio
async def test_idle_unload_sends_no_command_when_nothing_was_started() -> None:
    """Unloading an idle camera does not wake the mower with a stop command."""
    coordinator = _coordinator()
    camera = _registered_camera(coordinator)

    await coordinator.async_stop_camera_publisher_if_idle()

    assert _publisher_commands(coordinator) == []
    assert camera.has_pending_offer is False


@pytest.mark.asyncio
async def test_unload_cancels_pending_offer_before_unregistering_camera() -> None:
    """An offer cannot register a viewer after its camera entity is removed."""
    coordinator = _coordinator()

    async def start_stream(*, force):
        await coordinator.join_webrtc_channel()
        return MagicMock(), MagicMock()

    coordinator.async_check_stream_expiry = start_stream
    camera = _registered_camera(coordinator)
    negotiating = asyncio.Event()

    async def negotiate(*_args):
        negotiating.set()
        await asyncio.Event().wait()

    camera._perform_webrtc_negotiation = negotiate
    messages = []
    offer = asyncio.create_task(
        camera.async_handle_async_webrtc_offer("offer-sdp", "session", messages.append)
    )
    await negotiating.wait()

    with patch(
        "custom_components.mammotion.camera.MammotionCameraBaseEntity.async_will_remove_from_hass",
        new_callable=AsyncMock,
    ):
        await camera.async_will_remove_from_hass()

    assert offer.cancelled()
    assert camera._offer_tasks == set()
    assert camera._pending_sessions == set()
    assert not coordinator.has_active_camera_sessions
    assert coordinator._webrtc_session_controls == {}
    assert _publisher_commands(coordinator) == [START_PUBLISHER, STOP_PUBLISHER]

    messages = []
    await camera.async_handle_async_webrtc_offer(
        "offer-sdp", "after-unload", messages.append
    )
    assert messages[0].code == "503"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_task", [False, True])
async def test_offer_closed_during_session_registration_is_released(
    cancel_task: bool,
) -> None:
    """A close or cancellation cannot strand a registered coordinator viewer."""
    coordinator = _coordinator()

    async def start_stream(*, force):
        await coordinator.join_webrtc_channel()
        return MagicMock(), MagicMock()

    coordinator.async_check_stream_expiry = start_stream
    registered = asyncio.Event()
    finish_registration = asyncio.Event()
    original_register = coordinator.async_register_camera_session

    async def register_then_wait(camera_key, session_id):
        await original_register(camera_key, session_id)
        registered.set()
        await finish_registration.wait()

    coordinator.async_register_camera_session = register_then_wait
    camera = _registered_camera(coordinator)
    camera._perform_webrtc_negotiation = AsyncMock(return_value="answer-sdp")
    messages = []
    offer = asyncio.create_task(
        camera.async_handle_async_webrtc_offer("offer-sdp", "session", messages.append)
    )
    await registered.wait()

    if cancel_task:
        offer.cancel()
    else:
        camera.close_webrtc_session("session")
        finish_registration.set()
    await asyncio.gather(offer, return_exceptions=True)

    assert messages == []
    assert camera._sessions == set()
    assert not coordinator.has_active_camera_sessions
    assert _publisher_commands(coordinator) == [START_PUBLISHER, STOP_PUBLISHER]


@pytest.mark.asyncio
async def test_sibling_offer_waits_for_right_camera_refresh_and_negotiation() -> None:
    """A sibling cannot clear the right offer's dual-stream flag mid-refresh."""
    coordinator = _coordinator()
    right = _registered_camera(coordinator, "webrtc_camera_right", 2)
    left = _registered_camera(coordinator, "webrtc_camera", 1)
    right_waiting = asyncio.Event()
    left_started = asyncio.Event()
    release_right = asyncio.Event()

    async def refresh(*, force):
        if asyncio.current_task().get_name() == "right-offer":
            await coordinator.join_webrtc_channel()
            coordinator._dual_camera_stream_available = True
            right_waiting.set()
            await release_right.wait()
            return MagicMock(), MagicMock()
        left_started.set()
        coordinator._dual_camera_stream_available = False
        return None, None

    coordinator.async_check_stream_expiry = refresh
    right._perform_webrtc_negotiation = AsyncMock(return_value="answer-sdp")
    right_messages = []
    left_messages = []
    right_offer = asyncio.create_task(
        right.async_handle_async_webrtc_offer(
            "offer-sdp", "right-session", right_messages.append
        ),
        name="right-offer",
    )
    await right_waiting.wait()
    left_offer = asyncio.create_task(
        left.async_handle_async_webrtc_offer(
            "offer-sdp", "left-session", left_messages.append
        ),
        name="left-offer",
    )
    await asyncio.sleep(0)
    assert left.has_pending_offer
    assert not left_started.is_set()

    release_right.set()
    await asyncio.gather(right_offer, left_offer)

    assert left_started.is_set()
    assert len(right_messages) == 1
    assert right._attr_is_streaming is True
    assert left_messages[0].code == "503"
    assert _publisher_commands(coordinator) == [START_PUBLISHER]

    await right.async_close_webrtc_session("right-session")
    assert _publisher_commands(coordinator) == [START_PUBLISHER, STOP_PUBLISHER]


@pytest.mark.asyncio
async def test_failed_stop_keeps_publisher_marked_for_retry() -> None:
    """A failed command must not be reported as a confirmed publisher stop."""
    coordinator = _coordinator()
    coordinator._camera_publisher_on = True
    coordinator.async_send_command.return_value = False

    await coordinator.async_stop_camera_publisher_if_idle()

    assert coordinator._camera_publisher_on is True
    assert _publisher_commands(coordinator) == [STOP_PUBLISHER]

    coordinator.async_send_command.return_value = True
    await coordinator.async_stop_camera_publisher_if_idle()

    assert coordinator._camera_publisher_on is False
    assert _publisher_commands(coordinator) == [STOP_PUBLISHER, STOP_PUBLISHER]


@pytest.mark.asyncio
async def test_offline_stop_keeps_publisher_marked_for_retry() -> None:
    """An offline stop does not clear publisher state or mask offer cleanup."""
    coordinator = _coordinator()
    coordinator._camera_publisher_on = True
    coordinator.async_send_command.side_effect = DeviceOfflineException(
        "offline", "private-iot-id"
    )

    await coordinator.async_stop_camera_publisher_if_idle()

    assert coordinator._camera_publisher_on is True
    assert _publisher_commands(coordinator) == [STOP_PUBLISHER]

    coordinator.async_send_command.side_effect = None
    coordinator.async_send_command.return_value = True
    await coordinator.async_stop_camera_publisher_if_idle()

    assert coordinator._camera_publisher_on is False
    assert _publisher_commands(coordinator) == [STOP_PUBLISHER, STOP_PUBLISHER]


@pytest.mark.asyncio
@pytest.mark.parametrize("active_viewer", [False, True])
async def test_diagnostic_refresh_preserves_only_an_active_publisher(
    active_viewer: bool,
) -> None:
    """The refresh button stops an idle stream but keeps an existing view."""
    coordinator = _coordinator()
    if active_viewer:
        await coordinator.async_register_camera_session("webrtc_camera", "viewer")

    async def refresh(*, force):
        await coordinator.join_webrtc_channel()
        return MagicMock(), MagicMock()

    coordinator.async_check_stream_expiry = refresh

    await coordinator.async_refresh_camera_stream()

    assert _publisher_commands(coordinator) == (
        [START_PUBLISHER] if active_viewer else [START_PUBLISHER, STOP_PUBLISHER]
    )
