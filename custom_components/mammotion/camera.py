"""Mammotion camera entities."""

from __future__ import annotations

import asyncio
import collections
import functools
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websockets
from homeassistant.components.camera import (
    CameraEntityDescription,
    WebRTCAnswer,
    WebRTCError,
    WebRTCSendMessage,
)
from homeassistant.components.web_rtc import (
    async_register_ice_servers,
)
from homeassistant.core import (
    HomeAssistant,
    callback,
)
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from pymammotion.http.model.camera_stream import (
    StreamSubscriptionResponse,
)
from pymammotion.http.model.http import Response
from webrtc_models import RTCIceCandidateInit, RTCIceServer

from . import MammotionConfigEntry
from .agora_api import AgoraResponse
from .agora_websocket import AgoraWebSocketHandler
from .coordinator import MammotionBaseUpdateCoordinator, vision_camera_slots
from .entity import MammotionCameraBaseEntity

_LOGGER = logging.getLogger(__name__)

PLACEHOLDER = Path(__file__).parent / "placeholder.png"


@dataclass(frozen=True, kw_only=True)
class MammotionCameraEntityDescription(CameraEntityDescription):
    """Describes Mammotion camera entity."""

    key: str
    stream_fn: Callable[
        [MammotionBaseUpdateCoordinator], Response[StreamSubscriptionResponse] | None
    ]
    # Agora uid the mower publishes this feed under: cameraStates slot + 1.
    target_uid: int


# One description per cameraStates slot, in slot order.
CAMERAS: tuple[MammotionCameraEntityDescription, ...] = (
    MammotionCameraEntityDescription(
        key="webrtc_camera",
        stream_fn=lambda coordinator: coordinator.get_stream_data(),
        target_uid=1,
    ),
    MammotionCameraEntityDescription(
        key="webrtc_camera_right",
        stream_fn=lambda coordinator: coordinator.get_stream_data(),
        target_uid=2,
    ),
    MammotionCameraEntityDescription(
        key="webrtc_camera_rear",
        stream_fn=lambda coordinator: coordinator.get_stream_data(),
        target_uid=3,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: MammotionConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Mammotion camera entities."""
    mowers = entry.runtime_data.mowers
    entities: list[MammotionWebRTCCamera] = []

    for mower in mowers:
        slots = vision_camera_slots(mower.device.device_name)
        entities.extend(
            MammotionWebRTCCamera(mower.reporting_coordinator, entity_description, hass)
            for entity_description in CAMERAS[:slots]
        )
    async_add_entities(entities)


class MammotionWebRTCCamera(MammotionCameraBaseEntity):
    """Mammotion WebRTC camera entity."""

    entity_description: MammotionCameraEntityDescription
    _attr_capability_attributes = None

    def __init__(
        self,
        coordinator: MammotionBaseUpdateCoordinator,
        entity_description: MammotionCameraEntityDescription,
        hass: HomeAssistant,
    ) -> None:
        """Initialize the WebRTC camera entity."""
        super().__init__(coordinator, entity_description.key)
        self._cache: dict[str, Any] = {}
        self.access_tokens: collections.deque = collections.deque([], 2)
        self.async_update_token()
        self._create_stream_lock: asyncio.Lock | None = None
        self._join_lock = asyncio.Lock()
        self.coordinator = coordinator
        self._agora_handler = AgoraWebSocketHandler(
            hass,
            recover_stream=self._recover_stream,
            keepalive=self._fpv_keepalive,
            target_uid=entity_description.target_uid,
        )
        self.entity_description = entity_description
        self._attr_translation_key = entity_description.key
        self._stream_data: StreamSubscriptionResponse | None = None
        self._sessions: set[str] = set()
        self._teardown_lock = asyncio.Lock()
        self._attr_model = coordinator.device.device_name

    async def async_added_to_hass(self) -> None:
        """Register lifecycle cleanup and shared stream controls."""
        await super().async_added_to_hass()
        self.coordinator.register_webrtc_session_control(
            self, self.entity_description.key
        )
        self.async_on_remove(
            self.coordinator.async_add_listener(self._handle_coordinator_update)
        )
        unregister = async_register_ice_servers(self.hass, self.get_ice_servers)
        self.async_on_remove(unregister)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Refresh availability after coordinator or transport recovery."""
        self.async_write_ha_state()

    async def async_will_remove_from_hass(self) -> None:
        """Disconnect this camera without stopping a sibling camera's feed."""
        self.coordinator.register_webrtc_session_control(
            None, self.entity_description.key
        )
        sessions = tuple(self._sessions)
        self._sessions.clear()
        await self._agora_handler.disconnect()
        for session_id in sessions:
            await self.coordinator.async_release_camera_session(
                self.entity_description.key, session_id
            )
        if not sessions and not self.coordinator.has_active_camera_sessions:
            try:
                await self.coordinator.manager.stop_stream(
                    self.coordinator.device.device_name
                )
            except Exception as ex:  # noqa: BLE001 - unload cleanup is best effort
                _LOGGER.debug("Camera unload failed: %s", type(ex).__name__)
        self._set_streaming(False)
        await super().async_will_remove_from_hass()

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return a placeholder image for WebRTC cameras that don't support snapshots."""
        return await self.hass.async_add_executor_job(self.placeholder_image)

    @classmethod
    @functools.cache
    def placeholder_image(cls) -> bytes:
        """Return placeholder image to use when no stream is available."""
        return PLACEHOLDER.read_bytes()

    async def async_handle_async_webrtc_offer(
        self, offer_sdp: str, session_id: str, send_message: WebRTCSendMessage
    ) -> None:
        """Handle WebRTC offer by initiating WebSocket connection to Agora.

        This replaces the JavaScript SDK functionality and performs the WebRTC
        negotiation directly in Python.
        """

        try:
            await asyncio.wait_for(self._join_lock.acquire(), timeout=45)
        except TimeoutError:
            _LOGGER.warning(
                "Camera offer timed out waiting for another negotiation (%s)",
                session_id,
            )
            send_message(WebRTCError("503", "Camera is busy; please retry"))
            return

        try:
            (
                stream_data,
                agora_response,
            ) = await self.coordinator.async_check_stream_expiry(
                force=not self.coordinator.has_active_camera_sessions
            )
            self._agora_handler.candidates = []

            try:
                if stream_data is None or agora_response is None:
                    _LOGGER.warning("Camera stream is temporarily unavailable")
                    send_message(
                        WebRTCError(
                            "503",
                            "Camera stream is temporarily unavailable",
                        )
                    )
                    return

                if (
                    self.entity_description.target_uid != 1
                    and not self.coordinator.dual_camera_stream_available
                ):
                    send_message(WebRTCError("503", "Vision stream unavailable"))
                    return

                agora_data = stream_data

                # Start WebSocket connection and WebRTC negotiation
                answer_sdp = await self._perform_webrtc_negotiation(
                    offer_sdp, agora_data, session_id, agora_response
                )

                if answer_sdp:
                    await self.coordinator.async_register_camera_session(
                        self.entity_description.key, session_id
                    )
                    self._sessions.add(session_id)
                    send_message(WebRTCAnswer(answer_sdp))
                    self._set_streaming(True)
                    _LOGGER.info("WebRTC negotiation completed successfully")
                else:
                    if not self.coordinator.has_active_camera_sessions:
                        self.coordinator.clear_stream_data()
                    send_message(WebRTCError("500", "WebRTC negotiation failed"))

            except (
                websockets.exceptions.WebSocketException,
                json.JSONDecodeError,
            ) as ex:
                if not self.coordinator.has_active_camera_sessions:
                    self.coordinator.clear_stream_data()
                _LOGGER.error("WebRTC offer failed: %s", type(ex).__name__)
                send_message(WebRTCError("500", "WebRTC negotiation failed"))
        finally:
            self._join_lock.release()

    async def async_on_webrtc_candidate(
        self, session_id: str, candidate: RTCIceCandidateInit
    ) -> None:
        """Collect WebRTC candidates for inclusion in join message."""
        _LOGGER.debug("Received WebRTC candidate")

        # Collect candidates - they'll be included in the join message
        self._agora_handler.candidates.append(candidate)

    @callback
    def close_webrtc_session(self, session_id: str) -> None:
        """Schedule cleanup when the frontend ends a native WebRTC session."""
        if session_id not in self._sessions:
            return
        self.hass.async_create_task(self.async_close_webrtc_session(session_id))

    async def async_close_webrtc_session(self, session_id: str) -> None:
        """Close WebRTC session."""
        if session_id not in self._sessions:
            return
        self._sessions.discard(session_id)
        await self.coordinator.async_release_camera_session(
            self.entity_description.key, session_id
        )
        if not self._sessions:
            await self._agora_handler.disconnect()
        self._set_streaming(bool(self._sessions))

    async def async_teardown_stream(self, *, stop_device: bool = True) -> None:
        """Leave this camera's Agora session and optionally stop the mower."""
        async with self._teardown_lock:
            self._sessions.clear()
            await self._agora_handler.disconnect()
            if stop_device:
                try:
                    await self.coordinator.manager.stop_stream(
                        self.coordinator.device.device_name
                    )
                except Exception as ex:  # noqa: BLE001 - teardown is best effort
                    _LOGGER.debug("Camera stop failed: %s", type(ex).__name__)
            self._set_streaming(False)

    async def _fpv_keepalive(self) -> bool:
        """Re-arm the mower's video encoder on 4G; return False on WiFi.

        Invoked by AgoraWebSocketHandler every few seconds while a session is
        live. Over cellular the encoder stops publishing unless poked with
        ``refresh_fpv``; on WiFi the stream is continuous, so return False to
        stop the keep-alive loop without sending anything.
        """
        if not self.coordinator.is_on_4g:
            return False
        await self.coordinator.async_send_command("refresh_fpv")
        return True

    async def _recover_stream(self) -> None:
        """Re-establish the stream after the mower drops out of the Agora channel.

        Invoked by AgoraWebSocketHandler once the mower (peer) has been gone for
        its debounce window: nudge the device with a BLE sync, then refresh the
        stream subscription so it rejoins the channel.
        """
        stream_data, agora_response = await self.coordinator.async_check_stream_expiry(
            force=True
        )
        if stream_data is None or agora_response is None:
            self._set_streaming(False)

    async def _perform_webrtc_negotiation(
        self,
        offer_sdp: str,
        agora_data: StreamSubscriptionResponse,
        session_id: str,
        agora_response: AgoraResponse,
    ) -> str | None:
        """Perform WebRTC negotiation through Agora WebSocket.

        Args:
            self: The camera instance
            offer_sdp: The WebRTC offer SDP from the browser
            agora_data: Dict containing appid, channelName, token, uid
            session_id: Session ID for this WebRTC connection
            agora_response: AgoraResponse object containing ICE servers

        Returns:
            Answer SDP if successful, None otherwise

        """
        try:
            answer_sdp = await self._agora_handler.connect_and_join(
                agora_data, offer_sdp, session_id, agora_response
            )
        except (OSError, ValueError, TypeError) as ex:
            _LOGGER.error("WebRTC negotiation failed: %s", ex)
            return None
        else:
            if answer_sdp:
                _LOGGER.info("Successfully negotiated WebRTC through Agora")
                return answer_sdp

            _LOGGER.error(
                "Failed to get answer SDP from Agora negotiation, using handler fallback"
            )
            # Use the handler's fallback SDP generation as last resort
            return None

    def get_ice_servers(self) -> list[RTCIceServer]:
        """Return the ICE servers from Agora API."""
        return list(getattr(self.coordinator, "_ice_servers", []) or [])

    def _set_streaming(self, streaming: bool) -> None:
        """Update the camera's actual streaming state."""
        self._attr_is_streaming = streaming
        if self.hass is not None:
            self.async_write_ha_state()
