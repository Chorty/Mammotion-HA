"""Mammotion camera entities."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO

import voluptuous as vol
from aiohttp import ClientError
from homeassistant.components.camera import (
    Camera,
    CameraEntityDescription,
    StreamType,
    WebRTCSendMessage,
)
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from PIL import Image, ImageDraw
from pymammotion.http.model.camera_stream import (
    StreamSubscriptionResponse,
)
from pymammotion.utility.device_type import DeviceType

from . import MammotionConfigEntry
from .coordinator import MammotionBaseUpdateCoordinator
from .entity import MammotionBaseEntity
from .models import MammotionMowerData

_LOGGER = logging.getLogger(__name__)

DATA_SERVICES_REGISTERED = "camera_services_registered"
CAMERA_SERVICES = (
    "refresh_stream",
    "start_video",
    "stop_video",
    "get_tokens",
    "move_forward",
    "move_left",
    "move_right",
    "move_backward",
)
ENTITY_SERVICE_SCHEMA = vol.Schema({vol.Required("entity_id"): cv.entity_id})
MOVEMENT_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Required("entity_id"): cv.entity_id,
        vol.Optional("speed", default=0.4): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=1.0)
        ),
    }
)


@dataclass(frozen=True, kw_only=True)
class MammotionCameraEntityDescription(CameraEntityDescription):
    """Describes Mammotion camera entity."""

    key: str
    stream_fn: Callable[[MammotionBaseUpdateCoordinator], StreamSubscriptionResponse]


CAMERAS: tuple[MammotionCameraEntityDescription, ...] = (
    MammotionCameraEntityDescription(
        key="webrtc_camera",
        stream_fn=lambda coordinator: coordinator.get_stream_subscription(),
    ),
)


@dataclass(frozen=True, kw_only=True)
class MammotionMapCameraEntityDescription(CameraEntityDescription):
    """Describes Mammotion map camera entity."""

    key: str


MAP_CAMERAS: tuple[MammotionMapCameraEntityDescription, ...] = (
    MammotionMapCameraEntityDescription(
        key="map_camera",
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: MammotionConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Mammotion camera entities."""
    mowers = entry.runtime_data
    entities: list[Camera] = []
    for mower in mowers:
        entities.extend(
            MammotionMapCamera(mower.map_coordinator, description)
            for description in MAP_CAMERAS
        )
        if not DeviceType.is_luba1(mower.device.deviceName):
            _LOGGER.debug("Config camera for %s", mower.device.deviceName)
            try:
                # Try to get stream data
                stream_data = await mower.api.get_stream_subscription(
                    mower.device.deviceName, mower.device.iotId
                )
                if stream_data:
                    _LOGGER.debug("Received stream data: %s", stream_data)
                    entities.extend(
                        MammotionWebRTCCamera(
                            mower.reporting_coordinator, entity_description
                        )
                        for entity_description in CAMERAS
                    )
                else:
                    _LOGGER.error("No Agora data for %s", mower.device.deviceName)
            except (ClientError, TimeoutError) as err:
                _LOGGER.debug("Camera unavailable during setup: %s", err)

    async_add_entities(entities)
    await async_setup_platform_services(hass)


class MammotionMapCamera(MammotionBaseEntity, Camera):
    """Camera entity rendering the mower map."""

    entity_description: MammotionMapCameraEntityDescription
    _attr_has_entity_name = True
    _attr_content_type = "image/png"

    def __init__(
        self,
        coordinator: MammotionBaseUpdateCoordinator,
        entity_description: MammotionMapCameraEntityDescription,
    ) -> None:
        """Initialize the map camera entity."""
        super().__init__(coordinator, entity_description.key)
        Camera.__init__(self)
        self.coordinator = coordinator
        self.entity_description = entity_description
        self._attr_translation_key = entity_description.key
        self._attr_model = coordinator.device.deviceName

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return a rendered map image."""
        map_data = getattr(self.coordinator.data, "map", None)
        if not map_data:
            return None

        points = [
            (getattr(pt, "x", 0), getattr(pt, "y", 0))
            for plan in getattr(map_data, "plan", {}).values()
            for pt in getattr(plan, "data", [])
        ]

        if not points:
            return None

        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)

        width = width or 500
        height = height or 500
        scale_x = width / (max_x - min_x or 1)
        scale_y = height / (max_y - min_y or 1)

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        prev = None
        for x, y in points:
            px = int((x - min_x) * scale_x)
            py = int((y - min_y) * scale_y)
            if prev is not None:
                draw.line([prev, (px, py)], fill="green", width=2)
            prev = (px, py)

        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


class MammotionWebRTCCamera(MammotionBaseEntity, Camera):
    """Mammotion WebRTC camera entity."""

    entity_description: MammotionCameraEntityDescription
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: MammotionBaseUpdateCoordinator,
        entity_description: MammotionCameraEntityDescription,
    ) -> None:
        """Initialize the WebRTC camera entity."""
        super().__init__(coordinator, entity_description.key)
        self.coordinator = coordinator
        self.entity_description = entity_description
        self._attr_translation_key = entity_description.key
        self._stream_data: StreamSubscriptionResponse | None = None
        self._attr_model = coordinator.device.deviceName
        self.access_tokens = [secrets.token_hex(16)]
        self._webrtc_provider = None  # Avoid crash on async_refresh_providers()
        self._legacy_webrtc_provider = None
        self._supports_native_sync_webrtc = False
        self._supports_native_async_webrtc = False

    @property
    def frontend_stream_type(self) -> StreamType | None:
        """Return the type of stream supported by this camera."""
        return StreamType.WEB_RTC

    @property
    def content_type(self) -> str:
        """Return the content type of the camera image."""
        return "image/jpeg"

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return a still image response from the camera."""
        # WebRTC cameras typically don't support still images
        return None

    async def async_handle_async_webrtc_offer(
        self, offer_sdp: str, session_id: str, send_message: WebRTCSendMessage
    ) -> None:
        """Handle the WebRTC offer from the browser.

        This function is required by the Home Assistant interface,
        but it will not actually be used because we are using the Agora SDK.
        """
        _LOGGER.warning(
            "A native WebRTC offer from Home Assistant was received, "
            "but it will be ignored because we are using the Agora SDK directly in the frontend."
        )

        # Informs the frontend that it must use the Agora SDK
        send_message(
            '{"type":"error","error":"Use the Agora SDK for this camera","useAgoraSDK":true}',
            session_id,
        )


# Global
async def async_setup_platform_services(  # noqa: C901
    hass: HomeAssistant,
) -> None:
    """Register custom services for streaming."""

    domain_data = hass.data.setdefault("mammotion", {})
    if domain_data.get(DATA_SERVICES_REGISTERED):
        return

    def _get_mower_by_entity_id(entity_id: str) -> MammotionMowerData | None:
        entity_entry = er.async_get(hass).async_get(entity_id)
        if entity_entry is None or entity_entry.platform != "mammotion":
            return None
        for config_entry in hass.config_entries.async_entries("mammotion"):
            for mower in getattr(config_entry, "runtime_data", ()):
                if entity_entry.unique_id.startswith(f"{mower.name}_"):
                    return mower
        return None

    async def handle_refresh_stream(call) -> None:
        entity_id = call.data["entity_id"]
        mower = _get_mower_by_entity_id(entity_id)
        if mower:
            stream_data = await mower.api.get_stream_subscription(
                mower.device.deviceName, mower.device.iotId
            )
            _LOGGER.debug("Refresh stream data : %s", stream_data)

            mower.reporting_coordinator.set_stream_data(stream_data)
            mower.reporting_coordinator.async_update_listeners()

    async def handle_start_video(call) -> None:
        entity_id = call.data["entity_id"]
        mower = _get_mower_by_entity_id(entity_id)
        if mower:
            await mower.reporting_coordinator.join_webrtc_channel()

    async def handle_stop_video(call) -> None:
        entity_id = call.data["entity_id"]
        mower = _get_mower_by_entity_id(entity_id)
        if mower:
            await mower.reporting_coordinator.leave_webrtc_channel()

    async def handle_get_tokens(call: ServiceCall) -> ServiceResponse:
        entity_id = call.data["entity_id"]
        mower = _get_mower_by_entity_id(entity_id)
        if mower is not None:
            stream_data = mower.reporting_coordinator.get_stream_data()

            if not stream_data or stream_data.data is None:
                return {}
            # Return all the data needed for the Agora SDK
            return stream_data.data.to_dict()
        return {}

    async def handle_move_forward(call) -> None:
        entity_id = call.data["entity_id"]

        speed = call.data["speed"]

        mower = _get_mower_by_entity_id(entity_id)
        if mower:
            await mower.reporting_coordinator.async_move_forward(speed=speed)

    async def handle_move_left(call) -> None:
        entity_id = call.data["entity_id"]

        speed = call.data["speed"]

        mower = _get_mower_by_entity_id(entity_id)
        if mower:
            await mower.reporting_coordinator.async_move_left(speed=speed)

    async def handle_move_right(call) -> None:
        entity_id = call.data["entity_id"]

        speed = call.data["speed"]

        mower = _get_mower_by_entity_id(entity_id)
        if mower:
            await mower.reporting_coordinator.async_move_right(speed=speed)

    async def handle_move_backward(call) -> None:
        entity_id = call.data["entity_id"]

        speed = call.data["speed"]

        mower = _get_mower_by_entity_id(entity_id)
        if mower:
            await mower.reporting_coordinator.async_move_back(speed=speed)

    hass.services.async_register(
        "mammotion", "refresh_stream", handle_refresh_stream, ENTITY_SERVICE_SCHEMA
    )
    hass.services.async_register(
        "mammotion", "start_video", handle_start_video, ENTITY_SERVICE_SCHEMA
    )
    hass.services.async_register(
        "mammotion", "stop_video", handle_stop_video, ENTITY_SERVICE_SCHEMA
    )
    hass.services.async_register(
        "mammotion",
        "get_tokens",
        handle_get_tokens,
        ENTITY_SERVICE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        "mammotion", "move_forward", handle_move_forward, MOVEMENT_SERVICE_SCHEMA
    )
    hass.services.async_register(
        "mammotion", "move_left", handle_move_left, MOVEMENT_SERVICE_SCHEMA
    )
    hass.services.async_register(
        "mammotion", "move_right", handle_move_right, MOVEMENT_SERVICE_SCHEMA
    )
    hass.services.async_register(
        "mammotion", "move_backward", handle_move_backward, MOVEMENT_SERVICE_SCHEMA
    )
    domain_data[DATA_SERVICES_REGISTERED] = True


def async_remove_platform_services(hass: HomeAssistant) -> None:
    """Remove global camera services after the final entry unloads."""
    for service in CAMERA_SERVICES:
        hass.services.async_remove("mammotion", service)
    hass.data.setdefault("mammotion", {}).pop(DATA_SERVICES_REGISTERED, None)
