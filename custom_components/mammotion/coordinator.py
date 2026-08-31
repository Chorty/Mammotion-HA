"""Provides the mammotion DataUpdateCoordinator."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import json
import secrets
import time
from abc import abstractmethod
from collections import deque
from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

from aiohttp import ClientError
from habluetooth import BluetoothScanningMode
from habluetooth.models import BluetoothServiceInfoBleak
from homeassistant.components import bluetooth
from homeassistant.components.bluetooth import (
    BluetoothCallbackMatcher,
    BluetoothChange,
    async_register_callback,
)
from homeassistant.const import CONF_PASSWORD
from homeassistant.core import CALLBACK_TYPE, HassJob, HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryAuthFailed, HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.event import async_call_later, async_track_time_interval
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from mashumaro.exceptions import InvalidFieldValue
from pymammotion.aliyun.exceptions import (
    CloudSetupError,
    DeviceOfflineException,
    FailedRequestException,
    GatewayTimeoutException,
    TooManyRequestsException,
)
from pymammotion.aliyun.model.dev_by_account_response import Device
from pymammotion.client import MammotionClient
from pymammotion.data.model import GenerateRouteInformation
from pymammotion.data.model.device import (
    MowerDevice,
    MowerInfo,
    MowingDevice,
    PoolCleanerDevice,
    RTKBaseStationDevice,
)
from pymammotion.data.model.device_config import OperationSettings, create_path_order
from pymammotion.data.model.hash_list import Plan, SvgMessage
from pymammotion.data.model.pool_state import PoolPlan, SpinoToggle
from pymammotion.data.model.report_info import Maintain, NetUsedType
from pymammotion.data.mqtt.event import DeviceNotificationEventParams, ThingEventMessage
from pymammotion.data.mqtt.properties import ThingPropertiesMessage
from pymammotion.data.mqtt.status import StatusType, ThingStatusMessage
from pymammotion.http.model.camera_stream import (
    StreamSubscriptionResponse,
)
from pymammotion.http.model.http import ErrorInfo, Response
from pymammotion.mammotion.commands.mammotion_command import MammotionCommand
from pymammotion.proto import MulLanguage, MulSex
from pymammotion.state.device_state import DeviceShutdownEvent, DeviceSnapshot
from pymammotion.transport.base import (
    AuthError,
    BLEUnavailableError,
    CommandTimeoutError,
    ConcurrentRequestError,
    LoginFailedError,
    NoTransportAvailableError,
    ReLoginRequiredError,
    SessionExpiredError,
    Subscription,
    TransportType,
)
from pymammotion.transport.ble import BLETransport
from pymammotion.utility.constant import MOWING_ACTIVE_MODES, WorkMode
from pymammotion.utility.device_type import DeviceType
from pymammotion.utility.plan_id import make_copy_name, new_mower_plan_id
from webrtc_models import RTCIceServer

from .agora_api import SERVICE_IDS, AgoraAPIClient, AgoraResponse
from .config import MammotionConfigStore
from .connectivity import CloudConnectivityMonitor, WatchdogAction
from .const import (
    CONF_ACCOUNTNAME,
    CONF_CONNECT_DATA,
    CONF_HAS_CLOUD_ACCOUNT,
    CONF_MAMMOTION_DATA,
    DOMAIN,
    EXPIRED_CREDENTIAL_EXCEPTIONS,
    LOGGER,
    NO_REQUEST_MODES,
)

if TYPE_CHECKING:
    from . import MammotionConfigEntry

# Upper bound on the opportunistic BLE reconnect attempted during a report
# update. ``BLETransport.connect()`` is not bounded overall: it can run a
# self-managed scan (scan_timeout 10s) and then ``establish_connection`` retries
# up to MAX_CONNECT_ATTEMPTS (4) with backoff, so a bad link can block for tens
# of seconds. This runs inline in ``_async_update_data``, so an unbounded call
# stalls the coordinator tick and every entity update behind it (a BLE call was
# observed hanging 32.7s on this hardware, 2026-07-14). Reconnecting is
# best-effort -- cloud transport still serves the update -- so cap it and move on.
_BLE_RECONNECT_TIMEOUT_SECONDS = 15.0

MAINTENANCE_INTERVAL = timedelta(minutes=60)
DEFAULT_INTERVAL = timedelta(minutes=30)
REPORT_INTERVAL = timedelta(minutes=5)
DYNAMICS_LINE_INTERVAL = timedelta(seconds=10)
DEVICE_VERSION_INTERVAL = timedelta(weeks=1)
MAP_INTERVAL = timedelta(minutes=60)
RTK_INTERVAL = timedelta(hours=5)
SPINO_INTERVAL = timedelta(weeks=1)

# Possible states for ``MammotionReportUpdateCoordinator.map_sync_status`` and
# the ``map_sync_status`` diagnostic ENUM sensor that surfaces it.
MAP_SYNC_STATUSES = ("synced", "syncing", "out_of_sync")

#: Rolling window for the command-timeout diagnostic. 24 h deliberately matches
#: the cloud transport's own `sends_in_window()` window so the two sensors can
#: be read against each other without mentally rescaling one of them.
_COMMAND_TIMEOUT_WINDOW_SECONDS = 24 * 60 * 60
CLOUD_SEND_LIMIT_STATES = ("ok", "rate_limited")

# Cloud response code returned by the stream-subscription endpoint when the
# device is unreachable ("Device not responding. Please check the network
# connection").  Treated as a device-offline signal.
DEVICE_NOT_RESPONDING_CODE = 50504
STREAM_AUTH_ERROR_CODE = 401
STREAM_RETRY_DELAYS = (0, 2, 4)


class MammotionBaseUpdateCoordinator[DataT](DataUpdateCoordinator[DataT]):
    """Mammotion DataUpdateCoordinator."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: MammotionConfigEntry,
        device: Device,
        mammotion: MammotionClient,
        update_interval: timedelta,
        unique_name: str | None = None,
    ) -> None:
        """Initialize global mammotion data updater."""
        super().__init__(
            hass=hass,
            logger=LOGGER,
            name=DOMAIN,
            update_interval=update_interval,
            config_entry=config_entry,
        )
        self._ice_servers: list[RTCIceServer] = []
        self._agora_response: AgoraResponse | None = None
        self.service_info: BluetoothServiceInfoBleak | None = None
        assert config_entry.unique_id
        self.account = config_entry.data.get(CONF_ACCOUNTNAME, "")
        self.password = config_entry.data.get(CONF_PASSWORD, "")
        self.device: Device = device
        self.device_name = device.device_name
        self.unique_name = (
            unique_name if unique_name is not None else device.device_name
        )
        self.manager: MammotionClient = mammotion
        self._operation_settings = OperationSettings()
        self.update_failures = 0
        # Monotonic timestamps of CommandTimeoutError raised out of
        # `async_send_command`, the single funnel every queued command passes
        # through. Kept as a rolling 24 h window to match the cloud transport's
        # own `sends_in_window()` accounting, so the two diagnostics can be read
        # side by side. Monotonic, not wall clock: the window must survive a
        # clock step, and it resets on restart, which is the honest behaviour
        # for a counter that only ever lived in memory.
        self._command_timeouts: deque[float] = deque()
        self._stream_data: Response[StreamSubscriptionResponse] | None = (
            None  # Stream data [Agora]
        )
        self._stream_data_fetched_at: float = 0.0  # monotonic timestamp of last fetch
        self._STREAM_TOKEN_TTL: float = 300.0  # seconds before we re-fetch
        _mammotion_data = config_entry.data.get(CONF_MAMMOTION_DATA) or {}
        try:
            _user_account = int(
                _mammotion_data["data"]["userInformation"]["userAccount"]
            )
        except KeyError, TypeError, ValueError:
            _user_account = 0
        self.commands = MammotionCommand(device.device_name, _user_account)
        self._subscriptions: list[Subscription] = []
        # Position payloads are intentionally separate from normal coordinator
        # state updates.  A fresh payload may reduce to byte-for-byte identical
        # state, but it is still new safety evidence and must remain observable.
        self._position_sample_stream: Any | None = None
        self._position_sample_task: asyncio.Task[None] | None = None
        self._latest_position_sample: Any | None = None
        self._latest_position_consumed_at: float | None = None
        self._position_payload_intervals: deque[float] = deque(maxlen=100)
        self.map_offset_lat: float = 0.0
        self.map_offset_lon: float = 0.0
        self._bluetooth_enabled: bool = True
        self._cloud_enabled: bool = True
        self._connectivity_monitor = CloudConnectivityMonitor()
        self.last_map_sync: datetime.datetime | None = None
        #: ``bol_hash`` the last map-sync attempt was made against.  Used to
        #: back off a sync that is not converging — see ``_should_start_map_sync``.
        self.last_map_sync_bol_hash: int | None = None
        #: Name of the motion service currently holding this mower's manual-motion
        #: claim, or ``None``.  Set by ``services._wrap_exclusive_manual_motion``;
        #: read here so the coordinator never starts an exclusive map saga in the
        #: middle of a guarded motion run (the saga would block the mower's command
        #: queue and stall the run's pulses).
        self.manual_motion_owner: str | None = None
        #: Freshness tracking for the RTK report channel. On 2026-08-07 the whole
        #: RTK group froze for **three hours** while every other field kept
        #: updating: `rtk_position` held "float" from 15:40 to 18:39 and only
        #: moved once the base station was power-cycled. Nothing marked it stale,
        #: so a three-hour-old reading presented as current and the motion gate
        #: still reported `valid_for_motion: True`. A forced burst of 50 reports
        #: refreshed none of it, which is how the latch was caught.
        #:
        #: Detection follows the principle the linear phase already uses
        #: (`_streak_shows_dead_telemetry`): a live feed is never perfectly
        #: still. The RTK payload carries satellite counts and per-band signal
        #: quality, which drift continuously as the constellation moves -- when
        #: the base returned, they went 26 -> 23 and 35 -> 29 in the same tick.
        #: A byte-identical payload therefore means reports stopped arriving,
        #: not that reception is unusually steady.
        self._rtk_fingerprint: tuple[Any, ...] | None = None
        self._rtk_fingerprint_changed_at: float = time.monotonic()
        # The service layer owns the concrete ManualMotionSession type to avoid
        # coupling normal mower operation to experimental motion internals.
        self.manual_motion_session: Any | None = None
        self.last_manual_motion_session: Any | None = None
        self.last_task_sync: datetime.datetime | None = None
        self.last_map_task_error: str | None = None
        self.last_cloud_login_success: datetime.datetime | None = None
        self.last_token_refresh: datetime.datetime | None = None
        self.last_command_failure_reason: str | None = None
        self.last_camera_stream_failure_code: str | None = None

        mower_device = self.manager.get_device_by_name(self.device_name)
        assert mower_device is not None

        current_data = cast(DataT | None, getattr(self, "data", None))
        if current_data is None:
            setattr(self, "data", cast(DataT, mower_device))

    @property
    def has_cloud_account(self) -> bool:
        """Return True if cloud login is active for this entry."""
        config_entry = self.config_entry
        if config_entry is not None and CONF_HAS_CLOUD_ACCOUNT in config_entry.data:
            return bool(config_entry.data[CONF_HAS_CLOUD_ACCOUNT])
        return bool(self.account)

    @abstractmethod
    def get_coordinator_data(self, device: Any) -> DataT:
        """Get coordinator data."""

    async def async_check_stream_expiry(
        self, force: bool = False
    ) -> tuple[StreamSubscriptionResponse | None, AgoraResponse | None]:
        """Return cached Agora data or establish a fresh camera session."""
        now = time.monotonic()
        token_age = now - self._stream_data_fetched_at
        cached_data = self._stream_data

        if not force and (
            cached_data is not None
            and cached_data.data is not None
            and token_age < self._STREAM_TOKEN_TTL
            and self._agora_response is not None
        ):
            LOGGER.debug("Reusing cached stream token (age=%.0fs)", token_age)
            return cached_data.data, self._agora_response

        self.clear_stream_data()
        try:
            await self.async_send_command("send_todev_ble_sync", sync_type=3)
            await self.join_webrtc_channel()
        except (
            AuthError,
            ClientError,
            CommandTimeoutError,
            ConcurrentRequestError,
            DeviceOfflineException,
            FailedRequestException,
            GatewayTimeoutException,
            HomeAssistantError,
            NoTransportAvailableError,
            TimeoutError,
        ) as err:
            LOGGER.warning("Unable to start camera session: %s", type(err).__name__)
            self.last_camera_stream_failure_code = type(err).__name__
            return None, None

        stream_data: Response[StreamSubscriptionResponse] | None = None
        stream_auth_refreshed = False
        for delay in STREAM_RETRY_DELAYS:
            if delay:
                await asyncio.sleep(delay)
            try:
                stream_data = await self.manager.refresh_stream_subscription(
                    self.device_name, self.device.iot_id
                )
            except (
                AuthError,
                ClientError,
                DeviceOfflineException,
                FailedRequestException,
                GatewayTimeoutException,
                HomeAssistantError,
                TimeoutError,
            ) as err:
                LOGGER.warning("Camera token request failed: %s", type(err).__name__)
                self.last_camera_stream_failure_code = type(err).__name__
                self.clear_stream_data()
                return None, None
            if stream_data is not None and stream_data.data is not None:
                break
            if (
                stream_data is not None
                and stream_data.code == STREAM_AUTH_ERROR_CODE
                and not stream_auth_refreshed
                and self.manager.token_manager is not None
            ):
                stream_auth_refreshed = True
                LOGGER.warning(
                    "Camera stream token request returned 401; refreshing cloud credentials and retrying"
                )
                try:
                    await self.manager.token_manager.force_refresh(
                        TransportType.CLOUD_MAMMOTION
                    )
                    self.last_token_refresh = datetime.datetime.now(datetime.UTC)
                    self.store_cloud_credentials()
                except (
                    AuthError,
                    ClientError,
                    HomeAssistantError,
                    TimeoutError,
                ) as err:
                    LOGGER.warning(
                        "Camera stream credential refresh failed: %s",
                        type(err).__name__,
                    )
                    self.last_camera_stream_failure_code = type(err).__name__
                    self.clear_stream_data()
                    return None, None
                continue
            if stream_data is None or stream_data.code != DEVICE_NOT_RESPONDING_CODE:
                break

        if stream_data is None or stream_data.data is None:
            code = stream_data.code if stream_data is not None else "no_response"
            LOGGER.warning("Camera stream is temporarily unavailable (code %s)", code)
            self.last_camera_stream_failure_code = str(code)
            self.clear_stream_data()
            return None, None

        subscription = stream_data.data.to_dict()
        try:
            async with AgoraAPIClient() as agora_client:
                agora_response = await agora_client.choose_server(
                    app_id=subscription["appid"],
                    token=subscription["token"],
                    channel_name=subscription["channelName"],
                    user_id=int(subscription["uid"]),
                    service_flags=[
                        SERVICE_IDS["CHOOSE_SERVER"],
                        SERVICE_IDS["CLOUD_PROXY_FALLBACK"],
                    ],
                )
        except (
            ClientError,
            json.JSONDecodeError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ) as err:
            LOGGER.warning("Unable to configure camera relay: %s", type(err).__name__)
            self.last_camera_stream_failure_code = type(err).__name__
            self.clear_stream_data()
            return None, None

        if agora_response is None:
            self.last_camera_stream_failure_code = "missing_agora_response"
            self.clear_stream_data()
            return None, None

        try:
            self._ice_servers = [
                RTCIceServer(
                    urls=ice_server.urls,
                    username=ice_server.username,
                    credential=ice_server.credential,
                )
                for ice_server in agora_response.get_ice_servers(
                    use_all_turn_servers=False
                )
            ]
        except (AttributeError, TypeError, ValueError) as err:
            LOGGER.warning("Invalid camera relay response: %s", type(err).__name__)
            self.last_camera_stream_failure_code = type(err).__name__
            self.clear_stream_data()
            return None, None
        self._stream_data = stream_data
        self._stream_data_fetched_at = time.monotonic()
        self._agora_response = agora_response
        self.last_camera_stream_failure_code = None
        LOGGER.debug("Camera stream credentials refreshed")
        return stream_data.data, agora_response

    def set_stream_data(
        self, stream_data: Response[StreamSubscriptionResponse] | None
    ) -> None:
        """Set stream data."""
        self._stream_data = stream_data

    def get_stream_data(self) -> Response[StreamSubscriptionResponse] | None:
        """Return stream data."""
        return self._stream_data

    @property
    def is_on_4g(self) -> bool:
        """Return True when the device's active network interface is 4G/cellular."""
        device = self.manager.get_device_by_name(self.device_name)
        if device is None:
            return False
        try:
            return device.report_data.connect.used_net == NetUsedType.MNET
        except AttributeError:
            return False

    async def join_webrtc_channel(self) -> None:
        """Start stream command."""
        await self.async_send_command(
            "device_agora_join_channel_with_position", enter_state=1
        )

    async def leave_webrtc_channel(self) -> None:
        """End stream command."""
        try:
            await self.async_send_command(
                "device_agora_join_channel_with_position", enter_state=0
            )
        finally:
            self.clear_stream_data()

    def clear_stream_data(self) -> None:
        """Discard cached stream and relay credentials."""
        self._stream_data = None
        self._stream_data_fetched_at = 0.0
        self._agora_response = None
        self._ice_servers = []

    async def set_scheduled_updates(self, enabled: bool) -> None:
        """Enable or disable scheduled polling updates for this device."""
        device = self.manager.get_device_by_name(self.device_name)
        if device is None:
            return
        device.enabled = enabled
        if enabled:
            self.update_failures = 0
            if not device.online:
                device.online = True
        await self.manager.set_scheduled_updates(self.device_name, enabled=enabled)
        handle = self.manager.mower(self.device_name)
        if handle is not None:
            if enabled:
                await handle.restart_keep_alive()
            else:
                await handle.stop_polling()

    def is_online(self) -> bool:
        """Return True if the device currently has an active transport connection."""
        device = self.manager.get_device_by_name(self.device_name)
        if device is None:
            return False
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return bool(device.online)
        if handle.has_transport(TransportType.BLE) and (
            ble := handle.get_transport(TransportType.BLE)
        ):
            if ble.is_usable:
                return True
        if (
            self.has_cloud_account
            and self._cloud_enabled
            and handle.cloud_transport() is None
        ):
            # The cloud transport was detached (failed unbound migration) and
            # BLE can't cover the device: every send would raise, so report
            # offline honestly instead of pretending the device is reachable.
            return False
        return bool(not handle.availability.mqtt_reported_offline)

    @property
    def mqtt_transport_connected(self) -> bool:
        if handle := self.manager.mower(self.device_name):
            for t_type in (TransportType.CLOUD_ALIYUN, TransportType.CLOUD_MAMMOTION):
                if handle.is_transport_connected(t_type):
                    return True
        return False

    @property
    def mqtt_device_online(self) -> bool:
        device = self.manager.get_device_by_name(self.device_name)
        if device is None:
            return False
        if handle := self.manager.mower(self.device_name):
            return bool(not handle.availability.mqtt_reported_offline)
        return False

    def _cloud_transport(self) -> Any | None:
        """Return the connected cloud transport, or the first registered one.

        Two cloud transports can be registered (Aliyun and Mammotion) and only
        one is normally live, so prefer a connected one and fall back to
        whichever exists — a rate limit on an idle transport is still worth
        reporting.
        """
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return None
        cloud_types = (TransportType.CLOUD_ALIYUN, TransportType.CLOUD_MAMMOTION)
        fallback: Any | None = None
        for t_type in cloud_types:
            if not handle.has_transport(t_type):
                continue
            transport = handle.get_transport(t_type)
            if transport is None:
                continue
            if handle.is_transport_connected(t_type):
                return transport
            if fallback is None:
                fallback = transport
        return fallback

    @property
    def cloud_sends_in_window(self) -> int | None:
        """Cloud sends in the transport's rolling 24 h window, or None.

        Counts every PHYSICAL send including retries, so ACK degradation shows
        up here as an elevated rate before it shows up as a rate limit. None
        means no cloud transport is registered, which is not the same as zero.
        """
        transport = self._cloud_transport()
        if transport is None:
            return None
        try:
            return int(transport.sends_in_window())
        except AttributeError, TypeError, ValueError:
            return None

    @property
    def cloud_send_limit_state(self) -> str | None:
        """`rate_limited` while cloud sends are blocked, else `ok`.

        Covers both sources PyMammotion folds into `is_rate_limited`: a
        cloud-imposed 429 ban and the self-imposed rolling quota.
        """
        transport = self._cloud_transport()
        if transport is None:
            return None
        try:
            return "rate_limited" if bool(transport.is_rate_limited) else "ok"
        except AttributeError, TypeError:
            return None

    def record_command_timeout(self) -> None:
        """Note a command timeout for the rolling 24 h diagnostic."""
        self._command_timeouts.append(time.monotonic())

    @property
    def command_timeouts_in_window(self) -> int:
        """Command timeouts in the last 24 h.

        ⚠️ Counts timeouts raised out of `async_send_command` only. Commands
        dispatched by the motion executors in `services.py` do not pass through
        it, so this is a measure of the INTEGRATION's command health, not of a
        motion run's. Reads as 0 on a healthy link and resets on restart.
        """
        cutoff = time.monotonic() - _COMMAND_TIMEOUT_WINDOW_SECONDS
        while self._command_timeouts and self._command_timeouts[0] < cutoff:
            self._command_timeouts.popleft()
        return len(self._command_timeouts)

    @property
    def bluetooth_enabled(self) -> bool:
        """Return whether Bluetooth transport is enabled."""
        return self._bluetooth_enabled

    @property
    def cloud_enabled(self) -> bool:
        """Return whether Cloud transport is enabled."""
        return self._cloud_enabled

    @property
    def active_transport_state(self) -> str:
        """Return normalized active transport state for diagnostics."""
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return "none"
        active_transport: Any
        try:
            active_transport = handle.active_transport()
        except NoTransportAvailableError:
            # PyMammotion raises while every registered transport is currently
            # unusable (for example, cloud has reported the mower offline and
            # BLE has not connected yet). This is a normal fail-closed runtime
            # state, not an entity setup failure.
            return "none"
        except AttributeError, TypeError, ValueError:
            return "unknown"
        if active_transport is None:
            return "none"
        if active_transport == TransportType.BLE:
            return "ble"
        if active_transport == TransportType.CLOUD_ALIYUN:
            return "cloud_aliyun"
        if active_transport == TransportType.CLOUD_MAMMOTION:
            return "cloud_mammotion"

        active_transport_label = str(active_transport).lower()
        if "ble" in active_transport_label:
            return "ble"
        if "aliyun" in active_transport_label:
            return "cloud_aliyun"
        if "mammotion" in active_transport_label or "cloud" in active_transport_label:
            return "cloud_mammotion"
        return "unknown"

    @property
    def ble_only_fallback_mode(self) -> bool:
        """Return True when BLE is available while cloud transport is unavailable."""
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return False
        ble_usable = False
        if handle.has_transport(TransportType.BLE):
            ble = handle.get_transport(TransportType.BLE)
            ble_usable = bool(ble is not None and ble.is_usable)
        cloud_connected = any(
            handle.is_transport_connected(t_type)
            for t_type in (TransportType.CLOUD_ALIYUN, TransportType.CLOUD_MAMMOTION)
        )
        return ble_usable and not cloud_connected

    async def async_set_bluetooth_enabled(self, enabled: bool) -> None:
        """Enable or disable Bluetooth transport."""
        self._bluetooth_enabled = enabled
        handle = self.manager.mower(self.device_name)
        if handle is None:
            self._async_refresh_motion_gate_entities()
            return
        if not enabled:
            handle.set_prefer_ble(value=False)
            try:
                await handle.disconnect_transport(TransportType.BLE)
            finally:
                self._async_refresh_motion_gate_entities()
        else:
            handle.set_prefer_ble(value=True)
            try:
                await self._async_ensure_ble_client()
            except BLEUnavailableError as exc:
                # Enabling Bluetooth is a durable preference, not a guarantee
                # that the mower is advertising at this instant. Keep it
                # enabled so the advertisement callback can connect later;
                # ble_link_live remains fail-closed until that happens.
                LOGGER.debug(
                    "Bluetooth enabled for %s but no live connection is "
                    "currently available: %s",
                    self.device_name,
                    exc,
                )
            finally:
                self._async_refresh_motion_gate_entities()

    @callback
    def _async_refresh_motion_gate_entities(self) -> None:
        """Invalidate cached gate diagnostics and notify entities immediately."""
        # ``motion_gate_snapshot`` caches six related entities for five seconds.
        # A transport toggle is an explicit state transition, so retaining that
        # snapshot makes ble_link_live display the pre-toggle state until the
        # next coordinator tick.
        setattr(self, "_mammotion_gate_snapshot_monotonic", float("-inf"))
        self.async_update_listeners()

    async def async_set_cloud_enabled(self, enabled: bool) -> None:
        """Enable or disable Cloud transport."""
        self._cloud_enabled = enabled
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return
        if enabled:
            for t_type in (TransportType.CLOUD_ALIYUN, TransportType.CLOUD_MAMMOTION):
                await handle.connect_transport(t_type)
            await handle.restart_keep_alive()
        else:
            for t_type in (TransportType.CLOUD_ALIYUN, TransportType.CLOUD_MAMMOTION):
                await handle.disconnect_transport(t_type)

    async def async_refresh_login(self, exc: Exception | None = None) -> None:
        """Refresh login credentials asynchronously.

        LoginFailedError means the client already exhausted all recovery options
        (targeted refresh → force refresh → full re-login).  Raise ConfigEntryAuthFailed
        so HA tells the user to reconfigure the integration.

        For other auth errors, selectively refresh the affected transport:
        - SessionExpiredError: refreshes credentials for the specific transport.
        - AuthError (generic): performs a full login refresh.
        - Other/unknown: performs a full login refresh.
        """
        if not self.has_cloud_account:
            return

        if isinstance(exc, LoginFailedError):
            raise ConfigEntryAuthFailed(
                f"Login failed for Mammotion account: {exc}"
            ) from exc
        try:
            if (
                isinstance(exc, SessionExpiredError)
                and self.manager.token_manager is not None
            ):
                await self.manager.token_manager.refresh_aliyun_credentials()
                self.last_token_refresh = datetime.datetime.now(datetime.UTC)
            elif isinstance(exc, AuthError) and self.manager.token_manager is not None:
                await self.manager.token_manager.refresh_mqtt_credentials()
                self.last_token_refresh = datetime.datetime.now(datetime.UTC)
            else:
                await self.manager.refresh_login(self.account)
                self.store_cloud_credentials()
                self.last_token_refresh = datetime.datetime.now(datetime.UTC)
            self.last_cloud_login_success = datetime.datetime.now(datetime.UTC)
        except CloudSetupError as err:
            LOGGER.error("Aliyun cloud setup failed during re-login: %s", err)
            raise HomeAssistantError(
                translation_domain=DOMAIN, translation_key="cloud_setup_failed"
            ) from err
        except ReLoginRequiredError as err:
            raise ConfigEntryAuthFailed(
                f"Re-authentication required for Mammotion account: {err}"
            ) from err

    async def _async_connectivity_watchdog(self) -> None:
        """Detect a stuck cloud transport and attempt in-place recovery.

        Backstop for the push-driven recovery path: when the cloud transport
        stays disconnected across consecutive report ticks (and BLE can't
        cover the device), reconnect it in place.  A cloud transport that is
        missing entirely (detached by pymammotion after a failed unbound
        migration) cannot be re-attached from here — warn once so the state
        is visible instead of silently dropping every send.
        """
        if (
            self.hass.is_stopping
            or not self.has_cloud_account
            or not self._cloud_enabled
        ):
            return
        device = self.manager.get_device_by_name(self.device_name)
        if device is None or not device.enabled:
            return
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return

        ble = handle.get_transport(TransportType.BLE)
        ble_usable = bool(ble is not None and ble.is_usable and self._bluetooth_enabled)
        cloud_tt = handle.cloud_transport()
        cloud_transport = (
            handle.get_transport(cloud_tt) if cloud_tt is not None else None
        )
        auth_locked = bool(
            cloud_transport is not None
            and cloud_transport.is_unrecoverable_auth_failure
        )

        action = self._connectivity_monitor.tick(
            ble_usable=ble_usable,
            cloud_registered=cloud_tt is not None,
            cloud_connected=self.mqtt_transport_connected,
            auth_locked=auth_locked,
        )

        if self._connectivity_monitor.detached_warning_pending:
            self._connectivity_monitor.record_detached_warning()
            LOGGER.warning(
                "%s: cloud transport is no longer attached and cannot be "
                "restored in place — commands will fail until the Mammotion "
                "config entry is reloaded",
                self.device_name,
            )
            return

        if action is WatchdogAction.RECONNECT and cloud_tt is not None:
            LOGGER.warning(
                "%s: cloud transport %s disconnected across multiple update "
                "cycles — attempting reconnect",
                self.device_name,
                cloud_tt.value,
            )
            self._connectivity_monitor.record_reconnect_attempted()
            await self._async_reconnect_cloud(cloud_tt)

    async def _async_reconnect_cloud(self, transport_type: TransportType) -> None:
        """Reconnect the registered cloud transport in place.

        Issues no MQTT sends: ``connect_transport`` is a socket connect and
        ``restart_keep_alive`` re-arms pymammotion's own self-pacing loops.
        """
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return
        try:
            await handle.connect_transport(transport_type)
            await handle.restart_keep_alive()
        except EXPIRED_CREDENTIAL_EXCEPTIONS as exc:
            await self.async_refresh_login(exc)
        except (TimeoutError, OSError, HomeAssistantError) as exc:
            LOGGER.debug(
                "%s: cloud reconnect attempt failed: %s", self.device_name, exc
            )

    async def async_send_and_wait(
        self,
        command: str,
        expected_field: str,
        **kwargs: Any,
    ) -> None:
        """Send a command and wait for response with standard exception handling.

        Handles credential expiry, gateway/transport timeouts, and device-offline
        conditions uniformly.  Re-raises DeviceOfflineException after marking the
        device offline so callers can bail out of their update loops.
        """
        device = self.manager.get_device_by_name(self.device_name)
        if device is None or not self.is_online():
            return

        try:
            await self.manager.send_command_and_wait(
                self.device_name,
                command,
                expected_field,
                prefer_ble=self._bluetooth_enabled,
                **kwargs,
            )
        except EXPIRED_CREDENTIAL_EXCEPTIONS as exc:
            self.update_failures += 1
            await self.async_refresh_login(exc)
        except DeviceOfflineException:
            device = self.manager.get_device_by_name(self.device_name)
            if device is not None:
                self.device_offline(device)
        except TooManyRequestsException as exc:
            raise HomeAssistantError(
                translation_domain=DOMAIN, translation_key="api_limit_exceeded"
            ) from exc
        except NoTransportAvailableError as exc:
            LOGGER.debug(f"No Transport: {exc}")
        except (
            GatewayTimeoutException,
            CommandTimeoutError,
            ConcurrentRequestError,
        ):
            pass
        except asyncio.CancelledError:
            # bleak_retry_connector raises CancelledError when no BLE slot is
            # available (it cancels its own internal sleep).  Re-raise only when
            # the enclosing task is genuinely being cancelled; otherwise treat it
            # as a transient BLE failure and let setup continue.
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            LOGGER.debug(
                "BLE connection cancelled (no available slot) for %s — skipping",
                self.device_name,
            )

    @staticmethod
    def device_offline(device: MowingDevice | RTKBaseStationDevice) -> None:
        """Mark the device as offline in its state model."""
        device.online = False

    def store_cloud_credentials(self) -> None:
        """Store cloud credentials in config entry."""
        if config_entry := self.config_entry:
            cache = self.manager.to_cache()
            if not cache:
                return
            # Translate library key "connect_response" → HA key CONF_CONNECT_DATA
            translated = {
                (CONF_CONNECT_DATA if k == "connect_response" else k): v
                for k, v in cache.items()
            }
            self.hass.config_entries.async_update_entry(
                config_entry, data={**config_entry.data, **translated}
            )

    async def async_send_command(self, command: str, **kwargs: Any) -> bool | None:
        """Send command via MammotionClient command queue."""
        device = self.manager.get_device_by_name(self.device_name)
        if device is None or not self.is_online():
            return False

        try:
            await self.manager.send_command_with_args(
                self.device_name,
                command,
                prefer_ble=kwargs.pop("prefer_ble", self._bluetooth_enabled),
                skip_if_saga_active=False,
                **kwargs,
            )
            self.update_failures = 0
            self.last_command_failure_reason = None
            return True
        except FailedRequestException as exc:
            self.update_failures += 1
            self.last_command_failure_reason = f"{command}:{type(exc).__name__}"
        except EXPIRED_CREDENTIAL_EXCEPTIONS as exc:
            self.update_failures += 1
            self.last_command_failure_reason = f"{command}:{type(exc).__name__}"
            await self.async_refresh_login(exc)
        except CommandTimeoutError:
            # Count it and re-raise UNCHANGED. Callers already handle this --
            # several catch it and pass -- so swallowing it here to make the
            # accounting tidier would silently change command behaviour. The
            # counter is a diagnostic; it must not become a control-flow change.
            self.record_command_timeout()
            raise
        except GatewayTimeoutException as ex:
            LOGGER.error(f"Gateway timeout exception: {ex.iot_id}")
            self.update_failures = 0
            self.last_command_failure_reason = f"{command}:{type(ex).__name__}"
            return False
        except DeviceOfflineException as exc:
            self.last_command_failure_reason = f"{command}:{type(exc).__name__}"
            self.device_offline(device)
        except TooManyRequestsException as exc:
            self.last_command_failure_reason = f"{command}:{type(exc).__name__}"
            raise HomeAssistantError(
                translation_domain=DOMAIN, translation_key="api_limit_exceeded"
            ) from exc
        except NoTransportAvailableError as exc:
            self.last_command_failure_reason = f"{command}:{type(exc).__name__}"
            LOGGER.debug(
                "No transport connected yet for %s, command '%s' skipped",
                self.device_name,
                command,
            )
            raise HomeAssistantError(
                translation_domain=DOMAIN, translation_key="command_failed"
            ) from exc
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            LOGGER.debug(
                "BLE connection cancelled (no available slot) for %s — skipping",
                self.device_name,
            )
            self.last_command_failure_reason = f"{command}:CancelledError"
            return False
        return False

    async def async_refresh_camera_stream(self) -> None:
        """Refresh camera stream credentials and cache immediately."""
        stream_data, _ = await self.async_check_stream_expiry(force=True)
        if stream_data is None:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="camera_temporarily_unavailable",
            )

    async def async_refresh_cloud_session(self) -> None:
        """Refresh cloud session credentials for this device account."""
        if not self.has_cloud_account:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="command_failed",
            )
        await self.async_refresh_login()

    async def async_send_cloud_command(
        self, iot_id: str, command: bytes
    ) -> bool | None:
        """Send a raw cloud command via the device's active transport."""
        device = self.manager.get_device_by_name(self.device_name)
        if device is None or not self.is_online():
            return False
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return False

        try:
            await handle.send_raw(command)
            self.update_failures = 0
            return True
        except FailedRequestException:
            self.update_failures += 1
        except EXPIRED_CREDENTIAL_EXCEPTIONS as exc:
            self.update_failures += 1
            await self.async_refresh_login(exc)
        except GatewayTimeoutException as ex:
            LOGGER.error(f"Gateway timeout exception: {ex.iot_id}")
            self.update_failures = 0
            return False
        except DeviceOfflineException as ex:
            LOGGER.error(f"Device offline: {ex.iot_id}")
            self.device_offline(device)
            return False
        except NoTransportAvailableError:
            LOGGER.error("Device offline: no transport available")
            self.device_offline(device)
            return False
        except TooManyRequestsException as exc:
            raise HomeAssistantError(
                translation_domain=DOMAIN, translation_key="api_limit_exceeded"
            ) from exc
        except ReLoginRequiredError as err:
            raise ConfigEntryAuthFailed(
                f"Re-authentication required for Mammotion account: {err}"
            ) from err
        return False

    async def async_send_bluetooth_command(self, key: str, **kwargs: Any) -> None:
        """Send command via BLE transport."""
        await self.async_send_command(key, prefer_ble=True, **kwargs)

    async def check_firmware_version(self) -> None:
        """Check if firmware version is updated."""
        device = self.manager.get_device_by_name(self.device_name)
        if device is None:
            return
        device_registry = dr.async_get(self.hass)
        device_entry = device_registry.async_get_device(
            identifiers={(DOMAIN, self.device_name)}
        )
        if device_entry is None:
            return

        new_swversion = device.device_firmwares.device_version

        if new_swversion is not None and new_swversion != device_entry.sw_version:
            device_registry.async_update_device(
                device_entry.id, sw_version=new_swversion
            )

        if model_id := device.mower_state.model_id:
            if model_id is not None and model_id != device_entry.model_id:
                device_registry.async_update_device(device_entry.id, model_id=model_id)

    async def update_firmware(self, version: str) -> None:
        """Update firmware and clear cached version info so it is re-fetched after the upgrade."""
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return
        device = self.manager.get_device_by_name(self.device_name)
        if device is not None:
            device.clear_version_info()
        http = self.manager.mammotion_http
        if http is not None:
            await http.start_ota_upgrade(handle.iot_id, version)

    def _reported_bol_hash(self) -> int:
        """Return the device's currently reported map checksum, or 0."""
        device = self.manager.get_device_by_name(self.device_name)
        locations = getattr(getattr(device, "report_data", None), "locations", None)
        if not locations:
            return 0
        return int(getattr(locations[0], "bol_hash", 0) or 0)

    def _record_map_sync_attempt(self, bol_hash: int | None = None) -> None:
        """Stamp when a map sync ran and which checksum it targeted."""
        self.last_map_sync = datetime.datetime.now(datetime.UTC)
        self.last_map_sync_bol_hash = (
            self._reported_bol_hash() if bol_hash is None else bol_hash
        )

    def _should_start_map_sync(self, bol_hash: int) -> bool:
        """Return True when a background map sync is worth starting now.

        ``MapFetchSaga`` runs *exclusively* on the mower's command queue, and
        regular commands are ``Priority.NORMAL`` — they block on the exclusive
        slot until it finishes, and anything undispatched for ``_COMMAND_TTL``
        (120 s) is silently dropped.  So an unnecessary sync is not free: it can
        stall a guarded motion run's pulses and collapse the 200 ms refresh
        cadence the mower needs to keep driving.

        Two conditions suppress it:

        * a guarded motion run currently holds this mower — never contend with it
          (see :meth:`_raise_if_manual_motion_in_progress`, which covers the
          operator-triggered sync paths);
        * the previous attempt targeted the same ``bol_hash`` and was recent.
          ``is_map_synced()`` can stay False indefinitely on a map that is
          complete and perfectly usable (observed live 2026-07-24).

        A *changed* ``bol_hash`` always syncs immediately — that is a real
        device-side map edit and must not be delayed by the back-off.

        .. note::
           This predicate is **load-bearing as of 2026-07-25**.  Its call site
           in ``MammotionMapUpdateCoordinator._async_update_data`` was
           unreachable in steady state until
           :meth:`_async_short_circuit_update` stopped signalling "carry on"
           with an always-truthy value; it now runs on every map-coordinator
           tick (``MAP_INTERVAL``), which is exactly when the back-off below
           stops an unconverged ``is_map_synced()`` from re-running the
           exclusive saga forever.  The operator-triggered paths (the
           ``sync_maps`` button and ``force_map_resync``) remain guarded
           separately through :meth:`async_sync_maps`.
        """
        if self.manual_motion_owner is not None:
            return False
        if self.last_map_sync is None or self.last_map_sync_bol_hash != bol_hash:
            return True
        elapsed = datetime.datetime.now(datetime.UTC) - self.last_map_sync
        return elapsed >= MAP_INTERVAL

    def _raise_if_manual_motion_in_progress(self, action: str) -> None:
        """Refuse an exclusive device operation while a motion run owns the mower.

        ``MapFetchSaga`` runs at ``Priority.EXCLUSIVE`` and holds the mower's
        command queue until it completes.  Motion commands are
        ``Priority.NORMAL`` with ``skip_if_saga_active=False``, so they *block*
        on that slot rather than being dropped — and anything undispatched for
        ``_COMMAND_TTL`` (120 s) is discarded silently.  A sync started midway
        through a guarded run therefore stalls its pulses and collapses the
        200 ms refresh cadence the mower needs to keep driving, which makes it
        self-halt.

        This is the operator-facing half of the guard: the ``sync_maps`` button
        and ``force_map_resync`` are both reachable at any moment, including
        during a run.  Refusing loudly is better than queueing behind the saga,
        because the caller otherwise appears to succeed while the run degrades.
        """
        owner = self.manual_motion_owner
        if owner is None:
            return
        msg = (
            f"Cannot {action} while a guarded motion run is in progress "
            f"({owner}). A map sync holds the mower's command queue "
            "exclusively and would stall the run. Retry once it finishes."
        )
        raise HomeAssistantError(msg)

    async def async_sync_maps(self) -> None:
        """Get map data from the device."""
        self._raise_if_manual_motion_in_progress("sync maps")
        try:
            await self.manager.start_map_sync(self.device_name)
            self._record_map_sync_attempt()
            self.last_map_task_error = None

        except EXPIRED_CREDENTIAL_EXCEPTIONS as exc:
            self.update_failures += 1
            self.last_map_task_error = f"map_sync: {type(exc).__name__}"
            await self.async_refresh_login(exc)
            if self.update_failures < 5:
                await self.async_sync_maps()
        except Exception as exc:
            self.last_map_task_error = f"map_sync: {type(exc).__name__}"
            raise

    async def async_force_map_resync(self) -> dict[str, Any]:
        """Force a full map re-fetch and GeoJSON re-projection (recovery lever).

        Recovery for the "map stuck ``out_of_sync`` after a reload/restart"
        state, where the zone polygons never reload so click-to-path
        containment fails ``area_hash_not_found`` and the GeoJSON has no
        Polygon features.  A plain config-entry reload does not fix it — it
        restores the same cached map and never re-fetches the area frames.

        The sequence, in order, addresses each contributor found in the code:

        1. Refresh the RTK/dock reference — the map-sync saga's on-complete
           GeoJSON rebuild is skipped entirely when ``RTK.latitude == 0.0``.
        2. Fetch the area-name list — some cloud sessions never push
           ``toapp_all_hash_name`` on their own.
        3. Run the map-sync saga to (re)fetch the area frames.  Its on-complete
           handler restores ``root_hash_lists`` from the saga result, which is
           what lets a churning ``invalidate_maps`` cycle finally converge.
        4. Re-project the GeoJSON from the freshly-fetched frames.

        Non-destructive: the existing cache is left intact until the saga
        replaces it, so a failed resync never leaves the map worse off than it
        started.  Returns a step-by-step result for the caller/card to surface.

        Refused outright while a guarded motion run owns the mower — *every*
        step here enqueues device commands, and step 3's saga takes the command
        queue exclusively, so running this mid-run would stall the run's pulses.
        The refusal is reported as a normal result rather than raised, because
        this is a response service whose caller wants the diagnostics.
        """
        result: dict[str, Any] = {
            "map_sync_status_before": self.map_sync_status,
            "map_sync_diagnostics_before": self.map_sync_diagnostics(),
            "steps": [],
            "error": None,
        }
        if self.manual_motion_owner is not None:
            result["error"] = "manual_motion_in_progress"
            result["busy_owner"] = self.manual_motion_owner
            result["steps"].append("refused_manual_motion_in_progress")
            result["map_sync_status_after"] = self.map_sync_status
            result["map_sync_diagnostics_after"] = result["map_sync_diagnostics_before"]
            result["last_map_task_error"] = self.last_map_task_error
            return result
        try:
            await self.async_rtk_dock_location()
            result["steps"].append("rtk_dock_refreshed")
            try:
                await self.async_get_area_list()
                result["steps"].append("area_names_fetched")
            except (
                DeviceOfflineException,
                GatewayTimeoutException,
                NoTransportAvailableError,
                ConcurrentRequestError,
            ):
                # Non-fatal: the saga can still populate the area frames without
                # the name list; zones just fall back to generic labels.
                result["steps"].append("area_names_skipped")
            await self.async_sync_maps()
            result["steps"].append("map_synced")
            self.manager.regenerate_stale_geojson(self.device_name)
            result["steps"].append("geojson_regenerated")
        except Exception as exc:  # noqa: BLE001 - surface any failure as a field
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["map_sync_status_after"] = self.map_sync_status
        result["map_sync_diagnostics_after"] = self.map_sync_diagnostics()
        result["last_map_task_error"] = self.last_map_task_error
        return result

    async def async_sync_schedule(self) -> None:
        """Sync all scheduled mowing plans from the device via PlanFetchSaga."""
        try:
            await self.manager.start_plan_sync(self.device_name)
            self.last_task_sync = datetime.datetime.now(datetime.UTC)
            self.last_map_task_error = None
        except EXPIRED_CREDENTIAL_EXCEPTIONS as exc:
            self.update_failures += 1
            self.last_map_task_error = f"task_sync: {type(exc).__name__}"
            await self.async_refresh_login(exc)
            if self.update_failures < 5:
                await self.async_sync_schedule()
        except Exception as exc:
            self.last_map_task_error = f"task_sync: {type(exc).__name__}"
            raise

    async def async_fetch_audio_config(self) -> None:
        """Read current audio config (volume, language, gender) from device."""
        await self.async_send_and_wait("get_car_audio_cfg", "audio_cfg")

    async def async_set_voice_volume(self, volume: float) -> None:
        """Set robot voice volume (0–100)."""
        await self.async_send_and_wait(
            "set_car_volume", "set_audio", volume=int(volume)
        )

    async def async_set_prompt_volume(self, volume: float) -> None:
        """Set robot spoken-prompt volume (0–100).

        The app/APK exposes this multimedia audio field as ``au_switch`` on
        readback.  Pymammotion's command builder writes the same oneof field via
        ``set_car_volume`` / ``MulSetAudio.at_switch``.
        """
        await self.async_send_and_wait(
            "set_car_volume", "set_audio", volume=int(volume)
        )

    async def async_set_voice_on_off(self, on: bool) -> None:
        """Turn robot voice on (restores 50%) or off (sets volume to 0)."""
        await self.async_send_and_wait(
            "set_car_volume", "set_audio", volume=50 if on else 0
        )

    async def async_set_voice_gender(self, sex: str) -> None:
        """Set robot voice gender (MAN or WOMAN)."""
        await self.async_send_and_wait(
            "set_car_volume_sex", "set_audio", sex=MulSex[sex]
        )

    async def async_set_voice_language(self, language: str) -> None:
        """Set robot voice language."""
        await self.async_send_and_wait(
            "set_car_voice_language",
            "set_audio",
            language_type=MulLanguage[language],
        )

    async def async_run_camera_wiper(self, rounds: int = 2) -> None:
        """Run the camera wiper for the requested number of rounds."""
        await self.async_send_and_wait(
            "set_car_wiper", "set_wiper_ack", round_num=max(1, int(rounds))
        )

    async def async_set_device_wifi_enabled(self, enabled: bool) -> None:
        """Enable or disable the device Wi-Fi radio."""
        await self.async_send_command(
            "set_device_wifi_enable_status", new_wifi_status=enabled, prefer_ble=True
        )

    async def async_set_device_4g_enabled(self, enabled: bool) -> None:
        """Enable or disable the device 4G/mobile data radio."""
        await self.async_send_command(
            "set_device_4g_enable_status", new_4g_status=enabled, prefer_ble=True
        )

    async def async_start_stop_blades(
        self, start_stop: bool, blade_height: int = 60
    ) -> None:
        """Start stop blades."""
        if DeviceType.is_luba1(self.device_name):
            if start_stop:
                await self.async_send_and_wait(
                    "set_blade_control", "toapp_knife_status_change", on_off=1
                )
            else:
                await self.async_send_and_wait(
                    "set_blade_control", "toapp_knife_status_change", on_off=0
                )
        elif start_stop:
            if DeviceType.is_yuka(self.device_name) or DeviceType.is_yuka_mini(
                self.device_name
            ):
                blade_height = 0

            await self.async_send_command(
                "operate_on_device",
                main_ctrl=1,
                cut_knife_ctrl=1,
                cut_knife_height=blade_height,
                max_run_speed=1.2,
            )
        else:
            await self.async_send_command(
                "operate_on_device",
                main_ctrl=0,
                cut_knife_ctrl=0,
                cut_knife_height=blade_height,
                max_run_speed=1.2,
            )

    async def async_set_non_work_hours(self, start_time: str, end_time: str) -> None:
        """Set non work hours.

        start_time and end_time are in HH:MM format (24-hour).
        The proto field expects minutes-from-midnight as a string (e.g. "1320" for 22:00).
        """
        if start_time == end_time:
            await self.async_send_command("job_do_not_disturb_del")
            return

        def _to_minutes(hhmm: str) -> str:
            h, m = hhmm.split(":")
            return str(int(h) * 60 + int(m))

        await self.async_send_command(
            "job_do_not_disturb",
            unable_end_time=_to_minutes(end_time),
            unable_start_time=_to_minutes(start_time),
        )

    async def async_reset_blade_time(self) -> None:
        """Reset blade used time."""
        await self.async_send_and_wait(
            "reset_blade_time", "todev_reset_blade_used_time_status"
        )

    def _rw_expected_field(self, rw_id: int) -> str:
        """Return the expected response field for a read_write_device command.

        Mirrors the routing in MammotionCommand.read_write_device(): only
        rw_ids [3, 6, 7, 8, 10, 11] on Pro/X3 devices are sent via the nav
        adapter (nav_sys_param_cmd).  Every other rw_id — including 12 and 13
        used for wildlife safety — always goes through allpowerfull_rw() and
        responds on bidire_comm_cmd, regardless of device type.
        """
        if rw_id in (3, 6, 7, 8, 10, 11) and DeviceType.is_luba_pro(self.device_name):
            return "nav_sys_param_cmd"
        return "bidire_comm_cmd"

    async def async_set_rain_detection(self, on_off: bool) -> None:
        """Set rain detection."""
        await self.async_send_and_wait(
            "read_write_device",
            self._rw_expected_field(3),
            rw_id=3,
            context=int(on_off),
            rw=1,
        )

    async def async_read_rain_detection(self) -> None:
        """Read current rain detection state from device."""
        await self.async_send_and_wait(
            "read_write_device", self._rw_expected_field(3), rw_id=3, context=0, rw=0
        )

    async def async_set_sidelight(self, on_off: int) -> None:
        """Set Sidelight."""
        await self.async_send_and_wait(
            "read_and_set_sidelight",
            "todev_time_ctrl_light",
            is_sidelight=bool(on_off),
            operate=0,
        )

    async def async_read_sidelight(self) -> None:
        """Read current sidelight state from device."""
        await self.async_send_and_wait(
            "read_and_set_sidelight",
            "todev_time_ctrl_light",
            is_sidelight=False,
            operate=1,
        )

    async def async_set_manual_light(self, manual_ctrl: bool) -> None:
        """Set manual night light."""
        await self.async_send_and_wait(
            "set_car_manual_light", "set_lamp_rsp", manual_ctrl=manual_ctrl
        )

    async def async_read_manual_light(self) -> None:
        """Read current manual light state from device."""
        await self.async_send_and_wait("get_car_light", "get_lamp_rsp", ids=1126)

    async def async_set_night_light(self, night_light: bool) -> None:
        """Set night light."""
        await self.async_send_and_wait(
            "set_car_light", "set_lamp_rsp", on_off=night_light
        )

    async def async_read_night_light(self) -> None:
        """Read current night light state from device."""
        await self.async_send_and_wait("get_car_light", "get_lamp_rsp", ids=1123)

    async def async_set_traversal_mode(self, context: int) -> None:
        """Set traversal mode."""
        await self.async_send_and_wait(
            "traverse_mode", self._rw_expected_field(7), context=context
        )

    async def async_read_traversal_mode(self) -> None:
        """Read current traversal mode from device."""
        await self.async_send_and_wait(
            "read_write_device", self._rw_expected_field(7), rw_id=7, context=0, rw=0
        )

    async def async_set_wildlife_safety(self, mode: int) -> None:
        """Set wildlife safety mode (0=off, 1=stop mowing, 2=low-speed mowing).

        Sends rw_id=13 (status) first, then rw_id=12 (mode).  Both are sent
        via the device-appropriate channel (_rw_expected_field).
        """
        status = 0 if mode == 0 else 1
        await self.async_send_and_wait(
            "read_write_device",
            self._rw_expected_field(13),
            rw_id=13,
            context=status,
            rw=1,
        )
        await self.async_send_and_wait(
            "read_write_device",
            self._rw_expected_field(12),
            rw_id=12,
            context=mode,
            rw=1,
        )

    async def async_read_wildlife_safety(self) -> None:
        """Read current wildlife safety status and mode from device."""
        await self.async_send_and_wait(
            "read_write_device", self._rw_expected_field(13), rw_id=13, context=0, rw=0
        )
        await self.async_send_and_wait(
            "read_write_device", self._rw_expected_field(12), rw_id=12, context=0, rw=0
        )

    async def async_set_turning_mode(self, context: int) -> None:
        """Set turning mode."""
        await self.async_send_and_wait(
            "turning_mode", self._rw_expected_field(6), context=context
        )

    async def async_read_turning_mode(self) -> None:
        """Read current turning mode from device."""
        await self.async_send_and_wait(
            "read_write_device", self._rw_expected_field(6), rw_id=6, context=0, rw=0
        )

    async def async_blade_height(self, height: int) -> int:
        """Set blade height."""
        await self.async_send_and_wait(
            "set_blade_height", "toapp_knife_status_change", height=height
        )
        return height

    async def async_set_cutter_speed(self, mode: int) -> None:
        """Set cutter speed."""
        await self.async_send_and_wait(
            "set_cutter_mode", "cutter_mode_ctrl_by_hand", cutter_mode=mode
        )

    async def async_read_cutter_mode(self) -> None:
        """Query the current cutter mode and live RPM from the device."""
        await self.async_send_and_wait("get_cutter_mode", "current_cutter_mode")

    async def async_reset_blade_warning_time(self) -> None:
        """Reset blade used time to zero."""
        await self.async_send_and_wait(
            "reset_blade_time", "todev_reset_blade_used_time_status"
        )

    async def async_set_blade_warning_time(self, hours: int) -> None:
        """Set the blade warning time in hours."""
        await self.async_send_command("set_blade_warning_time", hours=hours)

    async def async_set_speed(self, speed: float) -> None:
        """Set working speed."""
        await self.async_send_and_wait(
            "set_speed", "bidire_speed_read_set", speed=speed
        )

    async def async_leave_dock(self) -> None:
        """Leave dock."""
        await self.send_command_and_update("leave_dock", "todev_taskctrl_ack")

    async def async_cancel_task(self) -> None:
        """Cancel task."""
        await self.send_command_and_update("cancel_job", "todev_taskctrl_ack")

    async def _async_ensure_ble_client(self) -> None:
        """Attach a BLE transport if we have an address but no client yet.

        Called before movement commands that prefer BLE so that a freshly
        discovered device (or one that was out of range at startup) gets a
        transport without waiting for the next full coordinator refresh.

        Short-circuits when the registered BLETransport already has the same
        BLEDevice address — avoids re-wiring on every 30 min refresh tick.
        Per-advertisement freshness is handled by the bluetooth callback in
        ``__init__.py``; this method only covers the case where no transport
        was wired (e.g. mower out of range at integration startup).
        """

        if not self._bluetooth_enabled:
            return

        device = self.manager.get_device_by_name(self.device_name)
        if device is None:
            return
        ble_mac = device.mower_state.ble_mac
        if not ble_mac:
            return
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return

        # If a BLE transport already exists and has the same address cached, do nothing.
        # The per-advertisement callback (_ble_seen) handles routine refreshes.
        if ble := handle.get_transport(TransportType.BLE):
            if not ble.is_connected:
                await ble.connect()

            if ble.is_connected:
                return

        ble_device = bluetooth.async_ble_device_from_address(
            self.hass, ble_mac.upper(), True
        )
        if ble_device is None:
            return

        await self.manager.add_ble_to_device(self.device_name, ble_device)

    async def async_move_forward(
        self, speed: float, use_wifi: bool = False
    ) -> bool | None:
        """Move forward. Prefer BLE unless use_wifi=True (lower latency for manual control)."""
        if not use_wifi:
            await self._async_ensure_ble_client()
        return await self.async_send_command(
            "move_forward", prefer_ble=not use_wifi, linear=speed
        )

    async def async_move_left(
        self, speed: float, use_wifi: bool = False
    ) -> bool | None:
        """Move left. Prefer BLE unless use_wifi=True."""
        if not use_wifi:
            await self._async_ensure_ble_client()
        return await self.async_send_command(
            "move_left", prefer_ble=not use_wifi, angular=speed
        )

    async def async_move_right(
        self, speed: float, use_wifi: bool = False
    ) -> bool | None:
        """Move right. Prefer BLE unless use_wifi=True."""
        if not use_wifi:
            await self._async_ensure_ble_client()
        return await self.async_send_command(
            "move_right", prefer_ble=not use_wifi, angular=speed
        )

    async def async_move_back(
        self, speed: float, use_wifi: bool = False
    ) -> bool | None:
        """Move back. Prefer BLE unless use_wifi=True."""
        if not use_wifi:
            await self._async_ensure_ble_client()
        return await self.async_send_command(
            "move_back", prefer_ble=not use_wifi, linear=speed
        )

    async def async_stop_manual_motion(self, use_wifi: bool = False) -> dict[str, Any]:
        """Stop low-level manual motion by sending zero linear/angular velocity."""
        if not use_wifi:
            await self._async_ensure_ble_client()
        linear_ok = await self.async_send_command(
            "move_forward", prefer_ble=not use_wifi, linear=0.0
        )
        angular_ok = await self.async_send_command(
            "move_left", prefer_ble=not use_wifi, angular=0.0
        )
        return {"linear_ok": linear_ok, "angular_ok": angular_ok}

    async def async_rtk_dock_location(self) -> None:
        """RTK and dock location."""
        await self.async_send_and_wait(
            "read_write_device",
            "bidire_comm_cmd",
            rw_id=5,
            rw=1,
            context=1,
        )

    async def async_get_area_list(self) -> None:
        """Fetch area names and wait for the toapp_all_hash_name response."""
        await self.async_send_and_wait(
            "get_area_name_list",
            "toapp_all_hash_name",
            device_id=self.device.iot_id,
        )

    async def async_set_area_name(self, hash_id: int, name: str) -> None:
        """Push a user-edited area name to the device.

        The device acks with a single toapp_map_name_msg (hash + name) which the
        pymammotion reducer applies to map.area_name, so no local write-back is
        needed here.
        """
        await self.async_send_and_wait(
            "set_area_name",
            "toapp_map_name_msg",
            device_id=self.device.iot_id,
            hash_id=hash_id,
            name=name,
        )

    async def async_relocate_charging_station(self) -> None:
        """Reset charging station."""
        await self.async_send_command("delete_charge_point")
        # fetch charging location?
        """
        nav {
          todev_get_commondata {
            pver: 1
            subCmd: 2
            action: 6
            type: 5
            totalFrame: 1
            currentFrame: 1
          }
        }
        """

    async def send_command_and_update(
        self, command_str: str, response: str | None = None, **kwargs: Any
    ) -> None:
        """Send command and update."""
        if response is not None:
            await self.async_send_and_wait(command_str, response, **kwargs)
        else:
            await self.async_send_command(command_str, **kwargs)
        await self.async_get_reports(count=5)

    def note_rtk_report_seen(self, device: MowingDevice | None) -> None:
        """Record whether the RTK report channel produced anything new.

        Called on every report refresh. Only the *changed-at* timestamp is
        stored, so a genuinely stable link keeps a small age while a latched one
        grows without bound.
        """
        rtk = getattr(getattr(device, "report_data", None), "rtk", None)
        if rtk is None:
            return
        fingerprint = (
            getattr(rtk, "status", None),
            getattr(rtk, "gps_stars", None),
            getattr(rtk, "age", None),
            getattr(rtk, "lat_std", None),
            getattr(rtk, "lon_std", None),
            tuple(getattr(rtk, "l1_satellites", None) or ()),
            tuple(getattr(rtk, "l2_satellites", None) or ()),
        )
        if fingerprint != self._rtk_fingerprint:
            self._rtk_fingerprint = fingerprint
            self._rtk_fingerprint_changed_at = time.monotonic()

    @property
    def rtk_report_age_seconds(self) -> float:
        """Seconds since the RTK report payload last changed."""
        return round(time.monotonic() - self._rtk_fingerprint_changed_at, 3)

    async def async_request_report_snapshot(self) -> None:
        """Fire a one-shot count=1 snapshot; no-op while BLE stream is active."""
        await self.manager.request_report_snapshot(self.device_name)

    async def async_start_report_stream(self, duration_ms: int = 300_000) -> None:
        """Start a transient continuous report window via the library."""
        await self.manager.start_report_stream(self.device_name, duration_ms)

    async def async_get_reports(self, count: int = 5) -> None:
        """Get reports from the device."""
        await self.manager.request_reports(self.device_name, count=count)

    async def async_start_continuous_reports(self, duration_ms: int = 30_000) -> None:
        """Subscribe to continuous device reports for *duration_ms*.

        ``start_report_stream`` is not enough on its own: pymammotion downgrades
        it to a single ``count=1`` snapshot unless the device is in ACTIVE mode
        (mowing or returning). A manually driven mower sits in MODE_READY, so it
        gets one report and then silence -- which is exactly what blinded the
        closed loop on hardware 2026-07-30, where every position sample across a
        whole pulse was bit-identical while the mower demonstrably moved 4
        inches.

        ``count=0`` with ``RPT_START`` is the same continuous subscription the
        Mammotion app uses, reported at the library's default 1000 ms period.
        The device-side ``timeout`` bounds the subscription so it expires on its
        own -- pymammotion arms no stop timer on this path, so nothing here
        depends on a teardown that might never run.
        """
        await self.manager.request_reports(
            self.device_name, count=0, timeout=duration_ms
        )

    async def async_stop_continuous_reports(self) -> None:
        """Stop a transient continuous report subscription explicitly."""
        await self.manager.request_iot_sync_continuous_stop(self.device_name)

    async def async_ensure_fresh_state(self) -> None:
        """Fire a one-shot snapshot if device state is older than 2 minutes."""
        await self.manager.ensure_fresh_state(self.device_name, max_age_s=120.0)

    async def send_svg_command(self, svg_message: SvgMessage) -> int | None:
        """Send an SVG tile to the device using the multi-frame saga protocol.

        Chunks *svg_message* into 500-character frames and sends them one at a
        time, waiting for a per-frame device ACK after each.  Returns the
        device-assigned ``data_hash`` for use in subsequent UPDATE or DELETE
        operations.

        Args:
            svg_message: Fully-populated message from
                         :func:`~pymammotion.utility.svg.build_svg_for_area` or
                         :func:`~pymammotion.utility.svg.build_svg_update`.

        Returns:
            Device-assigned ``data_hash``, or ``None`` on failure.

        """
        result: asyncio.Future[int | None] = asyncio.get_running_loop().create_future()

        async def _store_result(device_hash: int | None) -> None:
            if not result.done():
                result.set_result(device_hash)

        await self.manager.send_svg(
            self.device_name,
            svg_message,
            on_complete=_store_result,
        )
        return result.result() if result.done() else None

    def generate_route_information(
        self, operation_settings: OperationSettings
    ) -> GenerateRouteInformation:
        """Generate route information."""
        device: MowingDevice = cast(MowingDevice, self.data)
        if device.report_data.dev:
            dev = device.report_data.dev
            if dev.collector_status.collector_installation_status == 0:
                operation_settings.is_dump = False

        if DeviceType.is_yuka(self.device_name):
            operation_settings.blade_height = -10

        route_information = GenerateRouteInformation(
            one_hashs=list(operation_settings.areas),
            rain_tactics=operation_settings.rain_tactics,
            speed=operation_settings.speed,
            ultra_wave=operation_settings.ultra_wave,  # touch no touch etc
            toward=operation_settings.toward,  # is just angle (route angle)
            toward_included_angle=operation_settings.toward_included_angle  # demond_angle
            if operation_settings.channel_mode == 1
            else 0,  # crossing angle relative to grid
            toward_mode=operation_settings.toward_mode,
            blade_height=operation_settings.blade_height,
            channel_mode=operation_settings.channel_mode,  # single, double, segment or none (route mode)
            channel_width=operation_settings.channel_width,  # path space
            job_mode=operation_settings.job_mode,  # taskMode grid or border first
            edge_mode=operation_settings.mowing_laps,  # perimeter/mowing laps
            path_order=create_path_order(operation_settings, self.device_name),
            obstacle_laps=operation_settings.obstacle_laps,
        )

        if DeviceType.is_luba1(self.device_name):
            route_information.toward_mode = 0
            route_information.toward_included_angle = 0
        return route_information

    async def async_plan_route(
        self, operation_settings: OperationSettings
    ) -> bool | None:
        """Plan mow."""
        route_information = self.generate_route_information(operation_settings)

        # not sure if this is artificial limit
        # if (
        #     DeviceType.is_mini_or_x_series(device_name)
        #     and route_information.toward_mode == 0
        # ):
        #     route_information.toward = 0
        await self.async_send_and_wait(
            "generate_route_information",
            "bidire_reqconver_path",
            generate_route_information=route_information,
        )
        return True

    async def async_get_plan_route(self, operation_settings: OperationSettings) -> None:
        """Fetch the previously generated mow path from the device without replanning."""
        route_information = self.generate_route_information(operation_settings)
        await self.manager.start_mow_path_saga(
            self.device_name,
            zone_hashs=list(operation_settings.areas),
            route_info=route_information,
            skip_planning=True,
        )

    async def async_modify_plan_route(
        self, operation_settings: OperationSettings
    ) -> bool | None:
        """Modify plan mow."""

        if work := cast(MowingDevice, self.data).work:
            operation_settings.areas = list(dict.fromkeys(work.zone_hashs))
            operation_settings.toward = work.toward
            operation_settings.toward_mode = work.toward_mode
            operation_settings.toward_included_angle = work.toward_included_angle
            operation_settings.mowing_laps = work.edge_mode
            operation_settings.job_mode = work.job_mode
            operation_settings.job_id = work.job_id
            operation_settings.job_version = work.job_ver

        route_information = self.generate_route_information(operation_settings)

        return await self.async_send_command(
            "modify_route_information", generate_route_information=route_information
        )

    async def start_task(self, plan_id: str) -> None:
        """Start task."""
        await self.async_send_and_wait(
            "single_schedule", "todev_planjob_set", plan_id=plan_id
        )

    # ------------------------------------------------------------------
    # Mower task CRUD — backed by NavPlanJobSet on the wire.
    # All helpers look up the existing Plan from ``self.data.map.plan`` so
    # round-trip operations (enable / rename / edit / copy) preserve the
    # rest of the plan (reserved bytes, recurrence, areas, …) verbatim.
    # See ``docs/tasks_and_schedules.md`` § 1.
    # ------------------------------------------------------------------

    def _lookup_mower_plan(self, plan_id: str) -> Plan:
        """Return the stored mower Plan keyed by ``plan_id`` or raise."""
        plan = cast(MowingDevice, self.data).map.plan.get(plan_id)
        if plan is None:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="task_not_found",
                translation_placeholders={"plan_id": plan_id},
            )
        return plan

    async def async_create_mower_task(self, plan: Plan) -> None:
        """Create a brand-new mower schedule with a freshly generated plan_id.

        Caller passes a fully-populated Plan **without** a plan_id; this
        helper assigns one via :func:`new_mower_plan_id` so the device
        treats the write as a create rather than an edit.
        """
        plan_with_id = dataclasses.replace(plan, plan_id=new_mower_plan_id())
        await self.async_send_command("create_plan", plan=plan_with_id)

    async def async_edit_mower_task(self, plan: Plan) -> None:
        """Edit an existing mower schedule (``sub_cmd=4``)."""
        await self.async_send_command("edit_plan", plan=plan)

    async def async_rename_mower_task(self, plan_id: str, new_name: str) -> None:
        """Rename the mower schedule identified by ``plan_id`` to ``new_name``."""
        plan = self._lookup_mower_plan(plan_id)
        await self.async_send_command("rename_plan", plan=plan, new_name=new_name)

    async def async_set_mower_task_enabled(self, plan_id: str, enabled: bool) -> None:
        """Flip the enable flag (``reserved[2]``) on an existing mower schedule.

        The existing plan is round-tripped verbatim so the other reserved
        bytes (knife height, edge mode, …) are preserved.
        """
        plan = self._lookup_mower_plan(plan_id)
        await self.async_send_command("enable_plan", plan=plan, enabled=enabled)

    async def async_delete_mower_task(self, plan_id: str) -> None:
        """Delete the mower schedule identified by ``plan_id`` (``sub_cmd=3``)."""
        await self.async_send_command("delete_plan_by_id", plan_id=plan_id)

    async def async_copy_mower_task(
        self, plan_id: str, new_name: str | None = None
    ) -> None:
        """Duplicate the mower schedule under a new id + auto-generated name.

        Reuses :func:`make_copy_name` against the currently stored plans so
        successive copies produce ``Copy-1, Copy-2, …`` without collision.
        """
        plan = self._lookup_mower_plan(plan_id)
        existing_names = {
            p.task_name for p in cast(MowingDevice, self.data).map.plan.values()
        }
        resolved_name = new_name or make_copy_name(existing_names)
        await self.async_send_command(
            "copy_plan",
            plan=plan,
            new_name=resolved_name,
            new_plan_id=new_mower_plan_id(),
        )

    async def async_refresh_mower_tasks(self) -> None:
        """Re-fetch the mower schedule list via :class:`PlanFetchSaga`."""
        try:
            await self.manager.start_plan_sync(self.device_name)
            self.last_task_sync = datetime.datetime.now(datetime.UTC)
            self.last_map_task_error = None
        except EXPIRED_CREDENTIAL_EXCEPTIONS as exc:
            self.update_failures += 1
            self.last_map_task_error = f"task_sync: {type(exc).__name__}"
            await self.async_refresh_login(exc)
            if self.update_failures < 5:
                await self.async_refresh_mower_tasks()
        except Exception as exc:
            self.last_map_task_error = f"task_sync: {type(exc).__name__}"
            raise

    async def async_restart_mower(self) -> None:
        """Restart mower."""
        await self.async_send_command("remote_restart")

    def clear_update_failures(self) -> None:
        """Clear update failures and reconnect transports if needed."""
        self.update_failures = 0
        handle = self.manager.mower(self.device_name)
        if handle is None or not self._cloud_enabled:
            return
        cloud_tt = handle.cloud_transport()
        if cloud_tt is not None and not handle.is_transport_connected(cloud_tt):
            self.hass.async_create_task(self._async_reconnect_cloud_task(cloud_tt))

    async def _async_reconnect_cloud_task(self, transport_type: TransportType) -> None:
        """Run a cloud reconnect from a detached task.

        ``ConfigEntryAuthFailed`` only triggers reauth when raised inside the
        coordinator's update method — from a task it would just be an
        unhandled exception, so start the reauth flow explicitly instead.
        """
        try:
            await self._async_reconnect_cloud(transport_type)
        except ConfigEntryAuthFailed:
            self.config_entry.async_start_reauth(self.hass)

    @property
    def operation_settings(self) -> OperationSettings:
        """Return operation settings for planning."""
        return self._operation_settings

    async def async_modify_plan_if_mowing(self) -> None:
        """Re-plan the current mow route if the device is actively mowing."""
        _mdata = cast(MowingDevice, self.data)
        if (
            int(_mdata.report_data.work.bp_hash) in _mdata.work.zone_hashs
            and (_mdata.report_data.work.area >> 16) != 100
        ):
            await self.async_modify_plan_route(self.operation_settings)

    async def async_restore_data(self) -> None:
        """Restore saved data."""
        store: MammotionConfigStore = MammotionConfigStore(
            self.hass, version=1, minor_version=2, key=self.device_name
        )
        restored_data: Mapping[str, Any] | None = await store.async_load()

        handle = self.manager.mower(self.device_name)

        if restored_data is None:
            empty = MowingDevice()
            self.data = cast(DataT, empty)
            if handle is not None:
                handle.restore_device(empty)
            return

        try:
            if restored_data is not None:
                mower_state = MowingDevice().from_dict(restored_data)
                if handle is not None:
                    handle.restore_device(mower_state)
                    self.data = cast(DataT, mower_state)
                    # Re-project any restored map geometry now.  The cached
                    # GeoJSON is tied to the RTK yaw it was built with; after a
                    # restart the live heading may differ (or the cached GeoJSON
                    # may be stale/absent).  The only other regeneration triggers
                    # fire on the mowing report hot path, so an idle mower being
                    # repositioned would otherwise never re-project.  pymammotion
                    # guards this internally (skips when map.area is empty or the
                    # yaw/hashes are unchanged), matching its documented contract
                    # to call this after ``restore_device``.
                    self.manager.regenerate_stale_geojson(self.device_name)
        except InvalidFieldValue:
            empty = MowingDevice()
            self.data = cast(DataT, empty)
            if handle is not None:
                handle.restore_device(empty)

    async def async_save_data(self, data: MowingDevice | PoolCleanerDevice) -> None:
        """Store data."""
        store: Store = Store(
            self.hass, version=1, minor_version=2, key=self.device_name
        )
        await store.async_save(data.to_dict())

    async def remove_saved_data(self) -> None:
        """Remove saved coordinator data from persistent storage."""
        store: Store[dict[str, Any]] = Store(
            self.hass, version=1, minor_version=2, key=self.device_name
        )
        await store.async_remove()

    async def _async_short_circuit_update(self) -> DataT | None:
        """Run the checks shared by every coordinator's update.

        Returns the data to publish when the update must **stop here** (device
        gone, disabled, offline, mid-map-edit, or failing repeatedly), and
        ``None`` when the caller should carry on with its own update work.

        The ``None`` is load-bearing. This used to be ``_async_update_data`` and
        ended with ``return self.data``, and every subclass began with
        ``if data := await super()._async_update_data(): return data``. Because
        ``MowingDevice``/``MowerInfo``/``Maintain`` define neither ``__bool__``
        nor ``__len__``, that value was **always truthy**, so the early return
        fired on every healthy tick and everything after it in all five
        subclasses was unreachable in steady state -- it only ran once per HA
        start, while ``self.data`` was still ``None``.

        That silently disabled the per-tick map-sync check (so a device-side map
        edit was never picked up until a restart) and
        :meth:`MammotionReportUpdateCoordinator._async_opportunistic_ble_reconnect`
        (so the mower could sit on cloud transport indefinitely at healthy RSSI
        with its connect cooldown long expired -- the exact symptom that function
        was written for). Measured 2026-07-25: with DEBUG on, HA logged
        ``Finished fetching mammotion data in 0.000 seconds (success: True)``
        every tick while the ``LOGGER.debug`` three lines past the early return
        never appeared once.

        Callers must test ``is not None`` rather than truthiness, so a falsy but
        real payload can never be mistaken for "carry on".
        """
        device = self.manager.get_device_by_name(self.device_name)
        if device is None:
            return self.data

        if not device.enabled:
            return self.get_coordinator_data(device)

        handle = self.manager.mower(self.device_name)

        if not self.is_online():
            return self.get_coordinator_data(device)

        # Update BLE device address from HA bluetooth scanner if available
        if device.mower_state.ble_mac != "" and handle is not None:
            if ble_device := bluetooth.async_ble_device_from_address(
                self.hass, device.mower_state.ble_mac.upper(), True
            ):
                await self.manager.update_ble_device(self.device_name, ble_device)

        # Don't query the mower while users are doing map changes or it's updating.
        if device.report_data.dev.sys_status in NO_REQUEST_MODES:
            return self.get_coordinator_data(device)

        if self.update_failures > 5:
            async_call_later(
                self.hass,
                60,
                HassJob(lambda _: self.clear_update_failures()),
            )
            return self.get_coordinator_data(device)

        # Nothing short-circuited: the caller does its own update work.
        return None

    async def _async_update_notification(self, res: tuple[str, Any | None]) -> None:
        """Update data from incoming messages."""

    async def _async_update_properties(
        self, properties: ThingPropertiesMessage
    ) -> None:
        """Update data from incoming properties messages."""

    async def _async_update_status(self, status: ThingStatusMessage) -> None:
        """Update data from incoming status messages."""

    async def _async_update_event_message(self, event: ThingEventMessage) -> None:
        """Update data from incoming event messages."""

    async def _async_setup(self) -> None:
        handle = self.manager.mower(self.device_name)
        if handle is not None:
            self._subscriptions.extend(
                [
                    handle.subscribe_state_changed(
                        self._guarded(self._on_state_changed)
                    ),
                    handle.subscribe_device_status(
                        self._guarded(self._async_update_status)
                    ),
                    handle.subscribe_device_properties(
                        self._guarded(self._async_update_properties)
                    ),
                    handle.subscribe_device_event(
                        self._guarded(self._async_update_event_message)
                    ),
                    handle.subscribe_shutdown(self._guarded(self._on_device_shutdown)),
                ]
            )

    async def _on_device_shutdown(self, event: DeviceShutdownEvent) -> None:
        """React to a device-initiated power-off notification.

        The handle has already set mqtt_reported_offline=True (blocking further
        sends) and emitted a state-changed snapshot.  We force an immediate HA
        state write here so the entity availability reflects the shutdown before
        the debounce window or the next MQTT heartbeat timeout.
        """
        LOGGER.debug(
            "%s: device power-off notification (power_type=%d)",
            self.device_name,
            event.power_type,
        )
        handle = self.manager.mower(self.device_name)
        if handle is not None:
            self.async_set_updated_data(cast(DataT, handle.state_machine.current.raw))

    def _guarded(self, method: Any) -> Any:
        """Wrap a callback so it silently skips when HA is shutting down.

        During shutdown aiohttp's websocket layer may already be closing.
        Pushing state updates at that point raises ClientConnectionResetError
        inside shielded futures and logs noisy tracebacks.  Checking
        hass.is_stopping before every push prevents the error entirely.
        """

        async def _wrapper(*args: Any, **kwargs: Any) -> None:
            if self.hass.is_stopping:
                return
            await method(*args, **kwargs)

        return _wrapper

    def subscribe_map_updated(self, handler: Callable[[], None]) -> None:
        """Subscribe *handler* to map-updated events from the device handle.

        Fires only when ``toapp_all_hash_name`` is received or a ``MapFetchSaga``
        completes — not on every telemetry tick.  The subscription is kept alive
        for the lifetime of the coordinator and cancelled on shutdown.
        """
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return

        async def _on_map_updated() -> None:
            if not self.hass.is_stopping:
                handler()

        self._subscriptions.append(handle.subscribe_map_updated(_on_map_updated))

    async def async_shutdown(self) -> None:
        """Cancel all RAII subscriptions and delegate to HA coordinator shutdown."""
        if self._position_sample_stream is not None:
            self._position_sample_stream.close()
            self._position_sample_stream = None
        if self._position_sample_task is not None:
            self._position_sample_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._position_sample_task
            self._position_sample_task = None
        for sub in self._subscriptions:
            sub.cancel()
        self._subscriptions.clear()
        await super().async_shutdown()

    async def _on_state_changed(self, snapshot: DeviceSnapshot) -> None:
        """Push updated device data to HA."""
        cast(Any, self.device).online = True
        self.async_set_updated_data(cast(DataT, snapshot.raw))

    def find_entity_by_attribute_in_registry(
        self, attribute_name: str, attribute_value: Any
    ) -> tuple[str | None, er.RegistryEntry | None]:
        """Find an entity using the entity registry based on attributes."""
        entity_registry = er.async_get(self.hass)

        for entity_id, entity_entry in entity_registry.entities.items():
            entity_state = self.hass.states.get(entity_id)
            if (
                entity_state
                and entity_state.attributes.get(attribute_name) == attribute_value
            ):
                return entity_id, entity_entry

        return None, None

    def get_area_entity_name(self, area_hash: int) -> str | None:
        """Get string name of area hash."""
        if area_hash == 0:
            return None

        _mower_data = cast(MowingDevice, self.data)
        if area_hash not in _mower_data.map.area:
            return "path"

        # Prefer the user's HA-level entity name over the device-assigned name.
        entity_reg = er.async_get(self.hass)
        unique_id = f"{self.unique_name}_{area_hash}"
        entity_id = entity_reg.async_get_entity_id("switch", DOMAIN, unique_id)
        if entity_id and (entry := entity_reg.async_get(entity_id)) and entry.name:
            return entry.name

        for area in _mower_data.map.computed_areas:
            if area.hash == area_hash:
                return area.name

        return f"area {area_hash}"

    @property
    def map_sync_status(self) -> str:
        """Return the current map-sync status for diagnostics.

        One of :data:`MAP_SYNC_STATUSES`:

        * ``syncing`` — an exclusive sync saga (the map fetch) is running on
          the device command queue, so the cached map is mid-refresh.
        * ``synced`` — our local map fully matches the device's current area
          set (``map.is_map_synced`` against the latest reported ``bol_hash``).
        * ``out_of_sync`` — neither of the above: the cached map is stale or
          incomplete and a fresh ``async_sync_maps()`` is needed.
        """
        handle = self.manager.mower(self.device_name)
        if handle is not None and handle.queue.is_saga_active:
            return "syncing"

        if self.data is None:
            return "out_of_sync"

        mower_data = cast(MowingDevice, self.data)
        locations = mower_data.report_data.locations
        bol_hash = locations[0].bol_hash if locations else 0
        if mower_data.map.is_map_synced(bol_hash):
            return "synced"
        return "out_of_sync"

    def map_sync_diagnostics(self) -> dict[str, Any]:
        """Return why :attr:`map_sync_status` reads what it reads.

        ``is_map_synced()`` collapses three independent conditions into one
        boolean, so a mower can sit on ``out_of_sync`` indefinitely — re-running
        the sync saga on every coordinator tick — while its map is complete and
        perfectly usable for containment.  Observed live 2026-07-24: four areas
        with full polygon frames, containment passing, still ``out_of_sync``.

        Read-only; sends nothing to the device.
        """
        diagnostics: dict[str, Any] = {
            "status": self.map_sync_status,
            "reported_bol_hash": None,
            "computed_bol_hash": None,
            "bol_hash_matches": None,
            "incomplete_area_hashes": None,
            "area_names_covered": None,
            "area_frame_counts": None,
        }
        if self.data is None:
            return diagnostics

        mower_data = cast(MowingDevice, self.data)
        device_map = mower_data.map
        locations = mower_data.report_data.locations
        reported = locations[0].bol_hash if locations else 0
        try:
            computed = int(device_map.computed_bol_hash)
            incomplete = [str(h) for h in device_map.find_incomplete_hashes(0)]
            name_hashes = {a.hash for a in device_map.area_name}
            root_hashes = set(device_map.area_root_hashlist)
            diagnostics.update(
                {
                    "reported_bol_hash": str(reported),
                    "computed_bol_hash": str(computed),
                    "bol_hash_matches": bool(reported) and computed == reported,
                    "incomplete_area_hashes": incomplete,
                    "area_names_covered": (
                        not name_hashes or name_hashes.issubset(root_hashes)
                    ),
                    "area_frame_counts": {
                        str(area_hash): len(getattr(frames, "data", []) or [])
                        for area_hash, frames in device_map.area.items()
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        return diagnostics


class MammotionReportUpdateCoordinator(MammotionBaseUpdateCoordinator[MowingDevice]):
    """Mammotion report update coordinator."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: MammotionConfigEntry,
        device: Device,
        mammotion: MammotionClient,
        unique_name: str | None = None,
    ) -> None:
        """Initialize global mammotion data updater."""
        super().__init__(
            hass=hass,
            config_entry=config_entry,
            device=device,
            mammotion=mammotion,
            update_interval=REPORT_INTERVAL,
            unique_name=unique_name,
        )

        self._on_stop: list[CALLBACK_TYPE] = []

        self.poll_debouncer = Debouncer(
            hass,
            LOGGER,
            cooldown=60,
            immediate=True,
            function=self._add_ble_device,
            background=True,
        )

    @callback
    def _async_handle_bluetooth_event(
        self,
        service_info: BluetoothServiceInfoBleak,
        change: BluetoothChange,
    ) -> None:
        """Handle a bluetooth advertisement for this mower's MAC.

        Two responsibilities:

        1. Cache the latest ``service_info`` for downstream use (RSSI gates,
           freshness checks, etc.).
        2. Push the freshest ``BLEDevice`` into the existing BLETransport so
           ``bleak_retry_connector``'s ``ble_device_callback`` always has the
           most recent advertisement.  This is a synchronous pointer-swap
           (``BLETransport.set_ble_device``) — no event-loop work needed,
           safe to run in this ``@callback``-decorated handler.

        Initial transport wire-up (``add_ble_to_device``) is handled by
        :func:`_attach_ble_to_mower` in ``__init__.py``, which has access to
        the ``stay_connected_ble`` config flag.  Once the transport exists,
        every subsequent advertisement flows through this fast path.
        """
        self.service_info = service_info

        self.poll_debouncer.async_schedule_call()

    def _add_ble_device(self) -> None:
        if not self.service_info or not self._bluetooth_enabled:
            return
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return
        ble = handle.get_transport(TransportType.BLE)
        if ble is None:
            self.hass.create_task(self._async_ensure_ble_client())

        if ble := handle.get_transport(TransportType.BLE):
            if not ble.is_connected and self.data.enabled:
                cast(BLETransport, ble).set_ble_device(
                    self.service_info.device, self.service_info.rssi
                )
                self.hass.create_task(ble.connect())

    async def async_set_bluetooth_enabled(self, enabled: bool) -> None:
        """Enable or disable Bluetooth, reconnecting if re-enabled."""
        await super().async_set_bluetooth_enabled(enabled)
        if enabled:
            self._add_ble_device()

    @callback
    def _async_start(self) -> None:
        """Start the callbacks."""
        if self.data.mower_state.ble_mac != "":
            self._on_stop.append(
                async_register_callback(
                    self.hass,
                    self._async_handle_bluetooth_event,
                    BluetoothCallbackMatcher(
                        address=self.data.mower_state.ble_mac, connectable=True
                    ),
                    BluetoothScanningMode.ACTIVE,
                )
            )

    @callback
    def _async_stop(self) -> None:
        """Stop the callbacks."""
        for unsub in self._on_stop:
            unsub()
        self._on_stop.clear()

    def get_coordinator_data(self, device: MowingDevice) -> MowingDevice:
        """Get coordinator data."""
        return device

    async def _async_opportunistic_ble_reconnect(self) -> None:
        """Re-establish BLE during a report update when the link is usable again.

        pymammotion routes over MQTT while the BLE transport reports itself
        unusable (missing BLEDevice, RSSI under ``min_rssi``, or an armed connect
        cooldown). Once a cooldown lapses ``is_usable`` flips back but nothing
        reconnects until an advertisement happens to trigger ``_add_ble_device``,
        so the mower can sit on cloud transport long after BLE became viable --
        and real motion is gated on BLE (live 2026-07-19: repeatedly stuck on
        ``cloud_aliyun`` at healthy RSSI with the cooldown long expired).

        Best-effort and bounded: the update is already served by cloud transport,
        so a slow or failing connect must not stall the coordinator tick.

        .. note::
           This ran **only once per HA start** until 2026-07-25 -- it sits past
           the early return in :meth:`_async_update_data`, which fired on every
           healthy tick while :meth:`_async_short_circuit_update` still signalled
           "carry on" with an always-truthy value.  Measured the same day: the
           mower emits roughly one BLE advertisement burst per ~10 minutes, and
           although HA does push the fresh ``BLEDevice`` immediately on each one
           (``poll_debouncer`` -> ``_add_ble_device`` -> ``set_ble_device``),
           nothing then called ``connect()`` -- an advertisement at 18:56:56 was
           only followed by ``active_transport: ble`` at 19:04:47.  It now runs
           on every ``REPORT_INTERVAL`` tick, which is what closes that gap.
        """
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return
        ble = handle.get_transport(TransportType.BLE)
        if ble is None:
            return
        if not (
            handle.prefer_ble
            and ble.is_usable
            and not ble.is_connected
            and self._bluetooth_enabled
        ):
            return
        try:
            async with asyncio.timeout(_BLE_RECONNECT_TIMEOUT_SECONDS):
                await ble.connect()
        except BLEUnavailableError as exc:
            LOGGER.debug(
                "BLE unavailable for %s during update — continuing via cloud: %s",
                self.device_name,
                exc,
            )
        except TimeoutError:
            LOGGER.debug(
                "BLE reconnect for %s exceeded %.0fs — continuing via cloud",
                self.device_name,
                _BLE_RECONNECT_TIMEOUT_SECONDS,
            )

    async def _async_update_data(self) -> MowingDevice:
        """Get data from the device."""
        # Runs before _async_short_circuit_update's early-returns: the
        # watchdog must see every tick — a device with a detached or
        # disconnected cloud transport and no BLE is exactly the one
        # is_online() (and so the short-circuit) would skip.
        await self._async_connectivity_watchdog()

        if (data := await self._async_short_circuit_update()) is not None:
            return data

        device = self.manager.get_device_by_name(self.device_name)
        if device is None:
            LOGGER.debug("device not found")
            return self.data

        LOGGER.debug("Updated Mammotion device %s", self.device_name)
        self.update_failures = 0
        self.note_rtk_report_seen(device)
        await self.async_save_data(device)

        if self.data.mower_state.ble_mac != "" and len(self._on_stop) == 0:
            self._on_stop.append(
                async_register_callback(
                    self.hass,
                    self._async_handle_bluetooth_event,
                    BluetoothCallbackMatcher(
                        address=self.data.mower_state.ble_mac, connectable=True
                    ),
                    BluetoothScanningMode.ACTIVE,
                )
            )

        await self._async_opportunistic_ble_reconnect()

        return device

    async def _async_update_properties(
        self, properties: ThingPropertiesMessage
    ) -> None:
        """Update data from incoming properties messages."""
        if not self.data.enabled:
            return
        if not self.is_online():
            await self.set_scheduled_updates(True)
        if device := self.manager.get_device_by_name(self.device_name):
            self.async_set_updated_data(device)

    async def _async_update_status(self, status: ThingStatusMessage) -> None:
        """Update data from incoming status messages."""
        if not self.data.enabled:
            return
        if status.params.status.value == StatusType.CONNECTED:
            await self.set_scheduled_updates(True)
            self.hass.async_create_task(self.async_request_refresh())
        if device := self.manager.get_device_by_name(self.device_name):
            self.async_set_updated_data(device)

    async def _async_update_event_message(self, event: ThingEventMessage) -> None:
        """Update data from incoming event messages."""
        if not self.data.enabled:
            return
        if not self.is_online():
            await self.set_scheduled_updates(True)
        if device := self.manager.get_device_by_name(self.device_name):
            self.async_set_updated_data(device)

    async def _async_setup(self) -> None:
        await super()._async_setup()

        handle = self.manager.mower(self.device_name)
        open_stream = getattr(handle, "open_position_sample_stream", None)
        if callable(open_stream):
            self._position_sample_stream = open_stream(maxsize=1)
            self._position_sample_task = self.hass.async_create_task(
                self._consume_position_samples(),
                name=f"{DOMAIN}-{self.device_name}-position-pipeline",
            )

        # Common commands for all device types
        commands = [
            ("send_todev_ble_sync", {"sync_type": 3}),
            ("async_read_rain_detection", {}),
            ("async_read_sidelight", {}),
            ("async_read_turning_mode", {}),
            ("async_read_traversal_mode", {}),
        ]

        # Add device-specific commands
        if DeviceType.is_mini_or_x_series(self.device_name):
            commands.extend(
                [
                    ("async_read_manual_light", {}),
                    ("async_read_night_light", {}),
                    ("async_read_cutter_mode", {}),
                ]
            )

        if DeviceType.is_luba_pro(self.device_name):
            commands.extend(
                [
                    ("async_fetch_audio_config", {}),
                    ("async_read_wildlife_safety", {}),
                ]
            )

        # Final command for all devices
        commands.append(("async_request_report_snapshot", {}))

        # Execute all commands with unified exception handling
        for command_name, kwargs in commands:
            try:
                command_method = getattr(self, command_name, None)
                if command_method is None:
                    command_method = self.async_send_command
                    await command_method(command_name, **kwargs)
                else:
                    await command_method(**kwargs)
            except (
                DeviceOfflineException,
                NoTransportAvailableError,
                CommandTimeoutError,
                ConcurrentRequestError,
                BLEUnavailableError,
            ) as exc:
                LOGGER.debug(f"Command {command_name} failed with exception: {exc}")

        # Watch sys_status changes so we can refresh the full status when the
        # device transitions states.  Skipped when the BLE polling loop is
        # already feeding a continuous count=0 stream — the stream is fresher
        # than any count=1 poll we could fire.
        if (handle := self.manager.mower(self.device_name)) is not None:
            handle.watch_field(
                lambda s: cast(MowerDevice, s.raw).report_data.dev.sys_status,
                self._on_sys_status_changed_refresh,
            )

    async def _consume_position_samples(self) -> None:
        """Consume position evidence for presentation diagnostics only."""
        stream = self._position_sample_stream
        if stream is None:
            return
        previous_receipt: float | None = None
        while True:
            sample = await stream.queue.get()
            consumed_at = time.monotonic()
            if previous_receipt is not None:
                self._position_payload_intervals.append(
                    max(sample.received_at_monotonic - previous_receipt, 0.0)
                )
            previous_receipt = sample.received_at_monotonic
            self._latest_position_sample = sample
            self._latest_position_consumed_at = consumed_at

    def open_position_sample_stream(self, *, maxsize: int = 1) -> Any | None:
        """Open an independent safety-consumer position stream."""
        handle = self.manager.mower(self.device_name)
        open_stream = getattr(handle, "open_position_sample_stream", None)
        return open_stream(maxsize=maxsize) if callable(open_stream) else None

    def position_pipeline_diagnostics(self) -> dict[str, Any]:
        """Return non-sensitive position-pipeline timing diagnostics."""
        now = time.monotonic()
        sample = self._latest_position_sample
        if sample is None:
            return {
                "available": self._position_sample_stream is not None,
                "latest_sequence": None,
                "latest_epoch": None,
                "receipt_age_s": None,
                "pipeline_latency_s": None,
                "coordinator_latency_s": None,
                "payload_cadence_s": None,
                "presentation_stream_replacements": getattr(
                    self._position_sample_stream, "dropped_samples", 0
                ),
                "safety_stream_drops": None,
            }
        intervals = tuple(self._position_payload_intervals)
        return {
            "available": True,
            "latest_sequence": sample.sequence,
            "latest_epoch": sample.epoch,
            "source": sample.source,
            "transport": sample.transport,
            "valid_for_motion": sample.valid_for_motion,
            "rejection_reason": sample.rejection_reason,
            "receipt_age_s": max(now - sample.received_at_monotonic, 0.0),
            "pipeline_latency_s": max(
                sample.published_at_monotonic - sample.received_at_monotonic, 0.0
            ),
            "stage_latency_s": {
                "receipt_to_decode": max(
                    sample.decoded_at_monotonic - sample.received_at_monotonic,
                    0.0,
                ),
                "decode_to_broker": max(
                    sample.broker_completed_at_monotonic - sample.decoded_at_monotonic,
                    0.0,
                ),
                "broker_to_reducer": max(
                    sample.reducer_completed_at_monotonic
                    - sample.broker_completed_at_monotonic,
                    0.0,
                ),
                "reducer_to_state_apply": max(
                    sample.state_applied_at_monotonic
                    - sample.reducer_completed_at_monotonic,
                    0.0,
                ),
                "state_apply_to_publication": max(
                    sample.published_at_monotonic - sample.state_applied_at_monotonic,
                    0.0,
                ),
            },
            "coordinator_latency_s": (
                max(
                    self._latest_position_consumed_at - sample.published_at_monotonic,
                    0.0,
                )
                if self._latest_position_consumed_at is not None
                else None
            ),
            "payload_cadence_s": (
                sum(intervals) / len(intervals) if intervals else None
            ),
            "presentation_stream_replacements": getattr(
                self._position_sample_stream, "dropped_samples", 0
            ),
            # This coordinator stream is presentation-only. Safety consumers
            # own independent streams and report their drops per invocation.
            "safety_stream_drops": None,
        }

    async def _on_sys_status_changed_refresh(self, sys_status: int) -> None:
        """Trigger a one-shot count=1 poll on sys_status transitions when not streaming."""
        try:
            await self.async_request_report_snapshot()
        except DeviceOfflineException, NoTransportAvailableError:
            LOGGER.debug(
                "report-coordinator [%s]: skipping sys_status refresh — device offline / no transport",
                self.device_name,
            )


class MammotionMaintenanceUpdateCoordinator(MammotionBaseUpdateCoordinator[Maintain]):
    """Class to manage fetching mammotion data."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: MammotionConfigEntry,
        device: Device,
        mammotion: MammotionClient,
        unique_name: str | None = None,
    ) -> None:
        """Initialize global mammotion data updater."""
        super().__init__(
            hass=hass,
            config_entry=config_entry,
            device=device,
            mammotion=mammotion,
            update_interval=MAINTENANCE_INTERVAL,
            unique_name=unique_name,
        )

        mowing_device = self.manager.get_device_by_name(self.device_name)
        assert mowing_device is not None
        if self.data is None:
            self.data = mowing_device.report_data.maintenance
        self._prev_sys_status: int | None = None

    def get_coordinator_data(self, device: MowingDevice) -> Maintain:
        """Get coordinator data."""
        return device.report_data.maintenance

    async def _on_state_changed(self, snapshot: DeviceSnapshot) -> None:
        data = cast(MowerDevice, snapshot.raw)
        self.async_set_updated_data(data.report_data.maintenance)

    async def _on_sys_status_changed(self, sys_status: int) -> None:
        """Fetch maintenance data when the mower transitions from working to ready."""
        was_working = self._prev_sys_status in MOWING_ACTIVE_MODES
        self._prev_sys_status = sys_status
        if was_working and sys_status == WorkMode.MODE_READY:
            try:
                await self.async_send_command("get_maintenance")
            except DeviceOfflineException, GatewayTimeoutException:
                pass

    async def _async_update_data(self) -> Maintain:
        """Get data from the device."""
        if (data := await self._async_short_circuit_update()) is not None:
            return data

        _dev = self.manager.get_device_by_name(self.device.device_name)
        assert _dev is not None
        return _dev.report_data.maintenance

    async def _async_setup(self) -> None:
        """Set up maintenance coordinator."""
        await super()._async_setup()

        if handle := self.manager.mower(self.device_name):
            handle.watch_field(
                lambda s: cast(MowerDevice, s.raw).report_data.dev.sys_status,
                self._on_sys_status_changed,
            )

        try:
            await self.async_send_command("get_maintenance")
            await self.async_send_and_wait(
                "read_job_do_not_disturb", "todev_unable_time_set"
            )
        except DeviceOfflineException, GatewayTimeoutException:
            pass


class MammotionDeviceVersionUpdateCoordinator(
    MammotionBaseUpdateCoordinator[MowingDevice]
):
    """Class to manage fetching mammotion data."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: MammotionConfigEntry,
        device: Device,
        mammotion: MammotionClient,
        unique_name: str | None = None,
    ) -> None:
        """Initialize global mammotion data updater."""
        super().__init__(
            hass=hass,
            config_entry=config_entry,
            device=device,
            mammotion=mammotion,
            update_interval=DEFAULT_INTERVAL,
            unique_name=unique_name,
        )

        mowing_device = self.manager.get_device_by_name(self.device_name)
        if self.data is None:
            self.data = mowing_device

    def get_coordinator_data(self, device: MowingDevice) -> MowingDevice:
        """Get coordinator data."""
        return device

    async def _async_update_data(self) -> MowingDevice:
        """Get data from the device."""
        if (data := await self._async_short_circuit_update()) is not None:
            return data
        device = self.manager.get_device_by_name(self.device_name)
        assert device is not None
        handle = self.manager.mower(self.device_name)

        checks: list[tuple[str, str, bool]] = [
            (
                "get_device_version_main",
                "toapp_devinfo_resp",
                bool(device.mower_state.swversion),
            ),
            (
                "get_device_version_info",
                "toapp_dev_fw_info",
                bool(device.device_firmwares.main_controller),
            ),
            (
                "get_device_base_info",
                "toapp_devinfo_resp",
                bool(device.device_firmwares.device_version),
            ),
            (
                "get_device_product_model",
                "device_product_type_info",
                bool(device.mower_state.model_id),
            ),
        ]
        for command, expected_field, already_set in checks:
            if already_set:
                continue
            try:
                await self.async_send_and_wait(command, expected_field)
            except DeviceOfflineException:
                return device

        await self.check_firmware_version()

        if handle is not None and self.has_cloud_account:
            http = self.manager.mammotion_http
            if http is not None:
                try:
                    ota_info = await http.get_device_ota_firmware([handle.iot_id])
                except AuthError as err:
                    # has_cloud_account only means credentials are configured,
                    # not that the cloud session is currently live -- a dead
                    # refresh token must not block core (BLE) functionality
                    # over an optional version-check, same as the checks loop
                    # above treats DeviceOfflineException as non-fatal.
                    LOGGER.debug(
                        "%s: OTA firmware check skipped, cloud auth unavailable: %s",
                        self.device_name,
                        err,
                    )
                except DeviceOfflineException, GatewayTimeoutException:
                    pass
                else:
                    LOGGER.debug("OTA info: %s", ota_info.data)
                    if check_versions := ota_info.data:
                        for check_version in check_versions:
                            if check_version.device_id == handle.iot_id:
                                device.apply_version_check(check_version)

        if device.mower_state.model_id != "":
            self.update_interval = DEVICE_VERSION_INTERVAL

        return device

    async def _async_setup(self) -> None:
        """Set up device version coordinator."""
        await super()._async_setup()

        try:
            device = self.manager.get_device_by_name(self.device_name)
            if device is None:
                return

            checks: list[tuple[str, str, bool]] = [
                (
                    "get_device_version_main",
                    "toapp_devinfo_resp",
                    bool(device.mower_state.swversion),
                ),
                (
                    "get_device_version_info",
                    "toapp_dev_fw_info",
                    bool(device.device_firmwares.main_controller),
                ),
                (
                    "get_device_base_info",
                    "toapp_devinfo_resp",
                    bool(device.device_firmwares.device_version),
                ),
                (
                    "get_device_product_model",
                    "device_product_type_info",
                    bool(device.mower_state.model_id),
                ),
            ]
            for command, expected_field, already_set in checks:
                if already_set:
                    continue
                try:
                    await self.async_send_and_wait(command, expected_field)
                except DeviceOfflineException:
                    pass

            if not device.mower_state.wifi_mac:
                await self.async_send_command("get_device_network_info")

            handle = self.manager.mower(self.device_name)
            if handle is not None and self.has_cloud_account:
                http = self.manager.mammotion_http
                if http is not None:
                    try:
                        ota_info = await http.get_device_ota_firmware([handle.iot_id])
                    except AuthError as err:
                        # has_cloud_account only means credentials are
                        # configured, not that the cloud session is currently
                        # live -- a dead refresh token must not block core
                        # (BLE) setup over an optional version-check.
                        LOGGER.debug(
                            "%s: OTA firmware check skipped during setup, "
                            "cloud auth unavailable: %s",
                            self.device_name,
                            err,
                        )
                    except DeviceOfflineException, GatewayTimeoutException:
                        pass
                    else:
                        device = self.manager.get_device_by_name(self.device_name)
                        if device is not None and (check_versions := ota_info.data):
                            for check_version in check_versions:
                                if check_version.device_id == handle.iot_id:
                                    device.apply_version_check(check_version)

            self.async_set_updated_data(self.data)
        except DeviceOfflineException:
            pass


class MammotionMapUpdateCoordinator(MammotionBaseUpdateCoordinator[MowerInfo]):
    """Class to manage fetching mammotion data."""

    _dynamics_line_cancel: CALLBACK_TYPE | None = None

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: MammotionConfigEntry,
        device: Device,
        mammotion: MammotionClient,
        unique_name: str | None = None,
    ) -> None:
        """Initialize global mammotion data updater."""
        super().__init__(
            hass=hass,
            config_entry=config_entry,
            device=device,
            mammotion=mammotion,
            update_interval=MAP_INTERVAL,
            unique_name=unique_name,
        )

        mowing_device = self.manager.get_device_by_name(self.device_name)
        assert mowing_device is not None
        if self.data is None:
            self.data = mowing_device.mower_state

    def get_coordinator_data(self, device: MowingDevice) -> MowerInfo:
        """Get coordinator data."""
        return device.mower_state

    def _map_callback(self) -> None:
        """Trigger a resync when the bol hash changes."""
        # No direct bol-hash callback hook is exposed by pymammotion yet.
        # Map freshness is enforced in _async_update_data() via bol_hash checks.
        return

    async def _async_update_data(self) -> MowerInfo:
        """Get data from the device."""
        if (data := await self._async_short_circuit_update()) is not None:
            return data
        device = self.manager.get_device_by_name(self.device_name)
        assert device is not None

        try:
            # RTK/dock lat are radians with an exact-0.0 "unset" sentinel — compare to 0.0,
            # not round(lat, 0), which would treat everything within ~0.5 rad (~28°) of the
            # equator as unset and re-fetch the location on every update.
            if (
                device.location.RTK.latitude == 0.0
                or device.location.dock.latitude == 0.0
            ):
                await self.async_rtk_dock_location()

            bol_hash = (
                device.report_data.locations[0].bol_hash
                if device.report_data.locations
                else 0
            )
            if not device.map.is_map_synced(bol_hash) and self._should_start_map_sync(
                bol_hash
            ):
                self._record_map_sync_attempt(bol_hash)
                await self.manager.start_map_sync(self.device_name)

        except DeviceOfflineException as ex:
            if ex.iot_id == self.device.iot_id:
                self.device_offline(device)
                return device.mower_state
        except GatewayTimeoutException:
            pass
        except ConcurrentRequestError, NoTransportAvailableError:
            pass

        _d = self.manager.get_device_by_name(self.device_name)
        assert _d is not None
        return _d.mower_state

    def _device_supports_dynamics_line(self) -> bool:
        """Return True if this device supports the dynamics-line mow-progress stream."""
        device = self.manager.get_device_by_name(self.device_name)
        firmware = device.device_firmwares.main_controller if device else None
        return DeviceType.value_of_str(self.device_name).is_support_dynamics_line(
            firmware
        )

    def _ble_is_connected(self) -> bool:
        """Return True if BLE transport exists and is currently connected."""
        if handle := self.manager.mower(self.device_name):
            if ble := handle.get_transport(TransportType.BLE):
                return ble.is_connected
        return False

    def _stop_dynamics_line_poll(self) -> None:
        if self._dynamics_line_cancel is not None:
            self._dynamics_line_cancel()
            self._dynamics_line_cancel = None

    async def _on_sys_status_changed_dynamics(self, sys_status: int) -> None:
        """Start the dynamics-line poll when mowing over BLE; stop it otherwise."""
        if (
            sys_status in MOWING_ACTIVE_MODES
            and self._device_supports_dynamics_line()
            and self._ble_is_connected()
        ):
            if self._dynamics_line_cancel is None:
                self._dynamics_line_cancel = async_track_time_interval(
                    self.hass,
                    self._fetch_dynamics_line,
                    DYNAMICS_LINE_INTERVAL,
                )
        else:
            self._stop_dynamics_line_poll()

    async def _fetch_dynamics_line(self, _now: datetime.datetime) -> None:
        """Fetch the dynamics line; self-cancels if BLE has disconnected."""
        if not self._ble_is_connected():
            self._stop_dynamics_line_poll()
            return
        try:
            await self.manager.get_dynamics_line(self.device_name)
            device = self.manager.get_device_by_name(self.device_name)
            if device is not None:
                self.async_set_updated_data(device.mower_state)
        except (
            DeviceOfflineException,
            NoTransportAvailableError,
            GatewayTimeoutException,
        ):
            pass

    async def _async_setup(self) -> None:
        """Set up coordinator with initial call to get map data."""
        await super()._async_setup()
        device = self.manager.get_device_by_name(self.device_name)
        if device is None:
            return

        if handle := self.manager.mower(self.device_name):
            handle.watch_field(
                lambda s: cast(MowerDevice, s.raw).report_data.dev.sys_status,
                self._on_sys_status_changed_dynamics,
            )

        if not device.enabled or not device.online:
            return
        try:
            await self.async_rtk_dock_location()
        except DeviceOfflineException as ex:
            if ex.iot_id == self.device.iot_id:
                self.device_offline(device)
        except GatewayTimeoutException:
            pass
        except NoTransportAvailableError:
            LOGGER.debug(
                "No transport connected yet for %s, map data will be fetched on next update",
                self.device_name,
            )

    async def async_setup_map(self) -> None:
        """Run initial map setup without triggering a full first-refresh cycle."""
        await self._async_setup()


class MammotionDeviceErrorUpdateCoordinator(
    MammotionBaseUpdateCoordinator[MowingDevice]
):
    """Class to manage fetching mammotion data."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: MammotionConfigEntry,
        device: Device,
        mammotion: MammotionClient,
        unique_name: str | None = None,
    ) -> None:
        """Initialize global mammotion data updater."""
        super().__init__(
            hass=hass,
            config_entry=config_entry,
            device=device,
            mammotion=mammotion,
            update_interval=DEFAULT_INTERVAL,
            unique_name=unique_name,
        )
        mowing_device = self.manager.get_device_by_name(self.device_name)
        assert mowing_device is not None
        if self.data is None:
            self.data = mowing_device

    def get_coordinator_data(self, device: MowingDevice) -> MowingDevice:
        """Get coordinator data."""
        return device

    async def _async_update_event_message(self, event: ThingEventMessage) -> None:
        if (
            hasattr(event.params, "identifier")
            and event.params.identifier == "device_warning_code_event"
        ):
            event_params: DeviceNotificationEventParams = cast(
                DeviceNotificationEventParams, event.params
            )
            # '[{"c":-2801,"ct":1,"ft":1731493734000},{"c":-1008,"ct":1,"ft":1731493734000}]'
            try:
                warning_event = json.loads(event_params.value.data)
                LOGGER.debug("warning event %s", warning_event)
                await self._async_update_data()
                if device := self.manager.get_device_by_name(self.device_name):
                    self.async_set_updated_data(device)
            except json.JSONDecodeError:
                """Failed to parse warning event."""

    def get_error_code(self, number: int) -> int:
        """Get error code from an error code list."""
        try:
            return int(abs(next(iter(self.data.errors.err_code_list))))
        except StopIteration:
            return 0

    def get_error_time(self, number: int) -> datetime.datetime | None:
        """Get error time from an error code list."""
        try:
            return datetime.datetime.fromtimestamp(
                next(iter(self.data.errors.err_code_list_time)), datetime.UTC
            )
        except StopIteration:
            return None

    def get_error_message(self, number: int) -> str:
        """Return error message."""
        try:
            error_code: int = next(iter(self.data.errors.err_code_list))

            error_code = abs(error_code)
            error_info: ErrorInfo = self.data.errors.error_codes[f"{error_code}"]

            implication = (
                getattr(error_info, f"{self.hass.config.language}_implication")
                if hasattr(error_info, f"{self.hass.config.language}_implication")
                else error_info.en_implication
            )
            solution = (
                getattr(error_info, f"{self.hass.config.language}_solution")
                if hasattr(error_info, f"{self.hass.config.language}_solution")
                else error_info.en_solution
            )

            if implication == "":
                implication = error_info.en_implication

            if solution == "":
                solution = error_info.en_solution

            return f"{error_info.module}: {implication}, {solution}"

        except StopIteration:
            """Failed to get error code."""
            return "No Error"
        except KeyError:
            """Failed to get error message."""
            return "Error message not found"

    async def _async_update_data(self) -> MowingDevice:
        """Get data from the device."""
        if (data := await self._async_short_circuit_update()) is not None:
            return data
        device = self.manager.get_device_by_name(self.device_name)
        assert device is not None
        try:
            if not device.errors.error_codes and self.has_cloud_account:
                http = self.manager.mammotion_http
                if http is not None:
                    device.errors.error_codes = await http.get_all_error_codes()
        except DeviceOfflineException:
            return device

        return device

    async def _on_sys_status_changed(self, sys_status: WorkMode) -> None:
        """Handle sys status changed."""
        if sys_status in (
            WorkMode.MODE_WORKING,
            WorkMode.MODE_RETURNING,
            WorkMode.MODE_LOCK,
            WorkMode.MODE_PAUSE,
        ):
            await self.async_send_and_wait(
                "read_write_device", "bidire_comm_cmd", rw_id=5, rw=1, context=2
            )
            await self.async_send_and_wait(
                "read_write_device", "bidire_comm_cmd", rw_id=5, rw=1, context=3
            )

    async def _async_setup(self) -> None:
        """Set up the device-version coordinator."""
        await super()._async_setup()
        device = self.manager.get_device_by_name(self.device_name)
        if device is None:
            return
        if handle := self.manager.mower(self.device_name):

            def _extract_sys_status(snapshot: DeviceSnapshot) -> WorkMode:
                return cast(
                    WorkMode, cast(MowerDevice, snapshot.raw).report_data.dev.sys_status
                )

            handle.watch_field(
                _extract_sys_status,
                self._on_sys_status_changed,
            )

        try:
            await self.async_send_and_wait(
                "read_write_device", "bidire_comm_cmd", rw_id=5, rw=1, context=2
            )
            await self.async_send_and_wait(
                "read_write_device", "bidire_comm_cmd", rw_id=5, rw=1, context=3
            )
            if not device.errors.error_codes and self.has_cloud_account:
                http = self.manager.mammotion_http
                if http is not None:
                    device.errors.error_codes = await http.get_all_error_codes()

            self.async_set_updated_data(self.data)
        except DeviceOfflineException:
            pass


class MammotionRTKCoordinator(MammotionBaseUpdateCoordinator[RTKBaseStationDevice]):
    """Mammotion DataUpdateCoordinator for RTK base station devices."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: MammotionConfigEntry,
        device: Device,
        mammotion: MammotionClient,
        unique_name: str | None = None,
    ) -> None:
        """Initialize rtk mammotion data updater."""
        super().__init__(
            hass=hass,
            config_entry=config_entry,
            device=device,
            mammotion=mammotion,
            update_interval=RTK_INTERVAL,
            unique_name=unique_name,
        )

    def get_coordinator_data(
        self, device: RTKBaseStationDevice
    ) -> RTKBaseStationDevice:
        """Return the current RTK device state tracked by this coordinator."""
        return self.data

    async def async_restore_data(self) -> None:
        """Restore saved data."""
        store = MammotionConfigStore(
            self.hass, version=1, minor_version=2, key=self.device_name
        )
        restored_data: Mapping[str, Any] | None = await store.async_load()

        handle = self.manager.rtk_device(self.device_name)

        if restored_data is None:
            empty = RTKBaseStationDevice()
            self.data = empty
            if handle is not None:
                handle.restore_device(empty)
            return

        try:
            if restored_data is not None:
                rtk_state = RTKBaseStationDevice().from_dict(restored_data)
                if handle is not None:
                    handle.restore_device(rtk_state)
                    self.data = rtk_state
        except InvalidFieldValue:
            empty = RTKBaseStationDevice()
            self.data = empty
            if handle is not None:
                handle.restore_device(empty)

    async def _async_update_data(self) -> RTKBaseStationDevice:
        """Return current RTK state from the device handle's state machine.

        The state machine is kept up to date automatically by:
        - LubaMsg protobuf frames → RTKStateReducer.apply()
        - thing/properties JSON pushes → RTKStateReducer.apply_properties()
        - thing/status pushes → DeviceHandle.on_status_message()

        The only remaining polling work is the OTA firmware check, which is
        not pushed via MQTT and must be fetched from the HTTP API.
        """
        handle = self.manager.rtk_device(self.device_name)
        if handle is None:
            return self.data

        await self.async_send_command("send_todev_ble_sync", sync_type=3)
        await self.async_send_and_wait("basestation_info", "to_app")

        if self.has_cloud_account:
            http = self.manager.mammotion_http
            if http is not None:
                try:
                    ota_info = await http.get_device_ota_firmware(
                        [cast(Any, self.device).iot_id]
                    )
                    if check_versions := ota_info.data:
                        for check_version in check_versions:
                            if check_version.device_id == cast(Any, self.device).iot_id:
                                self.data.apply_version_check(check_version)
                except ReLoginRequiredError as err:
                    raise ConfigEntryAuthFailed(
                        f"Re-authentication required for Mammotion account: {err}"
                    ) from err
                except DeviceOfflineException, GatewayTimeoutException:
                    pass

        return self.data

    async def async_shutdown(self) -> None:
        """Cancel all RAII subscriptions and delegate to HA coordinator shutdown."""
        for sub in self._subscriptions:
            sub.cancel()
        self._subscriptions.clear()
        await super().async_shutdown()

    async def _async_setup(self) -> None:
        """Set up RTK device subscriptions and fetch one-time HTTP data."""
        await super()._async_setup()
        if handle := self.manager.rtk_device(self.device_name):
            updated = cast(Any, handle.snapshot.raw)
            updated.product_key = cast(Any, self.device).product_key
            updated.iot_id = cast(Any, self.device).iot_id
            updated.name = self.device.device_name
            snapshot, _ = handle.state_machine.apply(updated, handle.availability)

        if self.data.lat != 0:
            return

        if self.has_cloud_account:
            # Fetch lora version — only available via HTTP, not MQTT/protobuf.
            await self.manager.fetch_rtk_lora_info(self.device_name)

            if (
                gateway := self.manager.cloud_gateway
            ) and DeviceType.is_aliyun_product_key(self.data.product_key):
                await self.manager.fetch_rtk_properties(self.device_name)
                await gateway.get_device_status(cast(Any, self.device).iot_id)
        await self.async_send_command("send_todev_ble_sync", sync_type=3)
        await self.async_request_report_snapshot()
        await self.async_send_and_wait("basestation_info", "to_app")
        await self.async_send_and_wait(
            "get_device_network_info", "toapp_networkinfo_rsp"
        )
        self.data.online = True

    async def update_firmware(self, version: str) -> None:
        """Update firmware."""
        http = self.manager.mammotion_http
        if http is not None:
            await http.start_ota_upgrade(cast(Any, self.device).iot_id, version)


class MammotionSpinoCoordinator(MammotionBaseUpdateCoordinator[PoolCleanerDevice]):
    """Mammotion DataUpdateCoordinator for Spino pool cleaner devices."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: MammotionConfigEntry,
        device: Device,
        mammotion: MammotionClient,
        unique_name: str | None = None,
    ) -> None:
        """Initialize spino mammotion data updater."""
        super().__init__(
            hass=hass,
            config_entry=config_entry,
            device=device,
            mammotion=mammotion,
            update_interval=SPINO_INTERVAL,
            unique_name=unique_name,
        )

    async def _async_setup(self) -> None:
        """Subscribe to device events, then read the initial toggle states once.

        The buzzer/turbo/platform/waterline toggles aren't part of the regular
        push — the device only emits a ``bidire_comm_cmd`` for them in response
        to a read or write.  Issue one read per toggle here to seed initial
        state; subsequent changes (from our writes or the Mammotion app) arrive
        as ``bidire_comm_cmd`` responses applied by ``PoolStateReducer`` and
        pushed to entities via the inherited ``_on_state_changed`` callback.
        """
        await super()._async_setup()
        # Start the status report stream so the device pushes dev_statue_t
        # (sys_status / work_mode / battery) — the pool cleaner doesn't report
        # unsolicited otherwise. See get_report_cfg_spino / async_subscribe_status.

        try:
            with contextlib.suppress(
                GatewayTimeoutException,
                NoTransportAvailableError,
                HomeAssistantError,
            ):
                await self.async_subscribe_status()
            for toggle in SpinoToggle:
                with contextlib.suppress(
                    GatewayTimeoutException,
                    NoTransportAvailableError,
                ):
                    await self.async_send_and_wait(
                        "read_write_device",
                        "bidire_comm_cmd",
                        rw_id=int(toggle),
                        context=0,
                        rw=0,
                    )
            # Fetch pool geometry once at startup.  Responses arrive as unsolicited
            # APP_DOWNLINK_CMD frames and are reassembled by PoolStateReducer.
            for fetch in (self.async_fetch_pool_map, self.async_fetch_pool_line):
                with contextlib.suppress(
                    GatewayTimeoutException,
                    NoTransportAvailableError,
                ):
                    await fetch()
        except DeviceOfflineException:
            cast(Any, self.device).online = False

    def get_coordinator_data(self, device: PoolCleanerDevice) -> PoolCleanerDevice:
        """Return the current pool cleaner state tracked by this coordinator."""
        return self.data

    async def async_restore_data(self) -> None:
        """Restore saved data."""
        store = MammotionConfigStore(
            self.hass, version=1, minor_version=2, key=self.device_name
        )
        restored_data: Mapping[str, Any] | None = await store.async_load()

        handle = self.manager.mower(self.device_name)

        if restored_data is None:
            empty = PoolCleanerDevice()
            self.data = empty
            if handle is not None:
                handle.restore_device(empty)
            return

        try:
            spino_state = PoolCleanerDevice().from_dict(restored_data)
            if handle is not None:
                handle.restore_device(spino_state)
                self.data = spino_state
        except InvalidFieldValue:
            empty = PoolCleanerDevice()
            self.data = empty
            if handle is not None:
                handle.restore_device(empty)

    def get_error_code(self) -> int:
        """Return the absolute error code of the most recent fault, or 0."""
        try:
            return int(abs(self.data.pool_state.error_log[0].code))
        except IndexError:
            return 0

    def get_error_time(self) -> datetime.datetime | None:
        """Return the timestamp of the most recent fault as a UTC datetime, or None."""
        try:
            return datetime.datetime.fromtimestamp(
                self.data.pool_state.error_log[0].timestamp, datetime.UTC
            )
        except IndexError:
            return None

    def get_error_message(self) -> str:
        """Return a human-readable description of the most recent fault."""
        try:
            error_code = abs(self.data.pool_state.error_log[0].code)
            error_info: ErrorInfo = self.data.errors.error_codes[f"{error_code}"]
            implication = (
                getattr(error_info, f"{self.hass.config.language}_implication")
                if hasattr(error_info, f"{self.hass.config.language}_implication")
                else error_info.en_implication
            )
            solution = (
                getattr(error_info, f"{self.hass.config.language}_solution")
                if hasattr(error_info, f"{self.hass.config.language}_solution")
                else error_info.en_solution
            )
            if implication == "":
                implication = error_info.en_implication
            if solution == "":
                solution = error_info.en_solution
            return f"{error_info.module}: {implication}, {solution}"
        except IndexError:
            return "No Error"
        except KeyError:
            return "Error message not found"

    async def _async_update_data(self) -> PoolCleanerDevice:
        """Return current pool cleaner state from the device handle.

        Runtime state (sys_status, work_mode, battery, settings, map) is pushed
        into the state machine by ``PoolStateReducer`` as MQTT frames arrive, so
        the only polling work here is the HTTP OTA firmware check, which is not
        pushed over MQTT.
        """
        handle = self.manager.mower(self.device_name)
        if handle is None:
            return self.data

        if self.has_cloud_account:
            http = self.manager.mammotion_http
            if http is not None:
                try:
                    ota_info = await http.get_device_ota_firmware(
                        [cast(Any, self.device).iot_id]
                    )
                    if check_versions := ota_info.data:
                        for check_version in check_versions:
                            if check_version.device_id == cast(Any, self.device).iot_id:
                                self.data.apply_version_check(check_version)
                    if not self.data.errors.error_codes:
                        self.data.errors.error_codes = await http.get_all_error_codes()
                except ReLoginRequiredError as err:
                    raise ConfigEntryAuthFailed(
                        f"Re-authentication required for Mammotion account: {err}"
                    ) from err
                except DeviceOfflineException, GatewayTimeoutException:
                    pass

        await self.async_save_data(self.data)

        return self.data

    async def update_firmware(self, version: str) -> None:
        """Update firmware."""
        http = self.manager.mammotion_http
        if http is not None:
            await http.start_ota_upgrade(cast(Any, self.device).iot_id, version)

    # === Pool cleaner control helpers (called by control entities) ===

    async def async_subscribe_status(self) -> None:
        """Start the Spino status report stream (called once at setup).

        Subscribes to RIT_CONNECT + RIT_DEV_STA with count=0 (continuous) so the
        device pushes dev_statue_t frames; PoolStateReducer applies them.
        """
        await self.async_send_command("get_report_cfg_spino", count=1)

    async def async_request_status(self) -> None:
        """One-shot Spino status poll, backing the refresh-status button."""
        await self.async_send_command("get_report_cfg_spino", count=1)

    async def async_set_work_mode(self, work_mode: int) -> None:
        """Set the Spino cleaning work mode."""
        await self.async_send_command("set_swimming_work_mode", work_mode=work_mode)

    async def async_set_wall_material(self, material: int) -> None:
        """Set the pool wall material."""
        await self.async_send_command("sp_environment_update", material=material)

    async def async_set_bottom_type(self, bottom_type: int) -> None:
        """Set the pool bottom shape type."""
        await self.async_send_command("sp_set_bottom_type", bottom_type=bottom_type)

    async def async_set_floor_speed(self, speed: float) -> None:
        """Set the pool floor cleaning speed."""
        await self.async_send_command("sp_speed_update", speed=speed)

    async def async_fetch_pool_map(self) -> None:
        """Request the pool boundary map from the device."""
        await self.async_send_command("get_sp_map")

    async def async_fetch_pool_line(self) -> None:
        """Request the pool cleaning route from the device."""
        await self.async_send_command("get_sp_line")

    # ------------------------------------------------------------------
    # Spino task CRUD — backed by spino_ctrl.PlanJobSet on the wire.
    # See ``docs/tasks_and_schedules.md`` § 2.  All ``enabled`` arguments
    # are in NATURAL orientation; the builder inverts at the boundary.
    # ------------------------------------------------------------------

    def _lookup_spino_plan(self, jobid: int) -> PoolPlan:
        """Return the stored Spino PoolPlan keyed by ``jobid`` or raise."""
        plan = self.data.plans.get(jobid)
        if plan is None:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="task_not_found",
                translation_placeholders={"plan_id": str(jobid)},
            )
        return plan

    @staticmethod
    def _new_spino_jobid() -> int:
        """Generate a fresh 64-bit non-zero ``jobid`` for a new Spino plan.

        ``secrets.randbits(63)`` keeps the high bit clear (fits in a signed
        uint64-as-int comfortably); ``| 1`` avoids the all-zero corner.
        """
        return secrets.randbits(63) | 1

    async def async_create_spino_task(self, plan: PoolPlan) -> None:
        """Create a brand-new Spino schedule with a freshly generated jobid."""
        plan_with_id = dataclasses.replace(plan, jobid=self._new_spino_jobid())
        await self.async_send_command("create_spino_plan", plan=plan_with_id)

    async def async_edit_spino_task(self, plan: PoolPlan) -> None:
        """Edit an existing Spino schedule (``cmd = EDIT = 4``)."""
        await self.async_send_command("edit_spino_plan", plan=plan)

    async def async_rename_spino_task(self, jobid: int, new_name: str) -> None:
        """Rename the Spino schedule identified by ``jobid`` to ``new_name``."""
        plan = self._lookup_spino_plan(jobid)
        await self.async_send_command("rename_spino_plan", plan=plan, new_name=new_name)

    async def async_set_spino_task_enabled(self, jobid: int, enabled: bool) -> None:
        """Flip the enabled flag on an existing Spino schedule.

        Round-trips the stored plan; the wire inversion (``enable = 0 if
        enabled else 1``) happens in the pymammotion builder.
        """
        plan = self._lookup_spino_plan(jobid)
        await self.async_send_command("enable_spino_plan", plan=plan, enabled=enabled)

    async def async_delete_spino_task(self, jobid: int) -> None:
        """Delete the Spino schedule identified by ``jobid``."""
        await self.async_send_command("delete_spino_plan", jobid=jobid)

    async def async_copy_spino_task(
        self, jobid: int, new_name: str | None = None
    ) -> None:
        """Duplicate the Spino schedule under a new jobid + auto-generated name."""
        plan = self._lookup_spino_plan(jobid)
        existing_names = {p.jobname for p in self.data.plans.values()}
        resolved_name = new_name or make_copy_name(existing_names)
        await self.async_send_command(
            "copy_spino_plan",
            plan=plan,
            new_name=resolved_name,
            new_jobid=self._new_spino_jobid(),
        )

    async def async_refresh_spino_tasks(self) -> None:
        """Re-fetch every Spino schedule via :class:`SpinoPlanFetchSaga`.

        Used after ``delete_all`` (no per-plan echo) and on user request via
        the schedule-refresh service.
        """
        await self.manager.start_spino_plan_sync(self.device_name)

    async def async_set_pool_toggle(self, toggle: SpinoToggle, enabled: bool) -> None:
        """Write a Spino on/off toggle (buzzer / turbo / platform / waterline).

        Uses the generic ``read_write_device`` (``allpowerfullRW``) command: the
        toggle id with ``context`` 0/1 and ``rw=1`` (write).
        """
        await self.async_send_command(
            "read_write_device", rw_id=int(toggle), context=int(enabled), rw=1
        )
