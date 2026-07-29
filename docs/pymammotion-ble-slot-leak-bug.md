# pymammotion: BLE connections are abandoned without `disconnect()`, leaking proxy slots

**Affects:** `pymammotion` **0.8.12 / `main`** (verified against the tag — `ble.py`
at `v0.8.12` is byte-identical to `main`), file `pymammotion/transport/ble.py`.
Originally found on `0.8.8`; this integration now pins `0.8.12`.
**Found:** 2026-07-28, on a Luba `Luba-VSPLV397` via ESPHome BLE proxies.

> **Scope note.** Three sites were identified. **One of them (the `start_notify`
> path) is already fixed on `main`** by `5e18185a` (PR #176, 2026-07-08), which
> shipped in 0.8.12 — see "Prior art" below. It is kept here only because the
> integration was pinned to 0.8.8 when the failure was captured. **The
> other two are unfixed as of 0.8.12/`main`** and are what this report is asking
> for.
**Impact:** these paths can strand a proxy connection slot whenever the client
remains live from the proxy's perspective after the Python reference is dropped.
Repeated failures can exhaust the proxy's three slots. When no transport can
dispatch, the device queue may then accumulate commands—including a motion
command and its later stop—and replay them after recovery on a timeline the
caller no longer controls.

This is written up for filing upstream. Nothing in this repository can fix it —
the dependency is a pinned PyPI release, not a fork.

## Symptom

Counted over one hour of logs filtered to the mower's MAC:

| event | count |
|---|---|
| `Connection open` | **6** |
| `Disconnecting` | **0** |
| `out of connection slots / device unreachable` | **6** |

Six connections opened, none ever closed, six later attempts refused. An ESPHome
proxy has three BLE slots. Once they are gone:

```
20:56:48  out of connection slots — cooling down for 120s (is_usable now False; sends use MQTT)
21:01:16  out of connection slots — cooling down for 120s
21:03:57  BLETransport connecting
21:06:20  <- motion command issued; ONE send, then silence
21:06:40  BLETransport connecting          (reconnect succeeds)
21:06:41-43  21-send BURST                 (queue flushes)
21:07:16  mower has moved 1.0778 m         (~55 s after the command)
21:11:48  out of connection slots — cooling down for 120s
```

The 5 s `todev_ble_sync(2)` keepalive queues up along with everything else, which
is why the transport *looks* idle from above. It is **blocked, not dead**.

## Root cause

An ESPHome proxy frees a BLE slot only when it receives an explicit
`BluetoothDeviceRequest`/disconnect — which `bleak_esphome` sends from
`ESPHomeClient.disconnect()`. Dropping the Python reference to the client sends
nothing. The ESP32 keeps the connection until the client asks it to close, the
ESPHome API session drops, or the device reboots.

`BLETransport` abandons live clients without disconnecting them in three places.

### Site A — `_write_payload` nulls a possibly-live client (the main leak)

```python
# pymammotion/transport/ble.py:427-436
try:
    await self._message.post_custom_data_bytes(payload)
except (TimeoutError, BleakError, OSError) as exc:
    # Clear client refs immediately so is_connected returns False
    # before _on_disconnect_async runs — prevents the ble_loop from
    # retrying against a known-dead connection (GATT error 133 etc.).
    self._client = None          # <-- live client abandoned, never disconnected
    self._message = None
    await self._notify_availability(TransportAvailability.DISCONNECTED)
    raise TransportError(...) from exc
```

The comment's premise — that the connection is "known-dead" — does not hold for
the most common member of that except-tuple. A **write timeout does not tear down
the GATT link**; the proxy still holds it. The reference is dropped anyway.

This is unrecoverable by design, because nulling `_client` also disables the only
cleanup path. `DeviceHandle.disconnect_transport` is gated on `is_connected`:

```python
# pymammotion/device/handle.py:1548-1552
async def disconnect_transport(self, transport_type: TransportType) -> None:
    t = self._transports.get(transport_type)
    if t is not None and t.is_connected:      # <-- False, _client is None
        await t.disconnect()
```

and `BLETransport.is_connected` is `self._client is not None and ...`. So after
this handler runs, the transport reports "not connected", every caller believes
there is nothing to clean up, and the client object is unreachable — while the
proxy still counts the slot as in use. The next send calls `connect()` and takes
a **fresh** slot. Six sends that fail this way exhaust a three-slot proxy twice
over, which is exactly the 6-open / 0-disconnect count above.

### Site B — `connect()` leaks the slot when `start_notify` fails — ✅ FIXED in 0.8.12

**Already fixed on `main` / 0.8.12 by `5e18185a`.** Described here because 0.8.8
still carries it, and because the shipped fix's own rationale independently
confirms the mechanism this report is about. Skip to Site C for what is still
open.

```python
# pymammotion/transport/ble.py:277-313
self._client = await establish_connection(...)   # slot consumed here
self._message = BleMessage(self._client)
...
try:
    await self._client.start_notify(UUID_NOTIFICATION_CHARACTERISTIC, self._notification_handler)
except BleakError as exc:
    if "Notify acquired" in str(exc):
        ...
    else:
        await self._notify_availability(TransportAvailability.DISCONNECTED)
        self._record_connect_failure()
        raise BLEUnavailableError(...) from exc   # <-- no disconnect, no client reset
```

The connection already succeeded, so the slot is spent. On a `start_notify`
failure the transport announces DISCONNECTED, arms the 120 s cooldown, and
raises — but never calls `self._client.disconnect()` and never clears
`_client`/`_message`.

The fix applied upstream is exactly the teardown this report asks for at the
other two sites, and its comment states the same mechanism:

```python
# 5e18185a, in the start_notify failure branch
# Tear the link down: is_connected reads the live client, so leaving
# it connected here would wedge the transport into a state where
# writes succeed but responses never arrive (no notify subscription).
with contextlib.suppress(Exception):
    await self._client.disconnect()
self._client = None
self._message = None
```

That leaves a **half-open transport**, which is worse than a plain leak:
`is_connected` is `True` (the client really is connected) while `availability` is
`DISCONNECTED`. `_write_payload` will happily write over it, because its check is
`if self._client is None or not self._client.is_connected`. Commands go out on a
link that has no notify subscription, so nothing comes back.

### Site C — a non-`BleakError` from `start_notify` escapes `connect()` entirely

Both handlers in `connect()` catch only `BleakError`. Under `bleak_esphome` the
notify path can surface `aioesphomeapi` errors, and those are not in that
hierarchy:

```
>>> from aioesphomeapi.core import TimeoutAPIError
>>> TimeoutAPIError.__mro__
(TimeoutAPIError, APIConnectionError, Exception, BaseException, object)
>>> issubclass(TimeoutAPIError, bleak.exc.BleakError)
False
```

`TimeoutAPIError` is not a `BleakError` and not a builtin `TimeoutError`, so it
propagates straight out of `connect()`. The result is the worst of the three: the
slot is spent, `_client` is left set and live, no cooldown is armed, no
availability change is published, and the raw `aioesphomeapi` exception is handed
to whatever called `connect()`.

This is not hypothetical here — it is the failure recorded on 2026-07-25, where a
`start_notify` timeout escaped Home Assistant's `async_setup_entry` as a raw
`TimeoutAPIError` and left the integration in `setup_error` with no retry.

### Also: `disconnect()` cannot clean up a stranded slot

```python
# pymammotion/transport/ble.py:398-407
async def disconnect(self) -> None:
    if self._client is not None and self._client.is_connected:
        ...
```

Guarding the teardown on the client's own `is_connected` means that whenever
bleak believes the link is gone but the proxy has not been told, `disconnect()`
is a no-op. Teardown should be attempted on any non-`None` client and the result
ignored.

## Why this is severe, not cosmetic

For a passive integration a leaked slot is an availability bug. For anything that
sends **bounded** motion commands it is a safety bug.

The pattern this project relies on is: send a movement pulse, then send a
mandatory explicit stop that bounds it. When the queue is gated, the movement
commands *and the stop* accumulate and flush together, ~20–55 s later. The mower
then executes the whole batch on its own schedule — **after** the caller has
observed the command window close and concluded the mower never moved.

Measured on 2026-07-28: a command issued at 21:06:20 produced one send and then
silence. The caller sampled RTK position through the window, saw < 3 mm of
movement, and correctly reported "no actuation". The queue flushed at
21:06:41–43. The mower drove **1.0778 m** by 21:07:16, unattended.

The failure is also invisible from every status field an integration would
naturally check: `active_transport` keeps reporting `ble`, no GATT disconnect is
logged, RSSI reads a healthy −64 dBm, and `command_result.ok` is `True` (the
motion command uses `needAck=false`, so `ok` only means "handed to the queue").

## Prior art in this repository

This is not the first time the disconnect path has come up. Three pieces of
history are directly relevant.

**1. Routine BLE disconnection was deliberately removed** in `bf92c389`
(2026-05-04), whose message reads:

> centralize the report cfg reporting, **don't bother disconnecting from ble
> anymore, too many issues**

That commit deleted the entire idle-disconnect mechanism from `BLETransport` —
`_DISCONNECT_DELAY = 10`, `_disconnect_on_idle`, `_idle_disconnect_timer`,
`set_disconnect_strategy()`, `_reset_idle_disconnect_timer()`,
`_cancel_idle_disconnect_timer()` — along with the `disconnect_on_idle`
parameter on `MammotionClient.add_ble_to_device`.

The decision to hold the link open rather than cycle it is defensible on its own
(reconnects are expensive and were causing problems). But it means that from
2026-05-04 onward, **nothing in normal operation ever calls `disconnect()`**.
The only remaining callers are `DeviceHandle.disconnect_transport` — which is
gated on `is_connected`, and therefore cannot fire at Site A — and explicit
teardown in `client.py`. Every disconnect is now an *exception* path, which is
precisely where the missing teardown lives. Removing idle-disconnect did not
create the bug, but it removed the mechanism that had been masking it.

**2. The symptom has already been reported and closed as a hardware problem.**
Mammotion-HA issue **#810, "BLE Proxy issues"** (2026-06-29, closed) quotes the
identical log line from a different user on a different mower:

```
[pymammotion.transport.ble] BLETransport[Luba]: out of connection slots /
device unreachable — cooling down for 120s (is_usable now False; sends use MQTT)
```

The thread concluded that the reporter's Shelly proxies are passive-scan only and
cannot make GATT connections, so an ESPHome proxy was needed. That is a correct
explanation *for that reporter* — but it closed the issue without examining why
slots ran out, and the diagnosis does not transfer. **This deployment uses ESPHome
active proxies with three free slots and exhausts them anyway.** The log line is
therefore not a reliable indicator of proxy misconfiguration; it is also what a
slot leak looks like.

**3. Related, still open:** #797 (BLE never reconnects when the robot returns in
range; `set_ble_device()` never re-registered) and #672 (BLE never reconnects
after an account switch) are both "the BLE transport gets stuck and only a reload
recovers it" reports. They are not obviously the same defect and are not claimed
here as such.

Two open issues are *consistent* with slot exhaustion but carry no slot evidence,
so they are listed as suggestive only, not as corroboration: **#681** ("move
forward/left/right don't work... works for some minutes, then stops", ESPHome
proxy 3 m away) and **#778** ("commands often not working; reload the integration
and it works once or twice, then stops"). The "works briefly after a reload, then
stops" shape is what a leak that resets on reload would produce, but neither
thread contains the connection counts needed to confirm it.

## Suggested fix

**Site A** — attempt teardown before dropping the reference, and do it
unconditionally rather than trusting `is_connected`:

```python
except (TimeoutError, BleakError, OSError) as exc:
    client, self._client, self._message = self._client, None, None
    if client is not None:
        with contextlib.suppress(Exception):
            await client.disconnect()
    await self._notify_availability(TransportAvailability.DISCONNECTED)
    raise TransportError(...) from exc
```

**Site C (and Site B on 0.8.x)** — `5e18185a` fixed the `start_notify` branch it
could see, but the surrounding region is still not failure-atomic: anything that
is not a `BleakError` escapes with the slot spent. Wrap the whole
post-`establish_connection` region instead, so *any* exception releases the slot
it just took:

```python
try:
    self._message = BleMessage(self._client)
    with contextlib.suppress(Exception):
        await self._client.stop_notify(UUID_NOTIFICATION_CHARACTERISTIC)
    try:
        await self._client.start_notify(UUID_NOTIFICATION_CHARACTERISTIC, self._notification_handler)
    except BleakError as exc:
        if "Notify acquired" not in str(exc):
            raise
        _logger.debug("BLETransport: notify already acquired for %s — reusing", self._config.device_id)
except BaseException:
    client, self._client, self._message = self._client, None, None
    if client is not None:
        with contextlib.suppress(Exception):
            await client.disconnect()
    await self._notify_availability(TransportAvailability.DISCONNECTED)
    self._record_connect_failure()
    raise
```

Catching `BaseException` here is deliberate: a `CancelledError` during
`start_notify` strands a slot just as effectively as a timeout, and the handler
re-raises rather than swallowing.

**`disconnect()`** — drop the `is_connected` precondition:

```python
async def disconnect(self) -> None:
    client, self._client, self._message = self._client, None, None
    if client is not None:
        with contextlib.suppress(Exception):
            await client.disconnect()
    await self._notify_availability(TransportAvailability.DISCONNECTED)
```

## Suggested follow-up (optional, larger)

Two things would have made this diagnosable in minutes instead of weeks:

1. **A connect/disconnect balance counter on the transport.** The bug is a
   conservation-law violation — opens minus closes should trend to zero, and it
   never did. Nothing in the library exposed that; it had to be counted out of
   proxy logs.

2. **A public completion-aware BLE dispatch API.** `is_usable`
   answers a *routing-eligibility* question (a `BLEDevice` is cached, RSSI is
   acceptable, no cooldown is armed) and is easy to mistake for liveness.
   `last_send_monotonic` is also insufficient because BLE stamps it before the
   awaited GATT write. Consumers need a supported way to enqueue a command and
   await that specific dispatch—or cancel it before it can execute late.

Until then, consumers cannot distinguish "the mower went quiet" from "our own
commands are stuck in a queue" — and those call for opposite responses.
