# pymammotion: a lost BLE fragment poisons the next report frame

**Affects:** `pymammotion==0.8.8` (the version pinned when captured), file
`pymammotion/bluetooth/ble_message.py`.
**Found:** 2026-07-25, on a Luba `Luba-VSPLV397` via ESPHome BLE proxies.
**Impact:** one lost BLE packet mid-message silently corrupts and destroys **at least
two** device reports, and the corruption is delivered to the parser as a _complete_
frame rather than being detected at the transport layer.

**Status update 2026-07-31:** Mikey hand-applied the three reassembly-buffer
resets to upstream `main` as `68e0095`; PR #181 is superseded. It is not in an
official release. This integration pins the Chorty `0.8.12.post1` wheel carrying
that commit plus the teardown fix, and verifies both capabilities at runtime.
The analysis below is retained as the evidence for the fix.

## Symptom

`DeviceHandle` drops frames whose fields are the wrong _type_, with the offending value
being readable ASCII from an entirely different message:

```
ERROR [pymammotion.device.handle] ← Luba-VSPLV397  dropping frame:
  malformed report data failed deserialization (249 bytes):
  Field "pos_type" of type int in LocationData has invalid value
  [76, 117, 98, 97, 45, 86, 83, 80, 76, 86, 51, 57, 55]
```

```
Field "progress" of type int in WorkData has invalid value
  [97, 49, 76, 76, 109, 121, 49, 122, 99, 48, 106]
```

Decoding those byte runs as protobuf rather than as noise is what identifies the bug:

```
0x1a 0x0b "a1LLmy1zc0j"     field 3, len 11  -> the Aliyun product key
0x22 0x0d "Luba-VSPLV397"   field 4, len 13  -> the device name
```

Both are **well-formed fields from a device-identity message**, spliced into the middle
of a report. This is not corruption on the air — the bytes are intact and meaningful.
Two different messages were concatenated and handed over as one.

## Root cause

`BleMessage.parseNotification` accumulates fragments into `self.notification`
(a `BlufiNotifyData` wrapping an appending `BytesIO`), and **the only place that buffer
is ever cleared** is `BLETransport._notification_handler`, on the success path:

```python
# pymammotion/transport/ble.py
result = self._message.parseNotification(bytes(data))
if result != 0:
    # result == 1  → fragment received, waiting for more
    # result == 2  → duplicate sequence, already processed
    # result < 0   → parse error
    return                                   # <-- buffer left intact

payload = await self._message.parseBlufiNotifyData(return_bytes=True)
self._message.clear_notification()           # <-- only reset, only on success
```

So every non-zero return leaves a partial message sitting in the buffer:

| return | meaning                         | buffer state               |
| ------ | ------------------------------- | -------------------------- |
| `1`    | more fragments expected         | partial retained (correct) |
| `-4`   | checksum mismatch               | **partial retained (bug)** |
| `-100` | exception while parsing         | **partial retained (bug)** |
| (gap)  | a fragment never arrived at all | **partial retained (bug)** |

When the missing fragment simply never arrives, nothing ever resets the buffer. The
_next_ message's fragments are appended to the stale partial, and as soon as one arrives
with `hasFrag()` false, `parseNotification` returns `0` and the handler forwards
**stale-partial + new-message** as a single "complete" frame.

The most galling part: `parseNotification` already **detects** the loss and does nothing
about it. On a sequence discontinuity it resynchronises the counter and carries on with
the poisoned buffer:

```python
# ble_message.py, ~line 391
if sequence != (self.mReadSequence.increment_and_get() & 255):
    _LOGGER.debug(
        "parseNotification read sequence wrong %s %s",
        sequence,
        self.mReadSequence.get(),
    )
    # Set the value for mReadSequence manually
    self.mReadSequence.set(sequence)
```

## This is not a rare edge case

With `pymammotion.bluetooth.ble_message` at DEBUG on a healthy-looking link:

```
2026-07-25 19:04:32  parseNotification read sequence wrong 15 14
2026-07-25 19:04:43  parseNotification read sequence wrong 126 123
2026-07-25 19:05:54  parseNotification read sequence wrong 74 67
2026-07-25 19:05:56  parseNotification read sequence wrong 91 89
2026-07-25 19:06:32  parseNotification read sequence wrong 201 199
2026-07-25 19:07:24  parseNotification read sequence wrong 218 217
```

**11 sequence gaps in roughly 3 minutes of connected BLE**, losing 1–3+ packets each.
Only the gaps that land mid-fragment poison a frame, which matches the lower observed
rate of `dropping frame` (4 in 2.5 hours).

## Consequence for callers

A dropped frame is a _lost report_, and the loss is invisible above the transport: the
consumer sees telemetry that simply stops changing. In this integration that produced a
concrete false diagnosis — a guarded turn reported bit-identical `vision_heading` **and**
bit-identical `displacement_m` across two pulses while the operator watched the mower
physically turn ~4 inches, and the run aborted blaming the mower's actuation.

## Proposed fix

Reset the accumulation buffer whenever the reassembly is known to be broken. Minimal
change, entirely within `parseNotification`:

```diff
--- a/pymammotion/bluetooth/ble_message.py
+++ b/pymammotion/bluetooth/ble_message.py
@@
         # Compare with the second counter, mod 255
         if sequence != (self.mReadSequence.increment_and_get() & 255):
             _LOGGER.debug(
                 "parseNotification read sequence wrong %s %s",
                 sequence,
                 self.mReadSequence.get(),
             )
             # Set the value for mReadSequence manually
             self.mReadSequence.set(sequence)
+            # A sequence gap means at least one packet was lost. Anything already
+            # accumulated is now an unterminated fragment of a message we will never
+            # complete; keeping it makes the NEXT message parse as
+            # stale-partial + new-message concatenated, which deserialises as a
+            # valid frame carrying foreign field values. Discard it.
+            self.clear_notification()
@@
                 if respChecksum1 != calcChecksum1 or respChecksum2 != calcChecksum2:
                     _LOGGER.debug(
                         f"expect checksum: {respChecksum1}, {respChecksum2}\n"
                         f"received checksum: {calcChecksum1}, {calcChecksum2}"
                     )
+                    self.clear_notification()
                     return -4
@@
         except Exception as e:
             _LOGGER.debug(e)
+            self.clear_notification()
             return -100
```

Notes on correctness:

- Clearing at the sequence-gap site is safe because it happens **before** the
  `setType` / `setPkgType` / `setSubType` / `setFrameCtrl` calls, so the current packet
  still populates a fresh `BlufiNotifyData` normally.
- A gap on the _first_ packet of a new message is the common case and clearing is a
  no-op there (the buffer is already empty). The gap that matters is the mid-fragment
  one, and that is exactly the case this fixes.
- Clearing on `-4` / `-100` discards the type metadata too, which is correct: that
  message is being abandoned, and the next complete frame sets it again.

## Suggested follow-up (optional, larger)

The sequence check currently only emits at DEBUG. Since a gap now implies a _discarded
message_, it is worth surfacing — either a counter on the transport or a single INFO
log — so consumers can distinguish "the mower went quiet" from "we are losing packets".
The distinction matters: they look identical from above, and they call for opposite
responses.

---

# Second defect (same file): the chunk size exceeds BluFi's length field

**Found:** 2026-07-26, while investigating whether an MTU mismatch was causing the
packet loss above. It was not — but this turned up instead.

## The bug

`post_contains_data` splits outgoing payloads into 517-byte chunks:

```python
chunk_size = 517  # self.client.mtu_size - 3
```

`getPostBytes` then writes that chunk's length into a **single byte**, because that is
what the BluFi frame header allows:

```python
dataLength = 0 if data == None else len(data)
...
byteOS.write(dataLength.to_bytes(1, sys.byteorder))   # one byte -> max 255
```

A BluFi frame's `dataLen` field is one octet, so the maximum representable payload is
**255 bytes**. Any chunk between 256 and 517 bytes therefore raises

```
OverflowError: int too big to convert
```

inside `getPostBytes`, which propagates out through `post_contains_data` → `post` →
`post_custom_data_bytes` and fails the send. So **payloads larger than 255 bytes cannot
be transmitted at all** — not silently truncated, but hard-failed.

Note the comment (`# self.client.mtu_size - 3`) shows the intent was to derive this from
the negotiated MTU; `mBlufiMTU` is declared as `-1` and never set. The chunk size needs
to respect **both** limits:

```python
chunk_size = min(self.client.mtu_size - 3 - 4 - 2, 255)   # ATT header, BluFi header, checksum
```

## Practical impact today: none — but it is a live landmine

Measured on this deployment across the entire retained log, every outgoing BLE write is
small:

| payload size | occurrences |
| ------------ | ----------- |
| 27           | 64          |
| 28           | 141         |
| 35           | 1           |
| 48–54        | 80          |

**Largest send ever observed: 54 bytes.** So the 517 chunk size is never approached and
this defect is currently unreachable. It would bite the first command carrying a payload
over 255 bytes.

## What this _does_ explain about the reassembly bug

Because a BluFi frame can carry at most 255 bytes, **any device report larger than 255
bytes is necessarily fragmented**, regardless of the negotiated MTU. Fragmentation is the
precondition for the buffer-poisoning bug in the first half of this document — an
unfragmented frame returns 0 immediately and clears the buffer, so it can never be
poisoned.

The two frames observed being dropped were **249 and 260 bytes** — exactly at that
boundary. So the reports that break are precisely the ones large enough to require
fragmentation, which is consistent with the mechanism and explains why the corruption is
intermittent rather than constant.
