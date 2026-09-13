# The BLE write-latency mechanism, checked against the banked 2026-08-09 writes

**Written 2026-09-12. Phase 1 has not dispatched** — only the unscored setup leg
of predeclaration §15 has, at 17:06:54–17:07:39Z, **2.5 min after** this doc's
opening `sample_count: 0` read at 17:04:28Z. §1–§4.1 use **no** 2026-09-12
sample. §4.2 was measured after the setup leg ended; §4.3 deliberately uses the
setup leg's *excluded* samples as a mechanism cross-check only. **Nothing here
changes a threshold, and §15 closes §2–§14 to amendment anyway.**

Purpose: §10.5 of `predeclared-queue-timeout-measurement-20260911.md` raises the
prior that §3 fires ("p95 queue wait ≥ 1000 ms") on the strength of banked
refresh *write* latency, and §12.1 gives the "write latency binds" answer a
predeclared ratio test. This checks the mechanism behind those numbers.

---

## 1. 🗑️ My own proposed mechanism is REFUTED

I proposed that writes are slow because each command fragments into ~20-byte
Blufi chunks, each sent with an acknowledged GATT write plus a 10 ms inter-chunk
sleep. **That is wrong, and it is wrong twice.**

- `MAX_DATA_LEN = 255`, `FRAG_CONTENT_LEN = 253`
  (`pymammotion/bluetooth/ble_message.py:26-28`). The `DEFAULT_PACKAGE_LENGTH`
  / `MIN_PACKAGE_LENGTH = 20` I mistook for the fragment size are Blufi's
  *minimum negotiable package length*; `mBlufiMTU = -1` is unset.
- A `send_movement(linear_speed=400, angular_speed=0)` frame is **29 bytes**
  (measured against the installed `0.8.12.post4`). 29 < 255, so the `while`
  loop at `ble_message.py:587` runs **exactly once**, `frag` is `False`, and the
  `await sleep(0.01)` at `:605` **never executes**.

**One motion command = one `write_gatt_char(UUID_WRITE_CHARACTERISTIC, data,
True)` of 29 bytes** (`ble_message.py:351`). There is no fragmentation to blame.

What survives from the original observation is only that the write is
**acknowledged** (`response=True`), not fire-and-forget.

## 2. 🔑 What the distribution actually says

All `refresh_write_durations_ms` across the five `evidence-beta32-4segment-20260809T*`
runs, excluding the `phases[]` re-projections that duplicate the same records:

```
n=115  min=69.3  p10=98.8  p25=115.2  p50=206.5  p75=337.5
       p90=555.7  p95=783.6  max=2014.0   mean=283.9  sd=268.9
```

```
  50- 99  ############ 12
 100-149  ############################### 31   <- mode
 150-199  ########### 11
 200-249  ################ 16
 250-299  ######## 8
 300-349  ########## 10
 350-399  ### 3
 400-449  ######## 8
 450-499  # 1
 500-599  ###### 6
 600-799  ### 3
  >=800   ###### 6
```

🚨 **There is a hard floor at 69.3 ms. Zero samples below it.**

A single 29-byte write-with-response over BLE is one connection-event round
trip — tens of milliseconds at a typical ESP32-proxy connection interval. A
**floor of 69 ms with the mode at 100–149 ms is 2–5x that**, and it is a floor,
not a tail. That shape is a *fixed per-write overhead* on the path
HA → ESPHome API (TCP/WiFi) → proxy → GATT write → ack → back, **not** an
intrinsic property of the BLE link and **not** fragmentation.

The long right tail (6 samples ≥ 800 ms, max 2014.0) then sits **on top of**
that floor and is the contention/retry component.

## 3. 🚨 §10.5's figures do not reproduce, and the gap is in the unsafe direction

§10.5 (and the `_motion_refresh_window` comment at `services.py:6776-6783`)
quote **98 writes, p50 225.6 / p90 572.0 / p95 1029.2 / max 2014.0, 59 % over
200 ms**. Recomputing from the same five runs:

| | §10.5 quotes | recomputed |
|---|---|---|
| n | 98 | **115** |
| p50 | 225.6 | **206.5** |
| p90 | 572.0 | **555.7** |
| p95 | **1029.2** | **783.6** |
| max | 2014.0 | **2014.0** ✅ exact |
| over 200 ms | 59 % | **53 %** |

`max` matching to the decimal confirms the same underlying population family
(2014.0 appears only in `...T195940Z.json`). But **n never reaches 98** under
any partition tried: per-file counts are 13 / 35 / 0 / 15 / 52 (115 total;
155 if the `phases[]` duplicates are double-counted), and no subset sums to 98.
Splitting by phase gives 68 `linear_forward_to_target` + 47 unattributed.

⚠️ **p95 is the number §10.5 leans on, and it is the one most overstated
(1029.2 vs 783.6, +31 %).** §10.5 uses it to argue the prior favours §3. That
argument is weaker than written. **Pin the provenance of "98" before any Phase 1
conclusion cites §10.5**, or restate §10.5 against the recomputed figures.
This does not overturn §10.5's *direction* — writes really are slow relative to
the 200 ms interval — only its magnitude.

## 4. 🚨 The consequence for §12.1, which is the part that matters

§12.1 evaluates `ratio = p95(queue_wait_ms | pulse-open) / p95(write_ms | all
completed)`, with `ratio ≤ 1.5` at `Q ≥ 1000 ms` ⇒ "write latency binds and the
constant is NOT a candidate to move", `> 1.5` ⇒ real multi-item contention ⇒ §3.

That test is **sound but incomplete**, for a reason visible before the data:

- The queue is serialized, and the thing occupying it is *a previous write on the
  same path*. So `queue_wait_ms` and `write_ms` are **not independent
  quantities** — the ratio is, in effect, an **estimator of queue depth**
  (wait behind k in-flight writes ⇒ ratio ≈ k). At `k = 1` it reads ≈ 1.0.
- With p95(write) already at 784–1029 ms, **a single predecessor write nearly
  exhausts the 2.0 s budget on its own.** So the likely outcome is
  `Q ≥ 1000 ms` **and** `ratio ≈ 1.0` — §12.1's "write latency binds" branch.
- 🔑 **That branch correctly says "don't move the constant" and then names no
  remedy.** The legs still fail. Under that verdict the two live options are
  (a) stop queueing motion commands at all, and (b) reduce the per-write floor.

### 4.1 Two remedies that branch should name

- **(a) Direct send.** Motion dispatches go out at `Priority.NORMAL`
  (`services.py:8046`; only e-stop uses `EMERGENCY`), so they wait on the
  serialized queue by construction. Upstream's refactor branch added
  `Priority.USER` as a *direct-send* level that skips `queue.enqueue` — no TTL,
  no exclusive-slot wait, dispatched on the caller's task, with transport
  selection and the `send_raw` fallback chain untouched
  (`PyMammotion@large-scale-tidying-refactoring` `docs/decisions.md` D14).
  🛑 **It does not exist in our pinned `0.8.12.post4`** — verified: `Priority`
  is `EMERGENCY/EXCLUSIVE/NORMAL/BACKGROUND` only
  (`messaging/command_queue.py:31-41`). Adopting it is a backend migration,
  not a constant change.
- **(b) The floor is a proxy-path cost, not a link cost** (§2), so whatever
  occupies that path inflates *both* `write_ms` and `queue_wait_ms`. CLAUDE.md
  names `custom_components.omron` as that occupant. **Measured below — it is
  not.**

### 4.2 🗑️ The omron contention claim is REFUTED as stated (measured 2026-09-12 17:30-17:45Z)

CLAUDE.md says the omron integration is "sustained contention on the same proxy
radio" carrying the mower's GATT writes. From the HA container log:

| | 6 h window | 30 min window |
|---|---|---|
| omron connection-path enumerations | 823 (**137/h**) | 67 (**134/h**) |
| omron **actual connect attempts** | 101 | 9 (**~18/h**) |
| connect attempts via an ESP32 proxy | **0** | **0** |

🚨 **All 102 omron connect attempts in 6 h went `via source=D8:3A:DD:C3:CE:CD`
— `hci0`, the HA host's own built-in adapter.** Not one went through any ESP32
proxy. The cuff sits close to the host (RSSI −56 to −68 on `hci0`) and −94 to
−98 on the backyard proxies, so habluetooth always picks `hci0` and the attempt
fails there (`BleakNotFoundError ... Failed to connect after 4 attempt(s)`).

🔑 **The `failures=22` counters on the backyard proxies are habluetooth's
accumulated historical score, not evidence of ongoing attempts.** Reading them
as live contention — which the 2026-09-12 finding and CLAUDE.md both do — is the
error.

✏️ **And the mower is not on `hot-tub-backyard`.** `A8:B5:8E:2C:52:40 -
Luba-VSPLV397` ranks **`p1s-printer` first** (RSSI −49/−51 vs −54/−57 for
hot-tub-backyard), with **`failures=0` on all four paths**. CLAUDE.md's "the
mower normally holds `hot-tub-backyard`" is stale.

**So omron and the mower do not share a radio.** The 137/h log line is real and
ongoing, but it is a *path enumeration*, and the ~18/h real attempts land on a
different adapter. Any residual coupling is HA event-loop/bluetooth-stack time,
which is plausible but **unmeasured and much weaker than a shared radio**.

⚠️ **Consequence for §12.1:** the first concrete candidate occupant for the
falsifier branch is **gone**. The 69.3 ms floor and the ≥800 ms tail still need
an explanation, and omron is no longer it. Disabling that integration for the
measurement is **not** justified on this evidence — and doing it anyway would
spend a Phase 1 session controlling for a variable now shown to be off-path.

🛑 **What this does NOT refute:** that *some* occupant explains the tail. It
refutes only the named one.

✏️ **Do not chase `p1s-printer`'s `slots=1/3 free` as a lead.** An earlier draft
of this section did. Slot count governs whether a *new* connection can be
established; it says nothing about the latency of one already open, and it
cannot distinguish an idle second connection from a busy one.

🔑 **The floor and the tail need different explanations.** The 69.3 ms floor is
on *every* write with zero samples below it — contention is episodic and adds a
tail, it does not install a hard floor. The floor is a fixed per-write path cost
(HA → ESPHome API → connection interval → acknowledged-write ack → back) and is
already ~35 % of the 200 ms refresh interval before anything goes wrong. Only
the ≥ 800 ms tail (6 of 115) is a candidate for an occupant, and nothing
measured here identifies one.

### 4.3 Cross-check on an independent day (setup leg, EXCLUDED from §2)

`write_ms` from the 2026-09-12 setup leg, which §15 excludes from the population
and which is already published in
`docs/findings-setup-leg-and-classifier-validation-20260912.md`:

| | n | min | p50 | p95 | max |
|---|---|---|---|---|---|
| 2026-08-09 refresh writes | 115 | **69.3** | 206.5 | **783.6** | 2014.0 |
| 2026-09-12 non-stop writes | 55 | **70.6** | 186.5 | **307.8** | 611.5 |
| 2026-09-12 stop writes | 10 | 106.9 | 182.1 | 309.8 | 309.8 |

✅ **The floor reproduces** — 70.6 ms, zero samples below 69.3, a month apart,
under a different proxy (`p1s-printer`). That supports §2's fixed-path-cost
reading.

🚨 **The tail does NOT reproduce, and that weakens §4's prediction.** p95 fell
from 783.6 to **307.8 ms**. §4 argued a single predecessor write "nearly exhausts
the 2.0 s budget" and so the likely outcome was `Q ≥ 1000 ms` with `ratio ≈ 1.0`.
On this leg the predecessor is specifically the previous pulse's **stop** (per
the setup-leg findings §2), whose write p95 is 309.8 ms — and the pulse-open wait
p95 was 432.424 ms with ratio 1.396, i.e. **§5 inconclusive, not §3**.
✏️ **§4's "likely outcome" is withdrawn as a prediction.** Its structural point
— the ratio estimates queue depth, and the "write latency binds" branch names no
remedy — stands. n = 11 pulse-opens on one excluded leg supports no verdict.

## 5. 🛑 What this does NOT establish

- It does **not** move any threshold in
  `predeclared-queue-timeout-measurement-20260911.md`. Every criterion there
  stands exactly as committed.
- It does **not** show the 2.0 s bound is right or wrong. That still needs the
  Phase 1 measurement.
- The ~70 ms floor is two days (`n=115` refresh writes, `n=55` motion writes)
  under two proxy topologies. It is a shape, not a calibrated constant, and per standing
  decision 7 it must not be quoted as a rate.
- No claim here rests on the upstream docs' description of the refactored
  library, which is **not** the backend we ship.

## 6. Provenance

- Installed backend: `pymammotion 0.8.12.post4` (`.venv`), read directly.
- Banked writes: `docs/evidence-beta32-4segment-20260809T{170941,183129,192923,195940,210241}Z.json`.
- Live state at 2026-09-12T17:04Z: `real_motion_ready: on`, `blockers: []`,
  `zone_hash: 1343645155037768237` (non-zero), `device_position_type:
  area_inside`, `rtk_position: fix`, `position_level: 1`, `ble_link_live: on`
  at −62 dBm, battery 99 %, VIO 80 features, `sample_count: 0`.
- ✏️ **An earlier draft said "config entry options `{}` (gate disarmed)". That
  was wrong.** `/api/config/config_entries/entry` does not return `options`; the
  `{}` was the reading script's own default. `blockers: []` at 17:01:29Z means
  the gate was **armed** — legitimately, for the operator-approved setup leg.
  🔑 **Gate state is readable from `blockers` (look for
  `experimental_motion_disabled`) or raw `core.config_entries`, never from that
  endpoint.**
- Session end, 2026-09-12T23:28Z: gate **disarmed**, verified both ways —
  `blockers: ['experimental_motion_disabled', 'position_not_valid_for_motion']`
  and raw `enable_experimental_motion: False`. Mower docked (`charge_on` since
  21:54:40Z), 100 %. Timing history snapshot: 65 samples, identical window to
  `docs/evidence-setup-leg-timing-20260912.json`.

---

## 7. ✏️ Correction on committing (2026-09-13) — the mower's proxy MOVES

§4.2 says "the mower is not on `hot-tub-backyard`" and calls CLAUDE.md's
"normally holds `hot-tub-backyard`" stale. **Both statements are too strong.**
At 2026-09-13 01:58Z `bluetooth/subscribe_connection_allocations` showed
`A8:B5:8E:2C:52:40` allocated on `C4:DD:57:70:B6:96` — **`hot-tub-backyard`** —
as it was at 2026-09-12 03:0xZ, while this doc saw `p1s-printer` ranked first at
17:30Z. 🔑 **The connection moves between reconnects; neither doc should name a
fixed proxy.** Everything else in §4.2 was independently confirmed before commit:
omron is allocated on `hci0`, and §3's recomputation (n = 115, p95 783.6)
reproduces exactly from files unmodified since 2026-08-09.
