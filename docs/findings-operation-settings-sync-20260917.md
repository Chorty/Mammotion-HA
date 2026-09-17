# Mowing settings in HA are a local staging object, not the mower's job

**Investigated 2026-09-17, offline only.** No command was sent to the mower, no
`mammotion.*` service was called, no entity was changed, and Home Assistant was
neither restarted nor reloaded — a traced mow was running in a parallel session
throughout.

Backend read for every pymammotion citation below is the pinned wheel,
`pymammotion 0.8.12.post4` (`custom_components/mammotion/manifest.json:34`,
`requirements_test.txt:14`), unpacked and read directly. Paths are written
`pymammotion/…` and resolve under `.venv/lib/python3.14/site-packages/`.

---

## 0. Verdict on the prior session's code reading

The previous session's three claims were read from code and recorded as
unverified. All three **confirm**, with one material correction and two
additions that change the shape of the fix:

| Prior claim | Verdict |
| --- | --- |
| All three entities read/write a local `OperationSettings()` never filled from the device | ✅ **Confirmed** (§1, §2) |
| A mid-mow change re-sends the *whole* local settings object | ⚠️ **Confirmed but incomplete** — `async_modify_plan_route` already re-seeds **8** fields from the running job. The three that matter are exactly the three it omits (§3) |
| Units are worth checking; 2 ft/s ≈ 0.61 m/s exceeds HA's 0.6 m/s maximum | ✅ **Confirmed** — HA cannot represent the speed the job was started at (§5) |

Two things the prior reading did not have:

- 🔑 **The read path already exists on both sides.** The device's own
  `CurrentTaskSettings` carries `knife_height`, `channel_width` and `speed`
  (§4.1), and the pinned pymammotion already has the command that asks for them
  — `query_generate_route_information` (§4.3), which this integration already
  sends from one other place (`lawn_mower.py:306,312`).
- 🔑 **Upstream has already fixed half of this** and the fix is portable (§6).

---

## 1. The object: one local staging record, never filled from the device

`custom_components/mammotion/coordinator.py:184`

```python
self._operation_settings = OperationSettings()
```

Constructed once per coordinator with pymammotion's dataclass defaults
(`pymammotion/data/model/device_config.py`):

| field | line | default |
| --- | --- | --- |
| `speed` | :19 | `0.3` (m/s) |
| `channel_width` | :22 | `25` (cm) |
| `blade_height` | :24 | `0` (mm) |

Exposed read/write as a property at `coordinator.py:2139`. It is a **plan
builder**: `lawn_mower.py:257` copies it to start a new job, `switch.py`,
`select.py` and `number.py` all write into it. Nothing anywhere writes the
*device's* values into it.

**`async_restore_data` does not restore it either** (`coordinator.py:2152`). It
restores a whole `MowingDevice` from the store — which includes `work` — but
never touches `_operation_settings`. The comment at `lawn_mower.py:255-256`
("Merge onto coordinator's restored settings") is therefore misleading: the only
restore that reaches `operation_settings` is per-entity `RestoreNumber` in
`number.py:344-355`.

### 1.1 How the entities end up showing what they show

`MammotionConfigNumberEntity.__init__` (`number.py:300-326`):

- `number.py:309-311` — **the displayed value is seeded from the entity's own
  `native_min_value`**, not from anything the device said.
- `number.py:320-321` — if the description has a `get_fn`, the display is read
  from it.
- `number.py:322-326` — `elif set_fn` — the seeded minimum is **pushed into
  `operation_settings`**.

Per entity, on a LUBA:

| entity | `get_fn`? | what it displays at startup | what it writes into `operation_settings` at startup |
| --- | --- | --- | --- |
| `working_speed` (`number.py:194-207`) | **none** | `native_min_value` → **0.2 m/s** | `speed = 0.2` (via the `elif` at :322) |
| `path_spacing` (`number.py:208-218`) | **none** | `native_min_value` → **20 cm** | `channel_width = 20` (via the `elif` at :322) |
| `blade_height` (`number.py:174-189`) | `operation_settings.blade_height` (:188) | `0`, then clamped up to the slider minimum **25 mm** by `number.py:383-386` | **nothing** — the `elif` at :322 is skipped because `get_fn` is set |

🚨 **`blade_height` displays 25 mm while `operation_settings.blade_height` is
still 0.** They diverge from the first second. `RestoreNumber`
(`number.py:344-355`) then closes the gap on the next HA start, in the wrong
direction: it restores the *displayed* 25 and calls `set_fn`, so
`operation_settings.blade_height` becomes **25 mm ≈ 0.98″**.

That is the operator's "1 inch".

---

## 2. Write path, end to end

`number.py:404` `MammotionWorkingNumberEntity.async_set_native_value`
→ `set_fn` writes one attribute into `operation_settings` (`number.py:409-410`)
→ `set_async_fn` (`number.py:411-412`) → for `blade_height` (:185-187) and
`working_speed` (:201-203) that is `coordinator.async_modify_plan_if_mowing()`.
`path_spacing` has **no** `set_async_fn`, so it never pushes mid-mow.

`coordinator.py:2143-2150`:

```python
async def async_modify_plan_if_mowing(self) -> None:
    if (int(_mdata.report_data.work.bp_hash) in _mdata.work.zone_hashs
            and (_mdata.report_data.work.area >> 16) != 100):
        await self.async_modify_plan_route(self.operation_settings)
```

→ `async_modify_plan_route` (`coordinator.py:1998`)
→ `generate_route_information` (`coordinator.py:1932`)
→ `async_send_command("modify_route_information", …)`
→ pymammotion `navigation.py:556` builds `NavReqCoverPath(pver=1, sub_cmd=3, …)`
and sends it as `MctlNav(bidire_reqconver_path=…)`.

### 2.1 The exact protobuf fields sent, and where each value comes from

`navigation.py:556-575`, `modify_route_information`. **Every field is sent on
every call; there is no partial update on the wire.** "Job" = re-seeded from the
running job by `async_modify_plan_route`; "local" = whatever is in HA's staging
object.

| `NavReqCoverPath` field | wire # | value | unit | source | source line |
| --- | --- | --- | --- | --- | --- |
| `pver` | 1 | `1` | — | literal | `navigation.py:559` |
| `sub_cmd` | — | `3` (modify) | — | literal | `navigation.py:560` |
| `zone_hashs` | — | job's zones | — | **job** | `coordinator.py:2004` |
| `job_mode` | — | job's | — | **job** | `coordinator.py:2009` |
| `edge_mode` | — | job's `edge_mode` | laps | **job** | `coordinator.py:2008` |
| **`knife_height`** | 7 | `operation_settings.blade_height` | **mm** | 🚨 **local** | `coordinator.py:1955` |
| **`speed`** | — | `operation_settings.speed` | **m/s** | 🚨 **local** | `coordinator.py:1948` |
| `ultra_wave` | — | `operation_settings.ultra_wave` | enum | 🚨 **local** | `coordinator.py:1949` |
| **`channel_width`** | — | `operation_settings.channel_width` | **cm** | 🚨 **local** | `coordinator.py:1957` |
| `channel_mode` | — | `operation_settings.channel_mode` | enum | 🚨 **local** | `coordinator.py:1956` |
| `toward` | — | job's | deg | **job** | `coordinator.py:2005` |
| `reserved` (`path_order`) | — | `create_path_order(operation_settings, …)` | 8 bytes | 🚨 **local** | `coordinator.py:1960` |

`job_id` / `job_version` are copied from the job at `coordinator.py:2010-2011`
but 🔑 **`modify_route_information` does not put them on the wire at all** —
compare `navigation.py:559-572` with `generate_route_information`'s builder at
`navigation.py:536-550`, which also omits them. Copying them is inert for this
path.

🚨 **Zero/default values ARE sent for fields the user never touched.** There is
no "unset" sentinel: `blade_height` defaults to `0`, and the only reason the
operator saw 25 mm rather than 0 mm is the `RestoreNumber` round-trip in §1.1.
On a fresh install with no restored state, a mid-mow speed change would send
**`knife_height = 0`**.

### 2.2 The reported incident, reconstructed

1. Operator starts a mow **from the app** at blade 2.2″ (55.9 mm), 2 ft/s
   (0.61 m/s). Nothing in this integration learns those numbers (§4).
2. HA's `operation_settings` holds `blade_height = 25` (restored, §1.1),
   `channel_width = 20`, `speed` = whatever was last set in HA.
3. Operator changes **Working speed** in HA. `number.py:204-206` writes
   `speed`; `number.py:201-203` calls `async_modify_plan_if_mowing()`.
4. `sub_cmd=3` goes out carrying `knife_height=25` **and** `channel_width=20`
   alongside the new speed.
5. The mower's blade drops to 25 mm ≈ **1″**. Path spacing was pushed to 20 cm
   in the same message — 🔑 **unreported by the operator and not visible from
   HA, but it is on the wire by the same mechanism.**

---

## 3. What `async_modify_plan_route` already gets right

`coordinator.py:1998-2016` is **not** naive. It already re-seeds 8 fields from
the running job before sending:

| seeded from `self.data.work` | line |
| --- | --- |
| `areas` ← `work.zone_hashs` | :2004 |
| `toward` | :2005 |
| `toward_mode` | :2006 |
| `toward_included_angle` | :2007 |
| `mowing_laps` ← `work.edge_mode` | :2008 |
| `job_mode` | :2009 |
| `job_id` | :2010 |
| `job_version` ← `work.job_ver` | :2011 |

🔑 **The bug is an omission, not a design error.** The three settings the
operator can edit — `speed`, `channel_width`, `blade_height` — are precisely the
three this list leaves out, and `work` carries all three (§4.1). The intent to
preserve the running job was already there; it was applied to the fields nobody
edits and skipped on the fields everybody edits.

---

## 4. The read path that ought to exist — and mostly does

### 4.1 `CurrentTaskSettings` — the running job's own parameters

`pymammotion/data/model/work.py:9`, reachable as `coordinator.data.work`
(`pymammotion/data/model/device.py:115`):

| field | line | default | maps to |
| --- | --- | --- | --- |
| `knife_height` | :18 | `0` | `blade_height` (mm) |
| `channel_width` | :19 | `0` | `path_spacing` (cm) |
| `speed` | :23 | `0.0` | `working_speed` (m/s) |

pymammotion even ships the mapping already written:
`GenerateRouteInformation.from_current_task_settings`
(`pymammotion/data/model/generate_route_information.py:62-101`), documenting
`knife_height → blade_height` at :77 and :94.

### 4.2 How `work` gets filled

`pymammotion/device/state_reducer.py:408-411`, on the `bidire_reqconver_path`
nav sub-message:

```python
current_task = CurrentTaskSettings.from_dict(work_settings.to_dict(...))
device.work = current_task
```

⚠️ **Only that message fills it.** The ~1 Hz report handler
(`pymammotion/data/model/device.py:246-260`) touches `work.zone_hashs` and only
ever *clears* it; it never writes `speed`, `channel_width` or `knife_height`.
`MowingDevice` is persisted whole by `async_save_data` (`coordinator.py:2190`),
so `work` survives a restart — **a stale record and a fresh one are
indistinguishable at the field level.**

### 4.3 The command that asks for it

`pymammotion/mammotion/commands/messages/navigation.py:585`:

```python
def query_generate_route_information(self) -> bytes:
    build = NavReqCoverPath(pver=1, sub_cmd=2)
```

🔑 **It is in the pinned wheel — no pymammotion change is needed.** This
integration already sends it, but from one narrow place only: resuming or
starting from a breakpoint, `lawn_mower.py:306` and `lawn_mower.py:312`. It is
**never** sent for a job started from the app, which is exactly the reported
case.

### 4.4 Per-field verdict: live / cached / unreadable

| setting | live in the ~1 Hz report? | cached? | verdict |
| --- | --- | --- | --- |
| **blade height** | ✅ **yes** — `report_data.work.knife_height`, `RptWork` field **20** (`pymammotion/proto/__init__.py:4664`), surfaced as `WorkData.knife_height` (`report_info.py:464`), updated every report (`report_info.py:645`) | also in `work.knife_height` | 🟢 **LIVE** |
| **working speed** | ❌ **no** | `work.speed` only | 🟡 **CACHED** — needs `sub_cmd=2` |
| **path spacing** | ❌ **no** | `work.channel_width` only | 🟡 **CACHED** — needs `sub_cmd=2` |

🚨 **Two `RptWork` fields are decoys.** `man_run_speed` (field 18) is the
**manual-drive** speed, not the job's mowing speed; `cutter_width` is the
**physical cutting-deck width**, not path spacing. Neither answers the question,
and both would look plausible in a telemetry dump.

Fields verified against the live class, not the docs:

```
RptWork: plan, path_hash, progress, area, bp_info, bp_hash, bp_pos_x, bp_pos_y,
         real_path_num, path_pos_x, path_pos_y, ub_zone_hash, ub_path_hash,
         init_cfg_hash, ub_ecode_hash, nav_run_mode, test_mode_status,
         man_run_speed, nav_edit_status, knife_height, nav_heading_state,
         cutter_offset, cutter_width
```

**Nothing here is unreadable.** All three are obtainable; two of them cost one
round trip.

---

## 5. Units and ranges

| entity | HA native unit | HA range (`number.py`) | device unit on the wire |
| --- | --- | --- | --- |
| `blade_height` | mm (`number.py:177`) | 25–70 (`:179-180`) | `knife_height`, mm |
| `working_speed` | m/s (`number.py:197`) | 0.2–0.6 (`:199-200`) | `speed`, m/s |
| `path_spacing` | cm (`number.py:212`) | 20–35 (`:214-215`) | `channel_width`, cm |

Those static bounds are overridden per device at `number.py:370-372` from
`DeviceLimits` — either `handle.device_limits` or
`DeviceConfig().get_working_parameters(product_key)`
(`pymammotion/utility/device_config.py:2341`). That table holds 12 product
keys; **11 of them give `blade_height 30–70`, `working_speed 0.2–1.2`,
`path_spacing 20–35`**, and the twelfth (`a1ZU6bdGjaM`, `LubaAWD1000723`,
`device_config.py:30-40`) caps speed at **0.4** instead. So **the effective
range on the operator's mower cannot be established from source alone** — it
depends on the product key and on which branch produced `limits` at runtime
(`number.py:231-235`). See §9.

**Conversion is HA's, and it is display-only.** A `number` entity converts only
when the user has set a per-entity unit override: `unit_of_measurement`
(`homeassistant/components/number/__init__.py:403-408`) returns
`_number_option_unit_of_measurement`, which is populated **only** from the
entity registry option at `:525-540`. Unlike sensors, numbers are *not*
auto-converted by the unit system. Either way `async_set_native_value` receives
the **native** value, so US display units cannot themselves corrupt what is sent.

🚨 **But the range is genuinely wrong for this job.** 2 ft/s = **0.6096 m/s**,
above the 0.6 m/s maximum at `number.py:200`. If the static bounds are in force,
**HA cannot represent, let alone preserve, the speed the operator started the
mow at** — any HA-side speed edit necessarily lands below it. Blade height is
fine: 2.2″ = 55.9 mm, inside 25–70 mm.

---

## 6. Upstream status

Checked read-only by cloning; nothing was pushed, commented or opened.

**`mikey0000/Mammotion-HA` @ `d892896` (0.6.5-beta11) has fixed the write half.**
`custom_components/mammotion/coordinator.py:1741-1834` adds:

- `_is_route_job_running()` (:1741) — the running-job predicate, factored out of
  our inline condition at `coordinator.py:2143-2149`. Identical logic.
- `_seed_operation_settings_from_running_job()` (:1755) — our 8 fields **plus**
  `speed`, `channel_width`, `ultra_wave`, `channel_mode`, `blade_height`
  (`work.knife_height`) and `auto_change_direction`.
- `_apply_route_field_if_working(field)` (:1810) — read the new value, seed
  everything from the job, write the one field back, send.
- `async_change_blade_height_if_working` (:1786) and
  `async_change_speed_if_working` (:1828), wired at their `number.py:190-192`
  and `:205-206`.

Their `blade_height` path also branches on `DeviceType.is_luba_pro`: Luba 1 gets
a direct `setKnifeHight` nudge, Luba 2+ re-issues the route. `speed` and
`ultra_wave` send **nothing at all** on a Luba 1 (:1819-1822).

⚠️ **Not directly cherry-pickable.** `work.auto_change_direction`
(their :1779) does not exist on `CurrentTaskSettings` in our pinned
`0.8.12.post4`; upstream is on `pymammotion 0.9.0b9` (their `manifest.json`).
The field must be dropped when porting.

🚨 **Upstream has NOT fixed the read half.** Their `blade_height` `get_fn` still
reads `coordinator.operation_settings.blade_height` (their `number.py:192`) and
their `working_speed` / `path_spacing` still have no `get_fn` at all. **The
entities still show HA's staged values, and upstream never sends `sub_cmd=2` to
learn an app-started job's settings** — so seeding from a `work` record that was
never filled (all-zero, §4.2) would push `speed = 0.0`, `channel_width = 0`,
`knife_height = 0`. `_is_route_job_running()` happens to guard that case, because
an unfilled `work` has an empty `zone_hashs` and the predicate is then False —
but 🔑 **a `work` record left over from a *previous* job in the same zone passes
the predicate and seeds the wrong job's settings.** That gap is why the fix here
refreshes before seeding (§7).

`mikey0000/PyMammotion`: `query_generate_route_information` is unchanged from
our pinned copy; no fix needed or present there.

---

## 7. The fix

Two properties, from `docs/TODO.md`.

**Property 2 — changing one setting changes only that setting.** Port upstream's
shape (`_is_route_job_running`, seed-then-override, per-field entry points),
minus `auto_change_direction`, **plus a refresh**: before seeding, send
`query_generate_route_information` (§4.3) and require the resulting `work` record
to look filled in (`zone_hashs` non-empty, `speed > 0`, `channel_width > 0`).
`async_send_and_wait` swallows its own timeouts (`coordinator.py:1003-1007`), so
success cannot be judged from an exception — it is judged from the record.
🔑 **If the record still does not look filled, send nothing.** Refusing to act
beats pushing a guess at a running mower's blade height.

⚠️ **One pre-existing wrinkle is deliberately left alone.** When no job is
running, `blade_height`'s `get_fn` returns `operation_settings.blade_height`,
which is `0` until something writes it — below the slider's own minimum.
`MammotionWorkingNumberEntity.__init__` clamps that at construction
(`number.py:418-421`) but `_handle_coordinator_update` (`number.py:341-345`)
does not. That is unchanged from the old `get_fn`, which returned the same `0`,
so it is **not a regression** — but it is why the entity can briefly read below
its own minimum on an idle mower, and it is the next thing to look at if that is
ever reported.

**Property 1 — the entities show the job's real values, or say they don't.**
Give all three a `get_fn` that returns the running job's value when one is
running *and* the device has actually told us that value, and otherwise the
staged local value. Add a `value_source` state attribute reading `running_job`
or `staged`, so "value HA will send" is visible rather than inferred.

---

## 8. Hardware test plan — NOT RUN, needs a separate explicit operator go

Predeclared here before any hardware data exists. No part of it was executed;
this session sent nothing to the mower.

**Preconditions:** daylight, `rtk_position: fix`, mower in a mowing area, the
fix deployed, no other session driving the mower.

1. Record HA's `number.working_speed`, `number.blade_height`,
   `number.path_spacing` — state, `value_source` attribute, `min`, `max`, `unit`.
2. Start a mow **from the Mammotion app** at a blade height and speed that are
   both distinctive and **not** equal to any HA slider minimum (e.g. 45 mm and
   0.4 m/s — never 25 mm, never 0.2 m/s: a floor value cannot distinguish "read
   correctly" from "defaulted").
3. Within 60 s, re-read all three. **Pass:** each reads the app's value and
   `value_source: running_job`.
4. Change **Working speed only** in HA, by one step.
5. Re-read. **Pass:** speed is the new value; **blade height and path spacing are
   unchanged, at the mower** — confirm at the mower, not only in HA, since HA
   showing the right number is exactly the thing under test.
6. Repeat 4–5 changing **Blade height only**; speed and spacing must not move.
7. Dock. Confirm the entities fall back to `value_source: staged`.

**Falsifier:** if step 3 shows a slider minimum, or `value_source: staged` while
the mow runs, the read path is not working and the write path must not be
trusted — the refresh in §7 is the thing to instrument first.

⚠️ **Step 2's "not a slider minimum" rule is the one that makes this test
meaningful.** The original incident is indistinguishable from correct behaviour
if the chosen value happens to equal the default.

---

## 9. What could not be established

1. **Whether the device pushes `bidire_reqconver_path` unsolicited** when the
   app starts or edits a job (e.g. an MQTT broadcast to every connected client).
   If it does, `work` may be fresher in practice than §4.2 implies. Nothing in
   the reducer distinguishes a solicited reply from a pushed one, and no capture
   was available. **The fix does not depend on the answer** — it refreshes
   explicitly — but the answer would say whether the refresh is usually a no-op.
2. **The effective min/max on the operator's mower.** `number.py:370-372` takes
   them from `DeviceLimits` at runtime; the static 0.2–0.6 m/s and the LUBA table's
   0.2–1.2 m/s disagree, and which one is live decides whether §5's "HA cannot
   represent 2 ft/s" holds on this device. **One read of the entity's `max`
   attribute settles it** and was deliberately not taken — see 4.
3. **What the operator's entities actually read at the time.** Reconstructed
   from code in §2.2, not observed.
4. **No live HA access from this session.** `.env` is gitignored and absent from
   this cloud clone, so there was no token and no host — read-only API GETs were
   available in principle but not in fact. Items 2 and 3 are one authenticated
   GET each for a session that has one.
5. **The vendor APK decompile was not consulted.**
   `/Users/mattjoslin/mammotion-apk-decompile/src` is a path on the operator's
   Mac; this session runs in a Linux cloud container. Upstream's docstrings
   (their `coordinator.py:1789-1798`) cite `HomeMapFragment` /
   `WorkingOptionView.onConfirm` for the seed-then-override behaviour; that
   citation is **relayed, not verified here.**
6. **The Luba 1 branch is untestable here.** Upstream sends nothing on a mid-job
   speed change on a Luba 1 (their `coordinator.py:1819-1822`). The port keeps
   that shape, but the operator's mower is not a Luba 1 and no Luba 1 was
   available.
