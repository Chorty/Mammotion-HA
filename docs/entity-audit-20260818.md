# Entity duplication audit — 2026-08-18

**Result: there are no duplicated entities in the integration code.** Three
cosmetic problems exist in the live install; all three are logged below and none
has been changed, because every fix is user-visible and would break existing
automations, dashboards and history.

## Method

Three passes, because each catches a different kind of duplication:

1. **Static** — every `key="…"` across all eleven entity platforms, looking for
   repeats within and across files.
2. **Semantic** — entity descriptions reading the same underlying value.
3. **Live** — `/api/states` on the running install, looking for `_N` entity-id
   suffixes (the re-registration signature), colliding friendly names, and
   orphans.

⚠️ The naive static sweep produces **false positives**: `key="([^"]+)"` also
matches `gate_key=`, which invented four non-existent duplicates in
`binary_sensor.py` and `lawn_mower.py`. Use `(?<![a-z_])key="` instead.

## ✅ Confirmed NOT duplicates — do not re-audit these

| apparent duplicate | why it is fine |
| --- | --- |
| `cutting_angle_mode` ×2 in `select.py` | `LUBA1_SELECT_ENTITIES` vs `LUBA_PRO_SELECT_ENTITIES`, behind a mutually exclusive `if DeviceType.is_luba1(…) / else`. Exactly one ever registers. |
| `position_mode`, `rtk_latitude`, `rtk_longitude` ×2 in `sensor.py` | `SENSOR_TYPES` builds `MammotionSensorEntity` on the **mower** from `mower.reporting_coordinator`; `RTK_SENSOR_TYPES` builds `MammotionRTKSensorEntity` on the **RTK base station** from `rtk.coordinator`. Same measurement, two different devices, which is correct. |
| `real_motion_ready`, `ble_link_live`, `blade_safe_for_motion`, `position_valid_for_motion` | regex artefact — these are `gate_key=`, not `key=`. |
| 19 of 150 entities reading `unknown`/`unavailable` | mostly `last_*` diagnostics that are legitimately unknown until their event occurs, plus the four `emergency_nudge_*` buttons that are unavailable **by design** (`_nudge_available` returns `False`). Already documented in the deploy runbook as known-benign. |

---

# Open item 1 — two switches share a meaningless entity id

    switch.back_yard_clip_skywalker      "Clip Skywalker Device 4G"
    switch.back_yard_clip_skywalker_2    "Clip Skywalker Device Wi-Fi"

Two *different* functions, and neither id says which. The `_2` also reads as a
duplicate at a glance, which is how this audit started.

**The code is correct.** The entity registry shows distinct, well-formed
unique ids and names:

    unique_id: Luba-VSPLV397_device_4g_enabled     original_name: Device 4G
    unique_id: Luba-VSPLV397_device_wifi_enabled   original_name: Device Wi-Fi

`switch.py:248-265` gives both proper keys (`device_wifi_enabled`,
`device_4g_enabled`) and translations are present in **all twelve** locales.

**Cause: stale registry entries.** The ids were assigned at first registration
when the entities had no resolvable name, so HA fell back to the device slug and
appended `_2` to the collision — and **HA never renames an entity id after
creation**, so fixing the code afterwards could not fix the ids.

**Fix:** operator-side only. Settings → Devices → the entity → rename. Safe,
because `unique_id` is what binds registry to code. No code change achieves this.

⚠️ Renaming changes the entity id, which breaks any automation, dashboard card or
template referencing the old one, and starts fresh history.

**Severity:** low. Confusing, not wrong.

---

# Open item 2 — `Area Area 1`

    switch.back_yard_clip_skywalker_area_area_1   "Clip Skywalker Area Area 1"

The `area` entity key and the area's own name (`Area 1`) concatenate, doubling
the word in both the id and the friendly name.

**Fix:** drop the redundant word from the per-area switch name construction.
Touches a user-visible name and the entity id for every area switch.

**Severity:** cosmetic. Lowest value of the three; non-zero churn.

---

# Open item 3 — `blade_height` is a sensor AND a number, identically named

    sensor.back_yard_clip_skywalker_blade_height  1.96850393700787 in
    number.back_yard_clip_skywalker_blade_height  2.0              in

Both are named **"Clip Skywalker Blade height"**, both `device_class: distance`,
both in inches — and **they disagree**. The sensor reports the actual reported
knife height (50 mm); the number is the setpoint (50.8 mm).

In the UI that is two identically-named entities showing different values with
nothing to tell them apart. Of the three, this is the one that is a genuine
defect rather than historical debt: a read-only reported value and a settable
setpoint are different quantities and should not share a name.

**Fix options:**

- rename the sensor to something like "Blade height (reported)" — a translation
  change across **twelve** locale files plus `strings.json`, and a user-visible
  rename; or
- drop the sensor entirely, since an HA `number` already exposes its current
  value — but that removes an entity people may have in dashboards or history.

⚠️ Needs an explicit decision before either. The rename is the smaller change and
keeps both quantities visible.

**Severity:** the highest of the three, because it is actively misleading rather
than merely ugly.

---

## Why none of this was changed

Every fix here renames or removes a user-visible entity, which breaks
automations, dashboard cards, templates and history for anyone using them. That
is an operator decision, not a drive-by. Logged and left.
