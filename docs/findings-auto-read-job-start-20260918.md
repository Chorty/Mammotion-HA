# Findings: HA reads the running job by itself (2026-09-18)

**The auto-read works, and testing it on hardware found two real bugs in the
same evening's code — one of them caught by the operator, not by me.**

- **Build under test:** `0.6.4-beta112`, deployed 2026-09-18 01:52Z.
- **Evidence:** `docs/evidence-auto-read-job-start-20260918/entity_samples.jsonl`
  — 400 samples at ~5 s, read-only HA API GETs, 02:21:20Z–02:47Z.
- **Operator actions:** started the mows from the Mammotion app and made the
  HA changes. Nothing was sent from this session; the motion gate was not armed.
- Background: `docs/findings-operation-settings-sync-20260917.md` (the original
  defect), `docs/findings-operation-settings-hardware-20260917.md` (beta111).

---

## 1. PASS — an app-started job is read without anyone asking

At 02:21 the entities held HA's plan and the device reported 60 mm. The mow
started from the app at 02:27:25. **At 02:28:36, with no control touched:**

| | 02:28:31 | 02:28:36 |
| --- | --- | --- |
| Working speed | 1.21 ft/s (HA's plan) | **1.6** |
| Path spacing | 8.0 in (HA's plan) | **13.0 in** |
| `value_source` | `next_job_plan` | **`running_job`** |

`running_job`, not `running_job_after_ha_change`: HA queried the route itself,
at −76 dBm, about 70 s after the job began. This is what the app does on entry
to working state; HA now matches it.

⚠️ **Blade height could not discriminate on this run.** The operator's job was
at 1″ and HA's plan was 25 mm, so both read 1.0 in. Speed and spacing carry the
result. 🚨 **This cost a false alarm:** the device dropping 60 → 35 → 25 mm at
02:27:50 as the job began looked exactly like the beta110 defect re-appearing,
and it was the job's own setting. **A test where the job and the plan share a
value cannot tell "read the job" from "pushed the plan".**

## 2. PASS — a mid-job blade-height change, on hardware for the first time

| Time | Device blade | HA blade | HA speed | HA spacing |
| --- | --- | --- | --- | --- |
| 02:28:36 | 25 mm | 1.0 in | 1.6 | 13.0 |
| 02:30:23 | 25 mm | **2.0 in** (operator) | 1.6 | 13.0 |
| 02:30:33 | **50 mm** | 2.0 in | **1.6 kept** | **13.0 kept** |

The blade moved to the requested 2″ ten seconds later, and the job's speed and
spacing survived. Speed was then changed to 0.7 ft/s at 02:31:42 and **the blade
held 50 mm** while ground speed fell 0.490 → 0.190 m/s (0.7 ft/s = 0.213). Both
directions of the "change one setting, change only that setting" property now
hold on hardware.

## 3. PASS — the fail-closed path fired for real

An earlier attempt at the speed change, at **02:30:08Z on a −82 dBm link**,
could not read the job:

```
22:30:08 ERROR ... Error during service call to number.set_value:
Could not read the running job's settings from the mower, so the change was
not sent to the job. It will be used for the next job HA starts
```

(HA log timestamps are local EDT.) **Nothing was sent** — the device stayed at
25 mm and 1.6 ft/s — and the operator got the translated message. No traceback
from this integration; the surrounding ones in the log are `icloud3`. The
retry at −66 dBm 90 s later succeeded. This path had only ever been asserted in
tests.

⚠️ **UX gap, unfixed:** the refused value *is* kept in HA's plan, but the entity
keeps displaying the running job's value, so the slider appears to snap back and
the toast is the only signal. A refused change should probably show what was
picked, marked as not applied.

## 4. 🚨 BUG FOUND BY THE OPERATOR — a second mow showed the first one's settings

> "I started a mow on the app but the HA sensors do not reflect what the app
> started at."

Reproduced from the sample log:

| Time | Mode | Device blade | HA blade | `value_source` |
| --- | --- | --- | --- | --- |
| 02:34:01 | mode_ready | 50 mm | 2.0 in | `next_job_plan` ← correct |
| 02:34:50 | **mode_working** (new job) | 50 mm | 2.0 in | **`running_job_after_ha_change`** |
| 02:35:15 | mode_working | **25 mm** | 2.0 in | unchanged, stale |

**Cause:** the snapshot is keyed on `report_data.work.path_hash`, which
identifies the **route, not the run**. Mowing the same area again reuses the
route, so the hash matches. `running_job_settings()` only *hid* the snapshot
while the mower was idle and never cleared it, so the previous job's values —
and its `after_ha_change` label — reappeared for the next job.

**Fixed in `8d1f01c1`:** the snapshot and the read-attempt state are discarded
on the transition out of an active job, so the next job forces a fresh read. A
pause is not leaving the job, so resuming still costs no extra query. Shipped in
beta113.

🔑 **The lesson for the next test:** this only shows up on a *second* job. The
first job of a session always reads correctly, which is why the offline tests
and the first hardware run both passed. **Test the second mow, not just the
first.**

## 5. 🚨 BUG FOUND BY THE OPERATOR — the blade slider offered only 0/1/2/3 inches

> "blade height should be 1" to 2.8" but you can only select 0, 1, 2 or 3 on the
> slider"

Two independent HA behaviours, both in `number/__init__.py`:

- **The step is never unit-converted** (`_calculate_step`), so this entity's
  1 mm step read as **1 inch**.
- **The displayed min/max take their precision from the decimal count of the
  native value's string** (`_convert_to_state_value`), so an integer 25 mm
  floored to **0.0 in**. For the same reason a 60 mm job read as a flat 2.0 in.

**Fixed in `953bc297`**, shipped in beta113 and verified on the host:

| | before | after |
| --- | --- | --- |
| Blade height | 0.0–3.0 in, step 1 | **0.9–2.8 in, step 0.1** |
| Path spacing | 7–14 in, step 1 | **7.8–13.8 in, step 0.1** |

⚠️ **The displayed range is rounded outward** (2.8 in is 71.1 mm against a 70 mm
limit), so writes are now clamped to the model's native limits. That also closes
the hazard where the top slider position sent **76 mm**. Metric is unchanged at
25–70 mm in 1 mm steps. ⚠️ 0.1 in is 2.5 mm while the device is only observed to
move in 5 mm steps, so adjacent slider positions can land on the same height.

## 6. Not established

1. **Whether a change made in the app mid-job updates HA.** The read fires once
   per job; it re-reads only when the job ends or the route changes. Untested.
2. **Whether an app-side settings change alters `path_hash`.** If it does, HA
   re-reads; if not, it will not.
3. **Blade height's own auto-read discrimination** (§1) — the job and the plan
   shared a value.
4. **The backoff path** (3 attempts, 60 s) never ran to exhaustion; the single
   failure here was operator-initiated, not the automatic read.
