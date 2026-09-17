# Findings: the mowing-settings fix, verified on hardware (2026-09-17)

**PASS.** Changing Working speed during an app-started mow changed **only** the
speed. The blade held the app's height for 13 minutes across 157 consecutive
samples.

This closes item 3 of `docs/TODO.md`'s "done means". The defect, the code read
and the offline tests are in
`docs/findings-operation-settings-sync-20260917.md`; this file is the hardware
run only.

- **Build under test:** `0.6.4-beta111`, deployed 2026-09-17 21:30Z (fix commits
  `dd49e3b6`, merge `39f77483`). The host previously ran beta110, which has the
  defect.
- **Evidence:** `docs/evidence-operation-settings-hardware-20260917/entity_samples.jsonl`
  — 170 samples at ~5 s, read-only HA API GETs, 22:21:49Z–22:36:05Z.
- **Operator actions:** started the mow from the Mammotion app and made the one
  HA change. No command was sent from this session at any point.

---

## 1. Setup — every value differed from HA's plan

The mow was started from the app on **Backyard Right** at **2.2″ blade** and
**0.7 ft/s**. HA's own plan held different values for all three settings, which
is the condition that produced the 02:34Z incident.

| | Running job (device reports) | HA's plan (what beta110 would have sent) |
| --- | --- | --- |
| Blade height | **55 mm** (2.165″) | 1.0 in = **25 mm** |
| Working speed | 0.19–0.21 m/s (0.7 ft/s) | 1.61 ft/s = **0.49 m/s** |
| Path spacing | (not exposed pre-change) | 8.0 in = **20 cm** |

At 22:21:49Z, five minutes into the mow, all three entities read
`value_source: next_job_plan` — HA had not read the job. **That is expected and
is a known gap** (§4).

Conditions: RTK `fix`, `ble_link_live: on` at −78 dBm, battery 70 %, progress
6 %, `mode_working`.

## 2. The change

At **22:22:55Z** the operator changed **Working speed 1.61 → 1.2 ft/s** in HA.
Nothing else was touched.

## 3. Result — PASS on both properties

| | Before | After |
| --- | --- | --- |
| **Blade height (device)** | 55 mm | **55 mm — 157/157 samples, 22:22:55→22:36:05Z** |
| Ground speed (straight-line) | 0.16–0.21 m/s | **median 0.350, max 0.370 m/s** (1.2 ft/s = 0.366) |
| HA Working speed | 1.61 ft/s | 1.21 |
| HA Blade height | 1.0 in (HA's plan) | **2.0 in — the job's 55 mm** |
| HA Path spacing | 8.0 in (HA's plan) | **13.0 in — the job's real spacing** |
| `value_source` | `next_job_plan` | `running_job_after_ha_change` |

**Property 1 — the entities show the job's values.** Blade height and path
spacing corrected themselves from HA's stale plan to the job's real values the
moment HA read the job, and the attribute says which is being shown.

**Property 2 — one change changes one setting.** The blade never moved. On
beta110 the same action drove it 60 → 35 → 25 mm within 20 s
(`findings-operation-settings-sync-20260917.md` §2).

🔑 **Two things previously listed as unverified are now established:**
1. The `query_generate_route_information` (sub_cmd=2) round-trip **works on a
   real link at −78 dBm mid-mow**, and
2. its reply **carries usable blade height, speed and channel width** — the fix
   seeds from them and refuses if any is zero, and it did not refuse.

## 4. Limits of this run

- ⚠️ **HA still does not read an app-started job on its own.** For the first
  five minutes the entities showed HA's plan, labelled `next_job_plan`. The app
  queries the route whenever the mower enters working state
  (`HomeMapFragment.java:1221-1225`); HA does not. **Operator-reported the same
  evening.** Fix proposed, not built: query once per job on the transition into
  working, keyed on `path_hash`, best-effort and silent on failure.
- ⚠️ **The fail-closed path was never exercised** — the query succeeded, so
  `running_job_unreadable` has not been seen on hardware.
- ⚠️ **Blade height was not changed mid-job on hardware.** It runs the identical
  code path with a different field name and is covered offline. Not tested here
  because the inch-step issue below leaves no safe in-range value to pick.
- 🚨 **The blade-height slider steps 1 inch in US units**, so the only reachable
  values are 1.0 in (25 mm, scalps), 2.0 in (current) and 3.0 in — and 3.0 in
  converts to **76 mm, above the device maximum of ~70 mm**. HA validates
  against the *display* range and converts afterwards, and the clamp added in
  beta111 runs only at startup and restore, not on a user write. **Unfixed.**
- **Path spacing is still never sent to a running job**, by design. A mid-job
  change to it now visibly snaps back to the job's value on the next update,
  which reads as "ignored". A UX question, not a safety one.

## 5. Live state at the end

Mow continued normally and was later cancelled by the operator. `ble_link_live`
read `off` in the final samples while the mow ran on — the documented pattern
where HA keeps a stale allocation. All 157 hold samples predate it.
