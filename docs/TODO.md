# TODO

Open items the operator has asked to track. Newest first. Each entry says what
was observed, what is known about the cause, and what "done" means.

## Mowing settings in HA show HA's own values, not the mower's — and changing one mid-mow pushes the stale rest

**Reported by the operator, 2026-09-17 ~02:50Z, during an app-started mow.**

**Observed:**
- The mow was started from the Mammotion app at **blade height 2.2″** and
  **working speed 2 ft/s**. HA's *Working speed*, *Blade height* and *Path
  spacing* entities did not show those values.
- The operator changed **Working speed 2 → 1.6** in HA mid-mow. The mower's
  **blade height then changed to 1″** — the value HA's entity was showing, not
  one anyone chose.

**Known cause (read from the code 2026-09-17, not yet tested on hardware):**
- All three entities read and write `coordinator.operation_settings`, a local
  `OperationSettings()` created once in `MammotionBaseUpdateCoordinator.__init__`
  (`custom_components/mammotion/coordinator.py`) with pymammotion's defaults
  (`speed 0.3`, `channel_width 25`, `blade_height 0`). **Nothing ever fills it
  from the device**, so the entities show HA-side defaults or the last value
  typed in HA — never the running job's settings.
- `blade_height` and `working_speed` call `async_modify_plan_if_mowing()`, which
  sends the **whole** local `operation_settings` to the running job via
  `async_modify_plan_route`. So changing one setting re-sends every other stale
  local value — which is how the speed change also set the blade height to 1″
  (`blade_height`'s slider floor is 25 mm ≈ 1″).

**Why it matters:** a blades-on side effect. Touching one mowing setting in HA
can silently change the cut height (and path spacing) of a job started
elsewhere.

**Done means:**
1. The entities show the mower's actual job settings (read from the device's
   reported plan/work state), or are clearly marked "value HA will send" rather
   than the live value.
2. Changing one setting mid-mow changes **only that setting** on the running job;
   the others come from the job itself, not from local defaults.
3. Verified on hardware: start a mow from the app, change speed in HA, confirm
   the blade height is unchanged at the mower.

**Until fixed:** don't change mowing settings in HA while a job started from the
app is running; change them in the app.

### Status 2026-09-17 — traced offline, fixed on a branch, NOT yet verified on hardware

Full per-field evidence: `docs/findings-operation-settings-sync-20260917.md`.
Branch `claude/mammotion-operation-settings-sync-slr559`. **Not deployed.**

- ✅ All three of the previous session's code claims **confirm**, with one
  correction: `async_modify_plan_route` was never naive — it already re-seeded
  **8** fields from the running job (`coordinator.py:2004-2011`). The three it
  omitted are exactly the three the operator can edit. 🔑 **An omission, not a
  design error.**
- 🔑 **The read path already existed on both sides and nobody was using it.**
  `CurrentTaskSettings` carries `knife_height` / `channel_width` / `speed`
  (`pymammotion/data/model/work.py:18,19,23`), and the pinned
  `pymammotion 0.8.12.post4` already ships
  `query_generate_route_information` (`NavReqCoverPath sub_cmd=2`,
  `navigation.py:585`). This integration sent it from **one** place only — the
  breakpoint resume in `lawn_mower.py:306,312` — so **a job started in the app
  was never described to HA at all.**
- 🔑 **Blade height is LIVE in the ~1 Hz report** (`RptWork` field 20 →
  `report_data.work.knife_height`). **Speed and path spacing are not** — they
  can only come from `sub_cmd=2`. 🚨 `man_run_speed` and `cutter_width` are
  decoys: manual-drive speed and physical deck width, not the job's settings.
- ✅ **Upstream `mikey0000/Mammotion-HA` @ `d892896` fixed the WRITE half** and
  the shape was ported (seed-then-override, per-field entry points), minus
  `work.auto_change_direction` which does not exist in our pinned pymammotion.
  🚨 **Upstream did NOT fix the read half** and never sends `sub_cmd=2`, so its
  seed can come from a stale `work` record left by a previous job in the same
  zone. **The port adds a refresh-then-verify before seeding, and sends nothing
  if the device does not answer.**
- 🚨 **A second, unreported half of the incident:** the same message pushed
  `channel_width = 20` (HA's spacing floor). The operator only saw the blade.
- 🚨 **`path_spacing` had no `set_async_fn` at all**, so a mid-mow spacing
  change silently never reached the device. Now wired.
- ⚠️ **Units:** 2 ft/s = 0.6096 m/s, above `working_speed`'s 0.6 m/s maximum
  (`number.py:200`). If the static bounds are in force, **HA cannot represent
  the speed the mow was started at.** Whether they are depends on the runtime
  `DeviceLimits` and **was not established** — no HA access from that session.
- **Done means 1 and 2 are covered by 17 tests**
  (`tests/components/mammotion/test_operation_settings_sync.py`); the reported
  incident reproduces on the pre-fix tree as `assert 25 == 56`.
  🛑 **"Done means 3" (hardware) is still OPEN.** The test plan is written down
  in §8 of the findings doc and **deliberately not run** — it needs a separate
  explicit operator go. 🔑 **Its one load-bearing rule: pick a blade height and
  speed that are not slider minimums**, or a correct read is indistinguishable
  from a default.
