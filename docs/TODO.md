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

**✅ RESOLVED 2026-09-17.** Fixed, deployed as **beta111**, and **verified on
hardware the same evening**: during an app-started mow (blade 55 mm, 0.7 ft/s) the
operator changed Working speed 1.61 → 1.2 ft/s; the blade held **55 mm across 157
consecutive samples over 13 min** while the ground speed moved to the new value,
and all three entities switched from HA's plan to the job's real values. All three
"done means" items are met. Records:
`docs/findings-operation-settings-sync-20260917.md` (cause + fix),
`docs/findings-operation-settings-hardware-20260917.md` (the hardware run).

**Still open, smaller, tracked here:**
- HA does not read an app-started job on its own, so the entities show HA's plan
  (labelled `next_job_plan`) until the first change in HA. The app queries the
  route on entry to working; HA should too.
- 🚨 The blade-height slider steps **1 inch** in US units, and its top position
  (3.0 in) converts to **76 mm, above the device max ~70 mm**. HA validates the
  display value and converts afterwards; the beta111 clamp runs only at startup
  and restore, not on a user write.
- `lawn_mower.start_mow` with `modify: true` still fills unspecified fields from
  HA's plan — the same stale-fill hazard, on a different path.
- The fail-closed path (`running_job_unreadable`) has never been seen on hardware.

**Until fixed:** don't change mowing settings in HA while a job started from the
app is running; change them in the app.
