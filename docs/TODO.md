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

**Status 2026-09-17:** mechanism confirmed from HA's recorder (blade 60 → 25 mm
within 20 s of HA's speed write) and fixed on branch
`fix/operation-settings-running-job-sync`, **not deployed**; hardware check (done
item 3) not run. Record: `docs/findings-operation-settings-sync-20260917.md`.

**Until fixed:** don't change mowing settings in HA while a job started from the
app is running; change them in the app.
