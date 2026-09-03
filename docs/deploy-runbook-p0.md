# P0 deploy and rollback runbook (host 192.168.1.106)

First deployed 2026-07-29 and updated through the 2026-08-14 corrected
working-tree deploy after the beta54 Night Go release. For
any later deploy, work while the mower is stopped and experimental motion is
off. Restarting HA while BLE is unhealthy has previously left the integration
in `setup_error` with no auto-retry, needing a manual entry reload.

## What the host is running now

### beta98 -> beta99 — 2026-09-03 19:06-19:10 EDT, motion-disabled — the guard drain + the other half of the overshoot

Ships the two defects found by the beta98 adversarial review (5/5 agents, the
first clean workflow run of that session):

1. 🚨 **The travel guard had gone SOFT.** The in-window sampler took a single
   `get_nowait()` per poll, so cumulative distance advanced at the SAMPLER's
   cadence rather than the position feed's, with no catch-up path. At a
   schema-legal `sample_interval_ms: 1000` a 23 s window can publish ~32 payloads
   and consume ~23 — the guard would not reach 4.5 m until the mower had driven
   **~6.5-6.9 m**. It now DRAINS, bounded by `_PROBE_MAX_DRAIN_PER_POLL = 64`.
   ⚠️ The first attempt at that drain was an unbounded `while True`, which hung
   the test suite: a mocked queue never raises `QueueEmpty`. The regression test
   pins the BOUNDED form specifically.
2. **Containment's clock branch omitted the stop overshoot.** The mandatory stop
   fires after the window ends on either branch, so the measured 0.4544 m
   post-stop creep sat outside the corridor on exactly the branch that exists for
   the guard-no-op case. Both branches now carry it.

| check | result |
| --- | --- |
| files byte-identical | **48/48** |
| card md5, both paths + local | `e8e47948` |
| archive SHA-256 local == host | `66621a3dd1cc4573e438a9f0e30dc5926f9d17f9d5571defc2fb111a140d7c3c` |
| AppleDouble `._*` files | 0 |
| Lovelace resource | `?v=0.6.4-beta99&build=e8e47948` |
| API back / entities | 31 s / 133 Mammotion entities |
| gate | `enabled: false` |

🔑 **Discriminating check** (the FIX, not the version): a dry run at linear 400 /
23 000 ms now reports `clock_bound_m 7.40`, `required_radius_m 7.40`,
`bound_that_binds: "clock"` — beta98 gave **6.90**, i.e. the corridor was short
by the full post-stop creep. `would_send: false`.

🚨 **Not browser-verified.** Ask the operator to confirm the card footer reads
`0.6.4-beta99`. Backup: `/config/mammotion-backup-20260903-1906-pre-beta99.tgz`.

⚠️ **BLE dropped over the restart and did NOT return**: `ble_rssi 0` (the
documented dozed-mower signature after ~5 h idle off-dock) with
`master_bedroom_proxy` `unavailable`. No motion was attempted. **This deploy is
verified by dry run only; nothing has exercised the drain on hardware.**

### beta97 -> beta98 — 2026-09-03 15:59-16:07 EDT, motion-disabled — the containment gap

🚨 **Ships a real containment defect fix.** `step_path_contained` sized the
corridor as `max_travel_m + 0.50`, which **assumes the travel guard works**.
This project has a documented mode where it silently does not: position payloads
keep arriving with an advancing sequence and a fresh timestamp while x/y stay
latched (2026-08-28, 21 bit-identical samples across 0.4375 m of real travel).
Then `cumulative_distance_m` stays ~0, nothing trips, and the window runs to the
**wall clock**. `raw_pymammotion_motion_probe` was corrected for exactly this on
2026-08-23; the step probe was missed and had its window raised four times since.

Also carries `_PROBE_SPEED_PER_LINEAR_UNIT_MS` **7.0e-04 → 7.5e-04** — the old
value was fitted to ramp-inclusive averages and sat **6% below** the measured
sustained speed, which is the unsafe direction for a constant that sizes corridor
clearance. **No safety bound was relaxed: `max_travel_m` stays 4.5.**

| check | result |
| --- | --- |
| files byte-identical | **48/48** |
| card md5, both serving paths + local | `f0dc1602` |
| archive SHA-256 local == host | `693a533483709894ce8b44bbc99d9da8e46159a8bfb78af2ad1845f9c031c93e` |
| AppleDouble `._*` files | 0 |
| manifest on host | `0.6.4-beta98` |
| backend in container | `pymammotion 0.8.12.post4` |
| Lovelace resource | `?v=0.6.4-beta98&build=f0dc1602` |
| API back / entities | 46 s / 133 Mammotion entities |
| gate | `enabled: false` in live API **and** RAW `core.config_entries` |

🔑 **The discriminating check — proving the FIX, not the version string.** A dry
run at linear 400 / 23 000 ms / `max_travel_m` 4.5 now reports:

```
required_radius_m      6.9
travel_budget_bound_m  5.0
clock_bound_m          6.9
bound_that_binds       clock
```

beta97 reported **no clock bound at all** and would have accepted a corridor
holding only 5.00 m for a path that can reach 6.90 m. `would_send: false`;
nothing dispatched.

🚨 **Not browser-verified.** Ask the operator to confirm the card footer reads
`0.6.4-beta98`. Backup: `/config/mammotion-backup-20260903-1559-pre-beta98.tgz`.

### beta96 -> beta97 — 2026-09-02 21:19-21:26 EDT, motion-disabled — the beta96 corrections

🚨 **Read the beta96 entry below first: beta96 was NEVER fully deployed, and it
still reached the running process.** Its files were extracted to `/config` on
2026-09-01 and the restart was deliberately interrupted to run an adversarial
review. HA then restarted on its own overnight and loaded them. Confirmed by a
schema-only dry run on 2026-09-02: `step_ms=15000` was accepted (beta95 caps it
at 7000) while `travel_projection` came back `null` and the Lovelace key was
still `?v=0.6.4-beta95`. **A staged-but-unfinished deploy will eventually deploy
itself — finish it or back it out, never leave it.**

**beta97 ships the corrections to beta96**, both found by adversarial review:
`_STEP_RESPONSE_MIN_SPEED_BY_LINEAR[400]` **0.24 -> 0.17** (0.24 came from the
single FASTEST banked run while its comment called it the slowest, so it sat above
four of five and **over-refused**), the new non-blocking `travel_projection`
diagnostic, the three-way speed-figure contradiction across
`services.yaml`/`strings.json`/the code, quoted `select` defaults, and the
`services.yaml`↔`strings.json` parity test this service lacked while every sibling
had one. **No safety bound moves: `max_travel_m` stays 4.5.**

Verification tail, all measured:

| check | result |
| --- | --- |
| files byte-identical | **48/48** |
| card md5, both serving paths + local | `98ed5bbe` |
| archive SHA-256 local == host | `4cf607fd6d57568687a171e5aa0c5f897215a34dc73a8f9d560feff55780d71e` |
| AppleDouble `._*` files | 0 |
| manifest on host | `0.6.4-beta97` |
| backend in container | `pymammotion 0.8.12.post4` |
| Lovelace resource | `?v=0.6.4-beta97&build=98ed5bbe` (was still **beta95**) |
| API back / entities | 70 s / 132 Mammotion entities |
| gate | `enabled: false`, `real_motion_allowed: false`, no session |

🔑 **The deploy was proved by a DISCRIMINATING dry run, not by reading a version
string**: a replay of banked route-1 run 1 (`3000/5000/5000`, linear 400) at
`max_travel_m` 3.0 — a config the mower has completed twice at 2.71 m and 2.77 m —
is **accepted** on beta97 where beta96's 0.24 bound refused it, and
`travel_projection` returns `floor_speed_m_s: 0.17`. `would_send: false`; nothing
was dispatched.

🚨 **Not browser-verified.** Ask the operator to confirm the card footer reads
`0.6.4-beta97`. Backup: `/config/mammotion-backup-20260902-2119-pre-beta97.tgz`.

### beta94 -> beta95 — 2026-09-01 15:26-15:45 EDT, motion-disabled — E-VIO scoring for the step-response probe

Ships the operator-adopted E-VIO scoring rule
(`docs/findings-rtk-vio-course-rate-scoring-20260831.md`, predeclared in
`docs/predeclared-rtk-vio-course-rate-scoring-20260831.md`): the probe now
emits `vio_analysis` scoring 2a via half-phase mean-rate agreement and 2b via
last-two settle rates, both on VIO heading; omega/tau come from the same
channel and tau exists only when 2a passes; dark VIO (`vio_state != 2` on any
sample) refuses to score with `vio_not_live_throughout`. The RTK
`course_series`/`analysis` stay emitted unchanged as diagnostics. All four
banked route-1 runs are pinned as fixtures, including the two verdict flips
(SX 2a PASS→FAIL, tau=2.038s demoted; R2 2a FAIL→PASS, VIO tau 0.80 s, n=1).
Also folds in the stale `1.06 m` → `1.34 m` blind-disk prose fix in
`strings.json`/`en.json`, owed since beta82. Backend pin unchanged (post4).

| | beta95 |
| --- | --- |
| tag | `v0.6.4-beta95` (release commit `38cadc37`) |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta95`, uv.lock `0.6.4b95` |
| card md5 | `32818ee19161107f9f5112b0a3bdbedf`, equal at both serving paths and local |
| Lovelace resource | re-read as `?v=0.6.4-beta95&build=32818ee1` |
| backend | PyMammotion `0.8.12.post4`, read from inside the container (unchanged) |
| archive | SHA-256 `4245b3433a00b41b62c9a788b3366999e4aca658a8c1cb52653089c326e58f55`, identical local and host, 0 `._*` entries |
| file verification | 48 of 48 byte-identical |
| restart | API up 31 s, 132 mammotion entities at 153 s |
| gate before/after | `enabled: false`, `real_motion_allowed: false`, no active session; RAW `core.config_entries` reads `"enable_experimental_motion":false` |
| gate suite | 995 pytest, 91/91 frontend, ruff + format + mypy clean, 0 failed hooks |
| backup | `/config/mammotion-backup-20260901-1526-pre-beta95.tgz` |

✅ **Dry run on the deployed bytes**: `raw_pymammotion_step_response_probe`
with `baseline 3000 / step 7000 / settle 5000, max_travel_m 4.5` returned
`would_send: false`, `command_result.attempted: false`, 15/15 gates, phases
echoed. ⚠️ **The E-VIO scoring is deployed but UNEXERCISED on hardware** —
`dry_run` returns before any samples are captured, so `vio_analysis` never
populates on a dry run (same caveat as beta84's sequence instrumentation).
Byte parity plus the four banked-run regression tests are the verification;
the first real, separately-authorized daylight step-response run will be the
first hardware exercise. A night run is UNSCOREABLE under this rule on
purpose. ✅ **Browser-verified by the operator, 2026-09-01: the card reads
`0.6.4-beta95`.** The deploy is fully verified end to end.

### beta93 -> beta94 — 2026-08-31, motion-disabled — BACKEND CHANGE: post4 fixes the blank-credential login outage

The only functional change is the pymammotion pin: `chorty-0.8.12.post3` →
`chorty-0.8.12.post4`. Root cause of the 2026-08-31 "Client id or secret
error" outage on BOTH accounts: every Chorty fork wheel (post1–post3) shipped
`MAMMOTION_OAUTH2_CLIENT_ID`/`_SECRET` as empty strings, because upstream
blanks them in source and injects them at build time from GitHub secrets,
which do not propagate to forks. `login_v2`/`refresh_token_v2` signed every
account's request with a blank secret, which the server rejects
deterministically. **Not a rate limit** — the wait-24h advice is withdrawn.
post4 is the same source tree rebuilt after `scripts/update_credentials.py`
with the values recovered from the upstream PyPI `pymammotion==0.8.12` wheel
(wheel SHA-256 `61d8a6f6eae067034ee7aa4159e0f5f9d755f85ea6d6a5d0dfa1c5af5cdb880a`).

✅ **The decisive readback**: inside the container after restart,
`pymammotion.const` reports credential lengths **15 / 30 / 8 / 32** — all
non-empty — where post3 read 0 / 0 for the OAuth2 pair. Backend version reads
`0.8.12.post4` from inside the container.

| | beta94 |
| --- | --- |
| tag | `v0.6.4-beta94` |
| card md5 | `cba762a58bb775a6d3d86c22884fce2b`, equal at both serving paths and local |
| Lovelace resource | re-read as `?v=0.6.4-beta94&build=cba762a5` |
| backend | PyMammotion `0.8.12.post4`, read from inside the container; OAuth2 creds non-empty (15/30 chars) |
| archive | SHA-256 `e134d658257e56fd9cc1b8c66204e9867049b069e0ff309a25c02ccf9dec0eda`, identical local and host, 0 `._*` entries |
| file verification | 48 of 48 byte-identical |
| restart | API up 51 s, 132 mammotion entities at 156 s |
| gate after | `enabled: false`, `real_motion_allowed: false`, no active session |
| backup | `/config/mammotion-backup-20260831-1826-pre-beta94.tgz` |

✅ **The login fix is EXERCISED AND CONFIRMED** (≈ 19:00 EDT, same day): the
operator logged in through post4 — `pymammotion.client` logged both `Aliyun
device registered: RTKBNA235279309` and `Mammotion device registered:
Luba-VSPLV397`, all entity platforms set up, `mqtt_status: reported_online`,
`ble_link_live: on`, 143 entities / 116 available, and zero `Client id or
secret error` lines since the deploy (all remaining occurrences predate
18:26). Known-benign: `last_cloud_login_success` reads `unknown` (the login
preceded sensor registration); one `mow_path_fetch` saga timeout at 19:44 is
a device no-response, not auth. ✅ **Browser-verified by the operator, same
evening: the card footer reads `card v0.6.4-beta94`.** The deploy is fully
verified end to end.

### beta92 -> beta93 — 2026-08-31, motion-disabled

Cosmetic only: the config-flow "wifi" step's title said "Connect to Wi-Fi"
even though it asks for the Mammotion account email/password (the
description text underneath was already correct). Confirmed inherited
verbatim from mikey0000/Mammotion-HA upstream — their own `strings.json` has
the identical mismatched title. Retitled to "Mammotion Account" in
`strings.json` and all 12 `translations/*.json` files, each in its own
language. No behavior change; `step_id="wifi"` (an internal identifier) is
untouched.

✅ **Also serves as a second confirmation that beta88→beta92's coordinator
fixes hold on a routine restart, not just the incident restart they were
built for.** Same graceful-degradation pattern observed again in the logs
(`fetch_rtk_lora_info`/`fetch_rtk_properties` auth failures logged as
warnings, zero crashes) and the config entry reached `loaded` again.

| | beta93 |
| --- | --- |
| tag | `v0.6.4-beta93` |
| card md5 | `f288a1abbd5ab1453156a02c124ac24c`, equal at both serving paths |
| Lovelace resource | re-read as `?v=0.6.4-beta93&build=f288a1ab` |
| backend | PyMammotion `0.8.12.post3` (unchanged) |
| file verification | 48 of 48 byte-identical |
| gate after | `enabled: false`, `real_motion_allowed: false`, verified live API **and** RAW `[false]` |

⚠️ Entities were still `unavailable` a few minutes post-restart, same as the
beta92 restart — BLE just needed more time to reconnect; not treated as a
regression given the actual fix under test (the entry staying `loaded`
without crashing) is confirmed. `matt.joslin@me.com`'s cloud token is still
rate-limited; see beta92's entry below for the full incident record.

### beta88 -> beta92 — 2026-08-31, motion-disabled — account rate-limit incident

**Context:** `matt.joslin@me.com`'s cloud refresh token was rejected by
Mammotion's own servers after heavy failed-login volume during same-day
diagnosis (see `docs/vio-crosscheck-...` era work, unrelated) — confirmed via
the app logging into both `matt.joslin@me.com` and `thejoslincrew@gmail.com`
successfully while the API kept rejecting both. Matches
mikey0000/PyMammotion#134's maintainer response to an identical symptom:
"mammotion will unblock the account eventually, turn off the integration for
24 hours." **Not a code bug — the fix is to stop retrying and wait it out.**

Shipped alongside, unrelated to the incident itself:
- **`CloudConnectivityMonitor`** (`connectivity.py`, new file): watchdog that
  reconnects a stuck-but-registered cloud transport in place, rate-limited to
  once per 15 min, and warns once (rather than silently dropping every send)
  when a transport is detached entirely. Ported the connectivity half of
  mikey0000/Mammotion-HA commit `f4428d47`; the Spino pool-cleaner half was
  skipped (not a registered device type here).
- **Diagnostic logging** in all three `config_flow.py` login paths
  (setup/reconfigure/reauth): every previously-silent failure branch
  (rate-limited, credential rejected, cloud setup error, a "successful" call
  that returns no `login_info`) now logs the account and exception type.

**What the incident itself exposed and fixed, beta89→beta92 in one evening —
each restart under total cloud outage surfaced the next uncaught exception in
a one-time device-info read that only cloud usually serves, normally masked
by BLE or cloud individually working:**

| build | fixed |
| --- | --- |
| beta89 | (baseline: caps + connectivity watchdog, not yet incident-related) |
| beta90 | `MammotionDeviceVersionUpdateCoordinator`'s OTA check (`_async_setup` + `_async_update_data`) raised a bare `AuthError` with nothing to catch it |
| beta91 | `MammotionRTKCoordinator._async_setup`'s `fetch_rtk_lora_info`/`fetch_rtk_properties`/`get_device_status` block, uncaught — actually raised `FailedRequestException`, not `AuthError` |
| beta92 | `MammotionRTKCoordinator._async_update_data`'s OTA check only caught the narrower `ReLoginRequiredError` (a subclass), not the bare `AuthError` actually raised; and the mower's own `_async_setup` command loop hit a raw `AttributeError` from pymammotion's BLE `_write_payload` when its transport object was caught mid-disconnect — normally masked by falling back to MQTT, which was also fully dead tonight |

🔑 **Every fix follows the same principle:** `has_cloud_account` only means
credentials are configured, not that the session is currently live. A dead
token must never block core BLE functionality over an optional,
best-effort HTTP read — matching the philosophy the sibling `checks` loops in
the same methods already used for `DeviceOfflineException`. Each fix degrades
that one read gracefully instead of raising, using this project's own
established broad exception tuple (from the camera stream's best-effort
cloud calls) where the failure mode wasn't already narrowly diagnosed.

**Result:** the config entry now reaches and **stays** at `loaded` under
total cloud outage — no more crash/retry loop. Entities were still
`unavailable` at the end of this session because BLE *also* had its own
separate, apparently transient connectivity hiccup the same night (real
radio/proxy issue, unrelated to these fixes) — expected to clear on its own;
not chased further once the crash-loop itself was confirmed fixed, per the
standing rule against continuing to patch live indefinitely under unusually
adversarial conditions (both transports degraded at once).

| | beta92 (final of this incident) |
| --- | --- |
| tag | `v0.6.4-beta92` |
| quartet | `0.6.4-beta92` / `0.6.4b92` |
| card md5 | `39a332c31c0543fc2e7c6df5e69294aa`, equal at both serving paths |
| Lovelace resource | re-read as `?v=0.6.4-beta92&build=39a332c3` (bumped directly from beta88 — beta89-91's card cache key was never bumped mid-incident) |
| backend | PyMammotion `0.8.12.post3` (unchanged throughout) |
| file verification | 48 of 48 byte-identical at every step (beta89 through beta92) |
| gate after | `enabled: false`, `real_motion_allowed: false`, verified live API **and** RAW `[false]` |

⚠️ **Not fixed and not chased tonight:** `MammotionSpinoCoordinator`'s own
instance of this same bug class (no pool cleaner registered on this account,
so out of scope), and the RTK/error coordinators' `_async_update_data`
methods weren't audited beyond what actually fired in this incident — a
future full sweep of every cloud-only call across all coordinators would be
more thorough than this incident-driven, restart-and-discover pass.

### beta87 -> beta88 — 2026-08-30 19:08-19:12 EDT, motion-disabled

Ships the step-extension cap changes from
`docs/phase2-route1-step-extension-predeclared-20260830.md` (a further SAFETY
BOUND raise, not a convenience one) plus the `reason`-field bug fix from
commit `af5f547f`. `step_ms`'s schema ceiling moves 5000 -> **7000** ms and
`_STEP_RESPONSE_MAX_TOTAL_MS` moves 14000 -> **16000** ms, so a
`baseline 3000 / step 7000 / settle 5000` window (15000 ms) is now admissible
where it was refused before. `settle_ms` and `baseline_ms` stay unchanged, and
`max_travel_m`'s schema ceiling stays at the already-authorized 4.5 (no
further ceiling change — using it, not raising it). No `LUBA_ACCEPTANCE_PROFILE`
key touched.

| | beta88 |
| --- | --- |
| tag | `v0.6.4-beta88` (release commit `015883a1`) |
| deployed tree | **exactly the tag**, fast-forwarded from `875cc7dd` |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta88`, uv.lock `0.6.4b88` |
| backend | PyMammotion `0.8.12.post3`, read from inside the container (unchanged) |
| archive SHA-256 | `4ce62289fb00bfab8571d996030f0d3c954abcec8e34f2204046431c2fbd9f5f`, identical local and host |
| file verification | **47 of 47 byte-identical**, zero AppleDouble |
| card md5 | `d4d0f519a321081a621a3c92b4c2aa23`, equal at BOTH serving paths |
| Lovelace resource | re-read as `?v=0.6.4-beta88&build=d4d0f519` |
| host backup | `/config/mammotion-backup-20260830-1908-pre-beta88.tgz` |
| restart | API up in ~40 s; mammotion config entry `loaded` and 132 entities recovered ~4 min post-restart (same cloud-auth-cooldown-then-BLE-fallback pattern as the beta87 restart) |
| gate after deploy | `enabled: false`, `real_motion_allowed: false`, no active session; live API **and** RAW `[false]` |

✅ **THE NEW CAPS ARE CONFIRMED LIVE IN THE DEPLOYED BYTES.** A dry run of
`raw_pymammotion_step_response_probe` with
`baseline_ms=3000, step_ms=7000, settle_ms=5000, max_travel_m=4.5` was
accepted by the schema (`phases.total_ms: 15000`, `max_travel_m: 4.5` both
echoed back — the old 14000 ms / 5000 ms ceilings would have rejected this
call outright) and returned `would_send: false`, nothing dispatched. The one
failing gate, `step_path_contained`, is expected: the corridor used was a
placeholder for this schema check, not a real pre-run scan.

⚠️ **The `reason`-field fix also ships in this release but is UNEXERCISED on
the host** — no real run has been dispatched against beta88 yet, so nothing
has confirmed the fixed field reads correctly on live hardware. The two unit
tests pin it offline; that is not the same as seeing it emit a correct value
on a real run.

### beta86 -> beta87 — 2026-08-30 16:47-16:50 EDT, motion-disabled

Ships the route-1 cap changes from `docs/phase2-route1-predeclared-20260830.md`
(a SAFETY BOUND raise, not a convenience one) plus the already-committed
`maxsize=1` fix for the two remaining position streams. `max_travel_m`'s schema
ceiling moves 3.0 -> **4.5** (default stays 2.50) and `_STEP_RESPONSE_MAX_TOTAL_MS`
moves 12000 -> **14000** ms, so a `baseline 3000 / step 5000 / settle 5000` window
(13000 ms) is now admissible where it was refused before. `step_ms` stays capped
at 5000 and `linear_speed` stays pinned at 400, both deliberately unchanged. No
`LUBA_ACCEPTANCE_PROFILE` key touched.

🚨 **THE GATE WAS FOUND ARMED AND LIVE AT THE START OF THIS SESSION** —
`enabled: true`, `real_motion_allowed: true`, `blockers: []`, no active session.
Disarmed immediately, before any deploy step, and verified from both the live
API and RAW `core.config_entries` (`enable_experimental_motion":false`). No
motion was commanded before or during the disarm.

| | beta87 |
| --- | --- |
| tag | `v0.6.4-beta87` (release commit `0787f6b1`) |
| deployed tree | **exactly the tag**, fast-forwarded from `4f19a22e` |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta87`, uv.lock `0.6.4b87` |
| backend | PyMammotion `0.8.12.post3`, read from inside the container (unchanged) |
| archive SHA-256 | `f167a6e2567e8cf42432bc983f9624ddc6f401ff7c8cf590bd7de53a3571bc2c`, identical local and host |
| file verification | **47 of 47 byte-identical**, zero AppleDouble |
| card md5 | `743765579a4f10f35ee9f1f542a00964`, equal at BOTH serving paths |
| Lovelace resource | re-read as `?v=0.6.4-beta87&build=74376557` |
| host backup | `/config/mammotion-backup-20260830-1644-pre-beta87.tgz` |
| restart | API up in ~40 s; mammotion config entry `loaded` and 132 entities recovered ~110 s post-restart |
| gate after deploy | `enabled: false`, `real_motion_allowed: false`, no active session; live API **and** RAW `[false]` |

⚠️ **The restart briefly showed `cloud_mammotion permanently unavailable`** —
`Re-login required for account 'matt.joslin@me.com': refresh_login returned no
data`. This is the account's cloud MQTT token, unrelated to this change and to
the deployed tree; the integration proceeded on BLE (this project's primary
transport) and loaded normally. Not investigated further here — flag if it
recurs or if a cloud-path service is ever needed.

✅ **THE NEW CAPS ARE CONFIRMED LIVE IN THE DEPLOYED BYTES**, not just committed.
A dry run of `raw_pymammotion_step_response_probe` with
`baseline_ms=3000, step_ms=5000, settle_ms=5000, max_travel_m=4.0` was accepted
by the schema (`phases.total_ms: 13000`, `max_travel_m: 4.0` both echoed back —
the old 12000 ms / 3.0 m ceilings would have rejected this call outright) and
returned `would_send: false`, `command_result.attempted: false`. The one gate
that failed, `step_path_contained`, is expected: the corridor used was an
ad hoc placeholder for this schema check, not a real pre-run scan, and no
corridor scan, gate arming, or motion dispatch was performed as part of this
deploy.

### beta85 -> beta86 — 2026-08-29 17:01-17:07 EDT, motion-disabled

Ships the step-probe report-stream fix: the probe now starts the report stream
under its own lease and fails closed if no position payload arrives inside its own
generation. No backend change (PyMammotion `0.8.12.post3`), no
`LUBA_ACCEPTANCE_PROFILE` key touched.

| | beta86 |
| --- | --- |
| tag | `v0.6.4-beta86` (release commit `ce244772`) |
| deployed tree | **exactly the tag** |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta86`, uv.lock `0.6.4b86` |
| backend | PyMammotion `0.8.12.post3`, read from inside the container |
| archive SHA-256 | `19e316fc8c328db3ecd05884b8046445a90268487afdc95850cc34d5c0b1d602`, identical local and host |
| file verification | **47 of 47 byte-identical**, zero AppleDouble |
| card md5 | `c5a78cc90f6453982f2f11a44ea31587`, equal at BOTH serving paths |
| Lovelace resource | re-read as `?v=0.6.4-beta86&build=c5a78cc9` |
| host backup | `/config/mammotion-backup-20260829-1701-pre-beta86.tgz` |
| restart | API up after **45 s**, 132 entities at 184 s; BLE back immediately |
| fix in deployed bytes | `async_start_continuous_reports` count **15**, matching local |
| gate after deploy | `enabled: false`, `real_motion_allowed: false`; live API **and** RAW `[False]` |

✅ **THE FIX IS CONFIRMED ON HARDWARE, and it is the first time this probe measured
anything.** Two supervised runs, both signs: report stream `started`/`continuous`
true, `ready` true with no `readiness_reason`, `position_sequence` advancing
**4→13** and **16→24**, and **8 and 7 informative intervals** where every previous
run produced zero. Read `docs/evidence-dead-time-measured-20260829.json`.

🏁 **Result: τ ≥ 2.6–3.6 s of rotational dead time against a ~1 Hz control
period**, with onset lag ~1–2 s and no drivetrain asymmetry between signs.

⚠️ **Both runs tripped the travel guard with the mower still rotating**, so τ is
censored and should be quoted as a lower bound. A longer settle needs more
corridor, because `linear_speed` is schema-pinned at the measured 400.


### beta84 -> beta85 — 2026-08-29 11:37-11:50 EDT, motion-disabled

Ships `_in_window_ble_snapshot`: `is_connected`, `queue_depth`,
`queue_dispatch_paused` and `saga_active` recorded into every 100 ms in-window
telemetry sample, alongside the `position_sequence` / `position_epoch` added in
beta84. No backend change (PyMammotion `0.8.12.post3`), no
`LUBA_ACCEPTANCE_PROFILE` key touched.

| | beta85 |
| --- | --- |
| tag | `v0.6.4-beta85` (release commit `14a83001`) |
| deployed tree | **exactly the tag** |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta85`, uv.lock `0.6.4b85` |
| backend | PyMammotion `0.8.12.post3`, read from inside the container |
| archive SHA-256 | `d0f71202a34d7b03f1a00161f3f8623917da85501f12d83a51548849ae8c4ba5`, identical local and host |
| file verification | **47 of 47 byte-identical**, zero AppleDouble |
| card md5 | `1132c738d9dcd2f5422abef8dc70167a`, equal at BOTH serving paths |
| Lovelace resource | re-read as `?v=0.6.4-beta85&build=1132c738` |
| host backup | `/config/mammotion-backup-20260829-1137-pre-beta85.tgz` |
| restart | API up after **30 s**, 132 entities at 237 s |
| new code in deployed bytes | `_in_window_ble_snapshot` grep count **2**, matching local |
| gate after deploy | `enabled: false`, `real_motion_allowed: false`, no session; live API **and** RAW `[False]` |

🛑 **NO RUN WAS STARTED.** The operator asked for the release and deploy only.

🔌 **BLE took ~9.5 minutes to re-establish after the restart** (`active_transport`
`none` -> `ble` at 11:48:43), longer than beta82's ~10 min and beta84's recovery.
⚠️ **While it was down HA served stale state** — `charge_on` and battery 100% for
a mower that had been off the dock since 10:51. The true state (`AREA_INSIDE`,
battery 95%) only appeared once the link came back. **This is the 2026-08-24
failure shape: every field readable, all of it hours old.**

🔎 **DIRECT CONFIRMATION THAT `ble_link_live` LAGS, recorded because it changes how
its history may be read.** After settling, the ENTITY reported
`ble_link_reason: ble_client_not_connected` while a **live** recomputation of the
same gate listed only `experimental_motion_disabled` and `active_transport` read
`ble`. The entity is a coordinator-tick derived value; **its transitions cannot be
used to time anything**, which is exactly why beta85 records the raw BLE fields
in-window instead.

⚠️ **The new fields are DEPLOYED but UNEXERCISED** — `dry_run` returns before the
sampler starts, so nothing populates them until a real motion window runs.


### beta83 -> beta84 — 2026-08-28 21:09-21:15 EDT, motion-disabled

Ships one change: `_in_window_telemetry_sample` now records **`position_sequence`
and `position_epoch`**, the discriminator the 2026-08-28 step-probe abort could
not provide. No backend change (PyMammotion `0.8.12.post3`), no
`LUBA_ACCEPTANCE_PROFILE` key touched.

| | beta84 |
| --- | --- |
| tag | `v0.6.4-beta84` (release commit `d4ab2ad1`) |
| deployed tree | **exactly the tag** |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta84`, uv.lock `0.6.4b84` |
| backend | PyMammotion `0.8.12.post3`, unchanged, read from inside the container |
| archive SHA-256 | `fd9daa3a13326d802df60fe4f91736819b5d7fa5425eff69906018039d5fe6c1`, identical local and host |
| file verification | **47 of 47 byte-identical**, zero AppleDouble |
| card md5 | `12e493548625785b3be224ea82b5a1dc`, equal at BOTH serving paths |
| Lovelace resource | re-read as `?v=0.6.4-beta84&build=12e49354` |
| host backup | `/config/mammotion-backup-20260828-2109-pre-beta84.tgz` |
| restart | API up after **30 s**, 132 entities at 147 s |
| gate after deploy | `enabled: false`, `real_motion_allowed: false`, no session; live API **and** RAW `[False]` |

⚠️ **The new field is DEPLOYED but UNEXERCISED.** `dry_run` returns before the
sampler starts, so nothing populates `position_sequence` until a real motion
window runs. Byte parity is confirmed (grep count 2, matching local) and the code
path is unit-tested; **that is not the same as having seen it produce a value.**

🌙 **Deployed in the dark, deliberately.** VIO is fully collapsed
(`camera_brightness: dark`, `tracked_features: 0`, `signal_none`) and it does not
matter: there is **no VIO gate** anywhere in `_manual_velocity_pulse_gates` or in
the step probe's own four geometry gates. RTK held **Fix with 32 satellites**.


### beta82 -> beta83 — 2026-08-28 18:41-18:47 EDT, motion-disabled

Ships `raw_pymammotion_step_response_probe`, the open-loop dead-time probe, and
fixes two stale `services.yaml` descriptions that still said the corridor must
clear 1.06 m where the code has computed 1.34 m since 2026-08-27. No backend
change (PyMammotion `0.8.12.post3`), no `LUBA_ACCEPTANCE_PROFILE` key touched.

| | beta83 |
| --- | --- |
| tag | `v0.6.4-beta83` (release commit `5a3e96bc`) |
| deployed tree | **exactly the tag** — `git tag --points-at HEAD` = `v0.6.4-beta83` at deploy time |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta83`, uv.lock `0.6.4b83` |
| backend | PyMammotion `0.8.12.post3`, unchanged, read from inside the container |
| archive SHA-256 | `ea3dd599c25f0d6365ac9bc9c8ef179edd427662943eb1dc74ac00ed524ca04f`, identical local and host |
| file verification | **47 of 47 byte-identical**, zero AppleDouble |
| card md5 | `5cee2702d94a66a341a160baa4ab05ce`, equal at BOTH serving paths |
| Lovelace resource | re-read as `?v=0.6.4-beta83&build=5cee2702` |
| host backup | `/config/mammotion-backup-20260828-1841-pre-beta83.tgz` |
| restart | API up after **30 s**, 132 entities at 152 s |
| gate after deploy | `enabled: false`, `real_motion_allowed: false`, no session; live API **and** RAW `core.config_entries` |
| new service | registered live with all **16** fields, confirmed from `/api/services` |

🚨 **THE PROBE'S FIRST RUN ABORTED ON A FROZEN POSITION FEED AND MEASURED
NOTHING.** A supervised, operator-authorized `+120` run tripped
`travel_guard_tripped` at **2.218 s** with **zero informative intervals**. All 21
in-window samples read bit-identical `x 8.2832, y -7.3937, toward 126.8278`, and a
snapshot taken immediately after the run still read the pre-run position — yet the
mower had **travelled 0.4375 m** by the time the feed caught up (~0.197 m/s
implied). Read `docs/evidence-step-response-probe-aborted-20260828.json`.
✅ **The probe fail-closed exactly as designed** — cumulative distance was 0, so
this was beta72's stale-feed trip and not a distance trip — and the stop confirmed.
🛑 **The `-120` sign was NOT run.** Both signs still matter, but only once the feed
delivers position during motion.

⚠️ **This establishes NOTHING about Q1 or Q2.** Do not read "the step probe ran"
as progress on the dead-time question.


### beta81 -> beta82 — 2026-08-28 14:08-14:19 EDT, motion-disabled

Ships the acquisition-budget change `da1806e0` (`max_heading_acquisition_s`
2.0 -> 3.0 s), so the blind acquisition disk grows **1.06 -> 1.34 m**. No backend
change (PyMammotion `0.8.12.post3`), no `LUBA_ACCEPTANCE_PROFILE` key touched.

| | beta82 |
| --- | --- |
| tag | `v0.6.4-beta82` (release commit `56218a93`) |
| deployed tree | **exactly the tag** — `git tag --points-at HEAD` = `v0.6.4-beta82` at deploy time |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta82`, uv.lock `0.6.4b82` |
| backend | PyMammotion `0.8.12.post3`, unchanged, read from inside the container |
| archive SHA-256 | `e50a27eaacb3cdbb423f1ad1a47f6f561cb2f981ea3d8ee68c17f39f4f912ed3`, identical local and host |
| file verification | **47 of 47 byte-identical**, zero AppleDouble |
| card md5 | `54119a4b61e0f87d30a8e51c24a241f7`, equal at BOTH serving paths |
| Lovelace resource | re-read as `?v=0.6.4-beta82&build=54119a4b` |
| host backup | `/config/mammotion-backup-20260828-1408-pre-beta82.tgz` |
| restart | API up after **61 s**, 132 entities at 236 s, 0 unavailable |
| gate after deploy | `enabled: false`, `real_motion_allowed: false`, no session; live API **and** RAW `core.config_entries` |
| dry run | `continuous_motion_window` `would_send: false`, `command_result.attempted: false` |

🔑 **The 1.34 m disk is verified on the DEPLOYED build by a discriminating dry
run, not by reading the constant.** Two `continuous_motion_window` dry runs
against square corridors centred on the live start:

| corridor half-width | `required_radius_m` | `boundary_clearance_m` | gate |
| --- | --- | --- | --- |
| 1.20 m | **1.34** | 1.200 | **FAILS** `blind_heading_acquisition_contained` |
| 2.00 m | **1.34** | 2.000 | passes, `blockers: []` |

The 1.20 m case is the discriminator: it would have **passed** on beta81's
1.06 m disk and is refused here. Reported config on the host:
`max_safety_speed_mps 0.28`, `max_heading_acquisition_s 3.0`,
`stop_overshoot_m 0.5` -> `0.28 x 3.0 + 0.50 = 1.34`.

🚨 **The attempt-3 frozen corridor is now only 1.12x the disk, and that is the
binding constraint — not the yard.** The corridor in
`docs/phase2-steering-attempt3-design-20260827.md` /
`docs/evidence-phase2-steering-run3-blind-20260827.json` is a **3.0 m x 5.0 m**
rotated rectangle, so its maximum possible boundary clearance is **1.50 m**.
Recomputed with the shipped `blind_acquisition_feasibility` at the attempt-3
start `(4.9889, -3.0019)`:

```
required_radius_m    1.34
boundary_clearance_m 1.50
margin               0.16 m   (1.12x)      <- was 1.42x at the old 1.06 m disk
```

It still passes, but **only because that start sits on the corridor centreline**.
Any placement more than 0.16 m off-centre refuses the run.

⚠️ **"The 2026-08-27 spot had 5.07 m clearance, so it is ample" measures the
wrong thing.** That figure is the open-lawn clearance from
`docs/phase2-steering-refusal-recommendation-20260826.md`, not the frozen
corridor's. The **yard** is indeed ample — raw distance to the "Backyard Right"
boundary and to both keep-out polygons, measured from the live `export_map`:

| position | nearest area edge | nearest keep-out | clear radius | vs 1.34 m |
| --- | --- | --- | --- | --- |
| attempt-3 start `(4.9889, -3.0019)` | 4.0738 m | 5.8365 m | **4.0738 m** | 3.04x |
| live off-dock `(4.8213, -1.3620)` | 2.4347 m | 5.6788 m | **2.4347 m** | 1.82x |

**So attempt 4 needs a WIDER frozen corridor, not a different spot.** At 1.34 m
a corridor must be at least `2 x 1.34 = 2.68 m` wide before any placement
tolerance at all; attempt 3's 3.0 m leaves 0.16 m. Freeze a corridor of roughly
3.5-4.0 m width and prefer the attempt-3 area, where the yard gives 4.07 m.

⚠️ **`services.yaml` prose is stale.** The `continuous_motion_window`
description still says the corridor must "provide at least 1.06 m of boundary
clearance". The **code is correct at 1.34 m** — this is description text only,
and it is exactly the class of staleness `scripts/check_doc_symbols.py` cannot
catch (it checks names, not prose). Fix it in the next release; do not hand-patch
the host.

🔌 **BLE dropped across the restart and recovered on its own in ~10 minutes.**
Post-restart the gate read `ble_client_not_connected` with
`binary_sensor..._ble_link_live: off` and `active_transport: none`, while
`ble_rssi` read a healthy **-44** — another instance of the standing rule that
`ble_rssi` is self-reported and is **not** a liveness signal. It came back at
14:18:55 without a config-entry reload, and RTK went `float -> fix` at the same
time. No action was taken.

🔑 **The `blade_rpm_nonzero` latch did NOT recur.** `cutter_rpm 0`,
`blade_safe_for_motion: on` before and after the deploy.

✅ **Browser-verified by the operator, 2026-08-28: the card reads `0.6.4-beta82`.**
The deploy is now closed end to end — backend bytes, both serving paths, the
Lovelace cache key, and the rendered card.


### beta78 -> beta79 — 2026-08-27 17:05-17:12 EDT, motion-disabled

Adds the backend `current_orientation` field so the card's direction arrow can
render. No backend change (PyMammotion `0.8.12.post3`), no
`LUBA_ACCEPTANCE_PROFILE` key touched.

| | beta79 |
| --- | --- |
| tag | `v0.6.4-beta79` (commit `14b108cc`) |
| deployed tree | `6dc9bcbc` — **code byte-identical to the tag**, only doc commits since; verified with `git diff v0.6.4-beta79..HEAD -- custom_components/` |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta79`, uv.lock `0.6.4b79` |
| backend | PyMammotion `0.8.12.post3`, unchanged, read from inside the container |
| archive SHA-256 | `60754296bd2db1d2af44570c00c2a0f3e73c5b2ab9be782677d6b4f5a7677566` |
| file verification | **47 of 47 byte-identical**, zero AppleDouble |
| card md5 | `aac82658462827994699ca094d7f3c39`, equal at BOTH serving paths |
| Lovelace resource | `?v=0.6.4-beta79&build=aac82658` |
| host backup | `/config/mammotion-backup-20260827-1705-pre-beta79.tgz` |
| restart | API up after **45 s**, 133 entities at 165 s |
| gate after deploy | `enabled: false`, `real_motion_allowed: false`, no session; live API **and** RAW `core.config_entries` |

**The arrow verified live, not just present:** `current_orientation` returned
`trustworthy: true`, map heading **277.188°**, VIO 277.188 against compass mirror
276.964 — **0.224° apart**.

🔑 **Checked BLE freshness before restarting, after an overnight gap.** The
position sequence advanced 50 → 62 over 12 s, so the feed was live rather than a
latched session. Worth repeating: the 2026-08-24 incident looked healthy on every
status field while serving hours-stale state.

⚠️ **The mower was moved off the dock by the operator between the state check and
the deploy**, so post-restart it read `AREA_INSIDE` with a valid position and
`experimental_motion_disabled` as the ONLY blocker. That is the armed-would-be-empty
posture this project has flagged five times. It was confirmed as an intentional
manual move.

🛑 **NOT verified: a browser has not loaded the beta79 card.** The backend field
and both serving paths are confirmed; the arrow's on-screen rendering still needs
a human to look at it.

### beta77 -> beta78 — 2026-08-26 13:26-13:35 EDT, motion-disabled

Evidence-quality release. No backend change (still PyMammotion `0.8.12.post3`),
no `LUBA_ACCEPTANCE_PROFILE` key touched (profile reports ACCEPTED). Carries the
two fixes found while auditing the beta77 stationary runs.

| | beta78 |
| --- | --- |
| tag / HEAD | `v0.6.4-beta78` (release commit `7a04269f`) |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta78`, uv.lock `0.6.4b78` |
| backend | PyMammotion `0.8.12.post3`, unchanged, read back from inside the container |
| archive SHA-256 | `036a80a2a679cd09625609f9201cc6fead8efaf7c6739ac6858741b9b58a5a6b`, identical local and host |
| file verification | **47 of 47 byte-identical**, zero AppleDouble files |
| card md5 | `b0ea22d95efe7b1217a37006cb5db1bd`, equal at BOTH serving paths |
| Lovelace resource | re-read as `?v=0.6.4-beta78&build=b0ea22d9` |
| host backup | `/config/mammotion-backup-20260826-1326-pre-beta78.tgz` |
| restart | API up after **30 s**, 132 entities at 144 s |
| gate after deploy | `enabled: false`, `real_motion_allowed: false`, no active session; verified from the live API **and** RAW `core.config_entries` |
| dry run | `heading_acquisition_window` `would_send: false`, `command_result.attempted: false` |

**Both fixes verified in the DEPLOYED bytes, not just claimed:**
`grep -c include_raw_samples` on the host returns **0** (raw position evidence is
no longer stripped from matrix artifacts), and the named-predicate branch is
present. A live one-transition probe against the docked mower returned
`position_invalid_for_motion: zone_hash_unavailable` where beta77 returned a bare
`position_invalid_for_motion` with no cause.

⚠️ **That probe does NOT exercise the raw-evidence fix, and it should not be
recorded as if it did.** A failed readiness check short-circuits before the block
that builds `intervals_ms`/`pipeline_latencies_ms`, so the `intervals_ms` visible
in its output is the result skeleton, not the retention path. Confirming that fix
end to end needs a cell that COMPLETES, which needs the mower off the dock. The
code is deployed and hash-verified; it is simply unexercised.

✅ **Browser-verified by the operator, 2026-08-26: the card footer reads
`card v0.6.4-beta78`.** This deploy is verified end to end. The card's own runtime
panel independently read `PyMammotion backend 0.8.12.post3 (verified)`, which also
confirms the `installed_pymammotion_version` `lru_cache` added in beta77 reports
the live version correctly across a restart rather than caching a stale one.

### beta76 -> beta77 — 2026-08-25 23:23-23:40 EDT, motion-disabled

Ships the reviewed position-subscription lease work with PyMammotion
`0.8.12.post3`. No `LUBA_ACCEPTANCE_PROFILE` key touched
(`scripts/check_accepted_profile.py` reports ACCEPTED). Continuous steering
remains refused.

| | beta77 |
| --- | --- |
| tag / HEAD | `v0.6.4-beta77` (release commit `348c71a8`, feature commit `cd591a05`) |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta77`, uv.lock `0.6.4b77` |
| backend | PyMammotion `0.8.12.post3`, read back from inside the container |
| wheel SHA-256 | `cd3b0c3558d05c3ea6c7b6f2faad68c9c9eac523e70406bb930c5b01045a887a`, and `uv.lock` records the same hash independently |
| archive SHA-256 | `ffc220cda9dcc29cb226100bbc5dc15c5459baf7f196358281634c7b94024f77`, identical local and host |
| file verification | **47 of 47 byte-identical**, zero AppleDouble files |
| card md5 | `544adfbabc0fdff2cbd81f2bf9693cdc`, equal at BOTH serving paths |
| Lovelace resource | re-read as `?v=0.6.4-beta77&build=544adfba` |
| host backup | `/config/mammotion-backup-20260825-2323-pre-beta77.tgz` |
| restart | API up after **41 s**, 133 entities at 240 s |
| new service registered | `report_stream_sequence_probe` present via `/api/services` with fields `entity_id, observation_seconds, periods_ms, readiness_timeout_seconds` |
| gate after deploy | `enabled: false`, `real_motion_allowed: false`, no active session; verified from the live API **and** RAW `/config/.storage/core.config_entries` (`enable_experimental_motion: False`) |
| dry runs | `heading_acquisition_window` and `continuous_motion_window` both `would_send: false`, `command_result.attempted: false`, and both report the new `post_stop_observation_timeout_s: 3.5` -- proving the beta77 executor is the one loaded |

🔑 **This release exists because an independent review found a false-ready path
in the beta77 candidate.** The readiness evidence boundary was taken from the
return of `request_iot_sync_continuous`, which returns when the command is
**queued**, not sent. A position payload still arriving from the configuration
being replaced therefore satisfied the new generation's readiness. The boundary
is now the post-queue-settle flush. Pinned by
`test_isolated_probe_readiness_starts_at_the_start_flush_not_the_call`, which was
verified to accept the stale sample on the pre-fix code.

⚠️ **`requirements_test.txt` had drifted to post1 while the manifest shipped
post2**, so CI had been testing a different backend than the deployed one since
the post2 release. Now byte-identical to `manifest.json`. If you ever bump one,
bump both.

⚠️ **The version sites were deliberately NOT hand-bumped.** `Beta Release`
computes one above the highest of the manifest suffix and the newest tag, so the
candidate's hand-written beta77 bumps would have released **beta78**. They were
reverted before the release ran, and the workflow produced beta77 as intended.

⚠️ **Entity readback differs from this runbook's older "five benign
unavailable" note:** 11 of 133 read `unavailable`/`unknown` — four buttons plus
seven never-fired `last_*` timestamp sensors, all expected shortly after a
restart. No baseline was captured before the restart, so this is reported as
measured, not as a regression.

✅ **Browser-verified by the operator, 2026-08-26: the card loads as
`0.6.4-beta77`.** With backend bytes, both serving paths, the Lovelace cache key
and now the rendered card all confirmed, this deploy is verified end to end —
the one step the deployer cannot self-check is closed.

✅ **Both stationary gates were then run and PASSED on this build, 2026-08-26**
— 30/30 ownership transitions position-ready, and the randomized matrix passed
untouched with no retry substitution. No motion was commanded and the gate stayed
disarmed. See `docs/position-cadence-safety-followup-plan-20260825.md` →
"Stationary live results".

🛑 **Still NOT done and still needing its own authorization: any physical
motion.** Continuous steering remains refused in code.

⚠️ **Two fixes are committed but NOT on this host:** raw position evidence is no
longer stripped from matrix artifacts, and `position_invalid_for_motion` now names
its failing predicate. Cut a beta78 before the next matrix run if you want that
artifact fully auditable.

### beta72 → beta73 — 2026-08-23 20:23-20:29 EDT, motion-disabled

Ships the Phase 2 executor (`docs/phase2-executor-implemented-20260823.md`):
new service `continuous_motion_window`. No `LUBA_ACCEPTANCE_PROFILE` key
touched.

| | beta73 |
| --- | --- |
| tag / HEAD | `v0.6.4-beta73` |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta73`, uv.lock `0.6.4b73` |
| archive SHA-256 | `0e9b462f58ae2acfcfaf59803b50443478bf4f51d781cd14942e6d1f19ea6529`, identical local and host |
| file verification | **47 of 47 byte-identical**, zero AppleDouble files |
| card md5 | `4d854242e0fbac032a7594446fad55c3` |
| host backup | `/config/mammotion-backup-20260823-2023-pre-beta73.tgz` |
| restart | API up after **50 s**, 132 entities at 168 s, entry `loaded` |
| new service registered | confirmed via `/api/services`: `continuous_motion_window` present |

🚨 **A DEPLOY ERROR, self-caught after the operator reported the card still
read beta72.** I claimed "no card change" from a `git diff --stat` scoped to
my own feature commit, which correctly showed no card diff -- but the release
workflow's separate version-bump commit (`8d4fce9a`) always rewrites
`CARD_VERSION` regardless, so the file DID change. beta72's card hash was
`0be6d2ab164db8e053086f33be8f7ef9`; beta73's is `4d854242e0fbac032a7594446fad55c3`.
**Different, not unchanged.** The Lovelace resource query string was therefore
never bumped, and the browser kept serving its cached beta72 bundle from the
unchanged URL. Fixed: resource re-read as
`?v=0.6.4-beta72&build=0be6d2ab` immediately before the fix, updated and
verified as `?v=0.6.4-beta73&build=4d854242` after. **Lesson: check the
ACTUAL deployed file's hash, never a pre-release diff, to decide whether the
card cache key needs bumping** -- the version-bump commit is not visible to a
diff scoped before it lands.

**Dry run against LIVE coordinator state, not a fake:** `route_start` set to
the mower's actual live position, `dry_run: true`. **14 of 14 gates passed**,
`would_send: false`, `command_result.attempted: false`. This is the first time
the new executor has run against real telemetry shapes rather than the test
fixtures in `test_continuous_motion_window.py`.

🚨 **The motion gate was found ARMED after the restart, despite an explicit
disarm immediately before dispatching the release.** Sequence: gate disarmed
and verified `False` before the release was dispatched; release, deploy, and
restart took several minutes; the first post-restart check read
`enabled: true, real_motion_allowed: true, blockers: []`. Disarmed again
immediately and confirmed against the RAW on-disk config-entry storage
(`/config/.storage/core.config_entries`), not just the live API, which now
correctly reads `false`.

✅ **RESOLVED, operator-confirmed: deliberate.** The operator set experimental
motion on themselves, before the restart -- not a restart-path bug, not a
stale-save race. The restart correctly read what was actually on disk at that
moment; it happened to be the operator's own toggle, made after my disarm and
before the restart picked it up. Candidate 2 (a code defect re-reading a stale
value) is withdrawn. No motion was commanded either time.

The disarm automation's own `armed_but_blocked`/idle-and-ready triggers had
already fired once earlier the same evening (23:54:18 UTC, well before this
deploy sequence began) for a separate instance of the same pattern -- the
operator arming it and it being left, not a fresh occurrence each time.

🔎 **VIO fully collapsed during this deploy window**: `vio_tracked_features`
read **68** at the start of this session's check and **0** four minutes later,
with `camera_brightness: dark` and `visual_positioning_status: signal_none`.
Matches the documented cliff pattern exactly -- it does not fade, it drops to
zero. Confirms dusk had fully arrived; nothing depending on VIO should be
attempted tonight.

### beta71 → beta72 — 2026-08-23 13:12-13:20 EDT, motion-disabled

Two changes, both prerequisites for the Phase 1b arc:

1. **The travel guard's three runtime fail-open paths are closed** and its bounds
   resized. `_PROBE_TRAVEL_GUARD_OVERSHOOT_M` 0.35 → **0.50 m**;
   `corridor_must_cover_m` is now the **worst case** rather than the nominal one;
   a frozen feed, a missing position, or a dead sampler each trip the guard
   instead of silently returning the run to the wall clock.
2. **Phase 1 duration is expressed per control** — straight 4000 ms, arc
   **8000 ms** — so a Phase 1b capture can be scored at all.

| | beta72 |
| --- | --- |
| tag / HEAD | `v0.6.4-beta72` |
| quartet | manifest / pyproject / `CARD_VERSION` `0.6.4-beta72`, uv.lock `0.6.4b72` |
| archive SHA-256 | `385f9c080a08bdfc9b746dfb64670a58e012ce195ffa01cbb9eb4084cc417f3e`, identical local and host |
| file verification | **47 of 47 byte-identical**, zero AppleDouble files |
| card md5 | `0be6d2ab164db8e053086f33be8f7ef9` at both paths and locally |
| Lovelace resource | `?v=0.6.4-beta72&build=0be6d2ab`, verified by re-read |
| host backup | `/config/mammotion-backup-20260823-1313-pre-beta72.tgz` |
| restart | API up after **45 s**, 132 entities at 158 s |
| gate after | `enabled: false`, no session, `MODE_READY` |

🔑 **The guard change is verified live and the difference is measurable.** The
same dry run (`angular 180`, 8000 ms, `max_travel_m 1.5`) reports:

| | beta71 | **beta72** |
| --- | ---: | ---: |
| `expected_overshoot_m` | 0.35 | **0.50** |
| `corridor_must_cover_m` | 1.85 | **2.24** |
| `clock_bound_m` | *(absent)* | **2.24** |

**39 cm more corridor is now required** for the identical command, because the
number finally accounts for the guard doing nothing.

⚠️ **No mower command sent**, `would_send: false`, no blockers, gate disarmed
throughout. The mower is on the dock (`position_not_valid_for_motion`).

🚨 **Not browser-verified.** Ask the operator to confirm the card footer and
console banner read `0.6.4-beta72`.

### beta70 → beta71 — 2026-08-22 19:26-19:35 EDT, motion-disabled

Ships **one** change: the bounded raw motion probe's window is now bounded by
**distance** rather than by a time proxy. `duration_ms` was capped at 4000 ms
with the comment "the only thing limiting travel is the window"; that capped the
longest continuous run this project can perform at ~1.1 m, while the case for
continuous motion (4.88x) extrapolates a 4 s window to a 159 s route. The
in-window sampler is now also the guard: it measures displacement from where the
window started and aborts once `max_travel_m` is exceeded. Moves no
`LUBA_ACCEPTANCE_PROFILE` key; profile still ACCEPTED.

| | beta71 |
| --- | --- |
| tag / HEAD | `v0.6.4-beta71` at `1219de84` |
| feature commit | `8e22bfac` |
| version quartet | manifest `0.6.4-beta71`, pyproject `0.6.4-beta71`, `CARD_VERSION` `0.6.4-beta71`, uv.lock `0.6.4b71` |
| archive SHA-256 | `cd1f3d555257b36e8e3566c346a71125138a629f28695ddd0be678035c11bd65`, identical local and host |
| file verification | **47 of 47 byte-identical**, zero AppleDouble `._*` files |
| card md5 | `575e8510826fbbec178b1fc258a67a1b` at both serving paths and locally |
| Lovelace resource | `?v=0.6.4-beta71&build=575e8510`, verified by re-read |
| host backup | `/config/mammotion-backup-20260822-1927-pre-beta71.tgz` |
| restart | API up after **46 s**, 132 Mammotion entities at 134 s |
| config entry | `loaded` |
| backend | pymammotion `0.8.12.post1`, unchanged |
| gate after | `enabled: false`, `real_motion_allowed: false`, no session, `MODE_READY` |

**Zero-motion dry runs prove the new guard executes on the host**, all
`would_send: false`, `command_result.attempted: false`, 11 of 11 gates passed:

* `duration_ms: 8000` + `max_travel_m: 2.0` + 100 ms sampling → **no blockers**,
  81 planned samples, and
  `travel_guard: {"enabled": true, "max_travel_m": 2.0, "expected_overshoot_m": 0.35, "corridor_must_cover_m": 2.35, "tripped": false}`.
* `duration_ms: 8000` with sampling but **no** guard → refused
  `duration_over_4000ms_requires_max_travel_m`.
* `duration_ms: 8000` with a guard but **no** sampling → refused
  `duration_over_4000ms_requires_in_window_sampling`.
* `duration_ms: 4000`, no guard → **no blockers**. Every existing caller is
  unaffected and keeps the historic 4000 ms bound.

⚠️ **No mower command was sent and no physical long window has been run.** The
recommended first one is `duration_ms: 8000`, `max_travel_m: 2.0`, on a corridor
frozen to cover **≥3.5 m** — sized for the case where the new guard does
nothing, at the maximum observed 0.3762 m/s, because the guard is the thing
under test and must not also be the containment.

⚠️ **The runbook's "five entities read unavailable" note is stale.** Three do:
`start_camera_on_mower`, `image.…_last_event`, and `sensor.…_recognized_people`.
The four `emergency_nudge_*` buttons are now **available** because
`_nudge_available` returns `True` unconditionally by the operator's explicit
decision — that is the ungated-nudge change, not a regression.

🚨 **Not browser-verified.** Bytes and behaviour are confirmed on the host; no
browser has loaded beta71. Ask the operator to confirm the card footer and
console banner both read `0.6.4-beta71`.

### disarm automation installed — 2026-08-22, no integration change

Host config only. **No integration files changed, no version bump, no HA
restart, no mower command, and the motion gate was not armed.**

`docs/automations/disarm-motion-gate.yaml` was appended to
`/config/automations.yaml` and applied with the `automation/reload` service.

| step | result |
| --- | --- |
| backup taken first | `/config/automations.yaml.bak.claude-20260821-disarm`, md5 `5b08acc1ea54f2e3fe983155495b9795`, identical to the pre-change file |
| entity ids checked against the live install | `binary_sensor.back_yard_clip_skywalker_real_motion_ready` (`off`) and `lawn_mower.back_yard_clip_skywalker` (`paused`) — both match the YAML |
| appended | 54,526 → 55,686 bytes |
| host-side YAML re-parse | valid, 60 automations |
| applied | `POST /api/services/automation/reload` → HTTP 200 |
| loaded | `automation.mammotion_disarm_motion_gate_when_left_armed`, id `1755900000001`, state `on`, `last_triggered: None` |

Given an `id:` (the repo YAML has none) so HA treats it as a UI-editable entry
like the other 59.

🔒 **One-way by construction.** There is no arm service — arming stays behind
the options flow — and `disarm_experimental_motion` refuses while a session is
active, so this can never interrupt a supervised run. Worst case is one re-arm.
It notifies only when `result.changed` is true, so a nightly sweep that finds
the gate already closed is silent.

**Amended 2026-08-22 — third trigger `armed_but_blocked`.** The next morning
the gate was found `enabled: true` with the sole blocker
`position_not_valid_for_motion` (mower docked) — a fifth armed-at-rest
occurrence that **neither original trigger could see**, because
`real_motion_ready` reads `off` whenever any blocker fires. The danger is that
the dock blocker disappears as soon as the mower is moved, leaving a live gate
nobody armed. `enabled` is a config-entry option with no entity, so the new
trigger watches the readiness sensor's `blockers` attribute:
`experimental_motion_disabled` is in that list exactly when the gate is closed,
so its absence means armed. Rendered against live state via `/api/template`
before installing (returned `False` with the gate closed, as it must).
Reinstalled by truncating at the block's first line and re-appending; host-side
re-parse 60 automations, `automation/reload` HTTP 200, entity back `on` with
triggers `idle_and_ready` / `armed_but_blocked` / `nightly_sweep`. Backup:
`/config/automations.yaml.bak.claude-20260822-trigger`. 56,128 bytes.

**Rollback:** `cp /config/automations.yaml.bak.claude-20260821-disarm
/config/automations.yaml` (pre-automation) or
`/config/automations.yaml.bak.claude-20260822-trigger` (two-trigger version),
then call `automation/reload`. Or just disable the automation entity in the UI.

⚠️ A second Mammotion automation already existed and was **not** touched:
`Mammotion - resync map when it goes stale` presses `sync_maps` after
`map_sync_status` reads `out_of_sync` for 15 minutes (last triggered
2026-07-25). A `sync_maps` press once held the motion queue 12-17 s, but
`async_sync_maps` is now guarded against firing while `manual_motion_owner` is
set, so it is contained. Worth knowing it can fire unattended during a run
window.

### beta69 → beta70 — 2026-08-21 19:50-19:57 EDT, motion-disabled

Ships Phase 1 continuous-motion **instrumentation only** in the existing
bounded raw probe. `in_window_sample_interval_ms` is disabled by default; at
100 ms it samples coordinator cache while refreshed motion is open, without
extra in-window BLE report requests. There is still no continuous controller
executor or new dispatch command.

| | beta70 |
| --- | --- |
| Files | 47/47 byte-identical |
| Normalized per-file MD5-list SHA-256, local = host | `c8ed7e2aab0690dfbcaefbd81dd31bd3392b2ca60cdaa057285485098b4027b9` |
| Card md5 (both paths + local) | `ab85de303d6deef6f7f13c0f892302e0` |
| Archive SHA-256 local = host | `fd84013575efb83be969dcfb60db6b2f627b447a3c016fe51300a6817d5ebd15` |
| AppleDouble/bytecode entries | 0 |
| Backup | `/config/mammotion-backup-20260821-195101-pre-beta70.tgz` |
| Backup SHA-256 | `420e8418805f5e72ca2693227ebb8b8553e05453311a1fa7a436906aca46c772` |
| Tag | `v0.6.4-beta70` at version commit `a40fa32e` |
| Quartet on host | manifest and both card paths `0.6.4-beta70` |
| Lovelace resource | `?v=0.6.4-beta70&build=ab85de30` (read back) |
| Restart | API up 31 s; 133 Mammotion entities at 95 s |
| Config entry | `loaded` |
| Dependency | `pymammotion 0.8.12.post1`, backend verified, missing capabilities `[]` |
| Gate after | `enabled: false`, `real_motion_allowed: false`, no session, `MODE_PAUSE` |

Both exact Phase 1 plans executed as dry runs on the deployed integration:
straight `(linear 400, angular 0)` and shallow arc `(linear 400, angular 180)`,
each at 200 ms refresh, 100 ms cache sampling, and a 4,000 ms hard window. Both
returned `reason: dry_run`, `would_send: false`, `command_result.attempted:
false`, 41 planned samples, and zero extra in-window BLE report requests. No
stream was started and no mower command was sent. BLE had not reconnected after
restart, which independently kept real motion blocked.

Browser verification passed: footer and fresh console banner both loaded
beta70 from the new cache key. The final gate readback remained disabled with
no active session. Machine-readable record:
`docs/evidence-beta70-continuous-phase1-deploy-20260821.json`.

### beta68 → beta69 — 2026-08-21 14:01-14:11 EDT, motion-disabled

Ships **segment-level keep-out containment**. A path with legal endpoints whose
connecting leg crosses or touches a keep-out is now refused by the backend with
`path_legs_cross_keep_out_zone`; the card mirrors the geometry, blocks Real Go,
and keeps the crossing leg red/dashed. No `LUBA_ACCEPTANCE_PROFILE` key moved;
`check_accepted_profile.py` remains ACCEPTED.

The gate was found **ARMED at rest again** before deployment (`enabled: true`,
no active session, `MODE_PAUSE`). It was disarmed and verified before the
backup, immediately before deployment, and after all validation. **No motion
was commanded.**

| | beta69 |
| --- | --- |
| Files | 46/46 byte-identical |
| Normalized per-file MD5-list SHA-256, local = host | `495d65c5370a357b124bf1a1d0a525e9f4c6bcdb47d7b3ff97ea4bd96e163a45` |
| Card md5 (both paths + local) | `60dec49d1018c174715baca04de30f41` |
| Archive SHA-256 local = host | `bddb3941e735ef995b60653f7d4c3d8ebe7da752d08ee35c07c99aa9c36757e4` |
| AppleDouble entries | 0 |
| Backup | `/config/mammotion-backup-20260821-140148-pre-beta69.tgz` |
| Backup SHA-256 | `d9580aa3285eaaa0ca0ff423cd96ba5f1f949030bea27c239105ceae02c75ac9` |
| Tag | `v0.6.4-beta69` at version commit `6a8744c2` |
| Quartet on host | manifest `0.6.4-beta69`, CARD_VERSION `0.6.4-beta69` (both paths) |
| Lovelace resource | `?v=0.6.4-beta69&build=60dec49d` (read back) |
| Restart | API up 50 s; 133 Mammotion entities at 131 s |
| Config entry | `loaded` |
| Dependency | `pymammotion 0.8.12.post1` |
| Gate after | `enabled: false`, `real_motion_allowed: false`, no session |

**Live zero-motion validation against the real map:**

- Crossing `(9.0, -0.76) → (15.0, -0.76)`, both endpoints inside Backyard
  Right: `valid: false`, sole error `path_legs_cross_keep_out_zone`, zero point
  violations, leg 0 identified against obstacle `1529607395159402290`.
- Legal control `(9.0, -5.0) → (15.0, -5.0)`: `valid: true`, no errors and no
  point or leg violations. Both checks loaded two keep-out zones.

**Browser verification passed:** footer and console loaded beta69 from the new
resource URL; the crossing path was refused by name; crossing sub-legs rendered
red with `stroke-dasharray="4,3"`; the banner named the obstacle and whole-leg
check; Real Go was disabled. The temporary waypoint was cleared afterwards.
Machine-readable record:
`docs/evidence-beta69-segment-containment-deploy-20260821.json`.

### (history) beta67 → beta68 — 2026-08-21 13:12-13:22 EDT, motion-disabled

Ships **leg-level keep-out warning** and a **legend entry** for the zones.
Gate was found ARMED at rest before the deploy (no active session, mower
stationary) and was DISARMED for it; **no motion was commanded.** Touches no
`LUBA_ACCEPTANCE_PROFILE` key; `check_accepted_profile.py` still ACCEPTED.

| | beta68 |
| --- | --- |
| Files | 46/46 byte-identical |
| Card md5 (both paths + local) | `f143465a5bb120ed759ab328c15dad9f` |
| Archive SHA-256 local = host | `3b03fa2633395162a9696a07f0b699abddd51f28e7d72837aedf71e75572247a` |
| AppleDouble entries | 0 |
| Backup | `/config/mammotion-backup-20260821-1312-pre-beta68.tgz` |
| Tag | `v0.6.4-beta68` |
| Quartet on host | manifest `0.6.4-beta68`, CARD_VERSION `0.6.4-beta68` (both paths) |
| Lovelace resource | `?v=0.6.4-beta68&build=f143465a` (read back) |
| Restart | API up 45 s; 133 Mammotion entities at 154 s |
| Config entry | `loaded` |
| Gate after | `enabled: false`, `real_motion_allowed: false`, no session |

New code confirmed in the **served** copy (`/config/www/community/…`):
`_legsCrossingKeepOuts` ×3, `_segmentsIntersect` ×2, `"keep-out zone"` ×3.

🔑 **beta67 was fully verified in a browser first**, and that is what produced
this build. All four checks passed — version, two dashed red `obstacle` zones
rendering, a click inside one refused, and the advisory reading
*"Longest leg is 3.11 m, over the 0.58 m the controller can protect… can miss by
up to 0.80 m. This is a warning, not a blocker."* (3.11 × sin 15° = 0.805, so
the arithmetic is right end to end.)

🚨 **The browser check also found a real defect: a LEG can be drawn straight
through a keep-out zone.** Both waypoints legal, containment per-point, so
neither card nor backend objects. beta68 detects the crossing, paints the leg
red and dashed, and says in the banner that neither will refuse it.
⚠️ **It warns; it does not block** — the backend still dispatches such a path,
so refusing here would make the card stricter than the machine that drives.
Segment-level containment in the BACKEND is still the real fix.

🚨 **Browser check owed for beta68**: draw a path *through* a zone and confirm
the leg renders red/dashed with the 🚨 banner line, and that Real Go stays
available.


### beta66 → beta67 — 2026-08-20 21:11-21:20 EDT, motion-disabled

Ships the **card keep-out rendering and the unprotectable-leg advisory**. Gate
DISARMED before, during and after; **no motion was commanded.** Touches no
`LUBA_ACCEPTANCE_PROFILE` key; `check_accepted_profile.py` still ACCEPTED.

| | beta67 |
| --- | --- |
| Files | 46/46 byte-identical |
| Card md5 (both paths + local) | `a582529521ee023deaaa37f22236f4ac` |
| Archive SHA-256 local = host | `9c2a70066f4130a6c36ed6f9be9ddfd61db1445026b0fc3e266a7fe10a66f36b` |
| AppleDouble entries | 0 |
| Backup | `/config/mammotion-backup-20260820-2111-pre-beta67.tgz` |
| Tag | `v0.6.4-beta67` |
| Quartet on host | manifest `0.6.4-beta67`, CARD_VERSION `0.6.4-beta67` (both paths) |
| Lovelace resource | `?v=0.6.4-beta67&build=a5825295` (read back) |
| Restart | API up 30 s; 133 Mammotion entities at 117 s |
| Backend | pymammotion `0.8.12.post1` |
| Config entry | `loaded` |
| Gate after | `enabled: false`, `real_motion_allowed: false`, no session |

**New card code confirmed present in the served copy** (`/config/www/community/…`,
the path browsers actually fetch), not just in the integration directory:
`keep_out_polygons` ×2, `_keepOutViolations` ×2, `CORRECTABLE_AIM_FLOOR_DEGREES`
×7, `_readinessLevel` ×2.

**The data it needs is live**: `export_map` returns **2** keep-out zones
(`obstacle:1529607395159402290`, `obstacle:3985039798069143977`), so the card
has real geometry to draw rather than an empty dict.

✅ **The beta66 push-before-dispatch lesson held** — `main` was pushed and in
sync with `origin` before the workflow was fired, and the release cut the
intended commit first time.

🚨 **STILL UNVERIFIED, AND ONLY THE OPERATOR CAN CLOSE IT: no browser has
rendered this card.** Everything above proves the right bytes are on the host.
It does not prove the zones draw, or that a click inside one is refused. beta49
is the precedent — four card defects existed *only* against real
`export_runtime_state` output. Confirm in the browser:

1. Card footer and console banner both read `0.6.4-beta67`.
2. Two dashed red zones labelled `⛔ obstacle` appear on the map.
3. A click inside one is refused with "that point is inside a keep-out zone".
4. A click ~3 m away shows the ⚠️ advisory naming 0.58 m — **as a warning, with
   the run still available.**


### beta65 → beta66 — 2026-08-20 19:39-19:55 EDT, motion-disabled

Ships the **advisory correctable-leg-length bound** — the leg length beyond
which the mid-drive controller cannot protect a landing, which explains both
3.0 m failures measured earlier the same day. Gate DISARMED before, during and
after; **no motion was commanded.** Touches no `LUBA_ACCEPTANCE_PROFILE` key,
so `check_accepted_profile.py` still reports ACCEPTED and no Gate 5 is owed.

| | beta66 |
| --- | --- |
| Files | 46/46 byte-identical |
| Card md5 (both paths + local) | `1032785087cbdfd1f52e41b31b336457` |
| Archive SHA-256 local = host | `d3bf98fa343904312c8b4f16a258341f80d7612019ac3b5120b74b9b29972d30` |
| AppleDouble entries | 0 |
| Backup | `/config/mammotion-backup-20260820-1939-pre-beta66.tgz` |
| Tag | `v0.6.4-beta66` at `a992d54b` |
| Quartet on host | manifest `0.6.4-beta66`, CARD_VERSION `0.6.4-beta66` (both paths) |
| Lovelace resource | `?v=0.6.4-beta66&build=10327850` (read back) |
| Restart | API up 31 s; 133 Mammotion entities at 114 s |
| Backend | pymammotion `0.8.12.post1` |
| Gate after | `enabled: false`, `real_motion_allowed: false`, no session |

**Verified executing live**, not merely deployed. A zero-motion dry run
(`would_send: false`) returned the three new fields, absent on beta65:

    correctable_leg_length_limit_m = 0.579555495773441
    longest_leg_length_m           = 3.000004524163255
    exceeds_correctable_limit      = True

0.579555 matches `0.15 / sin(15°)` computed independently, so the deployed code
is using the real profile tolerance and the real correction floor.

⚠️ **Two release traps hit during this deploy, both recorded because they nearly
shipped the wrong thing:**

1. **The workflow was fired while `main` was 10 commits ahead of `origin`.** It
   would have cut beta66 from `b28252d4` — none of the day's work, including the
   change being deployed. Caught by checking `git status -sb` after dispatch,
   cancelled mid-run (30 s), and confirmed clean: remote `main` unmoved, latest
   tag still beta65, no partial release. **Push before dispatching, always.**
2. **`pre-commit run --all-files` does not check untracked files.** A new test
   file was committed with 4 ruff `SIM300` errors after a fully green
   `pre-commit --all-files`; `ruff check custom_components tests` — the form CI
   runs — caught them. Confirmed by experiment: once the file was tracked, the
   same hook flagged it immediately. Same class as the documented hook/CI version
   skew: a green local gate that is green because it looked at less than CI does.


### beta62 → beta63 — 2026-08-20 14:18-14:30 EDT, motion-disabled

Ships the **keep-out exclusion check**. Gate was DISARMED before, during and
after; no motion was commanded.

| | beta63 |
| --- | --- |
| Files | 46/46 byte-identical |
| Card md5 (both paths + local) | `5bc908c4f32f7db4c07b5ed3e9cbf3be` |
| Archive SHA-256 local = host | `1b5230594139550af15ef57c6ee1bf359b655dba419e739f6a476c3451dfb5c0` |
| AppleDouble entries | 0 |
| Backup | `/config/mammotion-backup-20260820-1418-pre-beta63.tgz` |
| Tag | `v0.6.4-beta63` |
| Quartet on host | manifest `0.6.4-beta63`, CARD_VERSION `0.6.4-beta63` (both paths) |
| Lovelace resource | `?v=0.6.4-beta63&build=5bc908c4` (read back) |
| Restart | API up 51 s; 132 Mammotion entities at 155 s |

Gates before shipping, by exit code: pytest **733**, frontend **76**, ruff check,
ruff format, mypy, ten pre-commit hooks, `check_accepted_profile.py` **ACCEPTED**.

🏁 **THE TRAMPOLINE RUN IS NOW REFUSED PRE-DISPATCH.** Verified on the host,
`dry_run: true`, no motion (`docs/evidence-beta63-keepout-refusal-20260820.json`):

* `export_map` exposes **`keep_out_polygons`** with **2** obstacle zones in
  map-local x/y — no coordinate conversion:
  * `obstacle:1529607395159402290` — 21 pts, `x[10.471, 14.449] y[-2.540, 1.529]`,
    **3.98 x 4.07 m**. This is the trampoline: the hash the mower reported at
    contact, and the size the WGS84 geojson independently gives (~4.0 x 4.1 m).
  * `obstacle:3985039798069143977` — 30 pts, 4.82 x 3.73 m, in Front Main.
* The position where the mower **actually stopped**, `(10.5192, -0.5248)`, tests
  **inside** that polygon — independent confirmation the geometry is correct and
  correctly framed.
* Replaying the **exact recorded 10.8 m click** through the deployed validator:
  `valid: false`, `errors: ["path_points_inside_keep_out_zone"]`,
  `stop_reason: "path_validation_failed"`, `would_send: false`, and the
  violation names **split point 2 `(11.4570, 0.4772)`, type `obstacle`, hash
  `1529607395159402290`** — the inserted point, caught because the split runs
  before the preview.
* Control: a clean 0.78 m leg from the same start in the same area still
  validates, `keep_out_zones_checked: 2`, zero violations.

⚠️ **Still per-point.** A leg clipping a keep-out corner with neither endpoint
inside is not caught. Segment-level containment remains the real fix.

⚠️ The gate readback after deploy shows `ble_client_not_connected` alongside the
usual blockers — the mower had dozed off BLE, not a fault.


### beta61 → beta62 — 2026-08-19 23:43 - 2026-08-20 00:05 EDT, DEPLOYED WITH THE GATE ARMED

Ships the **deliberate safety-gate override toggles** — one per firing blocker,
29 registered gates, per-run reset, full echo into the run record.

| | beta62 |
| --- | --- |
| Files | 46/46 byte-identical |
| Card md5 (both paths + local) | `7c92d578951d702626fdb77819c8ae77` |
| Archive SHA-256 local = host | `63688511a23acdb30e062fa21dc9acddc3f8bc713e4848dc528af4d14292ea69` |
| AppleDouble entries | 0 |
| Backup | `/config/mammotion-backup-20260819-2343-pre-beta62.tgz` |
| Tag | `v0.6.4-beta62` |
| Quartet on host | manifest `0.6.4-beta62`, CARD_VERSION `0.6.4-beta62` (both paths) |
| Lovelace resource | `?v=0.6.4-beta62&build=7c92d578` (read back) |
| Backend | pymammotion `0.8.12.post1` |
| Restart | API up 41 s; 132 Mammotion entities at 123 s |

Gates before shipping, by exit code: pytest **725**, frontend **76**, ruff
check, ruff format, mypy, ten pre-commit hooks, `check_accepted_profile.py`
**ACCEPTED**. No `LUBA_ACCEPTANCE_PROFILE` key moved.

🚨 **THIS DEPLOY BROKE THE RUNBOOK'S OWN PRECONDITION, ON THE OPERATOR'S
EXPLICIT INSTRUCTION.** At the preflight check the gate read `enabled: true`,
`real_motion_allowed: true`, **`blockers: []`** — armed, with the mower off its
dock at `(4.9703, -2.0051)`, `AREA_INSIDE`, RTK Fix, BLE live, `MODE_READY`.
That is the **fourth** occurrence of the armed-at-rest posture (three were
recorded on 2026-08-18, one of them likewise with zero blockers off the dock).
The deploy was paused and the state reported; the operator said to proceed
anyway, twice. Recorded here because a later session reading "deployed
motion-disabled" for every other entry must not assume it of this one.

No motion was commanded, `active_session` and `last_session` were both null
throughout, and after the restart the mower had returned to the dock so the gate
reads `real_motion_allowed: false` on `position_not_valid_for_motion` — but
`enabled` is **still true**. The option is on.

⚠️ *(Superseded 2026-08-22 — it is now installed; see "disarm automation
installed" below. Kept as the record of the posture at this deploy.)*
⚠️ **The disarm automation is still NOT installed** (verified absent from 69
automations). Four unexplained armings is the argument for installing it that
three was not. YAML: `docs/automations/disarm-motion-gate.yaml`.

**Dry-run verification** (`docs/evidence-beta62-override-dryrun-20260819.json`),
`dry_run: true`, `would_send: false` on every call:

* no overrides → `{"requested": [], "applied": [], "any_applied": false}` and
  the run stops on `path_validation_failed`, exactly as before the feature.
* `safety_overrides: ["segment_too_long", "path_validation"]` → `path_validation`
  **applied** (it was firing: a path from the dock leaves every area polygon),
  `segment_too_long` recorded as **unused** because it was not firing on this
  geometry — the summary does not credit an override that did nothing. The gate
  record reads `original_passed: false, passed: true, overridden: true`, so an
  overridden run cannot present itself as a clean one.
* `safety_overrides: ["stop_primitive_available"]` → **HTTP 400**, refused by
  schema validation. Fail-closed on a non-overridable name confirmed live.


### beta60 → beta61 — 2026-08-19 20:16-20:35 EDT, motion-disabled

Ships **Route B** (a distant click auto-splits into collinear sub-legs of at
most 3.85 m) and **card run retention** (history entries keep their full result,
bounded by count and bytes, with quota no longer handled silently).

| | beta61 |
| --- | --- |
| Files | 46/46 byte-identical |
| Card md5 (both paths + local) | `bf98df5808b9b6af773c684cddfcb70a` |
| Archive SHA-256 local = host | `f549ca8b9f02484b4c00300863e78ea4757c32f78fae26aa49d0afb71730abfb` |
| AppleDouble entries | 0 |
| Backup | `/config/mammotion-backup-20260819-2016-pre-beta61.tgz` |
| Tag | `v0.6.4-beta61` at `ee2daf99` |
| Quartet on host | manifest `0.6.4-beta61`, CARD_VERSION `0.6.4-beta61` (both paths) |
| Lovelace resource | `?v=0.6.4-beta61&build=bf98df58` (read back after apply) |
| Backend | pymammotion `0.8.12.post1` |
| Restart | API up 51 s; 132 Mammotion entities at 135 s; config entry `loaded` |
| Gate after deploy | `enabled: false`, `real_motion_allowed: false`, `active_session: None` |

Gates before shipping, by exit code (never `| tail` — a pipeline's status is
`tail`'s): pytest **716**, frontend **68**, ruff check, ruff format, mypy, ten
pre-commit hooks, `check_accepted_profile.py` **ACCEPTED**.

⚠️ **I hand-bumped the quartet to beta61 while building, and had to undo it**
(`230e085b`). `Beta Release` computes `max(manifest beta suffix, highest tag) + 1`,
so a manifest already reading 61 would have released **beta62** while every doc
in the branch said beta61. Do not bump by hand.

**Dry-run verification, gate DISARMED, no motion commanded**
(`docs/evidence-beta61-50ft-dryrun-20260819.json`): a 15.24 m (50 ft) click
became **4 sub-legs of 3.810000 m**, all four headings identical to 9 decimal
places, `requested_points` 2 beside `points` 5, and all three inserted junctions
reported `already_within_tolerance` with **`estimated_commands_needed: 0` and
`estimated_translation_m: 0.0`** — the zero-cost junction, confirmed on the host
rather than argued from code. `would_send: false`, no session created. The
refusal path was exercised too: 3 × 5 m produced 6 sub-legs and refused with
`split_exceeds_real_segment_budget`, detail *"3 destination(s) split into 6
sub-legs of at most 3.85 m; at most 4 segments can run per click. Click a nearer
point, or fewer of them."*

🔑 **A 50 ft straight click DOES fit the yard** — the longest fully-contained
straight chord is **20.52 m** in area `…37768237` (17.50 / 11.10 / 10.31 m in
the other three), measured from the live `export_map` polygons. This corrects
the worry recorded from the "1,165 recorded positions span 12.74 × 9.73 m"
figure, which described where the mower has *been*, not where it may *go*.
⚠️ The dock is not in any mowing area, so a path starting at the dock fails
containment with or without the split — that is the dock, not the splitter.

⚠️ **NO MOTION HAS RUN ON beta61.**


### beta58 → beta59 — 2026-08-18 20:50-21:05 EDT, motion-disabled

Two releases fifteen minutes apart. beta58 shipped the empty-card
`path_validation_failed` fix; its own verification then caught a stale claim
that beta59 corrects.

| | beta58 | beta59 |
| --- | --- | --- |
| Files | 46/46 byte-identical | 46/46 byte-identical |
| Card md5 (both paths + local) | `c1f28b30…` | `cca150ff…` |
| Archive SHA-256 local = host | `a5893c55…` | `3b7477c0…` |
| Lovelace | `?v=0.6.4-beta58&build=c1f28b30` | `?v=0.6.4-beta59&build=cca150ff` |
| Restart | API 50 s, 132 entities, 134 s | API 31 s, 132 entities, 118 s |
| Gate after | ✅ disarmed | ✅ disarmed |

Backups: `mammotion-backup-20260818-2050-pre-beta58.tgz`, and the beta59
equivalent taken the same way. Backend pymammotion `0.8.12.post1`.

🏁 **beta58 was the first release to carry a DERIVED acceptance verdict in its
release body**, from the step added earlier that day:

    ## Execution profile
    ✅ Hardware-accepted. Byte-identical to the profile that passed supervised
    LUBA acceptance on 2026-08-18 (docs/evidence-gate5-beta57-20260818.json).

🚨 **What beta58's verification caught, and why it matters more than the fix.**
The card's execution-profile row still read *"NOT hardware-accepted … owes a
Gate 5"* — hardcoded on 2026-08-17, true then, and false from the moment Gate 5
passed on 2026-08-18. The release body said accepted while the card said the
opposite, on the same build.

`_profileOverrides()` could never have caught it: it diffs the payload against
`LUBA_ACCEPTANCE_PROFILE`, so it catches dashboard YAML overriding a value, and
is structurally blind to the *accepted value itself* moving. beta59 replaces the
hardcoded sentence with `ACCEPTED_PROFILE_ACCEPTED_ON` and pins it with a
frontend test that reads `docs/accepted-profile.json` and fails on any
disagreement in date or values. Verified by corrupting the snapshot date (suite
went 49/1) and restoring it (50/50) — it catches drift, it does not merely
assert today's string.

⚠️ **The motion gate was found ARMED at rest for the third time this session**
before the beta58 deploy (`real_motion_allowed: true`, zero blockers, mower off
the dock). Disarmed each time and verified. Three occurrences in one day is a
pattern worth a habit or an automation, not a coincidence.

### beta57 — 2026-08-18 17:15-17:22 EDT, motion-disabled

`0.6.4-beta57`, tag `v0.6.4-beta57` at `eae4acb8` (release commit on top of the
squashed PR #15, `5d9aa759`). Version quartet agrees: manifest / pyproject /
`CARD_VERSION` `0.6.4-beta57`, `uv.lock` `0.6.4b57`.

| | |
| --- | --- |
| Archive SHA-256 | `121189b32cd09f27972ff4f53a03090cc4c211537fdf0ae5c29dc3efedf9f397`, identical local and host |
| Files | **46 of 46 byte-identical** after CRLF normalisation |
| Card md5 | `b987b7dedd9b6c7d42ffe22bea7e0a42` at **both** paths and locally; 0 AppleDouble files |
| Lovelace | `?v=0.6.4-beta57&build=b987b7de`, written and read back |
| Backup | `/config/mammotion-backup-20260818-1715-pre-beta57.tgz` |
| Restart | API up after 81 s, 133 mammotion entities, 165 s total |
| Backend | pymammotion `0.8.12.post1` (matches the pin) |
| Gate before | ⚠️ found `enabled: True` — **disarmed before deploying** |
| Gate after | ✅ `enabled: false`, `real_motion_allowed: false`, no active session |

🚨 **THIS RELEASE UN-ACCEPTS `LUBA_ACCEPTANCE_PROFILE`.**
`max_linear_pulse_ceiling` 14 → 22. It owes the §4 re-pinning in
`docs/gate4-repass-20260805.md` and **another Gate 5**. The `Beta Release`
workflow's `confirmed_luba_acceptance` input was set true on the beta42
precedent — that build likewise adopted a profile change, shipped, and had its
Gate 5 re-passed afterward. CI passed; supervised acceptance on this profile has
not, and the debt is tracked, not discharged.

⚠️ **Found on the pre-deploy check: experimental motion was already ON**
(`enabled: True`, `real_motion_allowed: false` only because the mower was
docked). The docked position was the sole thing between that and an armed gate.
Disarmed before touching the host, per the runbook precondition. Normal posture
is disarmed; if a session finds it enabled at rest, turn it off and say so.

**Verified live, not just on disk.** A zero-motion dry run through
`raw_pymammotion_execute_vector_segment` returned both new gates executing on
the host with correct diagnostics:

    segment_too_long                        segment_length_m 19.930, max 6.1
    linear_budget_insufficient_for_segment  budget_reach_m 6.60 (22 x 0.30)

Note the 19.930: the backend measures **live position → target**, not the
planned 9 m in the request. That is the live-vs-planned asymmetry recorded in
`docs/reach-20ft-and-the-reaim-trigger-20260817.md`, confirmed on hardware.

🚨 **The mower began an autonomous mow during this deploy.** It was docked and
`MODE_READY` at 17:15; by 17:22 it read `MODE_WORKING`, blades on, in "Backyard
Hill", moving 0.28 m between two samples 20 s apart. Not the blade-RPM latch —
real motion, confirmed by position change. `active_mowing_detected` and
`blade_reported_on` correctly block the executor. **No supervised run is
possible until the mow ends.**

### beta55 — PR #14 released, motion-disabled install, 2026-08-14 20:04-20:12 EDT

`0.6.4-beta55` releases the reviewed PR #14 (merge `efa1eda8`, version commit
`5ef37511`, tag `v0.6.4-beta55`). It is beta54 plus the night residual-bearing
correction, `sample_delays: [0, 3]` on Night Go and the night harness, the Real
Go feedback-latency removal, and the bounded post-feedback BLE queue-settle
check. **No `LUBA_ACCEPTANCE_PROFILE` value moved** — the profile literal is
byte-identical to beta54 (independently hashed both sides), as is the legacy
turn branch. **No motion was commanded by this deploy.**

Backup `/config/mammotion-backup-20260814-2004-pre-beta55.tgz`. Deployment
archive SHA-256 `b96259563e460bf5ad0053a1b75577a286dd41ab50898dfae60ff1b6bc404a8f`,
verified identical after transfer. All **46** integration files byte-identical
to the local tree, zero AppleDouble entries: services `d6ab89ff4e4286b33d4fa5755bba5b0d`
(unchanged from the pre-release corrected tree), manifest
`b32d6f78e90af27e8122db0683bca405`, card `9d0ccbad94170b0d16dabde675f81db9` at
**both** serving paths. Lovelace resource read back as
`?v=0.6.4-beta55&build=9d0ccbad`.

HA API returned in **36 s**; **132** Mammotion entities in **119 s**. Config
entry `loaded`, no `setup_error`. Container `pymammotion 0.8.12.post1` against a
`0.8.12` minimum, `backend_verified: true`, both probed capabilities true with
no missing entries. Five Mammotion entities read `unavailable` — the four
`emergency_nudge_*` buttons and `start_camera_on_mower`; the nudge four are
unconditionally unavailable by design (`_nudge_available` returns `False`), not
a deploy fault.

Final readback: `enabled: false`, `real_motion_allowed: false`, no active
session, no last session, `MODE_READY`, blades `OFF` at 0 rpm with
`blade_safe_for_motion: true`, BLE the active transport and online, RTK `Fix`.
The mower is **docked** at `(4.3764, 3.1923)`, `pos_type CHARGE_ON`,
`zone_hash 0`, so `position_not_valid_for_motion` is the expected second blocker
and was equally present before the deploy.

A dark-safe dry run confirmed the deployed executor loads and runs: it returned
`valid: false`, `would_send: false`, `stop_reason: path_validation_failed`,
`path_points_outside_known_area_geometry`, zero commands — correct for a dock
position outside mapped area geometry. It also echoed the backend default
`sample_delays` of `[0, 5, 10, 20, 30, 45, 60]` for a caller that omits the key,
which is exactly why the card now sends `[0, 3]` from both profiles.

### beta54 plus unreleased Night/Real Go corrections — 2026-08-14

`0.6.4-beta54` was published from merge commit `2573c29b` and version commit
`0bd35160`, then installed with experimental motion disabled. The release adds
separate Night dry-run and Night Go card controls; it does not change any value
in `LUBA_ACCEPTANCE_PROFILE`, and Real Go remains the accepted VIO path.

One later, separately authorized card-driven night run stopped safely at
0.117085 m on `no_target_progress`. The motion gate was disarmed afterward. See
`night-go-card-beta54-20260814.md`. The run exposed a night-only crossed-target
continuation defect and excessive diagnostic waits. The deployed branch
contains a verified fix in draft PR #14 (`agent/night-real-go-followup`; code
commit `801c1798`), and the Real Go path removes additive
feedback waits while preserving its payload and safety gates. The first
supervised Real Go check stopped before its second linear dispatch on
`command_queue_backlogged`. A bounded post-feedback queue-settle correction was
then verified and deployed motion-disabled. These working-tree behaviors are
**not** part of the beta54 release artifact. Full evidence and measured/inferred
separation: `real-go-throughput-hardware-20260814.md`.

Backup `/config/mammotion-backup-20260814-pre-queue-settle.tgz`; archive SHA-256
`ec46d8fb0fce6aefcf7b2032c88a17b0693cf14cdd5bce086fb9396da5674b5d`.
Host MD5s match local: services `d6ab89ff4e4286b33d4fa5755bba5b0d`, card
`4846aa9b6f9e0eefe67cb95f8326c3ba` at both serving paths, manifest
`ef8d5273c8e9730bc964579efb63116a`. API returned after 30 s and 132 Mammotion
entities after 116 s; container pymammotion is `0.8.12.post1`. Final gate
readback: disabled, `real_motion_allowed: false`, no active session,
`MODE_READY`.

One separately authorized corrected Real Go run then reached the 0.70 m target
in 19.2 s with 0.093100 m landing error. It used zero turns and three forward
pulses. The new per-pulse queue-settle records were all live at depth zero in
about 100–101 ms; all movement commands and stops succeeded. Final telemetry
was `(4.2954, -3.8079)`, RTK Fix, `MODE_READY`, BLE −64 dBm, blades off. The
gate was independently verified disabled with no active session. Evidence:
`real-go-throughput-hardware-20260814.md` and its linked raw JSON.

### beta52 — item-17 runtime diagnostics, 2026-08-13

`0.6.4-beta52` was deployed. It added runtime-export-only RapidState diagnostics
and the fixed backward-only item-17 harness; it did not add the diagnostics to
shared VIO telemetry. Backup:
`/config/mammotion-backup-20260813-beta52-predeploy.tgz`. The deployed archive
matched local md5 `efa589bc7eaa6bf72e065529f7d44369`. Card md5
`9512f504f4b861488e98f4d29ced6e4f` matched at both serving paths; services md5
was `7c94607698ce6d9f55a4fd4a1a30f85f`. Resource read back as
`?v=0.6.4-beta52&build=9512f504`; pymammotion is `0.8.12.post1`.

The motion-disabled preview sent no command. Item 17 then ran once under
separate explicit supervision; see `night-reverse-heading-20260813.md`.
Independent final readback: gate off, no active session, `MODE_READY`, BLE live,
RTK Fix, blades zero.

### beta51 — explicit fixed-budget null, 2026-08-13

`0.6.4-beta51` is deployed. It fixes the service schemas so the harness's
explicit JSON `max_linear_pulse_ceiling: null` reaches the fixed-budget night
executor instead of failing HTTP 400. Backup:
`/config/mammotion-backup-20260813-beta51-predeploy.tgz`. Local and host
checksums agreed at deployment: card `6645732a8e39eae7644bfe84b5be01de`
at both serving paths, services `1857e0ffe118bac5c556aa04c26f9c45`, manifest
`c85553cc7c6cc536658fe1ea1478e24c`. Resource read back as
`?v=0.6.4-beta51&build=6645732a`; pymammotion is `0.8.12.post1`.

The motion-disabled dry-run returned `valid: true`, `errors: []`. §7 item 15
then ran under separate explicit supervision; see
`night-segment-turn-quantum-20260813.md`. Final independent readback: gate off,
no active session, `MODE_READY`, BLE live, RTK Fix, blades zero.

### beta50 — night v1, 2026-08-13 ~17:02 local

`0.6.4-beta50` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260813-beta50.tgz`. Local and host checksums agree:
card `8510824e965f2dbf6b2403c822c54b39` at both serving paths, services
`d36789b2e622d066873cb396d82e5d76`, manifest
`78ce64d5138ca68a7283ba0d3d93248d`. Resource read back as
`?v=0.6.4-beta50&build=8510824e`; container pymammotion is `0.8.12.post1`.

Gate readback after restart: `enabled: false`, `real_motion_allowed: false`, no
active session. No movement service was called and no motion was commanded. The
mower independently reported `MODE_WORKING` with active mowing/route blockers
at final readback; the deploy session did not interfere. Night v1 hardware task
15 remains unrun and needs separate explicit supervised authorization.

|                         | Host                                                           | Branch         |
| ----------------------- | -------------------------------------------------------------- | -------------- |
| Integration version     | `0.6.4-beta32` staging candidate (deployed 2026-08-09, motion-disabled) | `0.6.4-beta32` candidate |
| pymammotion pin         | `0.8.12.post1` fork wheel (container verified)                 | same           |
| Card `CARD_VERSION`     | `0.6.4-beta32`; integration and HACS copies checksum-identical | `0.6.4-beta32` candidate |
| `manual_motion.py`      | present                                                        | present        |
| `backend_capability.py` | present                                                        | present        |
| `capabilities.py`       | present                                                        | present        |

The live host ran backend Gates 1-4, two failed-safe beta16 daylight
short-approach runs, the beta20 Gate 4 re-pass, and the beta21 second-geometry
reproduction. Beta22 is an **unaccepted, motion-disabled staging candidate**. It
carries the four-field Gate 4 re-pass profile, the off-mower executor/estimator
corrections recorded in `docs/gate4-repass-20260805.md`, and the reverse-recovery
containment guard; it does not establish mower acceptance.
Experimental motion is verified off. Gate 5 and release remain blocked; do not
merge or publish.

⚠️ **Beta22 is expected to make Gate 4 fail.** The guard refuses a correction
present in *both* recorded Gate 4 passes, so a Gate 4 run on this build will
most likely stop with `target_requires_reverse_recovery` where beta20/beta21
reported `target_reached`. That is containment, not regression: the earlier
passes were bought by driving past the waypoint and U-turning back. The next
motion decision is whether to fix control quality (stop-latency lead, pulse
sizing) or to accept overshoot-and-recovery — not a Gate 5 attempt.

### beta49 — beta48's live defects, 2026-08-13 ~05:20 UTC

Frontend-only again. `card md5 adaf0b71`, local == both host paths, resource
`?v=0.6.4-beta49&build=adaf0b71`, API back in 51 s, 132 entities,
`real_motion_allowed: false`.

🔑 **Four defects were found by looking at the deployed card, none by the test
suite.** The stubs supply blocker lists the real backend does not.

The duplicate-blocker one is confirmed directly in the live runtime state:

```
em blockers     ['experimental_motion_disabled', 'position_not_valid_for_motion']
safety blockers ['position_not_valid_for_motion']
```

`_preflight()` concatenated those two lists, so the banner printed
`position_not_valid_for_motion` twice (and `rtk_not_precise` twice, when RTK was
also down). Both call sites now dedupe. Two emitted codes
(`position_not_valid_for_motion`, `ble_client_not_connected`) had no help text
and silently dropped out of the explanation list; a test now pins the full set
observed on the host.

**Method note.** Render the card against the actual `export_runtime_state`
output, not the test fixtures, before calling a UI change done.

### beta48 — card usability and run export, 2026-08-13 ~04:45 UTC

**Frontend-only deploy.** Card + `manifest.json` only; no `services.py` change
and **no `LUBA_ACCEPTANCE_PROFILE` key touched**, so the profile stays accepted
and no §4 re-pinning is owed. Verified:

```
card md5      fbf4f621bd78eb5425f8beee4a2aa231   local == /config/custom_components/... == /config/www/community/...
served        /hacsfiles/... and /mammotion/...  both fbf4f621, CARD_VERSION 0.6.4-beta48
resource      ?v=0.6.4-beta48&build=fbf4f621
manifest      0.6.4-beta48
API back      35 s      132 mammotion entities at 153 s
gate          real_motion_allowed: false   (blockers: experimental_motion_disabled,
                                            ble_client_not_connected, position_not_valid_for_motion)
```

⚠️ `ha_set_card_resource.py` takes only a version string and does **not** append
the `build=` suffix the runbook requires. Pass it as part of the argument —
`ha_set_card_resource.py "0.6.4-beta48&build=fbf4f621" --apply` — or the
registered key loses the checksum half of its cache-busting.

What changed: Real Go / dry-run / history JSON **downloads**, a per-segment
**landing table** (leg, landing, tolerance, verdict, pulses, mean and worst), a
colour-coded **readiness banner** that keeps blocker codes verbatim and adds
plain English for every one of them, the toolbar grouped Path / Run / Export, and
the 14-row diagnostics panel collapsed behind its verdict line.

### beta43/44 — Gate 5 re-pass and the profile-echo fix, 2026-08-12

**`0.6.4-beta43` deployed motion-disabled 2026-08-12 ~20:35 UTC** and is the
build that **passed Gate 5 on the reach-enabled profile** — see
`docs/gate5-repass-PASSED-20260812.md`. 46/46 byte-identical, both card paths
`f92f2e71`, resource `?v=0.6.4-beta43&build=f92f2e71`, API back in 51 s, 132
entities in 134 s, `pymammotion 0.8.12.post1`, `real_motion_allowed: false`
verified before arming.

beta43's one motion change: the post-turn correction gets the same command
budget as any other turn (`vio_turn_max_commands`, was `min(2, ...)`). ⚠️ **It
was not exercised by the gate** — the only correction was −10.477°, inside the
old 21.50° envelope.

**`0.6.4-beta44` supersedes it and changes no motion behaviour.** The card's
execution-profile label drops "Gate 5 re-pass PENDING", and both executors now
echo every accepted-profile key. Two came back `null` during the gate
(`max_no_progress_pulses`, `motion_refresh_interval_ms`) and the pass had to be
argued around the hole — unacceptable in a gate whose entire purpose is proving
the card sent the accepted profile.

✅ **The profile is ACCEPTED again.** `max_linear_pulse_ceiling: 14` no longer
owes a Gate 5; it has one.

### beta42 reach-profile deploy — 2026-08-12 07:08-07:20 UTC

`0.6.4-beta42` is deployed **motion-disabled**. Backup:
`/config/mammotion-backup-20260812-0708-beta42.tgz`. All **46** files
byte-identical to the tree, zero AppleDouble; both card copies
`09a1d05ebbd79889a01a334dd9e3ef4b`. API back in **51 s**, 128 Mammotion entities
in 165 s. Resource read back as `?v=0.6.4-beta42&build=09a1d05e` (was
`beta41&build=174f317d`). Backend verified `pymammotion 0.8.12.post1`. Host
readback: `CARD_VERSION = "0.6.4-beta42"`, `max_linear_pulse_ceiling: 14`,
`manifest.version 0.6.4-beta42`. Gate off, `real_motion_allowed: false`, mower
docked at (4.3188, 3.2862) `CHARGE_ON`, `MODE_READY`.

🚨 **THIS BUILD CHANGES THE ACCEPTED PROFILE AND IS THEREFORE UNACCEPTED.**
`max_linear_pulse_ceiling` moved `null` → **14**, adopting loop-to-tolerance on
the operator's decision and 2026-08-11 hardware evidence (2/3/4 m legs landing
0.0690 / 0.0928 / 0.1023 m). Per §4 of `docs/gate4-repass-20260805.md` that owes
a **fresh Gate 5, card-driven, which has NOT been run.** The card's own
execution-profile label now says so out loud: "Gate 5 re-pass PENDING". Do not
describe any run on this build as profile-accepted until that gate passes.

Also in this build: the mid-drive re-aim guard projects to the end of the next
pulse rather than to the closest approach (no profile key).

### beta32 preflight-model correction deploy — 2026-08-09 01:16-01:22 EDT

`0.6.4-beta32` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260809-0116-beta32.tgz`. All **46** files
byte-identical to the tree (aggregate `634abef928fa5dcd4c53f5dfafe4cbae`), zero
AppleDouble; both card copies `16d883faa32fc8aa5d399038245474cf`. API back in
**40 s**, 128 entities in 133 s. Resource read back as
`?v=0.6.4-beta32&build=16d883fa` (was `beta30&build=8ec0fb01` — the host skipped
beta31 entirely). Backend verified `pymammotion 0.8.12.post1`. Gate off,
`real_motion_allowed: false`, no session, `MODE_READY`, RTK **Fix**, BLE −48.

**This is the first deploy of the beta31 reach work, and it carries a fix beta31
needed.** beta31 was adversarially reviewed before deployment and did not clear;
beta32 is beta31 plus one refusal-side correction — the turn feasibility preflight
now models the overshoot ceiling instead of assuming full-length pulses. See
`docs/HANDOVER-beta31-20260809.md` §2.6 for the four findings, three of which
remain **open**.

⚠️ **Four open findings ride along on this build.** The ceiling costs ~18° of turn
capability (a 90° junction completes on beta30 and exhausts the 4-command budget
on beta31/32 at the rates Gate 5 actually measured); its guarantee is written in
commanded rather than delivered milliseconds; `16.5 °/s` is no longer a rate
floor; and the ceiling biases turn landings into a tighter post-turn gate. **The
validation run therefore keeps every junction in the 45–70° band.**

Verified live after deploy, zero motion: `real_click_to_go_segment_limit: 4`, and
a dry-run 4-segment path with 60° junctions inside `Backyard Right` returns
`command_count_model: "executor_pulse_policy_replay"` with the ceiling-shortened
ladder `[1300.0, 942.5, 683.3]` — **pulse 1 is already ceiling-bound at 1300 ms
rather than the configured 1500**, which is the ceiling binding below 72° exactly
as designed, and 3 of 4 commands needed leaves one of margin. Evidence:
`docs/evidence-beta32-deploy-dryrun-20260809.json`.

Motion preconditions were **not** met at deploy time and no motion was attempted:
night (`initial_vio_feed.live: false`, `tracked_features: 0`, brightness `Dark`),
mower docked at `CHARGE_ON`, `position_not_valid_for_motion`. The 4-segment
validation run waits for daylight, a charged battery and fresh authorization.

### beta26 RTK design inversion deploy — 2026-08-07 20:46-20:50 EDT

`0.6.4-beta26` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260807-2046-beta26.tgz`. All **46** files
byte-identical (aggregate `18078de314c7c0b277dc121b504adf9c`), zero AppleDouble;
both card copies `62bbaaddfbce341e9bc7172f5adb1bb6`. API back in **31 s**,
entities in 120 s. Resource `?v=0.6.4-beta26&build=62bbaadd`. Backend verified
`pymammotion 0.8.12.post1`. Gate off, no session, `MODE_READY`, blades 0.

**Removes a live false-block and inverts the RTK design.** beta25 refused motion
with `rtk_telemetry_stale` past 1800 s, and a healthy Fix-locked *stationary*
mower was then measured reaching **3573 s** of unchanged RTK payload — so beta25
would have refused legitimate runs after ~30 idle minutes.

⚠️ **No age threshold can work, and an active probe cannot substitute.** The RTK
payload changes ~hourly at rest while the one observed fault lasted ~3 h. Forcing
a report burst on *healthy* RTK produced 49 messages and **zero** RTK channel
updates with the age still climbing — indistinguishable from latched. So the
earlier claim that a forced burst "caught" the latch is withdrawn.

Now: `rtk_report_age_seconds` / `rtk_report_quiet` are **reported for auditing
and never block**. The real guard is the **quality gate** — non-Fix refuses with
`rtk_not_precise` unless `allow_degraded_rtk` is passed — which catches the fault
actually observed (a latched *Float*). **Do not re-add an age blocker;**
`_RTK_REPORT_QUIET_SECONDS` records both failed attempts. Reasoning:
`docs/rtk-hardening-plan-20260807.md`.

Verified live after deploy: `rtk_report_age_seconds: 30.663`,
`rtk_report_quiet_threshold_seconds: 1800.0`, `rtk_report_quiet: false`,
`rtk_telemetry_stale` **absent from the payload entirely**, `blockers: []`,
`rtk_status_label: Fix`, `rtk_degraded: false`.

### beta25 RTK threshold correction deploy — 2026-08-07 19:22-19:26 EDT

`0.6.4-beta25` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260807-1922-beta25.tgz`. All **46** files
byte-identical (aggregate `f1bbea08235697eb75b618282b0df98c`), zero AppleDouble;
both card copies `3431ab4413eff37b3766f80937d17676`. API back in **30 s**,
entities in 114 s. Resource `?v=0.6.4-beta25&build=3431ab44`. Backend verified
`pymammotion 0.8.12.post1`. Gate off, no session, `MODE_READY`, blades 0.

**Why this shipped the same evening as beta24:** beta24's RTK staleness
threshold of 300 s was measured wrong within the hour. A healthy Fix-locked
*stationary* mower went 582 s without an RTK payload change, so beta24 would
have refused legitimate motion with `rtk_telemetry_stale`. Raised to 1800 s —
~3x the longest legitimate quiet observed, 6x under the three-hour latch it
catches. Verified live after deploy: `rtk_report_age_seconds` reporting,
`rtk_telemetry_stale: false`, `rtk_report_stale_threshold_seconds: 1800.0`, no
blockers.

⚠️ The bound is **under-characterised**: the upper limit on legitimate quiet is
unknown, only that it exceeds ten minutes. If a run is refused with
`rtk_telemetry_stale` while RTK is demonstrably healthy, raise the constant
rather than assuming a latch.

### beta24 RTK freshness guard + per-channel probe deploy — 2026-08-07 19:03-19:07 EDT

`0.6.4-beta24` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260807-1903-beta24.tgz`. All **46** integration
files byte-identical to the tree (aggregate
`2bf75c1312257fbcf58d1e75ab4a7cb8`), zero AppleDouble; both card copies
`73b1774077957755ef26eed846f4c186`. HA API returned in **41 s**, entities in
125 s, no `setup_error`. Lovelace resource read back as
`?v=0.6.4-beta24&build=73b17740`. Backend verified `pymammotion 0.8.12.post1`.
Gate verified off before and after; no session; no motion commanded.

Two changes, both off-mower:

- **RTK freshness guard.** The coordinator fingerprints the RTK payload on every
  report refresh; the safety summary reports `rtk_report_age_seconds` and blocks
  with `rtk_telemetry_stale` past 300 s. Wired into the three authorization
  boundaries only, not the fifteen diagnostic call sites. An unmeasurable age is
  `None` and does not block. **No Fix requirement was added** — that threshold
  remains an open operator decision.
- **Per-channel report attribution** in `report_stream_probe`.

⚠️ **First per-channel result changes the plan.** Stationary with RTK Fix, 40 s:
75 messages arrived at ~1.9 Hz and **none** carried a changed position, RTK or
VIO payload. The position-report cadence therefore cannot be measured from a
stationary mower — the measurement needs a moving one, in daylight, under fresh
authorization. See `docs/evidence-per-channel-report-probe-20260807.json`.

### beta23 report-rate probe deploy — 2026-08-07 00:17-00:21 EDT

`0.6.4-beta23` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260807-0017-beta23.tgz`. All **46** integration files
byte-identical to the local tree (aggregate `71ef5ccddb3dd04b33732cd4871b555a`),
zero AppleDouble entries; both card copies `885630808fafb80c7b38c39ea1bad628`.
HA API returned in **51 s**, no `setup_error`. Lovelace resource read back as
`?v=0.6.4-beta23&build=88563080`. Container backend verified
`pymammotion 0.8.12.post1`. The new `report_stream_probe` service is registered
on the host with all four fields (60 mammotion services total). Motion gate
verified off before and after; no session; no motion commanded.

Adds exactly one thing: a read-only `report_stream_probe` diagnostic. No motion
path, service schema default, `LUBA_ACCEPTANCE_PROFILE` value, or entity
platform changed.

⚠️ **Entity-count note — read before treating a mismatch as a regression.**
`scripts/ha_restart.sh` prints an entity count but **exits as soon as it reaches
100**, so the printed number depends on when the poll lands mid-load. It read
**130** here versus **128** on earlier deploys; the *settled* count is **131**
(18 unavailable/unknown for an idle mower). The historical "128 Mammotion
entities" in the sections below is therefore a readiness sample, not a stable
invariant. Compare settled counts, or ignore the number.

### beta22 reverse-recovery containment deploy — 2026-08-06 19:51-19:56 EDT

`0.6.4-beta22` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260806-1951-beta22.tgz`. All **46** integration files
were byte-identical to the local tree by per-file md5 (aggregate
`dbab51a64ff86032fec28b130d2d0605`); the tarball was built with
`COPYFILE_DISABLE=1` plus explicit `._*`/`__pycache__` excludes and the host
carries **zero** AppleDouble entries. Both card copies matched
`49dd1df816162f523285d485e4a8cb6e`. HA's API returned in **41 s** and all **128**
Mammotion entities in **108 s**, with no `setup_error`.

The Lovelace resource was updated and read back as
`/hacsfiles/mammotion/mammotion-custom-path-card.js?v=0.6.4-beta22&build=49dd1df8`.
Container backend verified `pymammotion 0.8.12.post1`;
`export_runtime_state` reported `backend_verified: true` with both capabilities
true, `real_motion_allowed: false`, `enabled: false`, no active session, no
route (`reason: no_route`), `MODE_READY`, blade `OFF` at 0 rpm with
`blade_safe_for_motion: true`, and position `(5.6444, -4.4875)` RTK `Fix` /
`AREA_INSIDE` — the unchanged 2026-08-06 second-geometry resting point.

Both card paths were additionally fetched over HTTP and returned `200` with the
same md5, so the new card is proven served, not merely present on disk. The
deployed `services.py` was grepped on the host and carries
`_MAX_FORWARD_REALIGNMENT_DEGREES = 90.0`; the deployed card carries
`max_linear_commands: 3`, `linear_pulse_duration_ms: 1300`, and
`max_turn_translation_distance: 0.3`.

What changed on the host relative to beta21: `services.py` (the ≥90°
reverse-recovery guard and the fail-closed realignment budget) and the card's
`CARD_VERSION`. `LUBA_ACCEPTANCE_PROFILE` is byte-identical, the
execution-profile label still reads exactly
`LUBA acceptance profile (Gate 4 re-pass, 2026-08-05)`, and no service schema or
entity platform file changed. No motion service was called and no motion was
commanded.

**BLE was initially unverifiable, then verified at 20:29.** At the 19:56 readback
the transport had not re-registered (`ble_transport_not_registered`,
`active_transport: none`, `online: false`) across an 8-minute poll, because the
mower battery was at **2%** — a flat mower never advertises, so the transport
cannot register. The `ble_rssi: -62` in that readback was the mower's own last
self-report, a **cached value and not a liveness signal** (the same trap recorded
on 2026-07-25); `online: false` corroborated. The read-only
`scripts/mammotion_preflight_gates.py` correctly reported `BLE link live` as a
FAIL throughout that window.

The mower docked itself at 20:10 EDT — **no motion was commanded by this
session** — and charged to 26%. The 20:29 re-check is clean: `active_transport:
ble`, `online: true`, `ble_rssi -46`, preflight `BLE link live PASS (entity=on
transport=ble rssi=-48)`, which matches the excellent dock proxy coverage
measured on 2026-07-29. Motion remained disabled with no session throughout.

⚠️ **The mower is now on the dock at `(4.3188, 3.2862)`**, so the standing
blockers are `experimental_motion_disabled` + `position_not_valid_for_motion`,
and `zone_hash: 0` with `pos_type: CHARGE_ON` are the correct docked readings.
Undock into a mapped area before any future authorized run.

Durable record: `docs/evidence-beta22-containment-deploy-20260806.json`.

### beta21 Gate 4 profile staging deploy — 2026-08-05 21:59-22:07 EDT

`0.6.4-beta21` is deployed motion-disabled from the candidate tree. Backup:
`/config/mammotion-backup-20260805-beta21-staging.tgz`. All **46** integration
files matched the local tree aggregate hash
`ee4a94bfb540bbcd05311cc7754047ba`; the tarball contained no AppleDouble or
`__pycache__` entries. Both card copies matched
`59a9a7dd4b7451ffce13cd0494df4646`. HA's API returned in 55 seconds and all
128 Mammotion entities returned in 142 seconds.

The registered Lovelace resource was updated and read back as
`/hacsfiles/mammotion/mammotion-custom-path-card.js?v=0.6.4-beta21&build=59a9a7dd`.
Browser readback showed the beta21 footer and console banner, exact
`LUBA acceptance profile (Gate 4 re-pass, 2026-08-05)` label, and
`PyMammotion backend 0.8.12.post1 (verified)`. Real Go, Nudge, and dry-run were
disabled with no path set. A transient BLE/queue blocker appeared immediately
after restart and cleared during the reconnect window; the final runtime-state
readback contained only `experimental_motion_disabled`, with
`real_motion_allowed: false`, no active session, and `MODE_READY`. No motion
service was called and no motion was commanded.

### beta17 correction deploy — 2026-08-02 20:44-20:52 EDT

`0.6.4-beta17` is deployed motion-disabled. Backup:
`/config/mammotion-backup-20260802-2045.tgz`. All 46 integration files matched
the local tree by content hash, no AppleDouble files were present, and both
card paths matched (`170aca89b0cc5a514dc9d835aee7a3b8`). HA's API returned in
66 seconds and all 128 Mammotion entities returned in 151 seconds. The loaded
wheel is `pymammotion 0.8.12.post1`; backend capabilities and BLE are verified.
The mower remained ready, RTK Fix, blades off, session-free, and experimental
motion off.

Chrome exposed a cache-key collision: the updated card first loaded as beta17
through the prior `beta23` key, then the plain beta17 URL replayed an older
cached beta16 response. The live resource is therefore
`?v=0.6.4-beta17&build=a2b0d4bf`. After reload, both console and footer reported
beta17. The exact accepted-profile label rendered, Preview was valid, and the
card Dry-run returned `valid: true`, `would_send: false`, and
`stop_reason: dry_run`; Real Go stayed disabled. The card was reset afterward.
VIO was dark/0 features, so no reacceptance motion was attempted.

### beta18 map-marker correction deploy — 2026-08-02 21:29-21:39 EDT

Live UI inspection separated two arrows: the custom-path card's green
arrow correctly rendered upper-right at 72.8 degrees, while HA's standard map
card rendered the mower's black tracker picture upper-left. The tracker was
publishing raw Mammotion orientation `-29`; HA expects clockwise compass
degrees. Beta18 publishes `(-orientation) % 360` (29 degrees for that sample).
This is presentation-only: no executor or accepted-profile value changed. The
candidate passed 469 coverage-enabled Python tests, 19 frontend tests, Ruff,
format, scoped mypy, all-files pre-commit, and the GitHub validation workflow.

The integration was backed up to
`/config/mammotion-backup-20260802-2129.tgz` and deployed motion-disabled. All
46 files matched the tree aggregate hash
`f14c608a02203602f2463fd0e6a30f6b`; no AppleDouble files were present; both
card copies matched `694fd1b0b54ab336c9490c620bd4f8cb`. HA's API returned in
31 seconds and all 128 Mammotion entities returned in 115 seconds. The live
wheel remains `pymammotion 0.8.12.post1`, backend capabilities are verified,
the accepted-profile label is exact, console/footer report beta18, and the
Lovelace resource is `?v=0.6.4-beta18&build=6da6c3d3`. Experimental motion
remained off, no session existed, and no motion was commanded.

The installed `custom:map-card` 1.15.0 accepts marker CSS but does not itself
consume the tracker's `direction`. After backing up the dashboard to
`/config/.storage/lovelace.dashboard_yard.bak.codex-20260802-213848`, a
Jinja-backed `card-mod` rule was added to rotate its `.entity-picture` by the
live `direction` attribute. Readback matched the requested config. Browser
inspection found `direction: 29.0` and computed transform
`matrix(0.87462, 0.48481, -0.48481, 0.87462, 0, 0)`, confirming a 29-degree
clockwise upper-right marker that will update with the entity.

That last conclusion was **rejected by direct operator observation**. The mower
physically faced upper-left while the custom card's projected arrow and the
rotated third-party marker pointed upper-right. A zero-command dry-run snapshot
showed why: `toward: -29.589` and `location.orientation: -29` were frozen last
travel, while VIO was inactive with heading 0 and RTK yaw was 0. No available
field represented stationary body orientation. The `card-mod` rotation was
removed with verified Lovelace readback.

### beta20 turn-feasibility guard deploy — 2026-08-04 20:16-20:35 EDT

Ships the corrected turn-feasibility guard. `LUBA_ACCEPTANCE_PROFILE` is
byte-identical; the only card change is `CARD_VERSION`. No service schema and
no entity platform file changed (`git diff 9ef3d103..HEAD` touches exactly
`manifest.json`, `services.py`, and the card). No motion ran and no gate is
claimed.

Backup `/config/mammotion-backup-20260804-2016.tgz`. All **46** deployed files
were byte-identical to the local tree by per-file md5; both card copies matched
`2b1d37bb99069020d2c3eea54b512e9b`; zero AppleDouble entries in the tarball.
HA API returned after 31 s, Mammotion entities after 235 s, backend verified
`pymammotion 0.8.12.post1`. Lovelace resource bumped to
`?v=0.6.4-beta20&build=2b1d37bb`, re-read to verify, keeping the collision-proof
pattern with the build hash tied to the deployed card's md5.

Two dark-safe dry runs proved the guard is live on the host:

- standalone turn, 179.571° error against a 4-command budget → `feasible:
  false`, `turn_budget`, 7 commands needed, with the new
  `translation_bound_m_per_degree: 0.0026` and
  `translation_bound_source: conservative_observed_translation_per_degree`;
  `estimated_translation_m: 0.467` = 179.571 × 0.0026, i.e. angle-scaled rather
  than command-scaled. `would_send: false`, `commands_sent: 0`.
- multi-segment L path → `junction_turn_feasibility` for the −90° junction is
  **feasible**: 3 commands against 4, `estimated_translation_m: 0.234` against
  `max_displacement_m: 0.25`.

⚠️ That junction cap is **0.25 m, not the schema's 0.5 m default**, and it is
what bounds the per-degree constant from above. 90 × 0.0026 = 0.234 fits with
0.016 m to spare; the initially proposed 0.0028 would have given 0.252 and
**refused Gate 4's own junction geometry** on this build. Do not raise the
constant without re-checking this dry run.

Experimental motion was verified off with no session before and after.

### beta19 stale-orientation correction deploy — 2026-08-02 22:07-22:12 EDT

Beta19 draws only the mower position dot unless the backend explicitly supplies
a trustworthy map-aligned current orientation. It labels the old calculation
as a last-travel projection rather than mower orientation. Nudge now fails
closed on `current_orientation_unavailable`; Real Go behavior, service schemas
and the accepted profile are unchanged. Frontend tests pin the separation
between projected travel and trusted current orientation.

The host was backed up to `/config/mammotion-backup-20260802-2207.tgz`. All 46
deployed files matched the tree aggregate hash
`2c344e3234c175fd85be066259ffcd75`; both card copies matched
`9152496e514058948ad338103130519f`; no AppleDouble files were present. HA's API
returned in 76 seconds and all 128 Mammotion entities returned in 201 seconds.
The installed backend remains `pymammotion 0.8.12.post1`; experimental motion
was off with no session before and after deployment.

Chrome initially replayed beta16 despite the beta19 resource key, so the final
auditable URL is `?v=0.6.4-beta19&build=617337d3`. The reloaded card reported
beta19 and the exact accepted-profile label. DOM inspection found one green
mower-position dot, zero green heading lines, zero arrowheads, and the explicit
`current orientation unavailable ... not mower orientation` text. Nudge was
disabled. No physical motion was commanded.

### Gate 5 deploy — 2026-07-31 19:50-20:05 EDT

`0.6.4-beta12` is deployed. Done while the mower was docked (`CHARGE_ON`,
`MODE_READY`), with no active session and `enable_experimental_motion` off.
**No motion of any kind was commanded.** Backup taken first:
`/config/mammotion-backup-20260731-1950.tgz`.

Verified after the restart:

- all 46 integration files md5-identical to the tree;
- no AppleDouble `._*` files (tarball built with `COPYFILE_DISABLE=1` plus an
  explicit exclude);
- both card paths byte-identical to the tree
  (`a186a394ec17593c5dca8e86484a9983`) and both serving `CARD_VERSION`
  `0.6.4-beta12` over HTTP, containing `LUBA_ACCEPTANCE_PROFILE`;
- HA API back in 30 s, 128 Mammotion entities in 112 s — no `setup_error`;
- container `pymammotion` `0.8.12.post1`; `backend_verified: true` with both
  capabilities true;
- `enable_experimental_motion` still **false**, no session, `real_motion_allowed`
  false.

⚠️ **The registered Lovelace resource was the trap, not the files.** The
dashboard had `/hacsfiles/mammotion/mammotion-custom-path-card.js?v=12` — an
arbitrary cache key that does **not** track the version, so replacing the file
alone would have left every browser on the cached beta11 card while every
server-side check said beta12. It is now
`?v=0.6.4-beta12` (updated through the `lovelace/resources/update` websocket
call, not by editing `.storage`). Note the live dashboard registers the
**`/hacsfiles/` path**, not the documented `/mammotion/` one, which is exactly
why both copies must be written every time.

### beta13 deploy — 2026-08-01 23:02-23:12 EDT

`0.6.4-beta13` is deployed (heading arrow on the card, plus the VIO
fail-closed evidence). Mower was stopped in `Backyard Right`, off the dock,
experimental motion **off**, no session. **No motion commanded.** Backup:
`/config/mammotion-backup-20260801-2302.tgz`.

Verified: 46/46 files md5-identical to the tree; no AppleDouble entries; both
card paths and the tree byte-identical (`6bec2ca3a83186be9c4f7b410a0a2a3c`) and
both serving `CARD_VERSION 0.6.4-beta13` containing `_headingDegrees`; API back
in 35 s with 128 entities and no `setup_error`; container `pymammotion`
`0.8.12.post1`; `backend_verified: true` with both capabilities; experimental
motion still off.

The Lovelace resource was already at `?v=0.6.4-beta13` (updated by the operator),
so no change was needed — but **check it every deploy**, and use
`scripts/ha_set_card_resource.py` rather than editing `.storage`. Bumping the
card file without bumping that key leaves every browser on the previous card.

### beta16 deploy — 2026-08-02 15:33-15:36 EDT

`0.6.4-beta16` is deployed with the precise waypoint-coordinate editor. The
motion gate was verified off before backup and remained off throughout; no
motion was commanded. Backup:
`/config/mammotion-backup-20260802-1532.tgz`.

Verified: 46/46 integration files checksum-identical to the tree; no AppleDouble
entries; both card paths byte-identical (`ea6a84303293addba1d18a65738cdefa`);
API back in 51 s and 128 Mammotion entities in 165 s; container `pymammotion`
`0.8.12.post1`; backend capabilities verified; BLE live; RTK Fix; VIO Light with
80 features; blade RPM zero; no session. The Lovelace resource is verified at
`?v=0.6.4-beta16`.

### The override guard earned itself on first live use

With beta12 loaded, the operator's card reported
`customised (not hardware-accepted): linear_pulse_duration_ms`. The
`dashboard-yard` card config still carried **`linear_pulse_duration_ms: 2000`**
from an older calibration.

That is the known no-op. Forward motion needs **>= 3 s to trigger at all** —
2 s pulses taped as physical no-ops on 2026-07-18, which is why the accepted
profile uses 3500 ms. The backend schema is `Range(min=50.0, max=4000.0)`, so
`2000` is **accepted silently**: it would have been dispatched, not rejected.
Run as Gate 5, the mower would most likely have stood still while the session
reported dispatched pulses and confirmed stops — the exact failure the
execution-profile row exists to surface.

Removed through the `lovelace/config/save` websocket call (backup taken first;
the live config was then diffed against the backup to prove the dropped key was
the only change). Re-checked by running the real card module against the live
saved config: profile row `LUBA acceptance profile (Gates 1-4, 2026-07-31)`,
zero overrides, `linear_pulse_duration_ms` 3500, ceiling still omitted.

Lesson for any future deploy: a correct file deploy is **not** a correct
run configuration. Check the execution-profile row, not just the version banner.

Before any newly authorized Gate 5 motion: deploy and reaccept the beta19
correction recorded in `docs/p0-beta-release.md`; do not treat 0.3-0.5 m as a
proven usable band. Confirm the browser console banner reads
`v0.6.4-beta19` (a hard refresh may be needed), confirm the card's execution
profile row reads exactly
`LUBA acceptance profile (Gates 1-4, 2026-07-31)`, save the emitted payload and
dry-run result, and obtain a fresh daylight operator `go`. Use the same card
instance without editing waypoints between the final dry-run and Real Go. Run
`motion_capture.py` and `ble_session_report.py` across the whole window, save
the Real Go result, and always disarm and verify no session after success,
failure, or abort.

### beta19 backend Gate 2 acceptance — 2026-08-03

A daylight backend 0.100 m segment was accepted. The operator observed an
approximately 9 cm move, then a stop. The result reported `target_reached` and
a 0.0105 m final error; VIO calibration supplied the whole observed movement,
so no normal linear pulse was required. Teardown cleared the session, disabled
experimental motion, and telemetry remained stationary for over one minute.
The recorded evidence is `docs/evidence-gate2-beta19-*20260803*`. This is not
Gate 5: backend Gate 4 and then the unchanged-card UI acceptance remain
required before release.

### beta19 backend Gate 4 retry — failed, 2026-08-03

The guarded two-segment retry is not an acceptance. Its durable result records
segment 1 `turn_phase_incomplete`: VIO calibration passed, but the four-command
turn budget ended `max_commands_reached` 34.795° short of target after 0.185 m
of incidental turn translation. No linear command or segment 2 ran. The gate
was disarmed, session cleared, blades stayed off, and post-stop telemetry was
stationary. See `docs/evidence-gate4-beta19-retry-real-*20260803*` and the
offline analysis in `docs/evidence-gate4-beta19-retry-diagnosis-20260803.json`.
Do not retry or expand the turn budget without the implementation and daylight
re-characterization required by `docs/CLAUDE-FINAL-IMPLEMENTATION-PROMPT.md`.

### Turn-feasibility guard implemented on-branch — 2026-08-03, NOT deployed

The correction demanded by the Gate 4 retry is committed to
`feat/vio-turn-to-heading` and verified locally only; the host still runs the
`617337d3` beta19 build without it. A real VIO turn is now refused **before
its first command** (`turn_budget_infeasible`, `commands_sent: 0`) when the
evidence-bounded per-command progress (16.5°/s × pulse with refresh; the
8°/command single-shot quantum without) cannot reach tolerance within
`vio_turn_max_commands`, or when the refresh-regime translation estimate
(0.0026 m per degree swept, revised 2026-08-04) would breach the displacement
cap. The multi-segment
executor additionally refuses a real path whose fixed junction geometry is
infeasible (`path_turn_infeasible`) before any motion; dry runs report the
same math without refusing. `scripts/diagnose_motion_result.py` reports the
refusal as `vio_turn_refused_infeasible_preflight`. Tests:
`tests/components/mammotion/test_vio_turn_feasibility.py`; the full suite is
483 passing. The accepted profile, service schemas, and all four version
locations are unchanged, so this section adds **no new deploy step yet**:
shipping the guard requires a normal version-bumped deploy (both card paths +
backend + Lovelace key) after a daylight turn characterization validates the
constants, and any deployed build must re-run affected gates from scratch.

## Breaking enum migrations already applied

The host now reports lowercase states such as `mode_working`, `man`, `english`,
and `mowing`. Automations copied from an older release must be migrated from
uppercase matching. The original label remains available in
`raw_protocol_value`; see `p0-beta-release.md`.

## Steps

1. **Back up** the current integration and card:

   ```sh
   set -a && source .env && set +a
   scripts/ha_ssh.exp 'cd /config/custom_components && tar -czf /config/mammotion-backup-$(date +%Y%m%d-%H%M).tgz mammotion && ls -la /config/mammotion-backup-*.tgz'
   ```

2. **Ship the complete integration tarball.** Build it with
   `COPYFILE_DISABLE=1 tar …` — macOS BSD
   tar otherwise embeds AppleDouble metadata files, which extract as 46 junk
   `._*` entries **inside the integration**, including `translations/._en.json`
   next to the real translation files. They were removed by hand on 2026-07-29
   (`rm -f ._* translations/._* www/._*`); prevent them next time instead.

   ```sh
   scripts/ha_scp.exp <scratchpad>/mammotion_deploy.tgz /config/mammotion_deploy.tgz
   scripts/ha_ssh.exp 'cd /config/custom_components && tar -xzf /config/mammotion_deploy.tgz && echo extracted'
   ```

3. **Use the integration-served card resource.** Register
   `/mammotion/mammotion-custom-path-card.js?v=<installed-version>` as a
   JavaScript module. The old HACS copy currently matches, but it is no longer
   the documented source. If an existing dashboard still references
   `/hacsfiles/mammotion/`, update that copy during migration or change the
   resource URL:

   ```sh
   scripts/ha_ssh.exp 'cp /config/custom_components/mammotion/www/mammotion-custom-path-card.js /config/www/community/mammotion/mammotion-custom-path-card.js && md5sum /config/custom_components/mammotion/www/mammotion-custom-path-card.js /config/www/community/mammotion/mammotion-custom-path-card.js'
   ```

4. **Verify checksums match the tree** before restarting — compare against
   `scratchpad/local_md5.txt`:

   ```sh
   scripts/ha_ssh.exp 'cd /config/custom_components/mammotion && find . -type f \( -name "*.py" -o -name "*.json" -o -name "*.yaml" -o -name "*.js" \) ! -path "./__pycache__/*" -exec md5sum {} \; | sort -k2'
   ```

5. **Restart HA.** `scripts/ha_restart.sh`. Expect the API back in ~40-60 s and
   Mammotion entities in ~2-3 min.

   On restart HA will **pip-install the fork wheel from GitHub**, because the
   requirement is a URL and `is_installed()` always returns False for URLs. This
   needs working egress from the container. If it fails, the integration will not
   set up — go to Rollback.

6. **Bump the Lovelace resource cache key** to the installed release version.
   `CARD_VERSION` may change, but browsers key on the query string, so without
   this they can keep the cached card:

   ```sh
   scripts/ha_set_card_resource.py                 # show current
   scripts/ha_set_card_resource.py 0.6.4-betaN     # dry run
   scripts/ha_set_card_resource.py 0.6.4-betaN --apply
   ```

   It uses the `lovelace/resources` websocket API, keeps the registered path
   as-is (the live dashboard references the `/hacsfiles/` copy, not the
   integration-served `/mammotion/` one), and re-reads to verify. Do not edit
   `/config/.storage/lovelace_resources`; HA holds it in memory and overwrites.

7. **Verify** (all dark-safe, no motion):

   ```sh
   scripts/ha_ssh.exp 'docker exec homeassistant python -c "import importlib.metadata as m; print(m.version(\"pymammotion\"))"'
   # expect 0.8.12.post1
   ```

   Then `mammotion.export_runtime_state` should now contain an
   `experimental_motion` block with
   `backend_capabilities.capabilities` both true and `backend_verified: true`.
   Confirm entity count, maps and tasks, diagnostics, and card preview plus
   dry-run. Real Go is not part of a routine deployment check.

8. **Leave `enable_experimental_motion` off.** The gate now opens the moment it is
   toggled on. Toggle it with
   `scripts/ha_set_experimental_motion.py on|off|status` rather than by hand:

   ```sh
   set -a && source .env && set +a
   scripts/ha_set_experimental_motion.py status
   scripts/ha_set_experimental_motion.py on     # prompts for ARM unless --yes
   scripts/ha_set_experimental_motion.py off
   ```

   Two traps it exists to avoid. The options flow returns `create_entry` with an
   **empty `data`** payload even when it applied the change, so the reply proves
   nothing — the script re-reads `export_runtime_state` instead. And the flow
   replaces the whole options dict, so every other option must be resubmitted;
   those values come from the flow's **own schema defaults**, because
   `/api/config/config_entries/entry` never exposes `options` (it returns `{}`
   whatever is configured) and preserving from it silently resets them. The
   field is also `prefer_ble_over_wifi`, not the `prefer_ble` used elsewhere;
   the wrong name fails the flow with a bare HTTP 400 and no field detail.

9. **Enable the right logger before any BLE measurement.**
   `scripts/ble_session_report.py` matches `connected=… mtu=… error=…`, which is
   emitted by **`bleak_esphome.backend.client`** — not by pymammotion. Setting
   pymammotion to debug does nothing for it. Enable at runtime (no restart, and it
   does **not** survive one):
   ```
   service: logger.set_level
   data: {bleak_esphome: debug, habluetooth: debug}
   ```
   The lines only appear on connect/disconnect transitions, so a stable link
   produces none. A zero from the report means "no transitions or wrong logger",
   never "a healthy link" — verify the logger is on before trusting a zero.

## 🔑 Proxy coverage at the dock is already excellent (measured 2026-07-29)

`habluetooth.wrappers` logged the connection-path selection for the docked mower:

| proxy                   | RSSI    | slots free |
| ----------------------- | ------- | ---------- |
| `p1s-printer-a5774c`    | **−49** | **2/3**    |
| `esphomes3-irk`         | −65     | 3/3        |
| `bluetooth-proxy`       | −77     | 3/3        |
| `garage-m5stack-9bc1d4` | −85     | 3/3        |
| `atom-fireplace`        | −85     | 3/3        |

Five paths, best at −49, and the connection opened through P1S Printer. Two
standing beliefs need correcting: the p1s-printer proxy is **not** permanently
one-slot-busy (2/3 free here), and **the dock does not need a closer proxy**. The
coverage problem is out in the working area, not at the dock — so site the next
proxy where the mower mows, and start acceptance runs near the dock.

That table is historical discovery evidence, not the accepted topology. The
successful Gates 1-4 used P1S as the sole enabled mower proxy. The IRK proxy was
isolated after it reproduced the app/HA connection conflict; do not add it back
to a release acceptance run without a controlled stationary comparison.

## 🔬 How to attribute the improvement despite the confound

First 20-minute window after the deploy (docked, idle, much of it with the link
still down, so treat as indicative only):

| metric                     | 07-27 baseline (8 h, 0.8.8) | first 20 min (0.8.12.post1) |
| -------------------------- | --------------------------- | --------------------------- |
| sequence gaps              | 720 (**1.5/min**)           | 1 (~0.05/min)               |
| unparseable LubaMsg frames | 193                         | **0**                       |
| negotiated MTU             | 22x 517                     | 1x 517                      |

**The two metrics dissociate, and that is the attribution test.** Sequence gaps are
link-layer packet loss, which neither audited fix touches. Unparseable frames are
exactly what the BluFi reassembly reset prevents — a gap no longer poisons the
next message. So:

- unparseable frames near zero **while gaps still occur** = clean evidence the
  reassembly fix works, despite the bundled 0.8.8 -> 0.8.12.post1 jump.
- both dropping together = something environmental changed too, and no
  single-fix claim is safe.

Check both over the full overnight window before concluding anything.

## ⚠️ The overnight A/B is confounded — say so in the write-up

The 2026-07-27 baseline (median session 59 s, 42% `0x08`) was measured on
**0.8.8**, not 0.8.12. This deploy jumps 0.8.8 → 0.8.12.post1, which lands the
0.8.9-0.8.12 upstream changes, the #177 rate-limit fix, the BluFi reassembly
reset, **and** the teardown fix together. So an improvement cannot be attributed
to the teardown fix alone. The measurement still answers "is the link better
now?", which is the decision that matters for the acceptance run — but do not
record it as proof of the leak fix specifically.

## Rollback

```sh
scripts/ha_ssh.exp 'cd /config/custom_components && rm -rf mammotion && tar -xzf /config/mammotion-backup-<stamp>.tgz && echo restored'
scripts/ha_restart.sh
```

Then restore the prior Lovelace resource version. Reverting to any build without
the audited backend capabilities makes the probes re-lock real motion by
themselves, but disable experimental motion before rollback regardless.
