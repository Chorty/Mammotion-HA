# Backend bumps and fork retirement

The shipping integration stays on its reviewed fork wheel while the replacement
is built offline. A green canary is software evidence, not permission to deploy
or move the mower. Hardware qualification follows completion of the offline
work and a separate operator go.

## Canary and weekly drift report

`Upstream backend canary` runs on PRs, main pushes, manual dispatch, and weekly
on Monday at 13:23 UTC. It creates the shipping test environment, resolves the
latest stable non-yanked `pymammotion` from PyPI, installs that candidate in the
runner, checks dependencies, and runs **all of `tests_ha`**. The existing
shipping validation workflow remains the blocking release check.

The advisory job retains logs, exact package versions, version sources, JUnit
XML, and failure groups for 30 days. It keeps the installed-pin assertion in
the raw suite and reports that intentional mismatch separately. A native-object
probe separately exercises identical report receipts, position epochs, warm-up,
and the audited backend capability check without starting a client. Installation,
dependency, and collection failures must be investigated; they do not count as
compatibility passes. Test doubles can hide missing native backend contracts.

The scheduled drift job compares the shipping wheel's upstream base with both
PyPI's latest stable backend and the latest upstream HA release's manifest pin.
For drift, it creates or updates one open `Upstream pymammotion drift` issue in
**Chorty/Mammotion-HA**. Identical reports leave the issue untouched. It never
writes to upstream repositories and never automatically bumps a pin. Issue
writes occur only on the trusted scheduled default-branch run; PRs and manual
dispatches have read-only permissions. Schedules become active after merge.

GitHub's [schedule behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)
and [job-level continue-on-error](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idcontinue-on-error)
define those CI boundaries.

## Backend bump checklist

- [ ] Record the shipping version, candidate version, source revision, upstream
  release, fork wheel base, and exact environment. Preserve baseline and
  candidate outputs, not just totals.
- [ ] Review upstream changes to transport lifecycle, positions, queue/saga
  admission, send ordering, settings, maps, and camera contracts.
- [ ] Reconcile upstream into the Chorty PyMammotion release branch before
  cutting a fork wheel. Verify the new wheel contains every previously accepted
  fork fix, especially failure-atomic BLE teardown. Do not cut from an unrelated
  branch merely because its version is newer.
- [ ] Update the same wheel pin in **all four places**:
  `custom_components/mammotion/manifest.json`, `requirements_test.txt`,
  `pyproject.toml`, and `uv.lock`. Regenerate the lock; do not hand-edit a partial
  package entry. Version changes are a separate PR from this canary.
- [ ] Refresh the actual test venv and Home Assistant component requirements.
  Read the `pymammotion:` pytest header. Run `pip check`; do not suppress conflicts.
- [ ] Run CI-equivalent checks: Ruff lint/format, Mypy with
  `--follow-imports=skip custom_components/mammotion`, `pytest tests_ha`, and
  `npm run test:frontend`. Review relevant followed-import API changes as well:
  the skip mode alone cannot detect backend API drift.
- [ ] Exercise native backend objects for BLE cleanup, identical position
  receipts, transport epochs, warm-up, queue ordering, and stop behavior.
  Passing fake-coordinator tests alone do not satisfy this item.
- [ ] Resolve or explicitly retain every canary failure. The expected installed
  pin mismatch is bookkeeping; a missing motion capability remains a blocker.
- [ ] Preserve accepted motion-profile values byte-for-byte. A deliberate
  control-law change requires its existing predeclaration/acceptance process.
- [ ] Prepare the deployment and rollback artifacts, then obtain a separate
  operator deployment go. Record host versions and integration file hashes.
- [ ] Verify entry loading, dark-safe dry run, and gate disarmed in **live API
  and raw config-entry storage**, with storage checked after its write delay.
- [ ] If transport, position stream, queue, or send paths changed, obtain a
  supervised-leg go and require a declared click-to-go PASS before calling the
  backend qualified. Do not replace this with a fixture-only test.

## Retained feature target

Operator decision, 2026-10-03: retain click-to-go, `real_motion_ready`,
`ble_link_live`, route overlay, raw-pulse operation, mowing settings, and other
settings wherever their real backend contracts permit. Preserve settings
behavior, including read-before-modify, refusal at device limits, and changing
one working field without overwriting the other fields. Do not silently drop a
feature or expose a control that cannot work.

| Feature group | Intended owner | Offline work / acceptance |
| --- | --- | --- |
| Click-to-go and stop/disarm actions | `mammotion_motion` companion | Move the accepted executor and gates; retain control-law values and emergency-stop ordering. |
| Readiness, live BLE, backend/blade/position safety, VIO diagnostics | Companion | Match existing definitions, freshness and fail-closed behavior; update dependent entity IDs explicitly. |
| Planned route and coverage overlay | Companion + card | Retain route normalization, route hash, serving paths, coordinate system, and zoom/click geometry. |
| Raw-pulse probe | Companion | Retain its gate wrapper and bounded stop path; qualification is separate from implementation. |
| Blade height, speed, spacing, obstacle handling, running-job reads, start/modify settings | Upstream where equivalent; companion fills proven gaps | Read real job values first; preserve untouched fields; validate model limits; mark unreadable jobs unavailable/refused. |
| Non-work hours, blade reminders/reset, task and schedule settings | Upstream where equivalent | Inventory every fork control and service against the target upstream release; test semantic parity, not just matching names. |
| Bluetooth, device Wi-Fi/4G, prompt volume, voice language | Upstream where equivalent; companion fills proven gaps | Check sibling-coordinator propagation and distinguish transport preferences from real device settings. |
| Camera/wiper/stream controls and other buttons | Upstream where equivalent | Keep supported settings/control behavior. Camera streaming and motion qualification remain separate. |
| Faults, obstacle/lock/fuse sensors, RTK source, health/cloud diagnostics | Upstream where equivalent; companion fills proven gaps | Preserve observable fields only; document missing telemetry rather than invent values. |
| Additional research probes | Existing closed-work decisions apply | Retain dependencies needed by the chosen features; do not reopen night or continuous-steering research. |

## Known unavailable contracts and loss register

The original beta123 comparison against 0.10.7 recorded 1,245 passes, 26
failures, and 22 skips, versus 1,271 passes and 22 skips on 0.9.6.post1.
One failure was the intentional installed-pin mismatch. These counts describe
that source and test population; later canaries may contain additional tests.

| Gap on stock 0.10.7 | Current impact | Required disposition |
| --- | --- | --- |
| Audited failure-atomic BLE cleanup absent | Real motion remains blocked | Implement and verify an owned compatibility layer or reviewed Chorty backend fix; never bypass the capability guard. |
| Native receipt stream and transport epochs absent | Comms-abort verification cannot confirm; native warm-up refuses | Provide every-report receipt sequences and link-generation epochs, including identical and solicited reports, with unload/reload/cancellation tests. |
| Brightness helper changed numeric meanings and labels | Bright-scene cold start is refused | Derive explicit semantics for this mower. Unknown values stay a refusal. Capitalization alone is insufficient. |
| `OperationSettings.rain_tactics` absent | Fork running-job modifications raise before sending | Reconcile the complete route-settings contract. The old field itself has no proven upstream write equivalent; do not guess one. |

**No user-facing item is yet proven impossible to retain.** The four groups
above are unresolved implementation/contract gaps. Keep a row for each control
that cannot be carried over, with: old entity/service, upstream equivalent (if
any), missing field or behavior, reproducible evidence, user-visible impact,
and proposed substitute. An unavailable control must remain unavailable; a
final removal requires an explicit operator decision. In particular, the
missing rain field does not authorize dropping all mowing settings.

## Offline build and qualification order

- [ ] Inventory all retained entities, settings, services, card consumers, and
  scripts against exact source revisions. Update older inventory conclusions:
  native position-stream support is now required by warm-up and stationary
  verification.
- [ ] Build in isolated companion/card branches. Reuse upstream's single
  client; do not open a second BLE connection. Re-resolve runtime objects on
  reload and release hooks/subscriptions on unload.
- [ ] Implement the compatibility contracts above, ownership of map/plan saga
  admission, and `zone_hash` derivation. Reachable monkeypatch seams are not
  evidence of safe production lifecycle behavior.
- [ ] Move executor closure by membership from the accepted beta123 source.
  Mark changed seams and compare the copied bodies/profile with the source.
- [ ] Complete setting/control parity and the loss register. Keep upstream
  features in upstream when equivalent; avoid duplicate writable controls.
- [ ] Verify native-object tests, HA setup/unload/reload, malformed/stale
  telemetry refusals, frontend geometry, and settings preservation. Build the
  installable artifacts and prepare exact rollback instructions.
- [ ] Only after the offline items are complete, prepare the one-step host
  cutover: upstream HA integration, companion, card. Qualification requires
  entry `loaded`, zero-motion stop checks 3/3, dark-safe dry run, gate disarmed
  (API + raw storage), and a supervised click-to-go PASS on operator go.

This canary/checklist PR does not perform the offline companion extraction or
the host cutover. It establishes the checks and retained scope for that build.
Writing to `mikey0000` repositories remains outside the authorized scope.
