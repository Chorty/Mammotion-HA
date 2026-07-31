# Codex working-plan index

Updated 2026-07-31. This file used to contain a long chronological scratchpad.
That history made old assumptions look current, including that VIO had not been
tested and multi-segment execution had not started. Those statements are no
longer true.

Use these sources instead:

- `docs/NEXT-SESSION.md` — concise Claude/Codex handoff, current blockers, safe
  next actions, and changed assumptions.
- `docs/p0-beta-release.md` — safety model, chronological hardware evidence,
  migration notes, verified limitations, and release criteria.
- `docs/deploy-runbook-p0.md` — current live-host state, safe deployment and
  rollback procedure, and BLE logging guidance.
- `docs/custom-path-execution-research.md` — architectural conclusion: no
  firmware arbitrary-waypoint upload was found; click-to-go uses guarded manual
  motion instead.
- `docs/archive/NEXT-SESSION-2026-07-28.md` — archived investigation details.

Current summary: the guarded backend and supervised LUBA Gates 1-4 have passed,
including abort and a corrected two-segment L path. Experimental motion is off.
The next release decision is not another backend primitive: reconcile the
card's older default Real Go profile with the conservative profile that passed
Gate 4, then close the explicitly listed P0 blockers and rerun the full CI
matrix. Do not push or write to a `mikey0000` repository.
