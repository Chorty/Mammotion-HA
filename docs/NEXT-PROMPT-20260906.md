# Working prompt — next session (handoff from 2026-09-05)

Read `CLAUDE.md` in full first (all live). Then, only if you pick up the item it
belongs to, read the linked findings doc for that item. **Verify every claim
here against the tree and live HA before acting** — this file was true at the
2026-09-05 handoff and the mower/HA state drifts.

---

## Live state at handoff (verify before acting — true 2026-09-05 ~16:30 UTC)

- Host runs **0.6.4-beta103**, backend `chorty-0.8.12.post4`. Deploy verified
  byte-identical; browser-confirmed at beta102 and the card is unchanged since.
- Mower **docked and charging**, ~80%+, RTK **Fix**, VIO 80 features,
  BLE back to **-56 dBm**. Gate **disarmed, verified live API AND RAW**.
- `main` is pushed and clean. `git status` shows only the untracked OTA capture
  artefacts (all gitignored) and the OTA probe script/test if not yet committed.
- 🔴 **Do NOT commit anything under `ota_tls_probe/`, `ota_work/`, or any
  `*.ota`** — real private key, a signed URL, and 213 MB of firmware. Gitignored
  this session; verified never committed. Stage by explicit path, never
  `git add -A`.

---

## What 2026-09-05 settled

1. 🏆 **The facing question is ANSWERED and the model works on the ground.**
   `runtime_state.map_facing` (shipped beta102) predicted the driven direction
   to a **mean 1.382°** over four scored forward legs (max 1.961°) against a
   predeclared 10° bar — **PASS 4/4**. Record:
   `docs/findings-facing-prediction-20260905.md`.
2. 🔴 **Device fault codes now reach the operator** (beta102). `1309` reads in
   full, all ten log slots are exposed, an unknown code (`5004`) degrades to the
   number rather than `"mcu: , "`. Confirmed live.
3. 🐛 **The travel guard was tripping at zero travel on every real run**; fixed
   in beta103 and confirmed on hardware. `max_travel_m` works now.
4. **The mower's OTA firmware was captured, and the OTA line was reaffirmed
   CLOSED by the operator with the firmware in hand** — see item 1, nothing to do.

---

## The work — pick what the operator wants; none of it is forced

### 1. 🛑 OTA firmware — CLOSED, do not reopen

The firmware was captured on 2026-09-05 (213 MB), and the operator was asked
whether that reopens the deliberately-closed OTA line. **It does not.** No OTA
capture, analysis, or decryption work is planned; a captured file is not a reason
to reopen it. The probe tool `scripts/ota_tls_probe.py` is kept OUT of the repo
by operator decision (untracked local tooling). The capture artefacts are
gitignored and must never be committed. Full record:
`docs/ota-firmware-capture-investigation-20260816.md`. **Nothing to do here —
listed only so it is not re-raised.**

### 2. If the operator wants more facing/motion confidence

The 4/4 PASS has a scope limit that is easy to over-read — **read §0 of
`docs/findings-facing-prediction-20260905.md` before quoting the result.** Every
leg ran within ~4° of `toward = 173.865°`, the heading where the compass mirror
and the additive offset are EQUAL, so the series does **not** discriminate the
two models. A series that wants to must start the mower near `toward = 353.865°`
(models ~180° apart). The case against the additive offset still rests entirely
on the banked 43-pulse data.

Two smaller, honest follow-ups from that findings doc (both need a mower,
daylight, a fresh predeclaration):

- **The +1.38° systematic bias.** All five legs drove clockwise of the estimate.
  Direction is established (5/5 same sign), magnitude is not (inside the per-leg
  noise floor). 🛑 Do not fit a correction — it needs *longer* legs, not more
  short ones.
- **The freshness-TTL friction.** The 300 s motion-confirmation TTL makes a
  per-leg operator go/no-go unrunnable as the protocol was written, and the
  facing tracker is in-memory so an HA restart makes the first dispatch
  unscorable. Both are design questions the next predeclaration should settle,
  not code bugs.

### 3. Loose diagnostic threads (low priority, offline unless noted)

- **`5004`** appeared twice during the run and is absent from all three code
  tables we have. Watch for a correlation with motion; identify it if it recurs.
- **`rtk_signal` and `age` both read 0** live. The sensors are implemented
  correctly; whether the device populates those fields is unverified. ~10 min
  against a moving mower if it ever matters.

---

## Boundaries (unchanged, all still binding)

- 🛑 **Standing decisions hold**: Phase 2 continuous steering (5), accuracy (3),
  night (4) all CLOSED; reliability stats beta57+ epoch only (7). OTA (6) is
  CLOSED and was reaffirmed 2026-09-05 with the firmware captured — do not reopen.
- 🛑 **`docs/accepted-profile.json` is untouched and stays that way** without a
  predeclaration and a Gate 5. Today's facing PASS authorizes no profile change.
- Never push to `mikey0000/*`; pass `-R Chorty/Mammotion-HA` to every `gh`.
  Stage by explicit path — **never `git add -A`** (the OTA secrets make this
  non-negotiable right now).
- Keep the predeclare-before-you-dispatch discipline in CLAUDE.md → "How this
  project works". It is what made today's results trustworthy, and it caught two
  of today's own missteps (the guard bug voided two dispatches cleanly; the
  4/4 result carries its own scope limit rather than being oversold).

---

## Housekeeping done 2026-09-05, so it is not re-litigated

- The "permanently dirty" `docs/agora_outbound_audio_probe.md` was two stray
  keystrokes, now fixed; the working tree is clean. CLAUDE.md's `git add -A`
  bullet was corrected — the rule stands, its old justification was a phantom.
- Both of today's armed-gate sightings were the operator using the click-to-go
  card, **not** the disarm-automation defect. Standing count stays at **six**.
  Ask before attributing an armed gate to the defect.
- `.vscode/` is now gitignored.
