# RTK: settle it properly — plan, 2026-08-07

Written after the 2026-08-07 Float episode and the guard work that followed. The
investigation behind it **changed the design substantially**, including
invalidating two conclusions I reached earlier the same day, so read §2 before
acting on anything already committed.

No motion is authorized by this document.

## 1. What actually happened, and what is verified

**The episode.** RTK read `Float` from 15:40 to 18:39 EDT. Rover reception was
healthy throughout (24 co-viewed satellites both bands, 26 tracked,
`rtk_over_datalink`) and the base station had not been moved. A rover-side
`sync_rtk_and_dock` did nothing. Power-cycling the **dock and base station**
restored `Fix` at 18:39:08, with the whole RTK group refreshing together
(`position_level` 2 → 1, so **1 = Fix, 2 = Float**).

**The cost of Float, measured:** a **13.9 cm** position jump with no command
sent, against **0.044 cm mean / 0.55 cm max** stationary jitter under Fix. A
single Float excursion exceeds the entire 0.08 m waypoint tolerance.

**This was the first recorded Float episode.** Mining every committed evidence
file: **534** recorded `Fix` states, **zero** `Float` or `Single` outside
today's own investigation file. (The 265 `None` records are the dead
`mowing_state` position candidate, which always reports `rtk_status 0`, not real
episodes.) **Past gate results are therefore not tainted** — every prior gate ran
on Fix. That was a live worry and it is now closed.

## 2. ⚠️ Two of my own conclusions from today were wrong

Both were load-bearing for the guard already committed, and both are refuted by
measurements taken this evening.

**2.1 — "A forced report burst caught the latch." It cannot catch anything.**
This afternoon I forced 50 reports, saw no RTK entity refresh, and concluded the
channel was latched. Repeating that test at 20:35 with RTK **verifiably healthy
and `Fix`**: 49 messages arrived, **zero** RTK/position/VIO channel updates, and
the payload age kept climbing 570.9 s → 601.1 s. A healthy feed and a latched
feed look **identical** under a forced burst. The test has no discriminating
power and the afternoon's inference from it was invalid.

**2.2 — The 1800 s staleness threshold will false-block. It is a live defect in
the deployed beta25.** Watching a healthy, stationary, Fix-locked mower, the RTK
payload age reached **3573 s (59.5 minutes)** before resetting. The RTK payload
changes roughly **once an hour** at rest. The deployed threshold is 1800 s, so
motion will be refused with `rtk_telemetry_stale` after ~30 idle minutes with
nothing wrong. This is the *second* time this constant has been set too low —
300 s failed the same way this afternoon.

**The structural conclusion:** legitimate quiet reaches ~1 h and the observed
fault lasted ~3 h. A 3x separation, from one sample of each, is far too poor to
site a timeout on. Combined with 2.1 — no passive threshold works, and no active
probe exists — **RTK freshness is not verifiable with any signal currently
available.** The guard as designed rests on a discrimination that does not exist.

## 3. The corrected design

**Freshness becomes advisory. Quality becomes the guard.**

The key realisation: the fault we actually observed was a latched **Float**
value. The quality gate (`rtk_not_precise`, non-Fix refuses) blocks that
directly, with no freshness reasoning required. Freshness was only ever needed
for a *different*, **hypothetical and never-observed** failure — a latched
**Fix** that silently stops being true. Guarding a hypothetical with a signal
that cannot discriminate, at the cost of false-blocking every run after 30 idle
minutes, is a bad trade.

| concern | mechanism | status |
| --- | --- | --- |
| RTK degraded (Float/Single) | `rtk_not_precise` blocker + `allow_degraded_rtk` override | committed, **not deployed** |
| RTK feed latched | `rtk_report_age_seconds` **reported, never blocking** | needs change — currently blocks |
| Run auditability | `rtk_status_label`, `rtk_degraded`, `rtk_degraded_override`, age in every result | committed |
| Base-station fault | operator runbook + recurrence watch | §5 |

## 4. Actions

### P0 — remove the live false-block (do first)

`_RTK_REPORT_STALE_SECONDS` currently gates motion in deployed beta25 and will
refuse legitimate runs. Change `rtk_telemetry_stale` from a blocker to a reported
diagnostic in `_runtime_motion_safety_summary` (`services.py`). Keep
`rtk_report_age_seconds` in the output — it is genuinely useful for auditing a
run after the fact, it just must not gate one. Record in the constant's comment
why a threshold cannot work, so it is not reintroduced a third time.

Ship together with the already-committed quality gate as **beta26**, since that
gate is the thing that actually protects a precision run and is not yet on the
host.

### P1 — verify the quality gate on hardware

The `rtk_not_precise` blocker has never run against a real degraded state. Once
deployed, confirm on the host that a normal Fix run is unaffected and that
`allow_degraded_rtk` is echoed in results. The refusal path itself can only be
exercised opportunistically — the next time RTK is genuinely Float, check that it
refuses and that the override permits.

### P2 — characterise, then decide whether freshness can ever be gated

The overnight watch (`scratchpad/rtk_watch.jsonl`, 6 h at 30 s) is collecting the
quiet-period distribution. If it shows legitimate quiet is bounded well below the
observed 3 h fault — say reliably under 90 min — a timeout becomes defensible at
a much higher value. If quiet is unbounded or highly variable, **freshness stays
advisory permanently** and that should be written down as settled, not revisited.

Also worth extracting from that log: whether satellite count or signal quality
drift *precedes* a Float transition, which is the only early-warning candidate
identified so far.

### P3 — root cause, which is still unknown

A base power cycle is a **remedy, not a diagnosis**. Ranked hypotheses, with what
would distinguish each:

1. **Base survey-in never completed or completed badly.** Most consistent with
   the evidence: rover fine, corrections arriving, only a base restart helps.
   Distinguisher — whether the base exposes survey state or a fixed reference
   position anywhere in its UI or telemetry.
2. **Base firmware hang** emitting stale or malformed corrections. Distinguisher
   — recurrence pattern and whether it correlates with uptime.
3. **Ambiguity-resolution failure from base-side multipath.** Distinguisher —
   whether the base has a clear sky view, and whether Float recurs at the same
   time of day (constellation geometry).

None of these is reachable from the mower's telemetry, which is the honest
limitation: **the integration cannot diagnose the base station.** The practical
step is a recurrence log — date, duration, what cleared it — so a pattern can
emerge. One episode cannot be root-caused.

### P4 — the operator runbook

Short, in `docs/deploy-runbook-p0.md`: if RTK reads Float, do not spend a session
on it. Confirm reception is healthy (co-viewed satellites non-zero), try
`sync_rtk_and_dock` once, then **power-cycle the dock and base station** and
allow ~20 minutes. That is the sequence that worked; the rover-side steps did
not.

## 5. What is deliberately not being done

- **No attempt to gate on freshness with a tuned constant.** Two attempts, two
  false-blocks. The signal does not support it.
- **No base-station integration work.** Nothing in the mower's telemetry reaches
  the base's internal state; building on that would be guesswork.
- **No change to `waypoint_tolerance`.** That decision belongs with the separate
  cadence finding (position updates ~1031 ms during motion, ~47 cm travelled
  between updates), not with RTK.

## 6. Verification

- Full matrix per `CLAUDE.md`: pytest with coverage, ruff check, ruff format
  check, scoped mypy, `npm run test:frontend`, all-files pre-commit.
- Tests must cover: freshness reported but never blocking; the quality gate
  refusing Float/Single/None; the override permitting while still recording
  `rtk_degraded: true`; unknown status staying advisory.
- Deploy verification per the runbook: 46/46 file md5 match, zero AppleDouble,
  both card paths identical, backend verified, gate off, no session.
- Post-deploy: confirm on the host that `rtk_telemetry_stale` no longer appears
  in `blockers` while `rtk_report_age_seconds` is still reported, and that a
  normal Fix state produces an empty blocker list.

## 7. Open questions

- Is legitimate RTK quiet bounded? (P2 answers this.)
- Is there any early-warning signal before a Float transition? (P2.)
- Why did the base fail? (P3 — needs recurrence data; one episode is not enough.)
- Does the mower ever report a latched **Fix** that has stopped being true? Never
  observed. If it never appears, the freshness concern can be closed entirely.
