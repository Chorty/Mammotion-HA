# Predeclared: first driven per-proxy BLE coverage collection (2026-09-16)

Written and committed **before any trace row exists** —
`scripts/ble_proxy_connected_trace_log.jsonl` does not exist at commit time.
Nothing below may be changed after the first dispatch; a changed rule needs a
new, dated amendment that names what it changes and does not rescore banked rows.

## 0. Question and scope

**Question:** under the current, unmodified RF set, which proxy does HA allocate
for a driven session at the Phase 1 anchor, and what connected RSSI does the
mower report from that proxy, cell by cell, along a known route?

**This is run 1 of a series, and its first job is to validate the instrument.**
The connected tracer has never produced a row, and a bug that would have
discarded every row it did produce was found only while planning this
(`build_ble_coverage_map.py` read `rssi`, the tracer writes `ble_rssi`; fixed and
tested in the same commit as this document). So breadth is deliberately
sacrificed for depth on a route that already runs clean.

**Out of scope, and not claimed by any outcome:** other proxies' coverage (one
run yields one proxy — whichever HA allocates); cells off the route; any change
to `_BLE_MOTION_QUEUE_START_TIMEOUT_SECONDS`, the accepted profile, or
reliability statistics (standing decision 7). Landings are recorded but **not
scored**.

## 1. RF configuration — frozen as found

All five scanners as they stand (`hci0` scan-only; `hot-tub-backyard`,
`p1s-printer`, `garage-m5stack`, `atom-fireplace` connectable), on the
2026-09-15 ESPHome firmware. **No proxy is disabled to force an allocation.**
Window D of `docs/findings-ble-proxy-adjacency-test-20260915.md` showed that
removing a proxy made the link worse (7 drop/reconnect cycles in 45 min), and a
run that needs a degraded link to choose its proxy measures the degradation.
Choosing *which* proxy a later run maps is a separate operator call.

The live scanner list is read and recorded immediately before the first leg.

## 2. Route — the Phase 1 S1–S12 targets, unchanged

Exactly the absolute targets of
`docs/predeclared-queue-timeout-measurement-20260911.md` §16.2 — N–S axis through
the anchor `(4.94, -3.82)`, every target ≤ 3.0 m from it, turn-backs at S4 and S9:

| leg | target | 1 m cell centre it lands in |
| --- | --- | --- |
| S1 | (4.94, -4.82) | (4.5, -4.5) |
| S2 | (4.94, -5.82) | (4.5, -5.5) |
| S3 | (4.94, -6.82) | (4.5, -6.5) |
| S4 | (4.94, -5.82) | (4.5, -5.5) |
| S5 | (4.94, -4.82) | (4.5, -4.5) |
| S6 | (4.94, -3.82) | (4.5, -3.5) |
| S7 | (4.94, -2.82) | (4.5, -2.5) |
| S8 | (4.94, -1.82) | (4.5, -1.5) |
| S9 | (4.94, -2.82) | (4.5, -2.5) |
| S10 | (4.94, -3.82) | (4.5, -3.5) |
| S11 | (4.94, -4.82) | (4.5, -4.5) |
| S12 | (4.94, -5.82) | (4.5, -5.5) |

**Route cells:** the six cells `x 4..5`, `y -7..-1`. A landing error up to
0.15 m can put samples in the `x 5..6` column; those count as off-route cells
(§5), never re-binned into the route column.

Why this route: it ran 12/12 first try with zero halts on 2026-09-15
(`docs/findings-phase1-repeat-20260915.md`), it stays inside the runner's
excursion and −76 dBm band checks without modification, and it crosses each
route cell 2–4 times, which is what makes ≥ 10 samples per cell reachable.

**Start:** the operator parks the mower at the anchor using the **dock-relative
offset** (≈1.1 m east, 7.1 m south of the dock), never an absolute lat/lon. The
executor cannot undock itself.

## 3. Procedure

1. **Pre-run, all required, all recorded:** gate disarmed in the live API AND
   RAW `core.config_entries`; `rtk_position` reads `fix` (and has held it for
   ≥ 5 min); after parking, a POSITIVE area check — `device_position_type` an
   area label and `zone_hash` non-zero; facing derived two ways; sun ≥ 10° (the
   runner enforces it); vision camera clean (fault 1068 history); battery
   ≥ 35 %.
2. Start the tracer **before** S1, in the background, at a 1 s poll:
   `.venv/bin/python scripts/ble_proxy_connected_trace.py 3600 --interval 1`.
   The log file must not exist beforehand, so every row belongs to this run.
3. Each leg through `scripts/phase1_leg_runner.py S<n> <tx> <ty> scored --out
   <evidence dir>`, **unmodified**, with its own daylight/VIO/corridor/excursion/
   band/dry-run checks. Gate armed immediately before each dispatch, disarmed
   and verified immediately after every leg, including a halted one (read
   `rc=$?` directly — zsh has no `PIPESTATUS`).
4. Wait ≥ 60 s after S4 and S9 (turn legs) before the next runner check.
5. Stop the tracer after S12 (or at a stop rule), record start/end UTC of every
   dispatch, then dock with `lawn_mower.dock` on operator go.

**Approval cadence** is the operator's to set before step 3: per-leg go, or a
standing go for S1–S12 as on 2026-09-14 session 2. Nothing here assumes the
standing go carries over.

## 4. Data rules — fixed now

Applied by `normalize_trace_rows` in `scripts/build_ble_coverage_map.py`, run
unmodified on the raw log. The raw log is banked in full; nothing is deleted.

- A row with proxy `?` or `disconnected` is **unattributed** and dropped.
- `ble_rssi` missing or `0` (dozed) is dropped.
- A row whose `x`, `y` and `ble_rssi` all equal the immediately preceding row is
  a **re-read** and dropped. This undercounts on purpose.
- A kept row is a **sample**. Cells are 1 m, binned by floor, per proxy.
- **Motion windows** are the dispatch start→end intervals recorded in step 5.

## 5. What counts as enough data

Let **P** be the proxy with the most samples in the run.

- A cell is **covered** for P when it holds **≥ 10 samples** tagged P
  (CLAUDE.md's per-cell target; within-cell sd was 5.5 dB in the 96 h data, so
  fewer cannot characterise a cell).
- **SUFFICIENT:** **≥ 5 of the 6 route cells** are covered for P, **and** P holds
  **≥ 80 %** of all samples.
- **SUFFICIENT, MIXED:** ≥ 5 route cells covered counting each proxy separately,
  but no proxy reaches 80 % (a mid-run handoff). Each proxy's cells stand as their
  own data; the run is not reported as one proxy's map.
- **INSUFFICIENT:** anything else — including any truncated run. The samples are
  still banked and still drawn; they are not called a coverage result.

Off-route cells are drawn if they exist and are **never** counted toward the bar.

## 6. Falsifiers — the instrument, not the yard

The run is **INVALID** (no coverage claim at all, whatever §5 says) if either:

1. **The RSSI is not live while driving.** Across all motion windows combined,
   samples arrive at **< 0.3 per second**. The history of
   `sensor.*_ble_rssi` on 2026-09-15 changed at roughly 1 Hz during motion and
   only every 5 min at rest; if the tracer cannot see the 1 Hz behaviour, the map
   would be built from stale values.
2. **Attribution is missing.** Unattributed rows exceed **10 %** of all rows
   inside motion windows.

## 7. Stop rules

- Every runner HALT stops the sequence (§16.3 of the 2026-09-11 predeclaration).
  Automatic retry only on `ble_client_not_connected` (nothing sent) or a
  tracked-feature dip with the sun still ≥ 10°, as on 2026-09-14 session 2.
- Battery below **30 %**, `rtk_position` leaving `fix`, a new device fault, or the
  gate found armed when it should not be — stop, disarm, verify.
- Any BLE drop is **data, not a reason to push on**: it stops the sequence; the
  rows up to it are banked.

## 8. What each outcome leads to

- **SUFFICIENT:** regenerate `ble-coverage.json`; P's layer on the card then
  holds real cells. A breadth run (different cells, still P or a chosen proxy) is
  a new predeclaration.
- **SUFFICIENT, MIXED:** same, and the handoff itself is recorded as a finding.
- **INSUFFICIENT:** bank, draw, no claim; decide whether a repeat is worth it.
- **INVALID §6.1:** the tracer's RSSI source is unfit for coverage mapping; stop
  collecting until a live connection RSSI is found. **INVALID §6.2:** fix
  attribution in the tracer before any repeat.

## 9. Pre-dispatch instrument fix (2026-09-16 22:00Z, before S1)

The tracer was started at 22:00:2xZ and tagged **every** row `disconnected` while
HA had the mower allocated on `atom-fireplace` (inventory 22:00:20Z). Raw capture:
`bluetooth/subscribe_connection_allocations` sends a full snapshot first, then
**one-scanner deltas**; the tracer decided attribution from each event alone, so
any unrelated scanner's update read as a disconnect. §6.2 would have voided the
run. Fixed (allocations merged per source) with tests, **before any dispatch**;
no rule above changed. The pre-fix rows are banked, excluded, as
`evidence-ble-connected-trace-20260916/aborted_prefix_trace_log.jsonl` and the
tracer restarted on an empty log.

## 10. Amendment — run 2 under a changed RF set (written 2026-09-16 ~22:20Z, before any run-2 row)

Run 1 is banked and scored (`docs/findings-ble-connected-trace-collection-20260916.md`,
INSUFFICIENT, instrument valid). **Nothing here rescores it.**

**What changed, by operator action after run 1:** the BLE proxy was removed from
`atom-fireplace` (the proxy that held run 1). At 22:16:10Z HA's scanner list was
`hot-tub-backyard`, `p1s-printer` (connectable) and `hci0` (scan-only);
**`garage-m5stack` was also absent**, cause not established. Operator decision:
run with the two connectable proxies. This **replaces §1 for run 2 only**, and
knowingly accepts the Window D risk (fewer listeners, slower reconnects).

**Unchanged for run 2:** §2 route (S1–S12), §3 procedure, §4 data rules, §5 bars,
§6 falsifiers, §7 stop rules, §8 outcomes — including the predeclared retries, now
implemented in the driver (`evidence-ble-connected-trace-20260916/drive.sh`).

**Run-2 specifics:**
- The tracer **appends** to the same log so the card map keeps run 1's layer.
  **Run-2 rows are exactly those with `t_utc` at or after the run-2 tracer start**,
  recorded in `evidence-ble-connected-trace-20260916/run2/`. §5 and §6 are
  evaluated on run-2 rows only; run-1 samples are never pooled into run 2's bar.
- **P** is determined from run-2 samples alone. A sample tagged `atom-fireplace`
  in run 2 would mean the removal did not take, and is reported, not counted.
- **Start** is wherever the mower stands after run 1's S9 halt (≈ (4.97, −1.65),
  inside the area). S1 is then a ~3.2 m leg rather than ~1.9 m; the runner's own
  segment, corridor, excursion and band checks decide whether it may run.
  Facing is derived two ways before S1, because S9 stopped mid-turn.
- The proxy inventory is recorded at start and end.
