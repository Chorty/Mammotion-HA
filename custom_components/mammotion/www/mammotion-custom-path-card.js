const MAX_WAYPOINTS = 7;
// Raised 2 -> 4 in beta31 to extend per-click reach. Mirrors the backend's
// REAL_CLICK_TO_GO_SEGMENT_LIMIT (manual_motion.py); keep the two in step or the
// card will offer a run the backend refuses. NOT a LUBA_ACCEPTANCE_PROFILE key.
const MAX_REAL_SEGMENTS = 4;
// Nudge is capped so a mistake is bounded by geometry. It is available only
// with trustworthy, map-aligned current orientation; frozen course-over-ground
// must never authorize it.
const MAX_NUDGE_METRES = 2.0;
// Night v1 is deliberately narrower than Real Go: the backend supports one
// forward-only segment of at most 1.0 m, with no loop-to-tolerance or junction
// chaining. Keep this independent from LUBA_ACCEPTANCE_PROFILE so adding the
// card control cannot alter the hardware-accepted daylight/VIO payload.
const MAX_NIGHT_SEGMENT_METRES = 1.0;
// Mirrors the backend's `_MAX_SEGMENT_LENGTH_M` (services.py). Keep the two in
// step or the card offers a leg the backend refuses -- the same trap that
// MAX_REAL_SEGMENTS documents. This is an AUTHORIZATION cap of 20 ft chosen on
// 2026-08-17, NOT a measured reach limit: the longest segment ever executed is
// 4.0 m. NOT a LUBA_ACCEPTANCE_PROFILE key.
const MAX_REAL_SEGMENT_METRES = 6.1;
// Mirrors the backend's `_BUDGET_CHECK_METRES_PER_PULSE` (services.py).
const BUDGET_CHECK_METRES_PER_PULSE = 0.3;
// Route B (2026-08-19). Any leg longer than this is split into collinear
// sub-legs, so a distant click is driven with only geometry that has been
// measured. Mirrors the backend's `_SPLIT_LEG_TARGET_LENGTH_M`; keep the two in
// step, the same trap MAX_REAL_SEGMENTS and MAX_REAL_SEGMENT_METRES document.
//
// ⚠️ NOT a LUBA_ACCEPTANCE_PROFILE key, deliberately. Adding it to that object
// would un-accept the hardware-accepted profile and owe another Gate 5 -- the
// exact cost Route B exists to avoid. It travels as a plain payload key, like
// `max_real_segments`.
//
// 3.85 rather than 3.81: `ceil(d / target)` is a step function and
// 15.24 / 3.81 = 4.000 exactly, so a centimetre of drift between this check and
// the backend's would flip the count to 5 and refuse the run. A true 50 ft
// click still splits into 4 sub-legs of 3.81 m.
//
// ⚠️ 3.81 m is 95% of the longest straight leg ever executed (4.0 m, n = 1).
// 4 x 3.85 = 15.40 m has never been driven.
const SPLIT_LEG_TARGET_METRES = 3.85;

// 🚨 The mid-drive controller only fires a correction once aim error reaches
// _MIN_CORRECTABLE_AIM_ERROR_DEGREES in services.py (post-turn tolerance 10 +
// deadband 5). An error just under that floor is NEVER corrected, whatever it
// costs, and it buys `distance * sin(floor)`. Setting that equal to
// waypoint_tolerance gives the longest leg whose landing can be protected:
//
//     limit = waypoint_tolerance / sin(CORRECTABLE_AIM_FLOOR_DEGREES)
//
// On the accepted profile (tolerance 0.15) that is 0.580 m. At 3.0 m the same
// floor permits an uncorrectable 0.776 m miss -- 5x tolerance -- which is why
// the measured-good regime is ~0.8 m and why 3.0 m legs miss.
//
// ⚠️ ADVISORY and deliberately pessimistic: it asks what happens if aim sits at
// the floor for a whole leg. Real legs correct repeatedly -- a 3.0 m sub-leg
// reached target at 0.094 m on 2026-08-20 while its sibling missed by 0.259 m.
// The card WARNS; it must never refuse on this. Mirrors
// `_correctable_leg_length_limit_m` in services.py; keep the two in step.
const CORRECTABLE_AIM_FLOOR_DEGREES = 15.0;
// Deliberate safety-gate overrides, 2026-08-19, at the operator's explicit
// request: every blocker gets a toggle so a restriction can be lifted ON
// PURPOSE instead of being worked around by editing a constant and redeploying.
//
// ⚠️ A toggle here is NOT enough on its own. The card's blockers are a courtesy
// copy; the BACKEND keeps its own gates. Every code below must also be a key of
// `_OVERRIDABLE_GATES` in services.py, or the card will read "Ready" while the
// backend refuses -- a card that lies, which is worse than no toggle at all.
// Keep the two in step.
//
// `tier` drives presentation order and colour only. `risk: "high"` marks the
// gates that protect against a physical hazard or a blind sensor rather than a
// chosen number; they are still overridable, they just say so loudly.
const OVERRIDE_TIERS = ["cap", "night", "sensing", "link", "physical"];
const OVERRIDABLE_BLOCKERS = Object.freeze({
  // chosen authorization numbers
  segment_too_long: { tier: "cap", risk: "low" },
  [`real_segment_limit_${MAX_REAL_SEGMENTS}`]: {
    tier: "cap",
    risk: "low",
    backend: "real_segment_limit",
  },
  real_segment_limit: { tier: "cap", risk: "low" },
  split_exceeds_real_segment_budget: { tier: "cap", risk: "low" },
  linear_budget_insufficient_for_segment: { tier: "cap", risk: "low" },
  point_count_2_to_8: { tier: "cap", risk: "low" },
  max_real_segments_positive: { tier: "cap", risk: "low" },
  one_segment_only: { tier: "cap", risk: "low" },
  // night
  night_segment_too_long: { tier: "night", risk: "medium" },
  night_requires_one_segment: {
    tier: "night",
    risk: "high",
    backend: "night_multi_segment_unsupported",
  },
  night_multi_segment_unsupported: { tier: "night", risk: "high" },
  night_linear_loop_unsupported: { tier: "night", risk: "medium" },
  night_requires_precise_rtk: { tier: "night", risk: "high" },
  // sensing
  rtk_not_precise: { tier: "sensing", risk: "high" },
  path_validation: { tier: "sensing", risk: "high" },
  path_validation_failed: {
    tier: "sensing",
    risk: "high",
    backend: "path_validation",
  },
  position_not_valid_for_motion: { tier: "sensing", risk: "medium" },
  live_map_position_available: { tier: "sensing", risk: "high" },
  map_position_nonzero: { tier: "sensing", risk: "high" },
  position_area_inside: { tier: "sensing", risk: "medium" },
  vio_feed_live: { tier: "sensing", risk: "high" },
  vio_feed_degraded: {
    tier: "sensing",
    risk: "high",
    backend: "vio_feed_live",
  },
  vio_active: { tier: "sensing", risk: "high" },
  live_heading_available: { tier: "sensing", risk: "high" },
  current_orientation_unavailable: {
    tier: "sensing",
    risk: "high",
    backend: "live_heading_available",
  },
  vio_heading_available: { tier: "sensing", risk: "high" },
  target_heading_available: { tier: "sensing", risk: "high" },
  // link
  ble_transport_required: { tier: "link", risk: "high" },
  ble_link_live: { tier: "link", risk: "high" },
  // physical
  mower_reports_blades_off: { tier: "physical", risk: "high" },
  blade_unsafe: {
    tier: "physical",
    risk: "high",
    backend: "mower_reports_blades_off",
  },
  mower_ready: { tier: "physical", risk: "medium" },
  not_docked_or_charging: { tier: "physical", risk: "medium" },
  runtime_not_mowing: { tier: "physical", risk: "high" },
  runtime_route_not_blocking: { tier: "physical", risk: "high" },
});
// Why each override matters, in the operator's terms, shown at the moment of
// flipping the toggle. A gate's NAME never says what it was protecting.
const OVERRIDE_WHY = Object.freeze({
  segment_too_long:
    "6.10 m is an authorization cap chosen 2026-08-17, not a measured limit. The longest segment ever executed is 4.0 m (n = 1).",
  real_segment_limit:
    "4 segments per click. Error does not compound with segment index (measured slope +0.017 m).",
  split_exceeds_real_segment_budget:
    "Runs more collinear sub-legs than a single click has ever driven.",
  linear_budget_insufficient_for_segment:
    "The pulse ceiling cannot reach this leg at a conservative 0.30 m/pulse. Risks stranding mid-leg — which stops safely.",
  point_count_2_to_8: "The executor chain is validated for 2 to 8 points.",
  max_real_segments_positive:
    "A real run with max_real_segments < 1 executes nothing.",
  one_segment_only: "This executor was validated on a single segment.",
  night_segment_too_long:
    "Night is capped at 1.0 m because the turn quantum is 48.15° ± 5.70 with NOTHING scaling it — 4 of 5 converging night turns landed inside tolerance by luck (margins 1.72 / 1.09 / 0.36°). No night landing-accuracy population exists.",
  night_requires_one_segment:
    "⚠️ NIGHT HAS NO JUNCTION FEASIBILITY MODEL. The preflight that refuses an impossible turn before segment 1 does not exist for night, so an infeasible junction is discovered AFTER motion starts.",
  night_multi_segment_unsupported:
    "⚠️ NIGHT HAS NO JUNCTION FEASIBILITY MODEL — an infeasible junction is discovered after motion has started.",
  night_linear_loop_unsupported:
    "Night runs a fixed pulse budget; loop-to-tolerance has never been exercised at night.",
  night_requires_precise_rtk:
    "Night steers on RTK alone — there is no VIO to fall back on. Float produced a 13.9 cm stationary jump (2026-08-07).",
  rtk_not_precise:
    "Non-Fix RTK. Float produced a 13.9 cm stationary jump against an 0.08 m tolerance (2026-08-07).",
  path_validation:
    "⚠️ CONTAINMENT. The path leaves every known area polygon — the mower may drive outside mapped geometry entirely.",
  path_validation_failed:
    "⚠️ CONTAINMENT. The path leaves every known area polygon — the mower may drive outside mapped geometry entirely.",
  position_not_valid_for_motion:
    "Typically docked, CHARGE_ON, or zone_hash 0. The dock sits outside every mowing area.",
  live_map_position_available:
    "No live map position — the controller steers on position.",
  map_position_nonzero:
    "Position reads (0, 0), usually a dead or unstarted feed.",
  position_area_inside: "The mower does not report itself inside a known area.",
  vio_feed_live:
    "⚠️ THE DUSK LATCH. vio_state reads active while tracked_features is 0 — the state field lies and the feed is already blind.",
  vio_feed_degraded:
    "⚠️ THE DUSK LATCH. vio_state reads active while tracked_features is 0 — the state field lies and the feed is already blind.",
  vio_active: "VIO is not active, and the vio turn mode closes on VIO heading.",
  live_heading_available:
    "No trustworthy current heading. Frozen course-over-ground is last travel, not orientation.",
  current_orientation_unavailable:
    "No trustworthy current heading. Frozen course-over-ground is last travel, not orientation.",
  vio_heading_available: "No VIO heading to close the turn loop on.",
  target_heading_available:
    "No target heading could be derived for this segment.",
  ble_transport_required:
    "Not on BLE. The position feed is BLE-only and stone dead on cloud.",
  ble_link_live:
    "⚠️ The link is not live. is_usable is routing eligibility, NOT liveness — commands can pile up undelivered, and the STOP may not arrive either.",
  mower_reports_blades_off:
    "🚨 THE MOWER REPORTS ITS BLADES ARE NOT OFF. The RPM register latches after a mow so this can be stale — but it can also be true. Confirm physically first.",
  blade_unsafe:
    "🚨 THE MOWER REPORTS ITS BLADES ARE NOT OFF. Confirm physically before overriding.",
  mower_ready: "Work mode is not MODE_READY or MODE_PAUSE.",
  not_docked_or_charging:
    "The mower reports charging. Driving off the dock under power can damage the dock.",
  runtime_not_mowing:
    "🚨 AN AUTONOMOUS MOW IS ACTIVE — this commands manual motion into a running vendor job.",
  runtime_route_not_blocking:
    "🚨 Live or ambiguous route data indicates the mower is executing a route.",
});
// Run retention. The card used to keep ten SUMMARIES plus exactly ONE full
// result, overwritten every run -- and `_segmentLandingRows()` needs the full
// result, so a summary-only entry renders as `[]`. The downloaded history was
// therefore NOT a recovery path, which is how the most informative dataset in
// the corpus came within one run of being lost (2026-08-18; it survived only
// because it had been manually downloaded).
//
// Two bounds, because either alone is wrong. A count bound alone would try to
// hold ten 150-280 KB results against a ~5 MB origin quota; a byte bound alone
// would let one enormous run evict every summary.
const HISTORY_MAX_ENTRIES = 10;
// Conservative against a typical ~5 MB localStorage origin quota, which this
// key SHARES with the last-run key and anything else on the dashboard origin.
// At 150-280 KB per full result this retains roughly 7-13 of them, and the
// shrink-to-fit loop below drops the oldest results first when it does not.
const HISTORY_MAX_BYTES = 2_000_000;
const NIGHT_GO_PROFILE = Object.freeze({
  prefer_ble: true,
  turn_mode: "night",
  max_linear_commands: 3,
  max_linear_pulse_ceiling: null,
  motion_refresh_interval_ms: 200,
  max_turn_commands: 4,
  heading_tolerance_degrees: 8,
  turn_pulse_duration_ms: 1500,
  max_turn_translation_distance: 0.3,
  ble_auto_recover: false,
  night_angular_speed: 500,
  toward_mirror_degrees: 90.13,
  // The backend's diagnostic default samples through 60 seconds after EVERY
  // command. On the first card-driven night run (2026-08-14), all three turn
  // heading changes were visible within 3 seconds, while the inherited default
  // stretched 3 turn + 3 linear commands to about 6.5 minutes. Night owns this
  // value so the frozen daylight profile and its dispatched payload do not move.
  sample_delays: [0, 3],
});
// Bump on EVERY deploy (date + b-counter) so the footer/console banner proves
// which build the browser actually loaded.
const CARD_VERSION = "0.6.4-beta67";

// The exact bounded execution profile that passed supervised LUBA acceptance
// Gate 4 re-pass on 2026-08-05 (three-write zero stop, bounded straight segment,
// active-session abort, 176 deg VIO regression, corrected two-leg L path;
// see docs/gate4-repass-20260805.md). It remains bounded: three linear commands
// per segment with NO loop-to-tolerance ceiling.
//
// `waypoint_tolerance` was raised 0.08 -> 0.15 on 2026-08-08 against its own
// hardware evidence (docs/evidence-slow-tier-validation-20260808.json): three
// 1.0 m segments landed 0.0882 / 0.1241 / 0.0317 m out, all UNDERSHOOTING, with
// zero turn commands and no reverse-recovery. That is the only value in this
// profile not sourced from the Gate 4 re-pass, hence the two dates in the label.
//
// This is the card's built-in default because it is the only Real Go profile
// any hardware has actually executed. Overriding ANY key below in the card
// YAML leaves the accepted profile; the card then labels the run "customised"
// and that payload is NOT hardware-accepted.
//
// `calibrated_forward_heading_offset_degrees` is a per-mower measurement (this
// value came from the acceptance LUBA). Re-derive it for another mower rather
// than assuming 102.4 transfers.
// The hardware acceptance this profile currently rests on. Kept beside the
// profile it describes, and pinned against docs/accepted-profile.json by a
// frontend test so the two cannot drift.
const ACCEPTED_PROFILE_ACCEPTED_ON = "2026-08-18";
const ACCEPTED_PROFILE_LABEL = `LUBA acceptance profile — Gate 5 passed ${ACCEPTED_PROFILE_ACCEPTED_ON}`;

const LUBA_ACCEPTANCE_PROFILE = Object.freeze({
  prefer_ble: true,
  turn_mode: "vio",
  max_turn_commands: 4,
  vio_turn_max_commands: 4,
  // Inert while loop-to-tolerance is on: the linear phase runs to the ceiling
  // below instead. Kept at 3 so turning loop-to-tolerance off anywhere falls
  // back to exactly the Gate 4/5 behaviour.
  max_linear_commands: 3,
  // 🏁 ADOPTED 2026-08-12 on hardware evidence, replacing `null`. This is the
  // change that makes per-click reach real for someone using the card: with
  // loop-to-tolerance disabled a segment stops after 3 pulses at roughly 1 m,
  // and four measured legs stopped 0.68 / 0.68 / 1.79 / 2.95 m short of their
  // waypoints. With it enabled the same geometry reached target:
  //
  //     2.0 m leg -> 0.0690 m in  5 pulses
  //     3.0 m leg -> 0.0928 m in  8 pulses
  //     4.0 m leg -> 0.1023 m in 11 pulses
  //
  // all three stopping on TOLERANCE rather than on the ceiling. Per-segment
  // reach goes ~1 m -> 4 m, per-click ~4 m -> ~16 m at four segments.
  // docs/loop-to-tolerance-reach-20260811.md.
  //
  // 🚨 RAISED 14 → 22 ON 2026-08-17, AND THIS PROFILE IS THEREFORE NO LONGER
  // HARDWARE-ACCEPTED. It owes the §4 re-pinning in docs/gate4-repass-20260805.md
  // and another Gate 5 before any run on it may be described as accepted. The
  // operator asked for a 20 ft (6.096 m) single-waypoint move and chose this
  // route knowing the cost; that decision is recorded in
  // docs/reach-20ft-and-the-reaim-trigger-20260817.md.
  //
  // **Why 22.** Same arithmetic that chose 14, applied to 6.096 m instead of
  // 4 m. The ceiling is a loop bound, not the runaway guard --
  // `linear_distance_ceiling_factor: 2.0` stops a segment at twice its leg
  // length whatever the pulse count does -- so the number only has to survive a
  // bad link. A healthy pulse travels ~0.41 m and a BLE-stalled one ~0.22 m
  // (measured, n=2), and the 4 m leg needed 11 pulses with 2 of 11 stalled. At
  // that stall rate 6.096 m needs ~17 pulses, at double it ~18, at a 50% stall
  // rate ~20. 22 clears all three.
  //
  // ⚠️ The backend refuses pre-dispatch if this ceiling cannot reach the leg
  // (`linear_budget_insufficient_for_segment`), so lowering it no longer
  // strands a run mid-leg -- it declines it up front.
  //
  // 🏁 What the SUPERSEDED value of 14 earned, kept because it is the last
  // accepted state: GATE 5 RE-PASSED, 2026-08-12, card-driven, four segments
  // target_reached at 0.0674 / 0.1032 / 0.0807 / 0.0607 m against a 0.15
  // tolerance — mean 0.0780, the best four-segment result on record. Zero
  // reverse-recovery, zero budget exhaustion, and the payload carried the key.
  // docs/evidence-gate5-repass-2-20260812.json.
  max_linear_pulse_ceiling: 22,
  max_no_progress_pulses: 3,
  heading_tolerance_degrees: 18,
  // 0.15, not 0.08, on hardware evidence from 2026-08-08: three 1.0 m segments
  // landed 0.0882 / 0.1241 / 0.0317 m from target with ZERO turn commands and
  // no reverse-recovery. Every one UNDERSHOT (travel ratio 0.88-0.98), whereas
  // Gate 4 at 0.08 overshot at 2.19x and needed a U-turn back -- which beta22
  // now refuses as `target_requires_reverse_recovery`. The position feed is
  // ~1031 ms stale and the mower covers 30-47 cm in that time, so at 0.08 it
  // cannot confirm arrival before it has already passed the point.
  // See docs/evidence-slow-tier-validation-20260808.json.
  waypoint_tolerance: 0.15,
  min_progress_distance: 0.0025,
  max_turn_translation_distance: 0.3,
  calibrated_forward_heading_offset_degrees: 102.4,
  turn_pulse_duration_ms: 1500,
  linear_pulse_duration_ms: 1300,
  motion_refresh_interval_ms: 200,
  final_approach_metres_per_pulse: 1.06,
  turn_degrees_per_second: 37,
  ble_auto_recover: false,
  sample_delays: [0, 3],
});

const PROFILE_KEYS = Object.freeze(Object.keys(LUBA_ACCEPTANCE_PROFILE));

// Plain-English text for the blocker codes the backend and preflight emit. The
// codes are the truth and are always still shown; this only adds the "so what".
// The commonest support question by far has been a disabled Real Go button with
// no visible reason, which is almost always experimental_motion_disabled.
const BLOCKER_HELP = Object.freeze({
  path_unset: "Click at least one destination on the map.",
  position_unavailable:
    "No live mower position yet — press Reload map/runtime, and wake the mower if BLE has gone idle.",
  // The backend's own code for this is position_not_valid_for_motion, which is
  // what actually appears in the banner; position_unavailable is the card's.
  // Both are kept because both reach the operator.
  position_not_valid_for_motion:
    "The mower's position is not usable for motion — it needs RTK Fix and a zone inside a mapped area.",
  ble_client_not_connected:
    "BLE is not connected. Wake the mower (it dozes after ~10 min idle) or move a Bluetooth proxy closer.",
  current_orientation_unavailable:
    "No trustworthy current facing. Course-over-ground is last-travel only and cannot see an in-place turn.",
  path_validation_failed:
    "The path failed validation — check the warnings above (usually a point outside the selected area).",
  experimental_motion_disabled:
    "Turn on the integration option 'Enable experimental BLE-only manual motion'.",
  experimental_motion_backend_not_ready:
    "The backend motion gate is closed. Check the integration options and that BLE is connected.",
  runtime_safety_blocked: "The backend refuses motion in the current state.",
  blade_not_safe: "Blades are not confirmed off or are still spinning down.",
  active_mowing: "A mowing job is running — stop or cancel it first.",
  active_route: "A planned route is active — cancel it first.",
  ble_link_not_live:
    "BLE is not live. Wake the mower, or move a Bluetooth proxy closer.",
  rtk_not_precise:
    "RTK is not at Fix. Wait for convergence, or pass allow_degraded_rtk deliberately.",
  night_requires_precise_rtk:
    "Night Go requires RTK Fix; its degraded-RTK override is intentionally unavailable.",
  night_requires_one_segment:
    "Night Go supports exactly one waypoint. Remove the additional destinations.",
  night_segment_too_long: `Night Go is limited to ${MAX_NIGHT_SEGMENT_METRES.toFixed(1)} m. Move the waypoint closer.`,
  segment_too_long: `A single leg is capped at ${MAX_REAL_SEGMENT_METRES.toFixed(1)} m (20 ft). Long legs are normally split automatically into ${SPLIT_LEG_TARGET_METRES.toFixed(2)} m sub-legs, so seeing this means the mower drifted past the cap mid-run. Move the waypoint closer.`,
  split_exceeds_real_segment_budget: `Too far for one click. Long legs are split into sub-legs of at most ${SPLIT_LEG_TARGET_METRES.toFixed(2)} m, and only ${MAX_REAL_SEGMENTS} segments can run per click (about ${(MAX_REAL_SEGMENTS * SPLIT_LEG_TARGET_METRES).toFixed(1)} m total). Click a nearer point, or fewer of them.`,
  linear_budget_insufficient_for_segment:
    "This leg is longer than its linear pulse budget can reach. Shorten it, or raise max_linear_pulse_ceiling (which leaves the accepted profile).",
  [`real_segment_limit_${MAX_REAL_SEGMENTS}`]: `Real Go runs at most ${MAX_REAL_SEGMENTS} segments. Remove waypoints or run it in two clicks.`,
});

console.info(
  `%c MAMMOTION-CUSTOM-PATH-CARD %c v${CARD_VERSION} `,
  "background:#22c55e;color:#000;font-weight:bold;",
  "background:#333;color:#fff;",
);

class MammotionCustomPathCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._hass = null;
    this._config = {};
    this._mapData = null;
    this._waypoints = [];
    this._runTicker = null;
    this._runStartedAt = null;
    this._submittingRealRun = false;
    this._livePosition = null;
    this._runtimeState = null;
    this._areaHash = "";
    this._mapT = null;
    this._draggingIndex = null;
    this._height = 520;
    this._status =
      "Load map/runtime, then click up to 7 destinations. Real Go is experimental and limited to four segments.";
    this._validation = null;
    this._dryRun = null;
    this._realRun = null;
    this._realRunAt = null;
    this._loadingMap = false;
    this._loadingRuntime = false;
    this._confirmBladesOff = false;
    this._confirmClearArea = false;
    this._confirmNightExperimental = false;
    this._nudgeDistance = 0.5;
    this._rendered = false;
    // Rendered under the status line -- NOT inside the history panel, which is
    // collapsed by default and absent entirely when history is empty. Nudge
    // persists a run without ever touching history, so a history-scoped warning
    // would be invisible on exactly the path that has no other copy. Null means
    // "nothing to say"; anything else means a run was not fully stored.
    this._storageWarning = null;
    // Deliberate per-run safety-gate overrides, as a Set of blocker codes.
    // ⚠️ RESET AFTER EVERY RUN, in the same `finally` that clears the blade and
    // clear-area confirmations. An override that outlives the reason it was set
    // is the `c196b8b1` failure shape -- state surviving its own intent -- and
    // the gate has already been found armed at rest three times (2026-08-18).
    // Deliberately NOT persisted to localStorage for the same reason.
    this._overrides = new Set();
  }

  _renderHistoryHtml() {
    const history = this._loadHistory();
    if (!history.length) return "";
    const rows = history
      .map((entry) => {
        const when = entry.at ? new Date(entry.at).toLocaleString() : "?";
        const mins = Math.floor((entry.elapsed_seconds || 0) / 60);
        const secs = String((entry.elapsed_seconds || 0) % 60).padStart(2, "0");
        const segs = (entry.segments || [])
          .map((seg) => {
            // Landing distance is the number that decides whether a run was
            // good; older history entries predate it, so keep them readable.
            const landing =
              typeof seg.landing === "number"
                ? ` ${seg.landing.toFixed(3)}m`
                : "";
            return `${seg.index}:${seg.passed ? "✓" : "✗"}${seg.stop_reason ? ` ${seg.stop_reason}` : ""}${landing}`;
          })
          .join(", ");
        const landings = (entry.segments || [])
          .map((seg) => seg.landing)
          .filter((value) => typeof value === "number");
        const mean = landings.length
          ? ` · mean ${(landings.reduce((sum, value) => sum + value, 0) / landings.length).toFixed(4)} m`
          : "";
        // Entries written before retention shipped, and entries whose result
        // was dropped to make room, are marked -- the download cannot recover
        // what is not there, and silence would imply it could.
        const retention = entry.result
          ? ""
          : entry.result_dropped
            ? " · summary only (dropped for space)"
            : " · summary only";
        return `<div class="history-row"><span class="history-when">${this._escapeHtml(when)} (${mins}:${secs})</span> <span class="history-outcome">${this._escapeHtml(entry.stop_reason || "?")}</span>${this._escapeHtml(mean)}${this._escapeHtml(retention)}${segs ? `<div class="history-segs">${this._escapeHtml(segs)}</div>` : ""}</div>`;
      })
      .join("");
    const withResults = history.filter((entry) => entry.result).length;
    // Name how many entries the download would actually carry. A summary-only
    // entry yields no landing rows, so "10 runs" without this reads as ten
    // recoverable runs when it may be one.
    return `<details><summary>Run history (${history.length}, ${withResults} with full result${withResults === 1 ? "" : "s"})</summary>${rows}<div class="history-actions"><button id="download-history" class="history-clear" type="button">Download history JSON</button><button id="clear-history" class="history-clear" type="button">Clear history</button></div></details>`;
  }

  _historyKey() {
    return `mammotion-path-card-history:${this._config.entity || "unknown"}`;
  }

  _lastRunKey() {
    return `mammotion-path-card-last-run:${this._config.entity || "unknown"}`;
  }

  _loadHistory() {
    try {
      const raw = localStorage.getItem(this._historyKey());
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (err) {
      return [];
    }
  }

  // Store the run, keeping its FULL result so the history download is a real
  // recovery path. Bounded by count and by bytes, dropping the OLDEST full
  // results first: a summary is worth little and costs almost nothing, while a
  // full result is the only thing `_segmentLandingRows()` can read.
  //
  // `fullResult` is optional so a failed call (which has no result) still
  // records its summary.
  _saveRunToHistory(entry, fullResult = null) {
    const stored = fullResult ? { ...entry, result: fullResult } : { ...entry };
    const history = [stored, ...this._loadHistory()].slice(
      0,
      HISTORY_MAX_ENTRIES,
    );
    // Shrink to fit: strip `result` from the oldest entries until the payload
    // is under budget, then keep stripping if the write is still refused. The
    // newest entry keeps its result longest because it is the one the operator
    // is about to want.
    for (let stripFrom = history.length; stripFrom >= 0; stripFrom -= 1) {
      const candidate = history.map((item, index) =>
        index >= stripFrom && item.result
          ? { ...item, result: undefined, result_dropped: true }
          : item,
      );
      const serialized = JSON.stringify(candidate);
      if (serialized.length > HISTORY_MAX_BYTES && stripFrom > 0) continue;
      try {
        localStorage.setItem(this._historyKey(), serialized);
        this._storageWarning =
          stripFrom < history.length
            ? `Storage is full — ${history.length - stripFrom} older run result${history.length - stripFrom === 1 ? "" : "s"} dropped to summaries. Download the history JSON to keep them.`
            : null;
        return true;
      } catch (err) {
        // Quota, most likely. Fall through and strip one more result.
        if (stripFrom === 0) {
          // ⚠️ NOT silent. There was no quota handling anywhere in this card,
          // and a change that stores MORE must not inherit that silence: the
          // operator has to know the run was not kept, because the only other
          // copy is the one they can still download from this page.
          this._storageWarning =
            "Could not save this run to history (browser storage is full). Download the run JSON now — it is not stored.";
          return false;
        }
      }
    }
    return false;
  }

  _lastRunAtKey() {
    return `mammotion-path-card-last-run-at:${this._config.entity || "unknown"}`;
  }

  // ⚠️ The timestamp is stamped only AFTER the write succeeds. It used to be
  // set FIRST, inside a catch commented "ignore quota failures", so on a quota
  // failure the card believed it had persisted a run it had not -- state set on
  // intent rather than on success, the same shape as the `c196b8b1` motion-gate
  // bug where "I called enable" was mistaken for "enable succeeded".
  _persistLastRun(result) {
    const at = new Date().toISOString();
    try {
      localStorage.setItem(this._lastRunKey(), JSON.stringify(result));
      // Stored separately so the restored result keeps its exact backend shape
      // -- wrapping it would break every reader that treats _realRun as the
      // service response, and older stored runs simply have no timestamp.
      localStorage.setItem(this._lastRunAtKey(), at);
      this._realRunAt = at;
      return true;
    } catch (err) {
      // Full results can be large. Leave `_realRunAt` alone: an unwritten run
      // must not read as stored, and `_runAgeLabel()` keys off it.
      this._storageWarning =
        "Could not save this run for reload (browser storage is full). Download the run JSON before leaving this page.";
      return false;
    }
  }

  _restoreLastRun() {
    try {
      const raw = localStorage.getItem(this._lastRunKey());
      this._realRunAt = localStorage.getItem(this._lastRunAtKey());
      return raw ? JSON.parse(raw) : null;
    } catch (err) {
      return null;
    }
  }

  // A restored run is shown on every page load, so it MUST say when it ran.
  // Without this the card presents a result from a previous day directly under
  // a "Map loaded" status line, which reads as "this just happened".
  _runAgeLabel() {
    if (!this._realRunAt) return "";
    const then = new Date(this._realRunAt);
    const ms = Date.now() - then.getTime();
    if (!Number.isFinite(ms)) return "";
    const minutes = Math.floor(ms / 60000);
    if (minutes < 1) return "just now";
    if (minutes < 60) return `${minutes} min ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} h ago`;
    return `${then.toLocaleString()}`;
  }

  _clearHistory() {
    try {
      localStorage.removeItem(this._historyKey());
      localStorage.removeItem(this._lastRunKey());
      // ⚠️ This key was ORPHANED: clearing history removed the run but left its
      // timestamp behind, so the next restore read a time for a run that no
      // longer existed and `_runAgeLabel()` dated whatever came next.
      localStorage.removeItem(this._lastRunAtKey());
    } catch (err) {
      // best-effort
    }
    this._realRunAt = null;
    this._storageWarning = null;
    this._status = "Run history cleared.";
    this._render();
  }

  // ---------------------------------------------------------------------
  // Result export
  //
  // The single most expensive gap in this project's investigation history was
  // a run whose per-command record existed ONLY in a browser pane and was lost
  // (see docs/turn-rate-variance-and-reach-analysis-20260808.md). Copy-to-
  // clipboard is not good enough for a 2000-line result on a phone. This
  // writes the whole response to a file the operator can hand straight over.
  // ---------------------------------------------------------------------

  _downloadSlug() {
    return String(this._config.entity || "mower").replace(/[^a-z0-9]+/gi, "-");
  }

  _downloadFilename(kind) {
    // Colons are illegal in filenames on Windows and awkward everywhere.
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    return `mammotion-${kind}-${this._downloadSlug()}-${stamp}.json`;
  }

  // Falls back to the clipboard rather than failing silently: a browser that
  // blocks object URLs should still let the operator get the record out.
  _downloadJson(payload, kind, label) {
    if (payload == null) {
      this._status = `No ${label} to download yet.`;
      this._render();
      return;
    }
    const text = `${JSON.stringify(payload, null, 2)}\n`;
    const filename = this._downloadFilename(kind);
    try {
      const blob = new Blob([text], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.style.display = "none";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      // Revoke on the next tick; revoking synchronously can cancel the
      // download in some browsers before it has started reading the blob.
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      this._status = `Downloaded ${filename}`;
      this._render();
    } catch (err) {
      this._copyText(text, `${label} (download unavailable, copied instead)`);
    }
  }

  // ---------------------------------------------------------------------
  // Run summary
  //
  // The number that decides whether a run was good is the landing distance
  // against waypoint_tolerance, and it is buried at
  // result.completion_status.waypoint_distances[last].distance. Every session
  // so far has extracted it by hand from raw JSON. Surface it.
  // ---------------------------------------------------------------------

  _landingDistance(segmentResult) {
    const distances = segmentResult?.completion_status?.waypoint_distances;
    if (!Array.isArray(distances) || !distances.length) return null;
    const last = distances[distances.length - 1];
    return typeof last?.distance === "number" ? last.distance : null;
  }

  _segmentLandingRows(result) {
    if (!result) return [];
    const segments =
      Array.isArray(result.segments) && result.segments.length
        ? result.segments
        : // A Nudge or single-segment vector run has no `segments` wrapper; it
          // IS the segment result. Present it the same way.
          this._landingDistance(result) != null
          ? [
              {
                index: 1,
                passed: result.stop_reason === "target_reached",
                result,
              },
            ]
          : [];
    return segments.map((segment) => {
      const inner = segment.result || {};
      const landing = this._landingDistance(inner);
      const tolerance =
        typeof inner.waypoint_tolerance === "number"
          ? inner.waypoint_tolerance
          : null;
      return {
        index: segment.index,
        passed: segment.passed === true,
        stopReason: inner.stop_reason ?? null,
        planned: typeof inner.distance === "number" ? inner.distance : null,
        landing,
        tolerance,
        inside:
          landing != null && tolerance != null ? landing <= tolerance : null,
        linearCommands: inner.linear_commands_sent ?? null,
        turnCommands: inner.turn_commands_sent ?? null,
      };
    });
  }

  _runSummaryHtml(result) {
    const rows = this._segmentLandingRows(result);
    if (!rows.length) return "";
    const landings = rows
      .map((row) => row.landing)
      .filter((value) => typeof value === "number");
    const mean = landings.length
      ? landings.reduce((sum, value) => sum + value, 0) / landings.length
      : null;
    const worst = landings.length ? Math.max(...landings) : null;
    const body = rows
      .map((row) => {
        const verdict =
          row.inside === null ? "—" : row.inside ? "inside" : "OUTSIDE";
        const verdictClass =
          row.inside === null ? "" : row.inside ? "ok" : "bad";
        const landing =
          row.landing == null ? "—" : `${row.landing.toFixed(4)} m`;
        const tolerance =
          row.tolerance == null ? "—" : `${row.tolerance.toFixed(2)} m`;
        const planned =
          row.planned == null ? "—" : `${row.planned.toFixed(2)} m`;
        const pulses = [
          row.turnCommands == null ? null : `${row.turnCommands}T`,
          row.linearCommands == null ? null : `${row.linearCommands}L`,
        ]
          .filter(Boolean)
          .join(" ");
        return `<tr>
          <td>${this._escapeHtml(row.index)}</td>
          <td class="${row.passed ? "ok" : "bad"}">${row.passed ? "✓" : "✗"}</td>
          <td>${this._escapeHtml(row.stopReason || "—")}</td>
          <td class="num">${this._escapeHtml(planned)}</td>
          <td class="num">${this._escapeHtml(landing)}</td>
          <td class="num">${this._escapeHtml(tolerance)}</td>
          <td class="${verdictClass}">${verdict}</td>
          <td class="num">${this._escapeHtml(pulses || "—")}</td>
        </tr>`;
      })
      .join("");
    const footer =
      mean == null
        ? ""
        : `<div class="summary-footer">mean landing ${mean.toFixed(4)} m · worst ${worst.toFixed(4)} m · ${landings.length} of ${rows.length} segments measured</div>`;
    const age = this._runAgeLabel();
    return `<div class="run-summary">
      <div class="title">Segment landings${age ? `<span class="run-age">${this._escapeHtml(age)}</span>` : `<span class="run-age">from a previous session</span>`}</div>
      <div class="summary-scroll"><table>
        <thead><tr><th>#</th><th>ok</th><th>stop reason</th><th>leg</th><th>landing</th><th>tol</th><th>verdict</th><th>pulses</th></tr></thead>
        <tbody>${body}</tbody>
      </table></div>
      ${footer}
    </div>`;
  }

  // ---------------------------------------------------------------------
  // Readiness banner
  // ---------------------------------------------------------------------

  // The leg-length advisory is appended to WHATEVER banner comes back, not
  // just the "ready" one. It was originally added inside the ready branch and
  // a test caught that immediately: readiness sits at "arming" or "blocked"
  // for most of the time an operator is actually choosing where to click, so
  // the warning was invisible during the only window in which it is useful.
  _readiness() {
    const banner = this._readinessLevel();
    const limit = this._correctableLegLimitMetres();
    const longest = this._longestPlannedLegMetres();
    if (limit && longest > limit) {
      const worst =
        longest * Math.sin((CORRECTABLE_AIM_FLOOR_DEGREES * Math.PI) / 180);
      // ⚠️ ADVISORY ONLY -- never downgrade `level`. A leg past this bound can
      // still land well (3.0 m reached target at 0.094 m on 2026-08-20) and a
      // leg inside it can still miss. The bound says the controller can no
      // longer GUARANTEE the landing, because aim error under the correction
      // floor is never corrected at all.
      banner.details = [
        ...(banner.details || []),
        `⚠️ Longest leg is ${longest.toFixed(2)} m, over the ${limit.toFixed(2)} m the controller can protect. ` +
          `Aim error below the ${CORRECTABLE_AIM_FLOOR_DEGREES.toFixed(0)}° correction floor is never corrected, so this leg can miss by up to ${worst.toFixed(2)} m. ` +
          `Measured-good is ~0.8 m. This is a warning, not a blocker.`,
      ];
    }
    return banner;
  }

  _readinessLevel() {
    if (this._motionRunActive()) {
      return {
        level: "busy",
        headline: "Run in progress — Real Go and editing are locked.",
        details: ["Use Abort / Stop to end it early."],
      };
    }
    const preflight = this._preflight();
    if (!preflight.safe) {
      // Every blocker gets its own explanation on its own line. Showing only
      // the first hides the deeper problem behind the shallow one -- "click a
      // destination" reads as the whole story while the motion gate is off --
      // and running them together as one paragraph is unreadable at five.
      const details = [
        ...new Set(
          preflight.blockers.map((code) => BLOCKER_HELP[code]).filter(Boolean),
        ),
      ];
      return {
        level: "blocked",
        headline: `Not ready: ${preflight.blockers.join(", ")}`,
        details,
      };
    }
    const missing = [];
    if (!this._confirmBladesOff) missing.push("blades off");
    if (!this._confirmClearArea) missing.push("clear area");
    if (missing.length) {
      return {
        level: "arming",
        headline: `Almost ready — confirm ${missing.join(" and ")}.`,
        details: ["Both confirmations are required before Real Go will send."],
      };
    }
    const destinations = this._segmentCount();
    const split = this._plannedSplit();
    const legs = split.subLegCount;
    // Name BOTH once a split is in play. "N segments" alone was true while a
    // click was always one leg; after the split it would be a lie in whichever
    // direction the reader took it.
    const headline = split.applied
      ? `Ready — Real Go will drive ${legs} leg${legs === 1 ? "" : "s"} to reach ${destinations} destination${destinations === 1 ? "" : "s"}.`
      : `Ready — Real Go will drive ${legs} segment${legs === 1 ? "" : "s"}.`;
    const details = split.applied
      ? [
          `Long legs are split into collinear sub-legs of at most ${SPLIT_LEG_TARGET_METRES.toFixed(2)} m; the extra junctions are 0° turns and cost no turn commands.`,
        ]
      : [];
    details.push(this._profileLabel());
    return {
      level: "ready",
      headline,
      details,
    };
  }

  setConfig(config) {
    if (!config.entity) {
      throw new Error("entity is required");
    }
    this._config = {
      speed: 0.2,
      blade_mode: "off",
      // Motion defaults are the accepted profile verbatim -- see
      // LUBA_ACCEPTANCE_PROFILE. Do not fork these values here.
      ...LUBA_ACCEPTANCE_PROFILE,
      ...config,
    };
    this._height = Number(this._config.card_height || 520);
    // Survive Lovelace element rebuilds (tab switches, app backgrounding):
    // restore the last completed run so its result stays visible.
    if (!this._realRun) {
      this._realRun = this._restoreLastRun();
    }
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._rendered) {
      this._render();
      this._rendered = true;
    }
    if (!this._mapData && !this._loadingMap) {
      this._loadMap();
    }
    if (!this._runtimeState && !this._loadingRuntime) {
      this._loadRuntimeState();
    }
  }

  getCardSize() {
    return Math.max(4, Math.ceil(this._height / 50));
  }

  static getStubConfig() {
    return {
      entity: "lawn_mower.my_mower",
      card_height: 520,
      speed: 0.2,
    };
  }

  _q(selector) {
    return this.shadowRoot.querySelector(selector);
  }

  async _callService(service, data) {
    const result = await this._hass.callService(
      "mammotion",
      service,
      { entity_id: this._config.entity, ...data },
      {},
      true,
      true,
    );
    return result?.response || result;
  }

  async _loadMap() {
    if (!this._hass || !this._config.entity) return;
    if (this._loadingMap) return;
    this._loadingMap = true;
    this._status = "Loading map…";
    this._render();
    try {
      this._mapData = await this._callService("export_map", {});
      const areaHashes = Object.keys(this._mapData?.area_polygons || {});
      this._areaHash = this._areaHash || areaHashes[0] || "";
      this._status = areaHashes.length
        ? "Map loaded. Click waypoints, then run dry-run or Real Go."
        : "Map loaded, but no area geometry is available.";
      this._render();
      await this._validateAndPreview();
    } catch (err) {
      this._status = `Map load failed: ${err?.message || err}`;
      this._render();
    } finally {
      this._loadingMap = false;
    }
  }

  async _loadRuntimeState() {
    if (!this._hass || !this._config.entity) return;
    if (this._loadingRuntime) return;
    this._loadingRuntime = true;
    try {
      this._runtimeState = await this._callService("export_runtime_state", {});
      const backendSession = this._activeBackendSession();
      if (backendSession && !this._runTicker) {
        this._startRunTicker(this._plannedLegCount() || "?");
      } else if (
        !backendSession &&
        this._runTicker &&
        !this._submittingRealRun
      ) {
        this._stopRunTicker();
      }
      this._render();
      await this._validateAndPreview();
    } catch (err) {
      this._status = `Runtime load failed: ${err?.message || err}`;
      this._render();
    } finally {
      this._loadingRuntime = false;
    }
  }

  _escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  _getAllPoints() {
    const polygons = this._mapData?.area_polygons || {};
    const points = Object.values(polygons).flat();
    const start = this._currentPositionPoint();
    if (start) points.push(start);
    points.push(...this._waypoints);
    return points;
  }

  // 🚨 Keep-out geometry, exposed by `export_map` since beta63. Until this the
  // card drew mowing areas ONLY, so a click landing in an obstacle zone looked
  // exactly like a legal one -- which is how a supervised 10.8 m run drove into
  // a trampoline on 2026-08-20. The backend refuses such a path at dispatch;
  // drawing it here refuses it at CLICK time, before daylight is spent.
  //
  // Sibling dicts of `map.area` in HashList: obstacle, no_go_zone,
  // virtual_wall, no_go_zone_variant, visual_obstacle_zone -- already flattened
  // by the backend into `keep_out_polygons`, keyed `"<kind>:<hash>"`.
  _keepOutPolygons() {
    const zones = this._mapData?.keep_out_polygons;
    if (!zones || typeof zones !== "object") return {};
    return zones;
  }

  // Ray-cast point-in-polygon, matching `_point_in_polygon` in services.py.
  // ⚠️ Boundary cases are not defined identically in the two implementations,
  // so this must never be the ONLY thing standing between a click and a zone --
  // the backend check remains authoritative. This exists to warn early.
  _pointInPolygon(point, polygon) {
    if (!Array.isArray(polygon) || polygon.length < 3) return false;
    const x = Number(point?.x);
    const y = Number(point?.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
    let inside = false;
    for (let i = 0; i < polygon.length; i += 1) {
      const a = polygon[i];
      const b = polygon[(i + 1) % polygon.length];
      const ax = Number(a.x);
      const ay = Number(a.y);
      const bx = Number(b.x);
      const by = Number(b.y);
      if (!Number.isFinite(ax) || !Number.isFinite(by)) continue;
      if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) {
        inside = !inside;
      }
    }
    return inside;
  }

  // Which of these points sit inside a keep-out, and which zone caught them.
  // ⚠️ PER-POINT, exactly like the backend's check -- a leg that clips a corner
  // with neither endpoint inside is NOT caught here either. Segment-level
  // containment is the real fix; this deliberately does not pretend otherwise.
  _keepOutViolations(points) {
    const zones = this._keepOutPolygons();
    const found = [];
    (points || []).forEach((point, index) => {
      for (const [name, polygon] of Object.entries(zones)) {
        if (this._pointInPolygon(point, polygon)) {
          found.push({ index, zone: name, point });
          break;
        }
      }
    });
    return found;
  }

  // Longest leg whose landing the controller can still protect. See
  // CORRECTABLE_AIM_FLOOR_DEGREES. Returns null when the tolerance is unknown,
  // so callers show nothing rather than inventing a bound.
  _correctableLegLimitMetres() {
    const tolerance = Number(this._profileValue("waypoint_tolerance"));
    if (!Number.isFinite(tolerance) || tolerance <= 0) return null;
    const floor = (CORRECTABLE_AIM_FLOOR_DEGREES * Math.PI) / 180;
    const sine = Math.sin(floor);
    if (!(sine > 0)) return null;
    return tolerance / sine;
  }

  // The longest sub-leg the current click would actually drive, AFTER splitting.
  _longestPlannedLegMetres() {
    const points = this._plannedSplit().points || [];
    let longest = 0;
    for (let i = 1; i < points.length; i += 1) {
      const length = Math.hypot(
        points[i].x - points[i - 1].x,
        points[i].y - points[i - 1].y,
      );
      if (Number.isFinite(length) && length > longest) longest = length;
    }
    return longest;
  }

  _currentPositionPoint() {
    const pos = this._livePosition || this._runtimeState?.position || {};
    if (pos.x == null || pos.y == null) {
      return null;
    }
    return {
      x: Number(pos.x),
      y: Number(pos.y),
    };
  }

  // Map-frame bearing the mower would drive forward along, in degrees, using
  // the same arithmetic the backend aims by:
  //
  //   target_map_heading = target_reported_heading + calibrated_offset
  //
  // so forward is (cos, sin) of that angle in map x/y. This remains useful for
  // Real Go control and last-travel diagnostics, but it is not rendered as
  // current body orientation.
  //
  // ⚠️ `toward` is course-over-ground, NOT a compass heading. While the mower is
  // stationary it is the bearing of the last movement and can be stale after a
  // turn. The offset is also a per-mower calibration.
  _headingDegrees() {
    const pos = this._livePosition || this._runtimeState?.position || {};
    const toward = Number(pos.toward);
    const offset = Number(
      this._profileValue("calibrated_forward_heading_offset_degrees"),
    );
    if (!Number.isFinite(toward) || !Number.isFinite(offset)) {
      return null;
    }
    return (((toward + offset) % 360) + 360) % 360;
  }

  // A course-over-ground bearing is not the mower body's current orientation:
  // it remains frozen after an in-place pivot. Only render or use an orientation
  // when the backend explicitly supplies a map-aligned, trustworthy value.
  // No current Mammotion field meets that contract while VIO is inactive, so
  // idle cards deliberately show a position dot without a directional arrow.
  _currentOrientationDegrees() {
    const orientation = this._runtimeState?.current_orientation || {};
    const degrees = Number(orientation.map_heading_degrees);
    if (orientation.trustworthy !== true || !Number.isFinite(degrees)) {
      return null;
    }
    return ((degrees % 360) + 360) % 360;
  }

  _segmentPoints() {
    const start = this._currentPositionPoint();
    if (!start || !this._waypoints.length) {
      return null;
    }
    return [start, ...this._waypoints].map((point) =>
      this._roundedPoint(point),
    );
  }

  _segmentCount() {
    return this._waypoints.length;
  }

  // Backend readiness only, with none of the waypoint/path checks in
  // _preflight(). Nudge has no waypoints and no validated path, so reusing the
  // full preflight would report irrelevant blockers -- but it still must not
  // offer a button the backend will refuse. The first Nudge build did exactly
  // that: it stayed enabled with the motion gate off, so clicking it looked
  // like the feature was broken when the backend was correctly saying no.
  _motionBackendBlockers() {
    const runtime = this._runtimeState || {};
    const experimental = runtime.experimental_motion || {};
    const safety = runtime.safety || {};
    const blockers = [];
    if (!this._currentPositionPoint()) {
      blockers.push("position_unavailable");
    }
    if (this._currentOrientationDegrees() == null) {
      blockers.push("current_orientation_unavailable");
    }
    if (experimental.real_motion_allowed !== true) {
      blockers.push(
        ...(Array.isArray(experimental.blockers) && experimental.blockers.length
          ? experimental.blockers
          : ["experimental_motion_backend_not_ready"]),
      );
    }
    if (safety.allowed_for_manual_motion === false) {
      blockers.push(
        ...(Array.isArray(safety.blockers) && safety.blockers.length
          ? safety.blockers
          : ["runtime_safety_blocked"]),
      );
    }
    // Same duplicate-source problem as _preflight(): the two backend lists
    // overlap, and this one feeds the Nudge tooltip.
    return [...new Set(blockers)];
  }

  // Blockers currently firing that the operator is allowed to override, in
  // registry order. Only firing blockers are offered: a toggle for a gate that
  // is not blocking anything invites arming something for no reason.
  // The override panel. Renders one toggle per FIRING overridable blocker,
  // grouped by tier, each carrying the reason the gate exists. Collapsed by
  // default and absent entirely when nothing is blocking -- this must never
  // read as a normal part of the run flow.
  _overridePanelHtml(preflight) {
    const items = this._overridableBlockers(preflight.blockers);
    if (!items.length) return "";
    const on = items.filter((item) => item.on);
    const rows = items
      .map(
        (
          item,
        ) => `<label class="override-row risk-${item.risk} ${item.on ? "on" : ""}">
          <input type="checkbox" class="override-toggle" data-override="${this._escapeHtml(item.code)}" ${item.on ? "checked" : ""}/>
          <span class="override-code">${this._escapeHtml(item.code)}</span>
          <span class="override-tier">${this._escapeHtml(item.tier)}${item.risk === "high" ? " · high risk" : ""}</span>
          <span class="override-why">${this._escapeHtml(item.why)}</span>
        </label>`,
      )
      .join("");
    const summary = on.length
      ? `⚠️ ${on.length} safety gate${on.length === 1 ? "" : "s"} overridden for the next run`
      : `Override blockers (${items.length} available)`;
    const active = on.length
      ? `<div class="override-active">These gates will NOT stop the next Real Go: <b>${this._escapeHtml(on.map((i) => i.code).join(", "))}</b>. They reset automatically when the run finishes. Every override is recorded in the run JSON.</div>`
      : "";
    return `<details class="override-panel${on.length ? " armed" : ""}" ${on.length ? "open" : ""}>
      <summary>${this._escapeHtml(summary)}</summary>
      ${active}
      <div class="override-note">A toggle lifts the gate in the BACKEND too, not just on this card. Dry runs ignore overrides on purpose, so use one to see the honest verdict first.</div>
      ${rows}
    </details>`;
  }

  _overridableBlockers(blockers) {
    const seen = new Set();
    return OVERRIDE_TIERS.flatMap((tier) =>
      (blockers || [])
        .filter((code) => {
          const meta = OVERRIDABLE_BLOCKERS[code];
          if (!meta || meta.tier !== tier || seen.has(code)) return false;
          seen.add(code);
          return true;
        })
        .map((code) => ({
          code,
          tier,
          risk: OVERRIDABLE_BLOCKERS[code].risk,
          backend: OVERRIDABLE_BLOCKERS[code].backend || code,
          why: OVERRIDE_WHY[code] || "",
          on: this._overrides.has(code),
        })),
    );
  }

  // The backend gate names to send. Card codes and backend gate names are not
  // always identical (`real_segment_limit_4` vs `real_segment_limit`,
  // `blade_unsafe` vs `mower_reports_blades_off`), and sending a card-only name
  // would be REFUSED by the schema -- which is the correct fail-closed
  // behaviour, but it would silently drop the override the operator asked for.
  _overridePayloadNames() {
    const names = new Set();
    for (const code of this._overrides) {
      const meta = OVERRIDABLE_BLOCKERS[code];
      if (meta) names.add(meta.backend || code);
    }
    return [...names];
  }

  _toggleOverride(code) {
    if (!OVERRIDABLE_BLOCKERS[code]) return;
    if (this._overrides.has(code)) {
      this._overrides.delete(code);
    } else {
      this._overrides.add(code);
    }
    this._render();
  }

  _clearOverrides() {
    if (!this._overrides.size) return;
    this._overrides = new Set();
  }

  _preflight() {
    const blockers = [];
    const runtime = this._runtimeState || {};
    const safety = runtime.safety || {};
    const experimental = runtime.experimental_motion || {};
    const start = this._currentPositionPoint();
    if (!start) {
      blockers.push("position_unavailable");
    }
    if (!this._waypoints.length) {
      blockers.push("path_unset");
    }
    if (this._segmentCount() > MAX_REAL_SEGMENTS) {
      blockers.push(`real_segment_limit_${MAX_REAL_SEGMENTS}`);
    }
    // Deliberately DISTINCT from real_segment_limit_N. That one means "you
    // clicked too many destinations"; this one means "the destinations are too
    // far apart to reach in the segment budget once split". Collapsing them
    // would give the operator advice for the wrong problem.
    const plannedSplit = this._plannedSplit();
    if (plannedSplit.applied && plannedSplit.subLegCount > MAX_REAL_SEGMENTS) {
      blockers.push("split_exceeds_real_segment_budget");
    }
    // Refuse an over-long leg HERE rather than letting the backend gate catch
    // it, so the reason appears in the readiness banner instead of arriving as
    // a failed run. The backend keeps its own copy: this one is a courtesy, not
    // the guard.
    // ⚠️ NO PLANNING MARGIN HERE, deliberately. A margin was tried on
    // 2026-08-17 to cover the card measuring PLANNED waypoint-to-waypoint
    // geometry while the backend measures LIVE POSITION to target -- they
    // differ by the previous segment's landing error. But a 0.2 m margin
    // refuses a 6.096 m (20 ft) leg, which is the exact length the cap exists
    // to allow, and segment 1 starts AT the live position so there is no drift
    // for it to cover. A later segment that drifts past the cap is refused by
    // the backend with `segment_too_long` and a diagnostics block naming the
    // measured length; that is the honest failure, and it beats pre-refusing a
    // legal leg.
    if (this._longestSegmentMetres() > MAX_REAL_SEGMENT_METRES + 1e-9) {
      blockers.push("segment_too_long");
    }
    // The backend's budget gate is per-segment and never reaches
    // `_preflight().blockers`, so its BLOCKER_HELP entry could never render.
    // Mirror the arithmetic here instead of shipping help text for a code the
    // banner cannot receive.
    if (this._linearBudgetReachMetres() < this._longestSegmentMetres()) {
      blockers.push("linear_budget_insufficient_for_segment");
    }
    if (experimental.real_motion_allowed !== true) {
      if (
        Array.isArray(experimental.blockers) &&
        experimental.blockers.length
      ) {
        blockers.push(...experimental.blockers);
      } else {
        blockers.push("experimental_motion_backend_not_ready");
      }
    }
    if (safety.allowed_for_manual_motion === false) {
      if (Array.isArray(safety.blockers) && safety.blockers.length) {
        blockers.push(...safety.blockers);
      } else {
        blockers.push("runtime_safety_blocked");
      }
    }
    // ⚠️ ONLY when there is a path to validate. `_validation` starts null and
    // stays null until a preview runs, so an unconditional test reported
    // "path_validation_failed" on an EMPTY card -- alongside `path_unset`,
    // which already says the real thing. Worse, its help text sent the
    // operator hunting for "a point outside the selected area" among zero
    // points. Observed on the live card 2026-08-18 with 0/7 points.
    //
    // Not validated yet is not the same as failed validation.
    if (this._waypoints.length && !this._validation?.valid) {
      blockers.push("path_validation_failed");
    }
    // Dedupe first (the backend reports position_not_valid_for_motion and
    // rtk_not_precise on BOTH its experimental_motion and safety lists), then
    // split into what still blocks and what the operator has deliberately
    // overridden. `blockers` stays the honest full list for display; `safe`
    // reflects what will actually dispatch.
    const all = [...new Set(blockers)];
    const overridden = all.filter((code) => this._overrides.has(code));
    const remaining = all.filter((code) => !this._overrides.has(code));
    return {
      safe: remaining.length === 0,
      blockers: all,
      remaining,
      overridden,
      runtime,
    };
  }

  _nightSegmentDistance() {
    const points = this._segmentPoints();
    if (!points || points.length !== 2) return null;
    return Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y);
  }

  // How far the configured linear pulse budget reaches, in metres. Mirrors the
  // backend's `linear_budget_insufficient_for_segment`, including its
  // loop-to-tolerance-only scope: with no ceiling the fixed-budget path fires
  // full ~1.06 m pulses and this arithmetic does not apply, so report Infinity
  // rather than a number that would refuse an accepted run.
  _linearBudgetReachMetres() {
    const ceiling = this._profileValue("max_linear_pulse_ceiling");
    if (ceiling == null) return Infinity;
    return Number(ceiling) * BUDGET_CHECK_METRES_PER_PULSE;
  }

  // Route B: mirror of the backend's `_split_long_legs`. Split every leg longer
  // than SPLIT_LEG_TARGET_METRES into `ceil(d / target)` equal sub-legs by
  // linear interpolation, so every inserted point sits exactly on the
  // operator's original line and every junction it creates is a 0 degree turn.
  //
  // The card computes this for DISPLAY and routing only -- the payload sends
  // the operator's clicks plus `split_leg_target_length_m`, and the backend
  // does the split itself, so the run JSON records what was actually asked for.
  // Duplicated arithmetic is this repo's accepted pattern for a mower-specific
  // number the card owns; keep the two in step.
  //
  // NO ROUNDING of the interpolated coordinates: rounding to 3 dp would inject
  // up to ~1.4 mm of non-collinearity, and collinearity is the entire basis for
  // treating these junctions as free.
  _plannedSplit() {
    const requested = this._segmentPoints();
    if (!requested || requested.length < 2) {
      return {
        applied: false,
        points: requested || [],
        requestedPoints: requested || [],
        requestedLegCount: 0,
        subLegCount: 0,
        insertedIndices: [],
      };
    }
    const points = [requested[0]];
    const insertedIndices = [];
    for (let i = 1; i < requested.length; i += 1) {
      const start = requested[i - 1];
      const end = requested[i];
      const length = Math.hypot(end.x - start.x, end.y - start.y);
      // A non-finite coordinate would make subLegs Infinity and hang the loop
      // inside a render. Leave such a leg unsplit; the length and validation
      // gates refuse it by name.
      const subLegs = Number.isFinite(length)
        ? Math.max(1, Math.ceil(length / SPLIT_LEG_TARGET_METRES))
        : 1;
      for (let step = 1; step < subLegs; step += 1) {
        const fraction = step / subLegs;
        insertedIndices.push(points.length);
        points.push({
          x: start.x + (end.x - start.x) * fraction,
          y: start.y + (end.y - start.y) * fraction,
        });
      }
      points.push(end);
    }
    return {
      applied: points.length > requested.length,
      points,
      requestedPoints: requested,
      requestedLegCount: requested.length - 1,
      subLegCount: points.length - 1,
      insertedIndices,
    };
  }

  // Number of legs the mower will actually drive, after the split. Distinct
  // from `_segmentCount()`, which counts the operator's destinations.
  _plannedLegCount() {
    return this._plannedSplit().subLegCount;
  }

  // Longest single leg in the planned path, in metres. The cap is per-SEGMENT,
  // not per-path: four legs of 6 m are four separate control problems, while
  // one leg of 24 m is the one thing no run has ever done.
  //
  // 🔑 This measures the SPLIT points, not destination-to-destination. Measuring
  // the destinations would have every long click trip `segment_too_long` in
  // _preflight() -- the split would never get a chance to run and Route B would
  // be dead on arrival.
  _longestSegmentMetres() {
    const points = this._plannedSplit().points;
    if (!points || points.length < 2) return 0;
    let longest = 0;
    for (let i = 1; i < points.length; i += 1) {
      longest = Math.max(
        longest,
        Math.hypot(
          points[i].x - points[i - 1].x,
          points[i].y - points[i - 1].y,
        ),
      );
    }
    return longest;
  }

  _nightPreflight({ dryRun = false } = {}) {
    const blockers = dryRun
      ? [
          ...(!this._currentPositionPoint() ? ["position_unavailable"] : []),
          ...(!this._waypoints.length ? ["path_unset"] : []),
          ...(this._waypoints.length && !this._validation?.valid
            ? ["path_validation_failed"]
            : []),
        ]
      : [...this._preflight().blockers].filter(
          (blocker) =>
            !blocker.startsWith("real_segment_limit_") &&
            // Real Go's length and budget gates are not night's. The backend
            // skips BOTH for `turn_mode: "night"` (night owns the tighter
            // `night_segment_too_long` at 1.0 m and runs a fixed budget), so
            // leaking them here blocks a legal night leg on a gate that would
            // never fire -- e.g. a 0.95 m night segment refused because a
            // configured Real Go ceiling of 3 reaches only 0.9 m.
            blocker !== "segment_too_long" &&
            // Night never splits: `_nightMotionPayload` omits the key entirely
            // and the backend caps night at one 1.0 m segment.
            blocker !== "split_exceeds_real_segment_budget" &&
            blocker !== "linear_budget_insufficient_for_segment",
        );
    if (this._segmentCount() !== 1) {
      blockers.push("night_requires_one_segment");
    }
    const rtkLabel = String(
      this._runtimeState?.position?.rtk_status_label ?? "",
    ).toLowerCase();
    if (!dryRun && rtkLabel !== "fix") {
      blockers.push("night_requires_precise_rtk");
    }
    const distance = this._nightSegmentDistance();
    if (distance != null && distance > MAX_NIGHT_SEGMENT_METRES + 1e-9) {
      blockers.push("night_segment_too_long");
    }
    return {
      safe: blockers.length === 0,
      blockers: [...new Set(blockers)],
      distance,
    };
  }

  _activeBackendSession() {
    return this._runtimeState?.experimental_motion?.active_session || null;
  }

  _motionRunActive() {
    return Boolean(this._submittingRealRun || this._activeBackendSession());
  }

  _runtimePreflightDetails() {
    const runtime = this._runtimeState || {};
    const safety = runtime.safety || {};
    const routeStatus = safety.active_route_status || {};
    const activeTransport = runtime.active_transport ?? "unknown";
    const bladeSafe = safety.blade_safe_for_motion === true;
    const activeMowing = safety.active_mowing_detected === true;
    // Note: a plain .includes("charging") check is wrong because "not_charging"
    // contains the substring "charging". Guard against the negated label, and
    // fall back to the numeric charge_state (0 == not charging) when no label.
    const chargeLabelRaw = String(
      runtime.charge_state_label ?? "",
    ).toLowerCase();
    const chargingNow = chargeLabelRaw
      ? chargeLabelRaw.includes("charging") && !chargeLabelRaw.startsWith("not")
      : typeof runtime.charge_state === "number" && runtime.charge_state !== 0;
    const routeBlocks = routeStatus.blocks_motion === true;
    return {
      activeTransport,
      bladeSafeLabel: bladeSafe ? "safe" : "unsafe",
      mowingReadinessLabel: activeMowing
        ? "blocked (active mowing detected)"
        : "ready",
      chargingReadinessLabel: chargingNow ? "charging now" : "not charging",
      routeBlockingLabel: routeBlocks
        ? `blocking (${routeStatus.reason || "unknown_reason"})`
        : `clear (${routeStatus.reason || "no_route"})`,
      haState: runtime.ha_state ?? "unknown",
      workMode: runtime.work_mode_label ?? runtime.work_mode ?? "unknown",
      chargeState:
        runtime.charge_state_label ?? runtime.charge_state ?? "unknown",
      motionEnabled:
        runtime.experimental_motion?.enabled === true ? "enabled" : "disabled",
      backendVerified:
        runtime.experimental_motion?.backend_verified === true
          ? "verified"
          : "unverified",
      backendVersion:
        runtime.experimental_motion?.installed_pymammotion_version ?? "unknown",
      motionBlockers:
        (runtime.experimental_motion?.blockers || []).join(", ") || "none",
      activeSession:
        runtime.experimental_motion?.active_session?.session_id ?? "none",
      sessionPhase:
        runtime.experimental_motion?.active_session?.phase ?? "idle",
      lastDispatch:
        runtime.experimental_motion?.active_session?.last_completed_dispatch
          ?.completed_at ?? "none",
      stopOutcome:
        runtime.experimental_motion?.active_session?.stop_result
          ?.stop_confirmed ??
        runtime.experimental_motion?.last_session?.stop_result
          ?.stop_confirmed ??
        "none",
    };
  }

  _computeMapTransform() {
    const svgEl = this._q("#path-map");
    if (!svgEl) return null;
    const rect = svgEl.getBoundingClientRect();
    const W = rect.width || svgEl.clientWidth || 600;
    const H = this._height;
    const allPts = this._getAllPoints();
    if (!allPts.length) {
      return {
        ppm: 20,
        padX: 40,
        padY: 40,
        W,
        H,
        bounds: { minX: 0, maxX: 10, minY: 0, maxY: 10 },
        toSX: (x) => x * 20,
        toSY: (y) => H - y * 20,
        toMX: (sx) => sx / 20,
        toMY: (sy) => (H - sy) / 20,
      };
    }
    const xs = allPts.map((p) => Number(p.x));
    const ys = allPts.map((p) => Number(p.y));
    const b = {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
    const pad = 40;
    const rangeX = b.maxX - b.minX || 1;
    const rangeY = b.maxY - b.minY || 1;
    const ppm = Math.min((W - pad * 2) / rangeX, (H - pad * 2) / rangeY);
    const padX = (W - rangeX * ppm) / 2;
    const padY = (H - rangeY * ppm) / 2;
    return {
      ppm,
      padX,
      padY,
      W,
      H,
      bounds: b,
      toSX: (mx) => padX + (mx - b.minX) * ppm,
      toSY: (my) => H - padY - (my - b.minY) * ppm,
      toMX: (sx) => b.minX + (sx - padX) / ppm,
      toMY: (sy) => b.minY + (H - padY - sy) / ppm,
    };
  }

  _centroid(points) {
    if (!points.length) return { x: 0, y: 0 };
    let area = 0;
    let cx = 0;
    let cy = 0;
    for (let i = 0; i < points.length; i += 1) {
      const j = (i + 1) % points.length;
      const cross = points[i].x * points[j].y - points[j].x * points[i].y;
      area += cross;
      cx += (points[i].x + points[j].x) * cross;
      cy += (points[i].y + points[j].y) * cross;
    }
    area /= 2;
    if (Math.abs(area) < 1e-10) {
      return {
        x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
        y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
      };
    }
    return { x: cx / (6 * area), y: cy / (6 * area) };
  }

  _svgPointFromEvent(event) {
    const svgEl = this._q("#path-map");
    const mt = this._mapT;
    if (!svgEl || !mt) return null;
    let sx;
    let sy;
    const screenTransform = svgEl.getScreenCTM?.();
    if (screenTransform && svgEl.createSVGPoint) {
      const svgPoint = svgEl.createSVGPoint();
      svgPoint.x = event.clientX;
      svgPoint.y = event.clientY;
      const transformed = svgPoint.matrixTransform(screenTransform.inverse());
      sx = transformed.x;
      sy = transformed.y;
    } else {
      const rect = svgEl.getBoundingClientRect();
      const scaleX =
        Number(svgEl.getAttribute("viewBox")?.split(" ")[2] || rect.width) /
        rect.width;
      const scaleY =
        Number(svgEl.getAttribute("viewBox")?.split(" ")[3] || rect.height) /
        rect.height;
      sx = (event.clientX - rect.left) * scaleX;
      sy = (event.clientY - rect.top) * scaleY;
    }
    return { x: mt.toMX(sx), y: mt.toMY(sy) };
  }

  async _validateAndPreview() {
    if (!this._hass) {
      this._validation = null;
      this._renderMap();
      return;
    }
    const points = this._segmentPoints();
    if (!points) {
      this._validation = null;
      this._renderMap();
      return;
    }
    try {
      const data = {
        points,
        speed: Number(this._config.speed || 0.2),
        blade_mode: "off",
      };
      if (this._areaHash) {
        data.area_hash = this._areaHash;
      }
      this._validation = await this._callService("preview_custom_path", data);
      this._status = this._validation.valid
        ? `Preview valid: ${this._validation.point_count} points, distance ${Number(this._validation.distance || 0).toFixed(2)}.`
        : `Invalid path: ${(this._validation.errors || []).join(", ")}`;
      this._render();
    } catch (err) {
      this._status = `Validation failed: ${err?.message || err}`;
      this._render();
    }
  }

  _onMapClick(event) {
    if (this._motionRunActive()) return;
    if (event.target?.dataset?.pointIndex != null || !this._mapT) return;
    const point = this._svgPointFromEvent(event);
    if (!point) return;
    if (this._waypoints.length >= MAX_WAYPOINTS) {
      this._status = `Maximum ${MAX_WAYPOINTS} waypoints reached. Remove one before adding another.`;
      this._render();
      return;
    }
    // 🚨 Refuse at CLICK time, not at dispatch. The backend refuses this path
    // too (`path_points_inside_keep_out_zone`, beta63), but only once the
    // operator has positioned the mower and spent the daylight getting there.
    // ⚠️ PER-POINT, like the backend: a leg CLIPPING a zone corner with neither
    // endpoint inside is still not caught by either. Do not read this as
    // segment-level containment.
    const blocked = this._keepOutViolations([point]);
    if (blocked.length) {
      const kind = String(blocked[0].zone).split(":")[0];
      this._status = `Refused: that point is inside a keep-out zone (${kind}). Pick a point outside the red zones.`;
      this._render();
      return;
    }
    this._waypoints.push(point);
    this._dryRun = null;
    this._realRun = null;
    this._validateAndPreview();
  }

  _onPointDown(event) {
    if (this._motionRunActive()) return;
    event.stopPropagation();
    const idx = Number(event.target?.dataset?.pointIndex);
    if (Number.isNaN(idx)) return;
    this._draggingIndex = idx;
    event.target.setPointerCapture(event.pointerId);
  }

  _onPointerMove(event) {
    if (this._draggingIndex == null) return;
    const point = this._svgPointFromEvent(event);
    if (!point) return;
    this._waypoints[this._draggingIndex] = point;
    this._dryRun = null;
    this._realRun = null;
    this._renderMap();
  }

  _onPointerUp() {
    if (this._draggingIndex == null) return;
    this._draggingIndex = null;
    this._validateAndPreview();
  }

  _setWaypointCoordinate(index, axis, rawValue) {
    if (this._motionRunActive()) return;
    const text = String(rawValue).trim();
    const value = Number(text);
    if (
      !Number.isInteger(index) ||
      index < 0 ||
      index >= this._waypoints.length ||
      !["x", "y"].includes(axis) ||
      !text ||
      !Number.isFinite(value)
    ) {
      this._status = "Waypoint coordinates must be finite numbers.";
      this._render();
      return;
    }
    this._waypoints[index] = {
      ...this._waypoints[index],
      [axis]: value,
    };
    this._dryRun = null;
    this._realRun = null;
    this._validateAndPreview();
  }

  _clearTarget() {
    if (this._motionRunActive()) return;
    this._waypoints = [];
    this._validation = null;
    this._dryRun = null;
    this._realRun = null;
    this._status = "Path cleared.";
    this._render();
  }

  _removeLastWaypoint() {
    if (this._motionRunActive()) return;
    this._waypoints.pop();
    this._dryRun = null;
    this._realRun = null;
    this._validateAndPreview();
  }

  _roundedPoint(point) {
    return {
      x: Number(point.x.toFixed(3)),
      y: Number(point.y.toFixed(3)),
    };
  }

  _previewPayload() {
    const points = this._segmentPoints();
    if (!points) return null;
    const payload = {
      entity_id: this._config.entity,
      speed: Number(this._config.speed || 0.2),
      blade_mode: "off",
      points,
    };
    if (this._areaHash) {
      payload.area_hash = String(this._areaHash);
    }
    return payload;
  }

  _dryRunPayload() {
    const payload = this._previewPayload();
    if (!payload) return null;
    return {
      ...payload,
      dry_run: true,
    };
  }

  // Resolve one profile key: explicit card YAML wins, otherwise the accepted
  // Gate 4 value. Never `||` -- 0 and false are legitimate profile values.
  _profileValue(key) {
    const configured = this._config?.[key];
    return configured === undefined || configured === null
      ? LUBA_ACCEPTANCE_PROFILE[key]
      : configured;
  }

  // Which profile keys the dashboard YAML overrode. A non-empty list means the
  // payload is NOT the profile that passed supervised LUBA acceptance.
  _profileOverrides() {
    return PROFILE_KEYS.filter((key) => {
      const configured = this._config?.[key];
      if (configured === undefined) return false;
      const accepted = LUBA_ACCEPTANCE_PROFILE[key];
      if (Array.isArray(accepted) || Array.isArray(configured)) {
        return JSON.stringify(configured) !== JSON.stringify(accepted);
      }
      return configured !== accepted;
    });
  }

  // Numeric orientation status plus last-travel diagnostics. It names the
  // source so stale course-over-ground is not mistaken for current heading.
  _headingLabel() {
    const orientation = this._currentOrientationDegrees();
    if (orientation != null) {
      const source =
        this._runtimeState?.current_orientation?.source || "trusted telemetry";
      return `${orientation.toFixed(1)}° current orientation (${source})`;
    }
    const heading = this._headingDegrees();
    if (heading == null) return "current orientation unavailable";
    const pos = this._livePosition || this._runtimeState?.position || {};
    return `current orientation unavailable; last-travel projection ${heading.toFixed(1)}° (course-over-ground ${Number(pos.toward).toFixed(1)}° + offset ${Number(this._profileValue("calibrated_forward_heading_offset_degrees")).toFixed(1)}°; not mower orientation)`;
  }

  // ⚠️ THIS STRING IS A CLAIM ABOUT HARDWARE AND IT HAS ALREADY GONE STALE ONCE.
  // On 2026-08-17 the un-acceptance was hardcoded here; Gate 5 then passed on
  // 2026-08-18 and the card went on telling the operator the profile "owes a
  // Gate 5" for hours. The card cannot read docs/accepted-profile.json at
  // runtime, so this constant is the only place the fact lives -- and a
  // frontend test now reads that file and fails if the two disagree. Update
  // both together or the test will stop you.
  //
  // `_profileOverrides()` diffs the payload against LUBA_ACCEPTANCE_PROFILE, so
  // it catches dashboard YAML overriding a value; it CANNOT catch the accepted
  // value itself moving, which is what happened when the ceiling went 14 -> 22.
  _profileLabel() {
    const overrides = this._profileOverrides();
    if (overrides.length) {
      return `customised (not hardware-accepted): ${overrides.join(", ")}`;
    }
    return `${ACCEPTED_PROFILE_LABEL} (max_linear_pulse_ceiling ${LUBA_ACCEPTANCE_PROFILE.max_linear_pulse_ceiling})`;
  }

  // Straight-line nudge along trustworthy CURRENT orientation only. The old
  // implementation used course-over-ground, but the 2026-08-02 upper-left
  // observation proved that value stays frozen after an in-place pivot. Until
  // a map-aligned orientation source exists, refuse rather than risking a blind
  // turn or driving the opposite way.
  //
  // ⚠️ `turn_mode: "legacy"` is required only to clear the `vio_active` gate,
  // which blocks up-front whenever turn_mode is "vio" regardless of whether a
  // turn is actually needed. Do not conclude that legacy is a night-capable
  // turn mode; it is not, and a blind pivot has no feedback at all.
  _nudgePayload(dryRun) {
    const start = this._currentPositionPoint();
    const heading = this._currentOrientationDegrees();
    if (!start || heading == null) return null;
    const metres = this._nudgeMetres();
    if (!(metres > 0)) return null;
    const rad = (heading * Math.PI) / 180;
    const target = this._roundedPoint({
      x: start.x + metres * Math.cos(rad),
      y: start.y + metres * Math.sin(rad),
    });
    // One linear command covers ~1.06 m (measured), so scale the budget to the
    // requested distance rather than silently stopping short. Schema caps at 3.
    const pulses = Math.min(
      3,
      Math.max(
        1,
        Math.ceil(
          metres /
            Number(this._profileValue("final_approach_metres_per_pulse")),
        ),
      ),
    );
    return {
      service: "raw_pymammotion_execute_vector_segment",
      payload: {
        entity_id: this._config.entity,
        points: [this._roundedPoint(start), target],
        dry_run: dryRun,
        // Blades-off is still gated by telemetry (`mower_reports_blades_off`
        // checks state AND cutter RPM); this is the operator half, and for a
        // bounded blades-off nudge the clear-area confirmation is the one that
        // carries the risk.
        confirm_blades_off: dryRun ? false : true,
        confirm_clear_area: dryRun ? false : this._confirmClearArea,
        prefer_ble: Boolean(this._profileValue("prefer_ble")),
        turn_mode: "legacy",
        max_turn_commands: Number(this._profileValue("max_turn_commands")),
        max_turn_translation_distance: Number(
          this._profileValue("max_turn_translation_distance"),
        ),
        max_linear_commands: pulses,
        max_no_progress_pulses: Number(
          this._profileValue("max_no_progress_pulses"),
        ),
        heading_tolerance_degrees: Number(
          this._profileValue("heading_tolerance_degrees"),
        ),
        min_progress_distance: Number(
          this._profileValue("min_progress_distance"),
        ),
        calibrated_forward_heading_offset_degrees: Number(
          this._profileValue("calibrated_forward_heading_offset_degrees"),
        ),
        turn_pulse_duration_ms: Number(
          this._profileValue("turn_pulse_duration_ms"),
        ),
        linear_pulse_duration_ms: Number(
          this._profileValue("linear_pulse_duration_ms"),
        ),
        waypoint_tolerance: Number(this._profileValue("waypoint_tolerance")),
        motion_refresh_interval_ms: Number(
          this._profileValue("motion_refresh_interval_ms"),
        ),
        final_approach_metres_per_pulse: Number(
          this._profileValue("final_approach_metres_per_pulse"),
        ),
        turn_degrees_per_second: Number(
          this._profileValue("turn_degrees_per_second"),
        ),
        ble_auto_recover: Boolean(this._profileValue("ble_auto_recover")),
        sample_delays: LUBA_ACCEPTANCE_PROFILE.sample_delays,
      },
    };
  }

  _nudgeMetres() {
    const raw = Number(this._nudgeDistance);
    if (!Number.isFinite(raw)) return 0;
    // Hard geometric cap: a mistake is bounded by distance, not by vigilance.
    return Math.min(MAX_NUDGE_METRES, Math.max(0, raw));
  }

  _motionPayload(dryRun) {
    const points = this._segmentPoints();
    if (!points) return null;
    const sampleDelays = this._profileValue("sample_delays");
    const pulseCeiling = this._profileValue("max_linear_pulse_ceiling");
    const payload = {
      entity_id: this._config.entity,
      points,
      dry_run: dryRun,
      confirm_blades_off: dryRun ? false : this._confirmBladesOff,
      confirm_clear_area: dryRun ? false : this._confirmClearArea,
      prefer_ble: Boolean(this._profileValue("prefer_ble")),
      max_turn_commands: Number(this._profileValue("max_turn_commands")),
      max_turn_translation_distance: Number(
        this._profileValue("max_turn_translation_distance"),
      ),
      max_linear_commands: Number(this._profileValue("max_linear_commands")),
      max_no_progress_pulses: Number(
        this._profileValue("max_no_progress_pulses"),
      ),
      heading_tolerance_degrees: Number(
        this._profileValue("heading_tolerance_degrees"),
      ),
      min_progress_distance: Number(
        this._profileValue("min_progress_distance"),
      ),
      calibrated_forward_heading_offset_degrees: Number(
        this._profileValue("calibrated_forward_heading_offset_degrees"),
      ),
      turn_mode: String(this._profileValue("turn_mode")),
      vio_turn_max_commands: Number(
        this._profileValue("vio_turn_max_commands"),
      ),
      turn_pulse_duration_ms: Number(
        this._profileValue("turn_pulse_duration_ms"),
      ),
      linear_pulse_duration_ms: Number(
        this._profileValue("linear_pulse_duration_ms"),
      ),
      waypoint_tolerance: Number(this._profileValue("waypoint_tolerance")),
      sample_delays: Array.isArray(sampleDelays)
        ? sampleDelays
        : LUBA_ACCEPTANCE_PROFILE.sample_delays,
      motion_refresh_interval_ms: Number(
        this._profileValue("motion_refresh_interval_ms"),
      ),
      final_approach_metres_per_pulse: Number(
        this._profileValue("final_approach_metres_per_pulse"),
      ),
      turn_degrees_per_second: Number(
        this._profileValue("turn_degrees_per_second"),
      ),
      ble_auto_recover: Boolean(this._profileValue("ble_auto_recover")),
    };
    // Only on a REAL run. A dry run is advisory and should keep reporting every
    // gate honestly -- overriding there would hide exactly what the dry run
    // exists to reveal.
    const overrideNames = dryRun ? [] : this._overridePayloadNames();
    if (overrideNames.length) {
      payload.safety_overrides = overrideNames;
    }
    // Omitted, not zeroed: the backend treats a missing ceiling as "no
    // loop-to-tolerance", which is what Gate 4 ran. Sending 0 would fail the
    // schema's Range(min=1).
    if (pulseCeiling !== null && pulseCeiling !== undefined) {
      payload.max_linear_pulse_ceiling = Number(pulseCeiling);
    }
    if (this._areaHash) {
      payload.area_hash = String(this._areaHash);
    }
    // Route B: a single distant click still needs the multi-segment executor,
    // because the backend will split it into several legs. Route on the number
    // of legs that will actually be DRIVEN, not on how many points were clicked.
    const split = this._plannedSplit();
    if (split.subLegCount > 1) {
      return {
        service: "raw_pymammotion_execute_multi_segment",
        payload: {
          ...payload,
          // `points` stays the operator's CLICKS, not the split path -- the
          // backend splits and echoes both, so the run JSON records what was
          // asked for as well as what was driven.
          split_leg_target_length_m: SPLIT_LEG_TARGET_METRES,
          // ⚠️ Clamp to MAX_REAL_SEGMENTS on BOTH branches. This used to send
          // `Math.min(points.length - 1, MAX_WAYPOINTS)` -- up to 7 -- into the
          // backend's `vol.Range(min=0, max=4)`, so a 5+-waypoint DRY RUN was
          // rejected by schema validation before it reached the handler.
          // Behaviour-neutral: `max_real_segments` is only read behind
          // `if not dry_run`.
          max_real_segments: MAX_REAL_SEGMENTS,
        },
      };
    }
    return {
      service: "raw_pymammotion_execute_vector_segment",
      payload,
    };
  }

  _nightMotionPayload(dryRun) {
    const points = this._segmentPoints();
    if (!points || points.length !== 2) return null;
    const payload = {
      entity_id: this._config.entity,
      points,
      dry_run: dryRun,
      confirm_blades_off: dryRun ? false : this._confirmBladesOff,
      confirm_clear_area: dryRun ? false : this._confirmClearArea,
      prefer_ble: NIGHT_GO_PROFILE.prefer_ble,
      turn_mode: NIGHT_GO_PROFILE.turn_mode,
      max_linear_commands: NIGHT_GO_PROFILE.max_linear_commands,
      motion_refresh_interval_ms: NIGHT_GO_PROFILE.motion_refresh_interval_ms,
      max_turn_commands: NIGHT_GO_PROFILE.max_turn_commands,
      heading_tolerance_degrees: NIGHT_GO_PROFILE.heading_tolerance_degrees,
      turn_pulse_duration_ms: NIGHT_GO_PROFILE.turn_pulse_duration_ms,
      max_turn_translation_distance:
        NIGHT_GO_PROFILE.max_turn_translation_distance,
      ble_auto_recover: NIGHT_GO_PROFILE.ble_auto_recover,
      night_angular_speed: NIGHT_GO_PROFILE.night_angular_speed,
      toward_mirror_degrees: NIGHT_GO_PROFILE.toward_mirror_degrees,
      sample_delays: NIGHT_GO_PROFILE.sample_delays,
    };
    const nightOverrides = dryRun ? [] : this._overridePayloadNames();
    if (nightOverrides.length) {
      payload.safety_overrides = nightOverrides;
    }
    // Deliberately OMIT max_linear_pulse_ceiling. Night v1 is fixed-budget;
    // sending the accepted daylight ceiling would be refused by the backend.
    if (this._areaHash) {
      payload.area_hash = String(this._areaHash);
    }
    return {
      service: "raw_pymammotion_execute_vector_segment",
      payload,
    };
  }

  _segmentProgressText(result) {
    if (!result) return "";
    if (!Array.isArray(result.segments) || !result.segments.length) {
      const blockers = result.blockers || [];
      return blockers.length
        ? `stop_reason=${result.stop_reason || "unknown"}, blockers=${blockers.join(", ")}`
        : `stop_reason=${result.stop_reason || "unknown"}`;
    }
    const total = result.total_segments ?? result.segments.length;
    const executed = result.segments.length;
    const failedIndex = result.failed_segment_index;
    if (failedIndex) {
      return `Stopped at segment ${failedIndex} of ${total}: ${result.stop_reason || "segment_failed"}`;
    }
    return `segment ${executed} of ${total}: ${result.stop_reason || "unknown"}`;
  }

  _payloadYaml() {
    const payload = this._previewPayload();
    if (!payload) return "";
    return this._yamlForPayload(payload);
  }

  _dryRunYaml() {
    const payload = this._dryRunPayload();
    if (!payload) return "";
    return this._yamlForPayload(payload);
  }

  _yamlForPayload(payload) {
    const lines = [`entity_id: ${payload.entity_id}`];
    if (payload.area_hash) {
      lines.push(`area_hash: "${payload.area_hash}"`);
    }
    lines.push(`speed: ${payload.speed}`);
    lines.push(`blade_mode: "${payload.blade_mode}"`);
    if (payload.dry_run != null) {
      lines.push(`dry_run: ${payload.dry_run ? "true" : "false"}`);
    }
    lines.push("points:");
    for (const point of payload.points) {
      lines.push(`  - x: ${point.x}`);
      lines.push(`    y: ${point.y}`);
    }
    return `${lines.join("\n")}\n`;
  }

  _payloadJson() {
    const payload = this._previewPayload();
    if (!payload) return "";
    return `${JSON.stringify(payload, null, 2)}\n`;
  }

  _dryRunJson() {
    const payload = this._dryRunPayload();
    if (!payload) return "";
    return `${JSON.stringify(payload, null, 2)}\n`;
  }

  async _copyText(text, label) {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.setAttribute("readonly", "readonly");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        this.shadowRoot.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
      this._status = `${label} copied. This is a preview payload only; no mower command was sent.`;
    } catch (err) {
      this._status = `Copy failed: ${err?.message || err}`;
    }
    this._render();
  }

  _copyYaml() {
    if (!this._waypoints.length) {
      this._status = "Add at least one waypoint before copying YAML.";
      this._render();
      return;
    }
    this._copyText(this._payloadYaml(), "YAML");
  }

  _copyJson() {
    if (!this._waypoints.length) {
      this._status = "Add at least one waypoint before copying JSON.";
      this._render();
      return;
    }
    this._copyText(this._payloadJson(), "JSON");
  }

  _copyDryRunYaml() {
    if (!this._waypoints.length) {
      this._status = "Add at least one waypoint before copying dry-run YAML.";
      this._render();
      return;
    }
    this._copyText(this._dryRunYaml(), "Dry-run YAML");
  }

  async _runDryRun() {
    const motion = this._motionPayload(true);
    if (!motion) {
      this._status =
        "Add at least one waypoint and ensure live mower position is available before dry-run.";
      this._render();
      return;
    }
    this._status = "Running guarded segment-chain dry-run…";
    this._render();
    try {
      this._dryRun = await this._callService(motion.service, motion.payload);
      this._status = `Dry-run complete. ${this._segmentProgressText(this._dryRun)}. No mower command was sent.`;
      this._render();
    } catch (err) {
      this._status = `Dry-run failed: ${err?.message || err}`;
      this._render();
    }
  }

  async _runNightDryRun() {
    await this._loadRuntimeState();
    await this._validateAndPreview();
    const preflight = this._nightPreflight({ dryRun: true });
    const motion = this._nightMotionPayload(true);
    if (!motion || !preflight.safe) {
      this._status = `Night dry-run blocked: ${preflight.blockers.join(", ") || "one waypoint and live mower position required"}.`;
      this._render();
      return;
    }
    this._status = "Running bounded Night Go dry-run…";
    this._render();
    try {
      this._dryRun = await this._callService(motion.service, motion.payload);
      this._status = `Night dry-run complete. ${this._segmentProgressText(this._dryRun)}. No mower command was sent.`;
    } catch (err) {
      this._status = `Night dry-run failed: ${err?.message || err}`;
    }
    this._render();
  }

  _startRunTicker(segmentCount, runLabel = "Real Go") {
    this._runStartedAt = Date.now();
    this._livePosition = null;
    const tick = async () => {
      const elapsed = Math.round((Date.now() - this._runStartedAt) / 1000);
      const mins = Math.floor(elapsed / 60);
      const secs = String(elapsed % 60).padStart(2, "0");
      let posText = "";
      try {
        const runtime = await this._callService("export_runtime_state", {});
        this._runtimeState = runtime;
        const pos = runtime?.position;
        if (pos && pos.x != null) {
          this._livePosition = pos;
          posText = ` — pos (${Number(pos.x).toFixed(2)}, ${Number(pos.y).toFixed(2)})${pos.toward != null ? ` hdg ${Number(pos.toward).toFixed(0)}°` : ""}`;
        }
      } catch (err) {
        // Read-only poll; ignore transient failures while the run is in flight.
      }
      const session = this._activeBackendSession();
      const phase =
        session?.phase || (this._submittingRealRun ? "submitting" : "waiting");
      const lastWrite = session?.last_completed_dispatch?.elapsed_seconds;
      const lastWriteText =
        lastWrite == null
          ? ""
          : ` — last write ${Number(lastWrite).toFixed(1)}s`;
      this._status = `Running ${runLabel} (${segmentCount} segment${segmentCount === 1 ? "" : "s"}, ${phase})… ${mins}:${secs}${posText}${lastWriteText}`;
      this._render();
    };
    this._runTicker = setInterval(tick, 5000);
    tick();
  }

  _stopRunTicker() {
    if (this._runTicker) {
      clearInterval(this._runTicker);
      this._runTicker = null;
    }
    this._runStartedAt = null;
  }

  async _runRealGo() {
    if (this._motionRunActive()) {
      this._status =
        "Real Go blocked: a manual-motion session is already active.";
      this._render();
      return;
    }
    this._submittingRealRun = true;
    this._status =
      "Refreshing runtime and path validation immediately before Real Go…";
    this._render();
    await this._loadRuntimeState();
    await this._validateAndPreview();
    const preflight = this._preflight();
    const motion = this._motionPayload(false);
    if (!motion) {
      this._status =
        "Add at least one waypoint and ensure live mower position is available before Real Go.";
      this._submittingRealRun = false;
      this._render();
      return;
    }
    if (!this._confirmBladesOff || !this._confirmClearArea) {
      this._status = "Real Go blocked: enable both confirmations first.";
      this._submittingRealRun = false;
      this._render();
      return;
    }
    if (!preflight.safe) {
      this._status = `Real Go blocked by preflight: ${preflight.blockers.join(", ")}`;
      this._submittingRealRun = false;
      this._render();
      return;
    }
    // The ticker counts DRIVEN legs, not clicked destinations -- after a split
    // they differ, and the ticker is the operator's only live progress signal.
    const segmentCount = this._plannedLegCount();
    const startedAt = new Date();
    this._startRunTicker(segmentCount);
    try {
      this._realRun = await this._callService(motion.service, motion.payload);
      this._stopRunTicker();
      this._status = `Real Go complete: ${this._segmentProgressText(this._realRun)}`;
      this._persistLastRun(this._realRun);
      this._saveRunToHistory(
        {
          at: startedAt.toISOString(),
          elapsed_seconds: Math.round(
            (Date.now() - startedAt.getTime()) / 1000,
          ),
          service: motion.service,
          waypoints: this._waypoints.length,
          stop_reason: this._realRun?.stop_reason ?? null,
          failed_segment_index: this._realRun?.failed_segment_index ?? null,
          segments: this._segmentLandingRows(this._realRun).map((row) => ({
            index: row.index,
            passed: row.passed,
            stop_reason: row.stopReason,
            landing: row.landing,
            tolerance: row.tolerance,
          })),
          summary: this._segmentProgressText(this._realRun),
        },
        this._realRun,
      );
    } catch (err) {
      this._stopRunTicker();
      this._status = `Real Go failed: ${err?.message || err}`;
      this._saveRunToHistory({
        at: startedAt.toISOString(),
        elapsed_seconds: Math.round((Date.now() - startedAt.getTime()) / 1000),
        service: motion.service,
        waypoints: this._waypoints.length,
        stop_reason: `call_failed: ${err?.message || err}`,
        failed_segment_index: null,
        segments: [],
        summary: String(err?.message || err),
      });
      this._render();
    } finally {
      this._submittingRealRun = false;
      this._confirmBladesOff = false;
      this._confirmClearArea = false;
      // Overrides die with the run that used them. See the constructor note.
      this._clearOverrides();
      this._stopRunTicker();
      await this._loadRuntimeState();
      this._render();
    }
  }

  async _runNightGo() {
    if (this._motionRunActive()) {
      this._status =
        "Night Go blocked: a manual-motion session is already active.";
      this._render();
      return;
    }
    this._submittingRealRun = true;
    this._status =
      "Refreshing runtime and path validation immediately before Night Go…";
    this._render();
    await this._loadRuntimeState();
    await this._validateAndPreview();
    const preflight = this._nightPreflight();
    const motion = this._nightMotionPayload(false);
    if (
      !motion ||
      !this._confirmBladesOff ||
      !this._confirmClearArea ||
      !this._confirmNightExperimental
    ) {
      this._status =
        "Night Go blocked: select exactly one waypoint and enable all three confirmations, including the night accuracy acknowledgement.";
      this._submittingRealRun = false;
      this._render();
      return;
    }
    if (!preflight.safe) {
      this._status = `Night Go blocked by preflight: ${preflight.blockers.join(", ")}`;
      this._submittingRealRun = false;
      this._render();
      return;
    }
    const startedAt = new Date();
    this._startRunTicker(1, "Night Go");
    try {
      this._realRun = await this._callService(motion.service, motion.payload);
      this._stopRunTicker();
      this._status = `Night Go complete: ${this._segmentProgressText(this._realRun)}`;
      this._persistLastRun(this._realRun);
      this._saveRunToHistory(
        {
          at: startedAt.toISOString(),
          elapsed_seconds: Math.round(
            (Date.now() - startedAt.getTime()) / 1000,
          ),
          service: motion.service,
          run_mode: "night",
          waypoints: 1,
          stop_reason: this._realRun?.stop_reason ?? null,
          failed_segment_index: null,
          segments: this._segmentLandingRows(this._realRun).map((row) => ({
            index: row.index,
            passed: row.passed,
            stop_reason: row.stopReason,
            landing: row.landing,
            tolerance: row.tolerance,
          })),
          summary: this._segmentProgressText(this._realRun),
        },
        this._realRun,
      );
    } catch (err) {
      this._stopRunTicker();
      this._status = `Night Go failed: ${err?.message || err}`;
      this._saveRunToHistory({
        at: startedAt.toISOString(),
        elapsed_seconds: Math.round((Date.now() - startedAt.getTime()) / 1000),
        service: motion.service,
        run_mode: "night",
        waypoints: 1,
        stop_reason: `call_failed: ${err?.message || err}`,
        failed_segment_index: null,
        segments: [],
        summary: String(err?.message || err),
      });
    } finally {
      this._submittingRealRun = false;
      this._confirmBladesOff = false;
      this._confirmClearArea = false;
      // Overrides die with the run that used them. See the constructor note.
      this._clearOverrides();
      this._confirmNightExperimental = false;
      this._stopRunTicker();
      await this._loadRuntimeState();
      this._render();
    }
  }

  disconnectedCallback() {
    this._stopRunTicker();
  }

  async _runNudge() {
    if (this._motionRunActive()) {
      this._status =
        "Nudge blocked: a manual-motion session is already active.";
      this._render();
      return;
    }
    if (!this._confirmClearArea) {
      this._status = "Nudge blocked: confirm the area is clear first.";
      this._render();
      return;
    }
    await this._loadRuntimeState();
    // Re-check against FRESH runtime, not the state the button was rendered
    // from. Naming the blocker matters: the commonest one is simply that the
    // experimental-motion option is off, which otherwise presents as "the
    // button does nothing".
    const blocked = this._motionBackendBlockers();
    if (blocked.length) {
      this._status = `Nudge blocked by the backend: ${blocked.join(", ")}. ${
        blocked.includes("experimental_motion_disabled")
          ? "Enable the integration option 'Enable experimental BLE-only manual motion'."
          : ""
      }`;
      this._render();
      return;
    }
    const nudge = this._nudgePayload(false);
    if (!nudge) {
      this._status =
        "Nudge needs trustworthy current orientation. Course-over-ground is only last travel and cannot observe an in-place turn.";
      this._render();
      return;
    }
    const metres = this._nudgeMetres();
    this._status = `Nudging ${metres.toFixed(2)} m along ${this._currentOrientationDegrees().toFixed(1)}°…`;
    this._render();
    try {
      const result = await this._callService(nudge.service, nudge.payload);
      this._realRun = result;
      this._persistLastRun(result);
      const reason = result?.stop_reason || "?";
      const sent = result?.linear_commands_sent ?? "?";
      const turns = result?.turn_commands_sent ?? "?";
      this._status = `Nudge finished: ${reason} (linear ${sent}, turns ${turns}). Turns must be 0 — anything else means it tried to steer blind.`;
      this._confirmClearArea = false;
      await this._loadRuntimeState();
    } catch (err) {
      this._status = `Nudge failed: ${err?.message || err}`;
    }
    this._render();
  }

  async _abortMotion() {
    this._status =
      "Aborting backend session and sending confirmed BLE stop sequence…";
    this._render();
    try {
      const abortResult = await this._callService("stop_manual_motion", {});
      this._realRun = {
        ...(this._realRun || {}),
        stop_result: abortResult,
        stop_reason: abortResult?.stop_confirmed
          ? "operator_stop"
          : "stop_unconfirmed",
      };
      const status =
        abortResult?.stop_confirmed === true ? "confirmed" : "not confirmed";
      this._status = `Abort result: ${status}; owner exited=${Boolean(abortResult?.owner_exited)}`;
      this._confirmBladesOff = false;
      this._confirmClearArea = false;
      this._confirmNightExperimental = false;
      await this._loadRuntimeState();
      this._render();
    } catch (err) {
      this._status = `Abort failed: ${err?.message || err}`;
      this._render();
    }
  }

  _renderMap() {
    const svgEl = this._q("#path-map");
    if (!svgEl) return;
    this._mapT = this._computeMapTransform();
    const mt = this._mapT;
    if (!mt) return;
    while (svgEl.firstChild) svgEl.removeChild(svgEl.firstChild);
    svgEl.setAttribute("viewBox", `0 0 ${mt.W} ${mt.H}`);

    const ns = "http://www.w3.org/2000/svg";
    const el = (name, attrs = {}) => {
      const node = document.createElementNS(ns, name);
      for (const [key, value] of Object.entries(attrs)) {
        node.setAttribute(key, String(value));
      }
      return node;
    };

    const polygons = this._mapData?.area_polygons || {};
    const areaNames = Object.fromEntries(
      (this._mapData?.areas || []).map((area) => [
        String(area.area_hash),
        area.name,
      ]),
    );
    for (const [hash, points] of Object.entries(polygons)) {
      if (points.length < 2) continue;
      const active = hash === String(this._areaHash);
      const polygon = el("polygon", {
        points: points
          .map(
            (point) =>
              `${mt.toSX(point.x).toFixed(1)},${mt.toSY(point.y).toFixed(1)}`,
          )
          .join(" "),
        fill: active ? "rgba(96,165,250,0.14)" : "rgba(55,65,81,0.25)",
        stroke: active ? "#60a5fa" : "#4b5563",
        "stroke-width": active ? "2" : "1",
        "stroke-linejoin": "round",
      });
      svgEl.appendChild(polygon);
      const c = this._centroid(points);
      const label = el("text", {
        x: mt.toSX(c.x).toFixed(1),
        y: mt.toSY(c.y).toFixed(1),
        "text-anchor": "middle",
        "dominant-baseline": "middle",
        fill: active ? "#bfdbfe" : "#9ca3af",
        "font-size": "12",
        "pointer-events": "none",
      });
      label.textContent = areaNames[hash] || hash.slice(-6);
      svgEl.appendChild(label);
    }

    // 🚨 Keep-outs AFTER areas, so an obstacle can never be painted over by the
    // mowing zone that contains it. Both are filled, and SVG paints in document
    // order: draw these first and a keep-out inside "Backyard Right" would be
    // invisible -- the precise failure mode that let a leg be clicked straight
    // through a trampoline on 2026-08-20.
    for (const [name, points] of Object.entries(this._keepOutPolygons())) {
      if (!Array.isArray(points) || points.length < 3) continue;
      svgEl.appendChild(
        el("polygon", {
          points: points
            .map(
              (point) =>
                `${mt.toSX(point.x).toFixed(1)},${mt.toSY(point.y).toFixed(1)}`,
            )
            .join(" "),
          fill: "rgba(239,68,68,0.22)",
          stroke: "#ef4444",
          "stroke-width": "2",
          "stroke-dasharray": "5,3",
          "stroke-linejoin": "round",
          "pointer-events": "none",
        }),
      );
      const centre = this._centroid(points);
      const marker = el("text", {
        x: mt.toSX(centre.x).toFixed(1),
        y: mt.toSY(centre.y).toFixed(1),
        "text-anchor": "middle",
        "dominant-baseline": "middle",
        fill: "#fca5a5",
        "font-size": "11",
        "pointer-events": "none",
      });
      // The kind ("obstacle", "no_go_zone", ...) is the informative half; the
      // hash after the colon is noise at a glance.
      marker.textContent = `⛔ ${String(name).split(":")[0]}`;
      svgEl.appendChild(marker);
    }

    const start = this._currentPositionPoint();
    // ⚠️ Draw and colour against the SPLIT path, not [start, ...waypoints].
    // `runResult.segments` is one entry per DRIVEN leg, so once a split turns
    // one click into four legs, indexing them against the clicks paints
    // segment 1's verdict over the whole leg and drops segments 2-4 entirely.
    const plannedSplit = this._plannedSplit();
    const pathPoints = start ? plannedSplit.points : [];
    const insertedIndices = new Set(plannedSplit.insertedIndices);
    const runResult = this._realRun || this._dryRun;
    const segments = Array.isArray(runResult?.segments)
      ? runResult.segments
      : null;

    if (pathPoints.length >= 2) {
      for (let i = 0; i < pathPoints.length - 1; i += 1) {
        let stroke = this._validation?.valid === false ? "#ef4444" : "#22c55e";
        let dashArray = null;
        if (segments) {
          const seg = segments[i];
          if (!seg || seg.passed == null) {
            stroke = "#6b7280";
            dashArray = "6,4";
          } else if (seg.passed === false) {
            stroke = "#ef4444";
          } else {
            stroke = "#22c55e";
          }
        }
        const segAttrs = {
          points: [pathPoints[i], pathPoints[i + 1]]
            .map(
              (point) =>
                `${mt.toSX(point.x).toFixed(1)},${mt.toSY(point.y).toFixed(1)}`,
            )
            .join(" "),
          fill: "none",
          stroke,
          "stroke-width": "4",
          "stroke-linecap": "round",
          "stroke-linejoin": "round",
        };
        if (dashArray) segAttrs["stroke-dasharray"] = dashArray;
        const path = el("polyline", segAttrs);
        svgEl.appendChild(path);
      }
    }

    if (start) {
      // Draw an arrow only for explicitly trustworthy current orientation.
      // Stale course-over-ground remains available as text diagnostics but must
      // not masquerade as the direction the stationary mower currently faces.
      const heading = this._currentOrientationDegrees();
      if (heading != null) {
        const rad = (heading * Math.PI) / 180;
        // Transform a point one metre ahead rather than rotating in screen
        // space: toSY flips the Y axis, so a screen-space rotation would point
        // the arrow at the mirror image of the real bearing.
        const sx = mt.toSX(start.x);
        const sy = mt.toSY(start.y);
        let dx = mt.toSX(start.x + Math.cos(rad)) - sx;
        let dy = mt.toSY(start.y + Math.sin(rad)) - sy;
        const len = Math.hypot(dx, dy);
        if (len > 1e-6) {
          dx /= len;
          dy /= len;
          // Perpendicular, for the arrowhead base.
          const px = -dy;
          const py = dx;
          const TAIL = 6;
          const NECK = 17;
          const TIP = 26;
          const HALF = 6;
          svgEl.appendChild(
            el("line", {
              x1: (sx + dx * TAIL).toFixed(1),
              y1: (sy + dy * TAIL).toFixed(1),
              x2: (sx + dx * NECK).toFixed(1),
              y2: (sy + dy * NECK).toFixed(1),
              stroke: "#22c55e",
              "stroke-width": "3",
              "stroke-linecap": "round",
            }),
          );
          const tip = `${(sx + dx * TIP).toFixed(1)},${(sy + dy * TIP).toFixed(1)}`;
          const left = `${(sx + dx * NECK + px * HALF).toFixed(1)},${(sy + dy * NECK + py * HALF).toFixed(1)}`;
          const right = `${(sx + dx * NECK - px * HALF).toFixed(1)},${(sy + dy * NECK - py * HALF).toFixed(1)}`;
          svgEl.appendChild(
            el("polygon", {
              points: `${tip} ${left} ${right}`,
              fill: "#22c55e",
              stroke: "#111827",
              "stroke-width": "1.5",
            }),
          );
        }
      }
      const startCircle = el("circle", {
        cx: mt.toSX(start.x).toFixed(1),
        cy: mt.toSY(start.y).toFixed(1),
        r: 7,
        fill: "#22c55e",
        stroke: "#111827",
        "stroke-width": "2",
      });
      svgEl.appendChild(startCircle);
    }

    // Split points the operator did not click: hollow, unlabelled and NOT
    // draggable -- they are derived from the two clicks around them and move
    // with them.
    pathPoints.forEach((point, index) => {
      if (!insertedIndices.has(index)) return;
      svgEl.appendChild(
        el("circle", {
          cx: mt.toSX(point.x).toFixed(1),
          cy: mt.toSY(point.y).toFixed(1),
          r: 5,
          fill: "none",
          stroke: "#fbbf24",
          "stroke-width": "2",
          "stroke-dasharray": "3,2",
          "pointer-events": "none",
        }),
      );
    });

    this._waypoints.forEach((point, index) => {
      const isLast = index === this._waypoints.length - 1;
      const circle = el("circle", {
        cx: mt.toSX(point.x).toFixed(1),
        cy: mt.toSY(point.y).toFixed(1),
        r: 8,
        fill: isLast ? "#f97316" : "#fbbf24",
        stroke: "#111827",
        "stroke-width": "2",
        "data-point-index": index,
        cursor: "grab",
      });
      circle.addEventListener("pointerdown", (event) =>
        this._onPointDown(event),
      );
      svgEl.appendChild(circle);
      const label = el("text", {
        x: mt.toSX(point.x).toFixed(1),
        y: mt.toSY(point.y).toFixed(1),
        "text-anchor": "middle",
        "dominant-baseline": "central",
        fill: "#111827",
        "font-size": "10",
        "font-weight": "700",
        "pointer-events": "none",
      });
      label.textContent = String(index + 1);
      svgEl.appendChild(label);
    });
  }

  _render() {
    const areas = this._mapData?.areas || [];
    const pathSet = this._waypoints.length > 0;
    const runActive = this._motionRunActive();
    const removeDisabled = pathSet && !runActive ? "" : "disabled";
    const preflight = this._preflight();
    const preflightText = preflight.safe
      ? "Preflight: safe — tap for runtime details"
      : `Preflight blockers: ${preflight.blockers.join(", ")} — tap for runtime details`;
    const readiness = this._readiness();
    const runtimePanel = this._runtimePreflightDetails();
    const realGoDisabled =
      !pathSet ||
      runActive ||
      !this._confirmBladesOff ||
      !this._confirmClearArea ||
      !preflight.safe;
    const nightPreflight = this._nightPreflight();
    const nightGoDisabled =
      runActive ||
      !this._confirmBladesOff ||
      !this._confirmClearArea ||
      !this._confirmNightExperimental ||
      !nightPreflight.safe;
    const nightTitle = nightPreflight.safe
      ? `One fixed-budget ${Number(nightPreflight.distance).toFixed(3)} m night segment; landing accuracy is not established.`
      : `Night Go unavailable: ${nightPreflight.blockers.join(", ")}`;
    const nudgeBlockers = this._motionBackendBlockers();
    if (!this._confirmClearArea) nudgeBlockers.push("confirm_clear_area");
    if (runActive) nudgeBlockers.push("motion_session_active");
    const nudgeDisabled = nudgeBlockers.length > 0;
    const nudgeTitle = nudgeDisabled
      ? `Nudge unavailable: ${nudgeBlockers.join(", ")}`
      : `Drive ${this._nudgeMetres().toFixed(2)} m along ${(this._currentOrientationDegrees() ?? 0).toFixed(1)}°. Straight line only — never turns.`;
    const segmentCount = this._segmentCount();
    const plannedLegCount = this._plannedLegCount();
    const coordinateEditor = this._waypoints.length
      ? `<div class="coordinate-editor">
          <div class="title">Precise waypoint coordinates (metres)</div>
          ${this._waypoints
            .map(
              (point, index) => `<div class="coordinate-row">
                <span>Waypoint ${index + 1}</span>
                <label>X <input id="waypoint-${index}-x" type="number" step="0.001" value="${Number(point.x).toFixed(3)}" ${runActive ? "disabled" : ""}/></label>
                <label>Y <input id="waypoint-${index}-y" type="number" step="0.001" value="${Number(point.y).toFixed(3)}" ${runActive ? "disabled" : ""}/></label>
              </div>`,
            )
            .join("")}
          <div class="hint">Changes re-run Preview. Run Dry-run again after the final coordinate edit.</div>
        </div>`
      : "";
    this.shadowRoot.innerHTML = `
      <style>
        ha-card { overflow: hidden; user-select: text; -webkit-user-select: text; }

        /* Readiness banner: one place that says whether Real Go will go, and
           if not, why. The blocker codes stay verbatim so they remain
           greppable against the backend, with plain English underneath. */
        .banner { margin: 12px 12px 0; padding: 9px 11px; border-radius: 6px; border-left: 4px solid; font-size: 13px; }
        .banner-headline { font-weight: 600; }
        .banner-detail { margin: 4px 0 0; padding-left: 18px; font-size: 12px; opacity: 0.85; }
        .banner-detail li { margin: 2px 0; }
        .banner.ready { border-color: #22c55e; background: rgba(34,197,94,0.12); }
        .banner.arming { border-color: #f59e0b; background: rgba(245,158,11,0.12); }
        .banner.blocked { border-color: #ef4444; background: rgba(239,68,68,0.12); }
        .banner.busy { border-color: #3b82f6; background: rgba(59,130,246,0.12); }

        /* Override panel. Deliberately visually loud when armed: an overridden
           gate must never look like a normal run. */
        .override-panel { margin: 10px 12px 0; padding: 6px 10px; border: 1px dashed rgba(239,68,68,0.5); border-radius: 6px; font-size: 12px; }
        .override-panel > summary { cursor: pointer; font-weight: 600; color: #ef4444; }
        .override-panel.armed { border-style: solid; border-width: 2px; background: rgba(239,68,68,0.10); }
        .override-active { margin: 6px 0; padding: 6px 8px; border-radius: 4px; background: rgba(239,68,68,0.16); }
        .override-note { margin: 6px 0; opacity: 0.8; font-size: 11px; }
        .override-row { display: grid; grid-template-columns: auto 1fr auto; gap: 4px 8px; align-items: start; padding: 6px; margin: 4px 0; border-radius: 4px; border: 1px solid rgba(127,127,127,0.25); cursor: pointer; }
        .override-row.on { border-color: #ef4444; background: rgba(239,68,68,0.10); }
        .override-row.risk-high .override-tier { color: #ef4444; font-weight: 700; }
        .override-code { font-family: ui-monospace, Menlo, monospace; font-size: 11px; }
        .override-tier { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.75; }
        .override-why { grid-column: 2 / span 2; font-size: 11px; opacity: 0.85; }

        .toolbar { display: flex; gap: 10px; align-items: stretch; padding: 12px; flex-wrap: wrap; }
        .group { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; padding: 6px 9px; border: 1px solid rgba(127,127,127,0.3); border-radius: 6px; }
        .group-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--secondary-text-color); font-weight: 700; }
        .confirm { display: inline-flex; gap: 4px; align-items: center; font-size: 12px; white-space: nowrap; }
        button.primary:not([disabled]) { background: #22c55e; color: #06240f; font-weight: 700; border: 1px solid #16a34a; border-radius: 4px; padding: 4px 12px; cursor: pointer; }
        button.danger:not([disabled]) { background: #ef4444; color: #fff; font-weight: 600; border: 1px solid #dc2626; border-radius: 4px; padding: 4px 10px; cursor: pointer; }
        .export-bar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; padding: 0 12px 12px; }

        /* Segment landings: the numbers every run has so far been read out of
           raw JSON by hand. landing vs waypoint_tolerance is the verdict. */
        .run-summary { margin: 0 12px 12px; padding: 8px 10px; border: 1px solid rgba(127,127,127,0.35); border-radius: 6px; font-size: 12px; }
        .run-summary .title { font-weight: 600; margin-bottom: 6px; display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }
        .run-age { font-weight: 400; font-size: 11px; color: var(--secondary-text-color); }
        .summary-scroll { overflow-x: auto; }
        .run-summary table { border-collapse: collapse; width: 100%; min-width: 460px; }
        .run-summary th { text-align: left; font-weight: 600; color: var(--secondary-text-color); font-size: 11px; padding: 2px 6px 4px 0; white-space: nowrap; }
        .run-summary td { padding: 3px 6px 3px 0; border-top: 1px solid rgba(127,127,127,0.2); white-space: nowrap; }
        .run-summary td.num { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; text-align: right; }
        .run-summary .ok { color: #22c55e; font-weight: 600; }
        .run-summary .bad { color: #ef4444; font-weight: 700; }
        .summary-footer { margin-top: 6px; color: var(--secondary-text-color); font-size: 11px; }

        .status { padding: 0 12px 12px; color: var(--secondary-text-color); font-size: 13px; }
        .card-version { padding: 4px 12px 10px; color: var(--secondary-text-color); font-size: 11px; opacity: 0.6; text-align: right; }
        .history-actions { display: flex; gap: 6px; flex-wrap: wrap; }
        .history-row { padding: 4px 0; border-bottom: 1px solid rgba(127,127,127,0.2); font-size: 12px; }
        .history-when { color: var(--secondary-text-color); }
        .history-outcome { font-weight: 600; }
        .history-segs { color: var(--secondary-text-color); font-size: 11px; padding-left: 8px; }
        .history-clear { margin-top: 8px; font-size: 11px; }
        .warnings { padding: 0 12px 12px; color: #f59e0b; font-size: 12px; }
        .waypoint-counter { font-size: 12px; color: var(--secondary-text-color); margin-left: auto; }
        .map-caption { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; padding: 0 12px 10px; font-size: 12px; color: var(--secondary-text-color); }
        .map-caption .legend { display: inline-flex; gap: 6px; align-items: center; }
        .map-caption .dot { width: 11px; height: 11px; border-radius: 50%; border: 1.5px solid #111827; display: inline-block; flex: none; }
        .coordinate-editor { margin: 0 12px 12px; padding: 8px 10px; border: 1px solid rgba(127,127,127,0.35); border-radius: 6px; font-size: 12px; }
        .coordinate-editor .title { font-weight: 600; margin-bottom: 6px; }
        .coordinate-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 3px 0; }
        .coordinate-row > span { min-width: 76px; }
        .coordinate-row input { width: 7em; font: inherit; }
        .coordinate-editor .hint { color: var(--secondary-text-color); margin-top: 5px; }
        .preflight-panel { margin: 0 12px 12px; padding: 8px 10px; border: 1px solid rgba(127,127,127,0.35); border-radius: 6px; font-size: 12px; color: var(--secondary-text-color); }
        .preflight-panel .title { font-weight: 600; margin-bottom: 6px; color: var(--primary-text-color); }
        .preflight-panel > summary { font-weight: 600; color: var(--primary-text-color); margin-bottom: 6px; }
        .preflight-row { display: flex; justify-content: space-between; gap: 10px; padding: 2px 0; }
        .preflight-row .label { opacity: 0.85; }
        .preflight-row .value { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; text-align: right; }
        details { padding: 0 12px 12px; color: var(--secondary-text-color); font-size: 12px; }
        summary { cursor: pointer; }
        pre { overflow: auto; max-height: 220px; padding: 8px; background: rgba(127,127,127,0.12); border-radius: 4px; user-select: text; -webkit-user-select: text; }
        .copy-result { margin: 6px 0; font-size: 11px; }
        svg { display: block; width: 100%; height: ${this._height}px; background: #0d1117; touch-action: none; cursor: crosshair; user-select: none; -webkit-user-select: none; }
        select, button { font: inherit; }
      </style>
      <ha-card header="Mammotion click/go (guarded segment chain)">
        <div class="banner ${readiness.level}">
          <div class="banner-headline">${this._escapeHtml(readiness.headline)}</div>
          ${
            readiness.details.length
              ? `<ul class="banner-detail">${readiness.details.map((line) => `<li>${this._escapeHtml(line)}</li>`).join("")}</ul>`
              : ""
          }
        </div>
        ${this._overridePanelHtml(preflight)}
        <div class="toolbar">
          <div class="group">
            <span class="group-label">Path</span>
            <label>Area
              <select id="area">
                ${areas.map((area) => `<option value="${this._escapeHtml(area.area_hash)}" ${String(area.area_hash) === String(this._areaHash) ? "selected" : ""}>${this._escapeHtml(area.name || area.area_hash)}</option>`).join("")}
              </select>
            </label>
            <button id="reload" type="button" title="Re-fetch the map and live mower state">Reload</button>
            <button id="undo" type="button" ${removeDisabled} title="Remove the last destination">Undo point</button>
            <button id="clear" type="button" ${runActive ? "disabled" : ""} title="Remove all destinations">Reset path</button>
            <span class="waypoint-counter">${this._waypoints.length}/${MAX_WAYPOINTS} points · ${plannedLegCount} leg${plannedLegCount === 1 ? "" : "s"}${plannedLegCount !== segmentCount ? ` (${segmentCount} destination${segmentCount === 1 ? "" : "s"}, auto-split)` : ""}</span>
          </div>
          <div class="group">
            <span class="group-label">Run</span>
            <button id="dry-run" type="button" ${pathSet && !runActive ? "" : "disabled"} title="Validate and plan without sending any mower command">Dry-run</button>
            <label class="confirm"><input id="confirm-blades-off" type="checkbox" ${this._confirmBladesOff ? "checked" : ""}/> blades off</label>
            <label class="confirm"><input id="confirm-clear-area" type="checkbox" ${this._confirmClearArea ? "checked" : ""}/> clear area</label>
            <button id="real-go" class="primary" type="button" ${realGoDisabled ? "disabled" : ""} title="${this._escapeHtml(realGoDisabled ? readiness.headline : "Drive the planned path for real")}">▶ Real Go</button>
            <button id="abort" class="danger" type="button" title="Stop any manual-motion session immediately">■ Abort / Stop</button>
          </div>
          <div class="group">
            <span class="group-label">Night v1 · experimental</span>
            <button id="night-dry-run" type="button" ${pathSet && !runActive ? "" : "disabled"} title="Validate one bounded night segment without sending motion">Night dry-run</button>
            <label class="confirm"><input id="confirm-night-experimental" type="checkbox" ${this._confirmNightExperimental ? "checked" : ""}/> night accuracy unproven</label>
            <button id="night-go" type="button" ${nightGoDisabled ? "disabled" : ""} title="${this._escapeHtml(nightTitle)}">▶ Night Go ≤ ${MAX_NIGHT_SEGMENT_METRES.toFixed(1)} m</button>
            <span class="hint">One forward-only segment · RTK Fix · fixed 3-pulse budget · landing accuracy unproven</span>
          </div>
          <div class="group">
            <span class="group-label">Nudge</span>
            <label title="Straight line only when trustworthy current orientation is available. Stale course-over-ground is refused.">
              <input id="nudge-distance" type="number" min="0.1" max="${MAX_NUDGE_METRES}" step="0.1" value="${this._nudgeMetres().toFixed(1)}" style="width:4.5em"/> m
            </label>
            <button id="nudge" type="button" title="${this._escapeHtml(nudgeTitle)}" ${nudgeDisabled ? "disabled" : ""}>Nudge forward</button>
          </div>
        </div>
        <div class="map-caption">
          <span class="legend"><span class="dot" style="background:#22c55e"></span>Green = mower position; arrow only with trusted live orientation</span>
          <span class="legend"><span class="dot" style="background:#f97316"></span>Click the map to add destinations (max ${MAX_WAYPOINTS}), driven in order</span>
          <span class="legend"><span class="dot" style="background:transparent;border:2px dashed #fbbf24"></span>Auto-inserted split point — a leg over ${SPLIT_LEG_TARGET_METRES.toFixed(2)} m is driven as collinear sub-legs (max ${MAX_REAL_SEGMENTS} per click)</span>
        </div>
        <svg id="path-map"></svg>
        ${coordinateEditor}
        <div class="status">${this._escapeHtml(this._status)}</div>
        ${this._storageWarning ? `<div class="warnings">⚠️ ${this._escapeHtml(this._storageWarning)}</div>` : ""}
        ${this._realRun ? this._runSummaryHtml(this._realRun) : ""}
        <div class="export-bar">
          <span class="group-label">Export</span>
          <button id="download-real-result" type="button" ${this._realRun ? "" : "disabled"} title="Save the complete Real Go response as a file">Download last run JSON</button>
          <button id="download-dry-result" type="button" ${this._dryRun ? "" : "disabled"} title="Save the complete dry-run response as a file">Download dry-run JSON</button>
          <button id="copy-yaml" type="button" ${removeDisabled}>Copy YAML</button>
          <button id="copy-json" type="button" ${removeDisabled}>Copy JSON</button>
          <button id="copy-dry-run-yaml" type="button" ${removeDisabled}>Copy dry-run YAML</button>
        </div>
        <details class="preflight-panel">
          <summary>${this._escapeHtml(preflightText)}</summary>
          <div class="preflight-row"><span class="label">execution profile</span><span class="value">${this._escapeHtml(this._profileLabel())}</span></div>
          <div class="preflight-row"><span class="label">mower orientation</span><span class="value">${this._escapeHtml(this._headingLabel())}</span></div>
          <div class="preflight-row"><span class="label">active_transport</span><span class="value">${this._escapeHtml(runtimePanel.activeTransport)}</span></div>
          <div class="preflight-row"><span class="label">blade-safe status</span><span class="value">${this._escapeHtml(runtimePanel.bladeSafeLabel)}</span></div>
          <div class="preflight-row"><span class="label">mowing readiness</span><span class="value">${this._escapeHtml(runtimePanel.mowingReadinessLabel)}</span></div>
          <div class="preflight-row"><span class="label">charging readiness</span><span class="value">${this._escapeHtml(runtimePanel.chargingReadinessLabel)}</span></div>
          <div class="preflight-row"><span class="label">route-blocking status</span><span class="value">${this._escapeHtml(runtimePanel.routeBlockingLabel)}</span></div>
          <div class="preflight-row"><span class="label">ha_state</span><span class="value">${this._escapeHtml(runtimePanel.haState)}</span></div>
          <div class="preflight-row"><span class="label">work_mode</span><span class="value">${this._escapeHtml(runtimePanel.workMode)}</span></div>
          <div class="preflight-row"><span class="label">charge_state</span><span class="value">${this._escapeHtml(runtimePanel.chargeState)}</span></div>
          <div class="preflight-row"><span class="label">experimental motion</span><span class="value">${this._escapeHtml(runtimePanel.motionEnabled)}</span></div>
          <div class="preflight-row"><span class="label">PyMammotion backend</span><span class="value">${this._escapeHtml(`${runtimePanel.backendVersion} (${runtimePanel.backendVerified})`)}</span></div>
          <div class="preflight-row"><span class="label">motion blockers</span><span class="value">${this._escapeHtml(runtimePanel.motionBlockers)}</span></div>
          <div class="preflight-row"><span class="label">active session</span><span class="value">${this._escapeHtml(`${runtimePanel.activeSession} (${runtimePanel.sessionPhase})`)}</span></div>
          <div class="preflight-row"><span class="label">last confirmed write</span><span class="value">${this._escapeHtml(runtimePanel.lastDispatch)}</span></div>
          <div class="preflight-row"><span class="label">stop confirmed</span><span class="value">${this._escapeHtml(runtimePanel.stopOutcome)}</span></div>
        </details>
        ${(this._validation?.warnings || []).length ? `<div class="warnings">Warnings: ${this._escapeHtml(this._validation.warnings.join(", "))}</div>` : ""}
        ${pathSet ? `<details><summary>Preview service YAML</summary><pre>${this._escapeHtml(this._payloadYaml())}</pre></details>` : ""}
        ${pathSet ? `<details><summary>Dry-run service YAML</summary><pre>${this._escapeHtml(this._dryRunYaml())}</pre></details>` : ""}
        ${this._dryRun ? `<details><summary>Last dry-run result</summary><button id="copy-dry-result" class="copy-result">Copy dry-run JSON</button><pre>${this._escapeHtml(JSON.stringify(this._dryRun, null, 2))}</pre></details>` : ""}
        ${this._realRun ? `<details><summary>Last Real Go result</summary><button id="copy-real-result" class="copy-result">Copy result JSON</button><pre>${this._escapeHtml(JSON.stringify(this._realRun, null, 2))}</pre></details>` : ""}
        ${this._renderHistoryHtml()}
        <div class="card-version">card v${CARD_VERSION}</div>
      </ha-card>
    `;
    this._q("#reload")?.addEventListener("click", async () => {
      await this._loadMap();
      await this._loadRuntimeState();
    });
    this._q("#undo")?.addEventListener("click", () =>
      this._removeLastWaypoint(),
    );
    this._q("#clear")?.addEventListener("click", () => this._clearTarget());
    this._q("#copy-yaml")?.addEventListener("click", () => this._copyYaml());
    this._q("#copy-json")?.addEventListener("click", () => this._copyJson());
    this._q("#copy-dry-run-yaml")?.addEventListener("click", () =>
      this._copyDryRunYaml(),
    );
    this._q("#dry-run")?.addEventListener("click", () => this._runDryRun());
    this._q("#real-go")?.addEventListener("click", () => this._runRealGo());
    this._q("#night-dry-run")?.addEventListener("click", () =>
      this._runNightDryRun(),
    );
    this._q("#night-go")?.addEventListener("click", () => this._runNightGo());
    this._q("#clear-history")?.addEventListener("click", () =>
      this._clearHistory(),
    );
    this.shadowRoot
      ?.querySelectorAll(".override-toggle")
      .forEach((el) =>
        el.addEventListener("change", (event) =>
          this._toggleOverride(event.target.dataset.override),
        ),
      );
    this._q("#download-real-result")?.addEventListener("click", () =>
      this._downloadJson(this._realRun, "real-go", "Real Go result"),
    );
    this._q("#download-dry-result")?.addEventListener("click", () =>
      this._downloadJson(this._dryRun, "dry-run", "dry-run result"),
    );
    this._q("#download-history")?.addEventListener("click", () =>
      this._downloadJson(
        {
          card_version: CARD_VERSION,
          entity: this._config.entity ?? null,
          runs: this._loadHistory(),
        },
        "run-history",
        "run history",
      ),
    );
    this._q("#copy-dry-result")?.addEventListener("click", () =>
      this._copyText(
        `${JSON.stringify(this._dryRun, null, 2)}\n`,
        "Dry-run result JSON",
      ),
    );
    this._q("#copy-real-result")?.addEventListener("click", () =>
      this._copyText(
        `${JSON.stringify(this._realRun, null, 2)}\n`,
        "Real Go result JSON",
      ),
    );
    this._q("#nudge")?.addEventListener("click", () => this._runNudge());
    this._q("#nudge-distance")?.addEventListener("change", (event) => {
      this._nudgeDistance = Number(event.target.value);
      this._render();
    });
    this._q("#abort")?.addEventListener("click", () => this._abortMotion());
    this._q("#confirm-blades-off")?.addEventListener("change", (event) => {
      this._confirmBladesOff = Boolean(event.target.checked);
      this._render();
    });
    this._q("#confirm-clear-area")?.addEventListener("change", (event) => {
      this._confirmClearArea = Boolean(event.target.checked);
      this._render();
    });
    this._q("#confirm-night-experimental")?.addEventListener(
      "change",
      (event) => {
        this._confirmNightExperimental = Boolean(event.target.checked);
        this._render();
      },
    );
    this._q("#area")?.addEventListener("change", (event) => {
      this._areaHash = event.target.value;
      this._validateAndPreview();
    });
    const svgEl = this._q("#path-map");
    svgEl?.addEventListener("click", (event) => this._onMapClick(event));
    svgEl?.addEventListener("pointermove", (event) =>
      this._onPointerMove(event),
    );
    svgEl?.addEventListener("pointerup", () => this._onPointerUp());
    svgEl?.addEventListener("pointercancel", () => this._onPointerUp());
    this._waypoints.forEach((_point, index) => {
      for (const axis of ["x", "y"]) {
        this._q(`#waypoint-${index}-${axis}`)?.addEventListener(
          "change",
          (event) =>
            this._setWaypointCoordinate(index, axis, event.target.value),
        );
      }
    });
    this._renderMap();
  }
}

// The card is served at two URLs (/mammotion/ and /hacsfiles/); if both end up
// registered as dashboard resources the second define() throws. Guard it.
if (
  typeof customElements !== "undefined" &&
  !customElements.get("mammotion-custom-path-card")
) {
  customElements.define("mammotion-custom-path-card", MammotionCustomPathCard);
}

if (typeof window !== "undefined") {
  window.customCards = window.customCards || [];
  window.customCards.push({
    type: "mammotion-custom-path-card",
    name: "Mammotion Click/Go (Guarded)",
    description:
      "Preview or dry-run up to seven destinations; guarded Real Go is limited to four segments.",
  });
}

export {
  CARD_VERSION,
  LUBA_ACCEPTANCE_PROFILE,
  ACCEPTED_PROFILE_ACCEPTED_ON,
  MAX_NIGHT_SEGMENT_METRES,
  MAX_REAL_SEGMENTS,
  MAX_REAL_SEGMENT_METRES,
  MAX_WAYPOINTS,
  NIGHT_GO_PROFILE,
  SPLIT_LEG_TARGET_METRES,
  CORRECTABLE_AIM_FLOOR_DEGREES,
  PROFILE_KEYS,
  MammotionCustomPathCard,
};
