import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

globalThis.HTMLElement = class {
  attachShadow() {
    return {
      appendChild() {},
      querySelector() {
        return null;
      },
    };
  }
};
globalThis.customElements = {
  _items: new Map(),
  define(name, value) {
    this._items.set(name, value);
  },
  get(name) {
    return this._items.get(name);
  },
};
globalThis.window = {};
// The stub used to be incapable of failing, which is why nothing about quota
// was ever tested -- and quota is exactly the path the card handled silently.
// `_quotaBytes` caps total stored size; `_failWrites` refuses every write.
globalThis.localStorage = {
  _items: new Map(),
  _quotaBytes: Infinity,
  _failWrites: false,
  reset() {
    this._items = new Map();
    this._quotaBytes = Infinity;
    this._failWrites = false;
  },
  _usedBytesExcluding(key) {
    let total = 0;
    for (const [name, value] of this._items) {
      if (name !== key) total += name.length + value.length;
    }
    return total;
  },
  getItem(key) {
    return this._items.get(key) ?? null;
  },
  setItem(key, value) {
    const text = String(value);
    if (
      this._failWrites ||
      this._usedBytesExcluding(key) + key.length + text.length >
        this._quotaBytes
    ) {
      // Same shape browsers throw.
      const err = new Error("QuotaExceededError");
      err.name = "QuotaExceededError";
      throw err;
    }
    this._items.set(key, text);
  },
  removeItem(key) {
    this._items.delete(key);
  },
};

const {
  ACCEPTED_PROFILE_ACCEPTED_ON,
  LUBA_ACCEPTANCE_PROFILE,
  MAX_NIGHT_SEGMENT_METRES,
  CARD_VERSION,
  MAX_REAL_SEGMENTS,
  MAX_REAL_SEGMENT_METRES,
  MAX_WAYPOINTS,
  NIGHT_GO_PROFILE,
  PROFILE_KEYS,
  SPLIT_LEG_TARGET_METRES,
  CORRECTABLE_AIM_FLOOR_DEGREES,
  MammotionCustomPathCard,
} =
  await import("../../custom_components/mammotion/www/mammotion-custom-path-card.js");

function card() {
  const element = new MammotionCustomPathCard();
  element._config = {
    entity: "lawn_mower.test",
    prefer_ble: true,
  };
  element._runtimeState = {
    position: { x: 1, y: 1, rtk_status_label: "Fix" },
    safety: { allowed_for_manual_motion: true, blockers: [] },
    experimental_motion: {
      real_motion_allowed: true,
      blockers: [],
    },
  };
  element._validation = { valid: true };
  element._render = () => {};
  return element;
}

test("acceptance profile is frozen for the night-mode change", () => {
  assert.deepEqual(LUBA_ACCEPTANCE_PROFILE, {
    prefer_ble: true,
    turn_mode: "vio",
    max_turn_commands: 4,
    vio_turn_max_commands: 4,
    max_linear_commands: 3,
    max_linear_pulse_ceiling: 22,
    max_no_progress_pulses: 3,
    heading_tolerance_degrees: 18,
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
  // Route B ships the split target as a plain payload key, NOT a profile key.
  // Putting it in this object would un-accept the hardware-accepted profile and
  // owe another Gate 5 -- the exact cost Route B exists to avoid.
  assert.equal(LUBA_ACCEPTANCE_PROFILE.split_leg_target_length_m, undefined);
  assert.equal(PROFILE_KEYS.includes("split_leg_target_length_m"), false);
});

test("Night Go has a separate frozen fixed-budget profile", () => {
  assert.equal(Object.isFrozen(NIGHT_GO_PROFILE), true);
  assert.deepEqual(NIGHT_GO_PROFILE, {
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
    sample_delays: [0, 3],
  });
  assert.equal(LUBA_ACCEPTANCE_PROFILE.turn_mode, "vio");
  assert.equal(LUBA_ACCEPTANCE_PROFILE.heading_tolerance_degrees, 18);
});

test("Night Go emits one backend vector segment and leaves Real Go unchanged", () => {
  const element = card();
  element._waypoints = [{ x: 1.7, y: 1 }];
  element._confirmBladesOff = true;
  element._confirmClearArea = true;
  element._confirmNightExperimental = true;

  const night = element._nightMotionPayload(false);
  const daylight = element._motionPayload(false);

  assert.equal(night.service, "raw_pymammotion_execute_vector_segment");
  assert.equal(night.payload.turn_mode, "night");
  assert.equal(night.payload.night_angular_speed, 500);
  assert.equal(night.payload.toward_mirror_degrees, 90.13);
  assert.equal(night.payload.heading_tolerance_degrees, 8);
  assert.equal(night.payload.max_linear_commands, 3);
  assert.equal(night.payload.motion_refresh_interval_ms, 200);
  assert.deepEqual(night.payload.sample_delays, [0, 3]);
  assert.equal("max_linear_pulse_ceiling" in night.payload, false);
  assert.equal(night.payload.confirm_blades_off, true);
  assert.equal(night.payload.confirm_clear_area, true);
  assert.equal(daylight.payload.turn_mode, "vio");
  assert.equal(daylight.payload.max_linear_pulse_ceiling, 22);
});

test("Night Go refuses multiple, long, and non-Fix paths in the card", () => {
  const element = card();
  element._waypoints = [
    { x: 1.6, y: 1 },
    { x: 1.8, y: 1 },
  ];
  let preflight = element._nightPreflight();
  assert.equal(preflight.safe, false);
  assert.ok(preflight.blockers.includes("night_requires_one_segment"));
  assert.equal(element._nightMotionPayload(false), null);

  element._waypoints = [{ x: 1 + MAX_NIGHT_SEGMENT_METRES + 0.001, y: 1 }];
  preflight = element._nightPreflight();
  assert.ok(preflight.blockers.includes("night_segment_too_long"));

  element._waypoints = [{ x: 1.7, y: 1 }];
  element._runtimeState.position.rtk_status_label = "Float";
  preflight = element._nightPreflight();
  assert.ok(preflight.blockers.includes("night_requires_precise_rtk"));
});

test("Night Go real execution refreshes checks and resets confirmations", async () => {
  const element = card();
  element._waypoints = [{ x: 1.7, y: 1 }];
  element._confirmBladesOff = true;
  element._confirmClearArea = true;
  element._confirmNightExperimental = true;
  const order = [];
  element._loadRuntimeState = async () => order.push("runtime");
  element._validateAndPreview = async () => order.push("preview");
  element._callService = async (_service, payload) => {
    order.push(payload.turn_mode);
    return { stop_reason: "no_target_progress", completion_status: {} };
  };
  element._startRunTicker = () => {};
  element._stopRunTicker = () => {};
  element._saveRunToHistory = () => {};

  await element._runNightGo();

  assert.deepEqual(order.slice(0, 3), ["runtime", "preview", "night"]);
  assert.equal(element._realRun.stop_reason, "no_target_progress");
  assert.equal(element._confirmBladesOff, false);
  assert.equal(element._confirmClearArea, false);
  assert.equal(element._confirmNightExperimental, false);
  assert.match(element._status, /Night Go complete/);
});

test("Night Go needs its own experimental acknowledgement", async () => {
  const element = card();
  element._waypoints = [{ x: 1.7, y: 1 }];
  element._confirmBladesOff = true;
  element._confirmClearArea = true;
  let called = false;
  element._loadRuntimeState = async () => {};
  element._validateAndPreview = async () => {};
  element._callService = async () => {
    called = true;
  };

  await element._runNightGo();

  assert.equal(called, false);
  assert.match(element._status, /all three confirmations/);
});

test("Night dry-run works with the motion gate off and sends no confirmations", async () => {
  const element = card();
  element._waypoints = [{ x: 1.7, y: 1 }];
  element._runtimeState.experimental_motion = {
    real_motion_allowed: false,
    blockers: ["experimental_motion_disabled"],
  };
  element._runtimeState.position.rtk_status_label = "Float";
  let sent = null;
  element._loadRuntimeState = async () => {};
  element._validateAndPreview = async () => {};
  element._callService = async (service, payload) => {
    sent = { service, payload };
    return { stop_reason: "dry_run" };
  };

  await element._runNightDryRun();

  assert.equal(sent.service, "raw_pymammotion_execute_vector_segment");
  assert.equal(sent.payload.turn_mode, "night");
  assert.equal(sent.payload.dry_run, true);
  assert.equal(sent.payload.confirm_blades_off, false);
  assert.equal(sent.payload.confirm_clear_area, false);
  assert.match(element._status, /No mower command was sent/);
});

test("map clicks accept seven destinations and refuse an eighth", () => {
  const element = card();
  element._mapT = {};
  element._svgPointFromEvent = () => ({ x: 2, y: 3 });
  element._validateAndPreview = () => {};

  for (let index = 0; index < MAX_WAYPOINTS + 1; index += 1) {
    element._onMapClick({ target: {} });
  }

  assert.equal(element._waypoints.length, 7);
  assert.match(element._status, /Maximum 7 waypoints/);
});

test("precise coordinate edits invalidate stale runs and re-preview", () => {
  const element = card();
  element._waypoints = [{ x: 2, y: 3 }];
  element._dryRun = { stop_reason: "dry_run" };
  element._realRun = { stop_reason: "target_reached" };
  let previews = 0;
  element._validateAndPreview = () => {
    previews += 1;
  };

  element._setWaypointCoordinate(0, "x", "2.375");

  assert.deepEqual(element._waypoints, [{ x: 2.375, y: 3 }]);
  assert.equal(element._dryRun, null);
  assert.equal(element._realRun, null);
  assert.equal(previews, 1);
});

test("precise coordinate edits reject invalid values and active sessions", () => {
  const element = card();
  element._waypoints = [{ x: 2, y: 3 }];
  element._validateAndPreview = () => assert.fail("must not preview");

  element._setWaypointCoordinate(0, "y", "not-a-number");
  assert.deepEqual(element._waypoints, [{ x: 2, y: 3 }]);
  assert.match(element._status, /finite numbers/);

  element._setWaypointCoordinate(0, "y", "");
  assert.deepEqual(element._waypoints, [{ x: 2, y: 3 }]);

  element._runtimeState.experimental_motion.active_session = {
    session_id: "active",
  };
  element._setWaypointCoordinate(0, "y", "4.25");
  assert.deepEqual(element._waypoints, [{ x: 2, y: 3 }]);
});

test("seven-point dry-run is retained but real payload is capped at the limit", () => {
  const element = card();
  element._waypoints = Array.from({ length: 7 }, (_, index) => ({
    x: index + 2,
    y: index + 2,
  }));

  const dry = element._motionPayload(true);
  const real = element._motionPayload(false);

  // Was MAX_WAYPOINTS (7), which the backend schema's vol.Range(min=0, max=4)
  // rejects -- a 5+-waypoint DRY RUN failed validation before reaching the
  // handler, and this assertion pinned the broken value. Clamping is
  // behaviour-neutral: max_real_segments is only read behind `if not dry_run`.
  assert.equal(dry.payload.max_real_segments, MAX_REAL_SEGMENTS);
  assert.ok(dry.payload.max_real_segments <= MAX_REAL_SEGMENTS);
  assert.equal(real.payload.max_real_segments, MAX_REAL_SEGMENTS);
  assert.equal(real.payload.motion_refresh_interval_ms, 200);
  assert.equal(real.payload.final_approach_metres_per_pulse, 1.06);
  assert.equal(real.payload.turn_degrees_per_second, 37);
});

// The card must emit the exact bounded profile that passed supervised LUBA
// Gate 4 re-pass on 2026-08-05 -- see docs/gate4-repass-20260805.md. These are
// the values the hardware actually executed; drifting from them silently would
// make the card's Real Go an untested profile again.
test("default Real Go payload is the Gate 4 accepted profile", () => {
  const element = card();
  element._waypoints = [{ x: 2, y: 2 }];

  const { payload } = element._motionPayload(false);

  assert.equal(payload.max_turn_commands, 4);
  assert.equal(payload.vio_turn_max_commands, 4);
  assert.equal(payload.max_linear_commands, 3);
  assert.equal(payload.linear_pulse_duration_ms, 1300);
  assert.equal(payload.max_turn_translation_distance, 0.3);
  assert.equal(payload.waypoint_tolerance, 0.15);
  assert.equal(payload.min_progress_distance, 0.0025);
  assert.equal(payload.calibrated_forward_heading_offset_degrees, 102.4);
  assert.equal(payload.motion_refresh_interval_ms, 200);
  assert.equal(payload.ble_auto_recover, false);
  assert.equal(payload.turn_mode, "vio");
});

test("the accepted profile enables loop-to-tolerance and sends the ceiling", () => {
  const element = card();
  element._waypoints = [{ x: 2, y: 2 }];

  const { payload } = element._motionPayload(false);

  // Adopted 2026-08-12. This is the key that makes reach real for a card user:
  // without it a segment stops after three pulses at roughly 1 m.
  assert.equal(LUBA_ACCEPTANCE_PROFILE.max_linear_pulse_ceiling, 22);
  assert.equal(payload.max_linear_pulse_ceiling, 22);
  // max_linear_commands stays at the Gate 4/5 value so that turning the ceiling
  // off anywhere falls back to exactly the accepted fixed-budget behaviour.
  assert.equal(payload.max_linear_commands, 3);

  element._config.max_linear_pulse_ceiling = 30;
  const opted = element._motionPayload(false).payload;
  assert.equal(opted.max_linear_pulse_ceiling, 30);
  assert.deepEqual(element._profileOverrides(), ["max_linear_pulse_ceiling"]);
});

test("an explicitly null ceiling falls back to the accepted value, not omission", () => {
  // `_profileValue` treats null as "unset" and returns the profile value, so a
  // dashboard cannot silently disable loop-to-tolerance by nulling the key --
  // it has to be a deliberate different number.
  const element = card();
  element._waypoints = [{ x: 2, y: 2 }];
  element._config.max_linear_pulse_ceiling = null;

  const { payload } = element._motionPayload(false);
  assert.equal(payload.max_linear_pulse_ceiling, 22);
});

test("the card's acceptance claim matches docs/accepted-profile.json", () => {
  // 🚨 THIS CLAIM HAS ALREADY GONE STALE ONCE. The un-acceptance was hardcoded
  // into _profileLabel() on 2026-08-17; Gate 5 passed on 2026-08-18 and the
  // card kept telling the operator the profile "owes a Gate 5". The card
  // cannot read the snapshot at runtime, so this test is the only thing tying
  // the two together.
  //
  // If this fails, do NOT edit the constant to match blindly -- work out which
  // side is right. The snapshot is regenerated only by
  // scripts/check_accepted_profile.py --write-accepted after a real Gate 5.
  const snapshot = JSON.parse(
    readFileSync(new URL("../../docs/accepted-profile.json", import.meta.url)),
  );
  assert.equal(
    ACCEPTED_PROFILE_ACCEPTED_ON,
    snapshot.accepted_on,
    "card acceptance date disagrees with docs/accepted-profile.json",
  );
  assert.deepEqual(
    LUBA_ACCEPTANCE_PROFILE,
    snapshot.profile,
    "shipped profile differs from the hardware-accepted snapshot",
  );
});

test("profile label states acceptance by default and names any override", () => {
  const element = card();

  assert.match(element._profileLabel(), /LUBA acceptance profile/);
  assert.match(
    element._profileLabel(),
    new RegExp(ACCEPTED_PROFILE_ACCEPTED_ON),
  );
  assert.doesNotMatch(element._profileLabel(), /owes a Gate 5/);
  assert.doesNotMatch(element._profileLabel(), /NOT hardware-accepted/);

  element._config.waypoint_tolerance = 0.25;
  element._config.ble_auto_recover = true;

  assert.deepEqual(element._profileOverrides(), [
    "waypoint_tolerance",
    "ble_auto_recover",
  ]);
  assert.match(element._profileLabel(), /customised \(not hardware-accepted\)/);
  assert.match(element._profileLabel(), /waypoint_tolerance/);
});

test("falsy profile values survive resolution instead of falling back", () => {
  const element = card();
  element._waypoints = [{ x: 2, y: 2 }];
  element._config.motion_refresh_interval_ms = 0;
  element._config.min_progress_distance = 0;

  const { payload } = element._motionPayload(false);

  assert.equal(payload.motion_refresh_interval_ms, 0);
  assert.equal(payload.min_progress_distance, 0);
  assert.equal(payload.ble_auto_recover, false);
});

// README documents the accepted profile a third time, in YAML an operator can
// paste. Nothing else stops that copy drifting from the values the hardware ran,
// and a README that quietly disagrees with the card is how an operator ends up
// pasting a "documented default" that is not the accepted profile.
test("README's written-out defaults are the accepted profile", () => {
  const readme = readFileSync(
    new URL("../../README.md", import.meta.url),
    "utf8",
  );
  const block = readme.match(
    /The built-in defaults, written out[\s\S]*?```yaml\n([\s\S]*?)```/,
  );
  assert.ok(block, "README no longer contains the written-out defaults block");

  const documented = new Map();
  let listKey = null;
  for (const line of block[1].split("\n")) {
    const item = line.match(/^ {2}- (.+)$/);
    if (item && listKey) {
      documented.get(listKey).push(item[1].trim());
      continue;
    }
    const pair = line.match(/^([a-z_]+):\s*(.*)$/);
    if (!pair) continue;
    if (pair[2] === "") {
      listKey = pair[1];
      documented.set(listKey, []);
    } else {
      listKey = null;
      documented.set(pair[1], pair[2].trim());
    }
  }

  for (const key of PROFILE_KEYS) {
    // `max_linear_pulse_ceiling` used to be the one key README had to OMIT,
    // because setting it left the accepted profile. It was adopted 2026-08-12,
    // so it is now documented like every other key.
    assert.ok(documented.has(key), `README omits ${key}`);
    assert.equal(
      documented.get(key)?.toString(),
      LUBA_ACCEPTANCE_PROFILE[key].toString(),
      `README disagrees with the accepted profile for ${key}`,
    );
  }
});

// Course-over-ground plus the calibrated offset is useful for backend aiming,
// but it is not the mower's current orientation after an in-place turn.
test("last-travel projection is not exposed as current orientation", () => {
  const element = card();
  element._runtimeState.position = { x: 1, y: 1, toward: 169.7755 };

  // 169.7755 + 102.4 = 272.1755
  assert.ok(Math.abs(element._headingDegrees() - 272.1755) < 1e-6);
  assert.equal(element._currentOrientationDegrees(), null);
  assert.match(element._headingLabel(), /272\.2°/);
  assert.match(element._headingLabel(), /last-travel projection/);
  assert.match(element._headingLabel(), /not mower orientation/);

  element._runtimeState.current_orientation = {
    trustworthy: true,
    map_heading_degrees: -45,
    source: "test",
  };
  assert.equal(element._currentOrientationDegrees(), 315);
  assert.match(element._headingLabel(), /315\.0° current orientation/);

  // Wraps past 360 rather than running off the end.
  element._runtimeState.position = { x: 1, y: 1, toward: 300 };
  assert.ok(Math.abs(element._headingDegrees() - 42.4) < 1e-9);

  // No usable position must not fabricate a bearing.
  element._runtimeState.position = { x: 1, y: 1 };
  assert.equal(element._headingDegrees(), null);
  element._runtimeState.current_orientation = null;
  assert.match(element._headingLabel(), /orientation unavailable/);
});

test("trusted orientation arrow is computed in map space, not screen space", () => {
  const element = card();
  element._runtimeState.current_orientation = {
    trustworthy: true,
    map_heading_degrees: 0,
  };

  // A transform with a FLIPPED y axis, as the real one has (toSY = H - ...).
  const mt = { toSX: (x) => 100 + x * 10, toSY: (y) => 500 - y * 10 };
  const rad = (element._currentOrientationDegrees() * Math.PI) / 180;
  const dx = mt.toSX(Math.cos(rad)) - mt.toSX(0);
  const dy = mt.toSY(Math.sin(rad)) - mt.toSY(0);
  assert.ok(dx > 0, "map +x must render as screen +x");
  assert.ok(Math.abs(dy) < 1e-9, "pure +x heading must not acquire screen y");

  // Map +y must render as screen -y under the flip. Rotating in screen space
  // instead would point the arrow at the mirror image of the real bearing.
  element._runtimeState.current_orientation.map_heading_degrees = 90;
  const rad2 = (element._currentOrientationDegrees() * Math.PI) / 180;
  assert.ok(Math.abs(element._currentOrientationDegrees() - 90) < 1e-9);
  assert.ok(mt.toSY(Math.sin(rad2)) - mt.toSY(0) < 0, "map +y is screen -y");
});

// Nudge is the night / no-VIO escape hatch. Its entire safety argument is that
// the target sits ON the heading ray, so the turn phase has nothing to do and
// no blind pivot can happen. These pin that.
test("nudge targets the heading ray so no turn is ever required", () => {
  const element = card();
  element._runtimeState.position = { x: 10, y: 5, toward: 72 };
  element._runtimeState.current_orientation = {
    trustworthy: true,
    map_heading_degrees: 0,
  };
  element._nudgeDistance = 1.5;

  const { payload } = element._nudgePayload(false);
  const [from, to] = payload.points;

  assert.deepEqual(from, { x: 10, y: 5 });
  // Bearing 0 => straight along +x, y unchanged.
  assert.ok(Math.abs(to.x - 11.5) < 1e-6, `got ${to.x}`);
  assert.ok(Math.abs(to.y - 5) < 1e-6, `got ${to.y}`);

  // legacy only to clear the up-front vio_active gate; it must never turn.
  assert.equal(payload.turn_mode, "legacy");
  // 1.5 m at ~1.06 m per command => 2 commands.
  assert.equal(payload.max_linear_commands, 2);
  assert.equal("max_linear_pulse_ceiling" in payload, false);
});

test("nudge is distance-capped and refuses without a heading", () => {
  const element = card();
  element._runtimeState.position = { x: 0, y: 0, toward: 0 };
  element._runtimeState.current_orientation = {
    trustworthy: true,
    map_heading_degrees: 102.4,
  };

  element._nudgeDistance = 99;
  assert.equal(element._nudgeMetres(), 2.0, "must clamp to MAX_NUDGE_METRES");
  element._nudgeDistance = -5;
  assert.equal(element._nudgeMetres(), 0);
  assert.equal(
    element._nudgePayload(false),
    null,
    "zero distance sends nothing",
  );

  // Stale course-over-ground without trusted current orientation must refuse.
  element._nudgeDistance = 1;
  element._runtimeState.position = { x: 0, y: 0, toward: 0 };
  element._runtimeState.current_orientation = null;
  assert.equal(element._currentOrientationDegrees(), null);
  assert.equal(element._nudgePayload(false), null);
});

test("nudge requires clear-area but not the blades checkbox", () => {
  const element = card();
  element._runtimeState.position = { x: 0, y: 0, toward: 0 };
  element._runtimeState.current_orientation = {
    trustworthy: true,
    map_heading_degrees: 102.4,
  };
  element._nudgeDistance = 0.5;

  // Blades-off stays asserted to the backend (telemetry gates it separately),
  // while the operator's clear-area confirmation is still carried through.
  element._confirmClearArea = false;
  assert.equal(element._nudgePayload(false).payload.confirm_blades_off, true);
  assert.equal(element._nudgePayload(false).payload.confirm_clear_area, false);

  element._confirmClearArea = true;
  assert.equal(element._nudgePayload(false).payload.confirm_clear_area, true);

  // A dry run must never carry confirmations.
  const dry = element._nudgePayload(true).payload;
  assert.equal(dry.confirm_blades_off, false);
  assert.equal(dry.confirm_clear_area, false);
});

// Regression: the first Nudge build stayed enabled while the motion gate was
// off, so clicking it looked like a broken feature when the backend was
// correctly refusing. Nudge must be gated on backend readiness like Real Go.
test("nudge is blocked when the backend will refuse it", () => {
  const element = card();
  element._runtimeState.position = { x: 0, y: 0, toward: 0 };
  element._runtimeState.current_orientation = {
    trustworthy: true,
    map_heading_degrees: 102.4,
  };
  element._confirmClearArea = true;
  assert.deepEqual(element._motionBackendBlockers(), []);

  element._runtimeState.experimental_motion = {
    real_motion_allowed: false,
    blockers: ["experimental_motion_disabled"],
  };
  assert.deepEqual(element._motionBackendBlockers(), [
    "experimental_motion_disabled",
  ]);

  // Runtime safety blockers surface too, not just the motion option.
  element._runtimeState.experimental_motion = { real_motion_allowed: true };
  element._runtimeState.safety = {
    allowed_for_manual_motion: false,
    blockers: ["blade_unsafe"],
  };
  assert.deepEqual(element._motionBackendBlockers(), ["blade_unsafe"]);

  // No current orientation is its own named blocker, not a silent no-op.
  element._runtimeState.safety = { allowed_for_manual_motion: true };
  element._runtimeState.current_orientation = null;
  assert.ok(
    element
      ._motionBackendBlockers()
      .includes("current_orientation_unavailable"),
  );
});

test("backend blockers and the real-segment limit lock Real Go", () => {
  const element = card();
  // One past the limit, derived from the constant so this test follows a future
  // change instead of pinning a stale literal (it pinned 2 until beta31).
  element._waypoints = Array.from(
    { length: MAX_REAL_SEGMENTS + 1 },
    (_unused, index) => ({ x: index + 2, y: index + 2 }),
  );
  element._runtimeState.experimental_motion = {
    real_motion_allowed: false,
    blockers: ["pymammotion_backend_unverified"],
  };

  const preflight = element._preflight();

  assert.equal(preflight.safe, false);
  assert.ok(
    preflight.blockers.includes(`real_segment_limit_${MAX_REAL_SEGMENTS}`),
  );
  assert.ok(preflight.blockers.includes("pymammotion_backend_unverified"));
});

test("a path exactly at the real-segment limit is not blocked for it", () => {
  const element = card();
  element._waypoints = Array.from(
    { length: MAX_REAL_SEGMENTS },
    (_unused, index) => ({ x: index + 2, y: index + 2 }),
  );
  element._runtimeState.experimental_motion = {
    real_motion_allowed: true,
    blockers: [],
  };

  const preflight = element._preflight();

  assert.ok(
    !preflight.blockers.some((blocker) =>
      blocker.startsWith("real_segment_limit_"),
    ),
  );
});

test("active backend session prevents editing and duplicate runs", async () => {
  const element = card();
  element._waypoints = [{ x: 2, y: 2 }];
  element._runtimeState.experimental_motion.active_session = {
    session_id: "active",
    phase: "running",
  };
  element._clearTarget();
  await element._runRealGo();

  assert.equal(element._waypoints.length, 1);
  assert.match(element._status, /already active/);
});

test("Abort uses stop_manual_motion without safety confirmations", async () => {
  const element = card();
  const calls = [];
  element._callService = async (service, payload) => {
    calls.push([service, payload]);
    if (service === "stop_manual_motion") {
      return { stop_confirmed: true, owner_exited: true };
    }
    return element._runtimeState;
  };
  element._loadRuntimeState = async () => {};

  await element._abortMotion();

  assert.deepEqual(calls, [["stop_manual_motion", {}]]);
  assert.match(element._status, /confirmed/);
  assert.equal(element._confirmBladesOff, false);
  assert.equal(element._confirmClearArea, false);
});

test("fresh preflight is fetched and confirmations reset after failure", async () => {
  const element = card();
  element._waypoints = [{ x: 2, y: 2 }];
  element._confirmBladesOff = true;
  element._confirmClearArea = true;
  const order = [];
  element._loadRuntimeState = async () => {
    order.push("runtime");
  };
  element._validateAndPreview = async () => {
    order.push("preview");
  };
  element._callService = async () => {
    order.push("execute");
    throw new Error("write failed");
  };
  element._startRunTicker = () => {};
  element._stopRunTicker = () => {};

  await element._runRealGo();

  assert.deepEqual(order.slice(0, 3), ["runtime", "preview", "execute"]);
  assert.equal(element._confirmBladesOff, false);
  assert.equal(element._confirmClearArea, false);
  assert.match(element._status, /failed/);
});

test("successful Real Go reloads runtime once before and once after execution", async () => {
  const element = card();
  element._waypoints = [{ x: 2, y: 2 }];
  element._confirmBladesOff = true;
  element._confirmClearArea = true;
  let runtimeLoads = 0;
  element._loadRuntimeState = async () => {
    runtimeLoads += 1;
  };
  element._validateAndPreview = async () => {};
  element._callService = async () => ({
    stop_reason: "target_reached",
    completion_status: { complete: true },
  });
  element._startRunTicker = () => {};
  element._stopRunTicker = () => {};
  element._persistLastRun = () => {};
  element._saveRunToHistory = () => {};

  await element._runRealGo();

  assert.equal(runtimeLoads, 2);
  assert.match(element._status, /Real Go complete/);
});

// ---------------------------------------------------------------------------
// Run summary, readiness banner and result export (beta48)
// ---------------------------------------------------------------------------

function segment(index, landing, { passed = true, tolerance = 0.15 } = {}) {
  return {
    index,
    passed,
    result: {
      stop_reason: passed ? "target_reached" : "max_linear_commands_reached",
      distance: 0.8,
      waypoint_tolerance: tolerance,
      linear_commands_sent: 2,
      turn_commands_sent: 1,
      completion_status: {
        // The real payload lists the START point first and the waypoint last;
        // reading [0] would report the distance already travelled, not the miss.
        waypoint_distances: [
          { index: 0, distance: 0.867 },
          { index: 1, distance: landing },
        ],
      },
    },
  };
}

test("segment landings are read from the LAST waypoint distance", () => {
  const element = card();
  const rows = element._segmentLandingRows({
    segments: [segment(1, 0.0674), segment(2, 0.1032)],
  });

  assert.equal(rows.length, 2);
  assert.equal(rows[0].landing, 0.0674);
  assert.equal(rows[1].landing, 0.1032);
  assert.equal(rows[0].tolerance, 0.15);
  assert.equal(rows[0].inside, true);
  assert.equal(rows[0].planned, 0.8);
});

test("a landing outside tolerance is flagged even when the segment passed", () => {
  const element = card();
  const rows = element._segmentLandingRows({
    segments: [segment(1, 0.1797)],
  });

  assert.equal(rows[0].inside, false);
  assert.match(
    element._runSummaryHtml({ segments: [segment(1, 0.1797)] }),
    /OUTSIDE/,
  );
});

test("a single-segment result with no segments wrapper still summarises", () => {
  const element = card();
  const rows = element._segmentLandingRows(segment(1, 0.09).result);

  assert.equal(rows.length, 1);
  assert.equal(rows[0].landing, 0.09);
});

test("run summary reports the mean landing across measured segments", () => {
  const element = card();
  const html = element._runSummaryHtml({
    segments: [
      segment(1, 0.0674),
      segment(2, 0.1032),
      segment(3, 0.0807),
      segment(4, 0.0607),
    ],
  });

  // The Gate 5 re-pass mean, to four places.
  assert.match(html, /mean landing 0\.0780 m/);
  assert.match(html, /worst 0\.1032 m/);
  assert.match(html, /4 of 4 segments measured/);
});

test("run summary is empty rather than broken when there is nothing to show", () => {
  const element = card();
  assert.equal(element._runSummaryHtml(null), "");
  assert.equal(element._runSummaryHtml({ segments: [] }), "");
  assert.deepEqual(
    element._segmentLandingRows({ stop_reason: "safety_gates_failed" }),
    [],
  );
});

test("readiness names the blocker code AND explains it", () => {
  const element = card();
  element._waypoints = [{ x: 2, y: 2 }];
  element._runtimeState.experimental_motion = {
    real_motion_allowed: false,
    blockers: ["experimental_motion_disabled"],
  };

  const readiness = element._readiness();

  assert.equal(readiness.level, "blocked");
  assert.match(readiness.headline, /experimental_motion_disabled/);
  assert.match(
    readiness.details.join(" "),
    /experimental BLE-only manual motion/,
  );
});

test("every blocker is explained, not just the first", () => {
  const element = card();
  // No waypoints AND the motion gate closed. Explaining only path_unset would
  // send the operator off to click the map while the real problem is the gate.
  element._runtimeState.experimental_motion = {
    real_motion_allowed: false,
    blockers: ["experimental_motion_disabled"],
  };

  const readiness = element._readiness();

  assert.match(readiness.headline, /path_unset/);
  assert.equal(readiness.details.length, 2, "one line per blocker");
  assert.match(readiness.details.join(" "), /Click at least one destination/);
  assert.match(
    readiness.details.join(" "),
    /experimental BLE-only manual motion/,
  );
});

test("readiness distinguishes missing confirmations from a real blocker", () => {
  const element = card();
  element._waypoints = [{ x: 2, y: 2 }];

  const arming = element._readiness();
  assert.equal(arming.level, "arming");
  assert.match(arming.headline, /blades off and clear area/);

  element._confirmBladesOff = true;
  element._confirmClearArea = true;
  const ready = element._readiness();
  assert.equal(ready.level, "ready");
  assert.match(ready.headline, /1 segment\./);
});

test("readiness reports a live run instead of offering to start another", () => {
  const element = card();
  element._waypoints = [{ x: 2, y: 2 }];
  element._confirmBladesOff = true;
  element._confirmClearArea = true;
  element._runtimeState.experimental_motion.active_session = "session-1";

  assert.equal(element._readiness().level, "busy");
});

test("download filenames are filesystem-safe and name the entity", () => {
  const element = card();
  const filename = element._downloadFilename("real-go");

  assert.match(filename, /^mammotion-real-go-lawn-mower-test-[\dTZ-]+\.json$/);
  // Colons are illegal in Windows filenames, and a dot in the stem would make
  // the timestamp's milliseconds look like a second extension.
  assert.ok(!filename.includes(":"));
  assert.equal(filename.split(".").length, 2);
});

test("downloading with no result reports it instead of writing an empty file", () => {
  const element = card();
  let copied = null;
  element._copyText = (text) => {
    copied = text;
  };

  element._downloadJson(null, "real-go", "Real Go result");

  assert.match(element._status, /No Real Go result to download/);
  assert.equal(copied, null);
});

test("history entries carry the landing distance, not just pass/fail", () => {
  const element = card();
  const rows = element._segmentLandingRows({ segments: [segment(1, 0.0674)] });
  const entry = {
    at: "2026-08-13T02:00:00.000Z",
    elapsed_seconds: 95,
    stop_reason: "path_complete",
    segments: rows.map((row) => ({
      index: row.index,
      passed: row.passed,
      stop_reason: row.stopReason,
      landing: row.landing,
      tolerance: row.tolerance,
    })),
  };
  element._loadHistory = () => [entry];

  const html = element._renderHistoryHtml();

  assert.match(html, /0\.067m/);
  assert.match(html, /mean 0\.0674 m/);
  assert.match(html, /download-history/);
});

test("blocker codes are deduplicated across the two backend lists", () => {
  // The backend reports the same condition on BOTH experimental_motion.blockers
  // and safety.blockers. Concatenating them printed position_not_valid_for_motion
  // and rtk_not_precise twice each in the live banner (observed on beta48).
  const element = card();
  element._waypoints = [{ x: 2, y: 2 }];
  element._runtimeState.experimental_motion = {
    real_motion_allowed: false,
    blockers: ["position_not_valid_for_motion", "rtk_not_precise"],
  };
  element._runtimeState.safety = {
    allowed_for_manual_motion: false,
    blockers: ["position_not_valid_for_motion", "rtk_not_precise"],
  };

  const { blockers } = element._preflight();

  assert.deepEqual(blockers, [
    "position_not_valid_for_motion",
    "rtk_not_precise",
  ]);
  assert.equal(new Set(blockers).size, blockers.length);
});

test("nudge blockers are deduplicated too", () => {
  const element = card();
  element._runtimeState.experimental_motion = {
    real_motion_allowed: false,
    blockers: ["ble_link_not_live"],
  };
  element._runtimeState.safety = {
    allowed_for_manual_motion: false,
    blockers: ["ble_link_not_live"],
  };

  const blockers = element._motionBackendBlockers();

  assert.equal(
    blockers.filter((code) => code === "ble_link_not_live").length,
    1,
  );
});

test("a restored run is labelled with its age, never shown as current", () => {
  const element = card();
  element._realRun = { segments: [segment(1, 0.0674)] };

  // No stored timestamp: older persisted runs must still be marked as stale
  // rather than silently reading as "this just happened".
  assert.match(
    element._runSummaryHtml(element._realRun),
    /from a previous session/,
  );

  element._realRunAt = new Date(Date.now() - 90 * 60 * 1000).toISOString();
  assert.equal(element._runAgeLabel(), "1 h ago");

  element._realRunAt = new Date(Date.now() - 5 * 60 * 1000).toISOString();
  assert.equal(element._runAgeLabel(), "5 min ago");

  element._realRunAt = new Date().toISOString();
  assert.equal(element._runAgeLabel(), "just now");
  assert.match(element._runSummaryHtml(element._realRun), /just now/);
});

test("persisting a run stamps the time it completed", () => {
  const element = card();
  element._persistLastRun({ stop_reason: "path_complete" });

  assert.ok(element._realRunAt, "the run must carry a completion timestamp");
  assert.equal(element._runAgeLabel(), "just now");
});

test("the blocker codes the backend actually emits all have help text", () => {
  // Observed live on the host, 2026-08-13 beta48 (docked, just restarted).
  // A code with no entry silently drops out of the banner's explanation list,
  // which is exactly the "disabled button, no reason" failure this replaced.
  const observed = [
    "path_unset",
    "experimental_motion_disabled",
    "position_not_valid_for_motion",
    "rtk_not_precise",
    "path_validation_failed",
    "ble_client_not_connected",
  ];
  const element = card();
  element._runtimeState.experimental_motion = {
    real_motion_allowed: false,
    blockers: observed.filter((code) => code !== "path_unset"),
  };

  const { details } = element._readiness();

  // path_unset is added by the card itself, so all six should be explained.
  assert.equal(details.length, observed.length);
  assert.match(details.join(" "), /RTK Fix and a zone inside a mapped area/);
  assert.match(details.join(" "), /dozes after ~10 min idle/);
});

test("an empty card does not claim the path FAILED validation", () => {
  // Observed on the live card 2026-08-18: with 0/7 points the banner read
  // "path_unset, experimental_motion_disabled, path_validation_failed" and told
  // the operator to hunt for "a point outside the selected area" among zero
  // points. `_validation` starts null and stays null until a preview runs, so
  // an unconditional `!this._validation?.valid` fired on an empty card.
  //
  // Not validated yet is not the same as failed validation.
  const element = card();
  element._validation = null;
  element._waypoints = [];

  const { blockers } = element._preflight();

  assert.ok(blockers.includes("path_unset"));
  assert.equal(blockers.includes("path_validation_failed"), false);
});

test("a path that really did fail validation still reports it", () => {
  // The paired assertion: suppressing the false positive must not suppress the
  // true one, or the guard would hide a genuinely out-of-area waypoint.
  const element = card();
  element._validation = { valid: false };
  element._waypoints = [{ x: 1.5, y: 1 }];

  const { blockers } = element._preflight();

  assert.ok(blockers.includes("path_validation_failed"));
  assert.equal(blockers.includes("path_unset"), false);
});

test("Night dry-run does not inherit the empty-path validation false positive", () => {
  const element = card();
  element._validation = null;
  element._waypoints = [];

  const { blockers } = element._nightPreflight({ dryRun: true });

  assert.ok(blockers.includes("path_unset"));
  assert.equal(blockers.includes("path_validation_failed"), false);
});

// --- Route B: a distant click auto-splits into collinear sub-legs ------------
//
// What these do NOT prove: that a 3.81 m leg lands accurately, or that
// 4 x 3.85 = 15.40 m is drivable. The longest straight leg ever executed is
// 4.0 m, n = 1. These pin the card's arithmetic, its routing, and the fact that
// nothing about the accepted profile moved.

test("a 50 ft click splits into four collinear sub-legs", () => {
  const element = card();
  // Mower at (1, 1); a single click 15.24 m away on +x.
  element._waypoints = [{ x: 16.24, y: 1 }];

  const split = element._plannedSplit();

  assert.equal(split.applied, true);
  assert.equal(split.requestedLegCount, 1);
  assert.equal(split.subLegCount, 4);
  assert.equal(split.points.length, 5);
  for (let i = 1; i < split.points.length; i += 1) {
    const dx = split.points[i].x - split.points[i - 1].x;
    const dy = split.points[i].y - split.points[i - 1].y;
    assert.ok(Math.abs(Math.hypot(dx, dy) - 15.24 / 4) < 1e-9);
    // Collinear: every sub-leg is on the original bearing. The whole route
    // rests on this -- a non-collinear junction is not a free turn.
    assert.ok(Math.abs(dy) < 1e-9);
  }
});

test("_longestSegmentMetres measures the SPLIT points, not the destinations", () => {
  // The single most important card edit. Measuring destination-to-destination
  // would trip `segment_too_long` on every long click and the split would never
  // get a chance to run.
  const element = card();
  element._waypoints = [{ x: 16.24, y: 1 }];

  assert.ok(element._longestSegmentMetres() < MAX_REAL_SEGMENT_METRES);
  assert.ok(Math.abs(element._longestSegmentMetres() - 15.24 / 4) < 1e-9);
  assert.equal(
    element._preflight().blockers.includes("segment_too_long"),
    false,
  );
});

test("a short click is untouched and still dispatches as one vector segment", () => {
  const element = card();
  element._waypoints = [{ x: 1.8, y: 1 }];

  const split = element._plannedSplit();
  assert.equal(split.applied, false);
  assert.equal(split.subLegCount, 1);

  const { service, payload } = element._motionPayload(false);
  assert.equal(service, "raw_pymammotion_execute_vector_segment");
  assert.equal(payload.split_leg_target_length_m, undefined);
});

test("a split click routes to multi-segment and sends the clicks, not the split", () => {
  const element = card();
  element._waypoints = [{ x: 16.24, y: 1 }];

  const { service, payload } = element._motionPayload(false);

  assert.equal(service, "raw_pymammotion_execute_multi_segment");
  assert.equal(payload.split_leg_target_length_m, SPLIT_LEG_TARGET_METRES);
  // Provenance pin: the backend splits and echoes both, so the run JSON records
  // what the operator actually asked for.
  assert.equal(payload.points.length, 2);
  assert.equal(payload.max_real_segments, MAX_REAL_SEGMENTS);
});

test("too far for one click is refused by name, distinctly from too many clicks", () => {
  const element = card();
  // 30 m needs 8 sub-legs against a budget of 4.
  element._waypoints = [{ x: 31, y: 1 }];

  const { blockers } = element._preflight();

  assert.ok(blockers.includes("split_exceeds_real_segment_budget"));
  // Distinct from real_segment_limit_N: one destination was clicked, so that
  // code would be advice for the wrong problem.
  assert.equal(
    blockers.includes(`real_segment_limit_${MAX_REAL_SEGMENTS}`),
    false,
  );
});

test("night is unaffected by the split gates", () => {
  const element = card();
  element._waypoints = [{ x: 31, y: 1 }];

  const { blockers } = element._nightPreflight();

  assert.equal(blockers.includes("split_exceeds_real_segment_budget"), false);
  assert.ok(blockers.includes("night_segment_too_long"));

  const night = element._nightMotionPayload(false);
  assert.equal(night.payload.split_leg_target_length_m, undefined);
});

test("too many short clicks reports only the click limit, not the split code", () => {
  // The two codes are distinct and must stay distinct in BOTH directions.
  // Five short destinations are an over-click, not an over-reach; firing the
  // split code here would give advice for a problem the operator does not have.
  const element = card();
  element._waypoints = Array.from({ length: 5 }, (_, index) => ({
    x: 1 + 0.8 * (index + 1),
    y: 1,
  }));

  const { blockers } = element._preflight();

  assert.ok(blockers.includes(`real_segment_limit_${MAX_REAL_SEGMENTS}`));
  assert.equal(blockers.includes("split_exceeds_real_segment_budget"), false);
  assert.equal(element._plannedSplit().applied, false);
});

// --- Run retention ----------------------------------------------------------
//
// The card kept ten SUMMARIES and exactly ONE full result, overwritten every
// run. `_segmentLandingRows()` needs the full result, so a summary-only entry
// renders as [] and the downloaded history was NOT a recovery path. None of the
// behaviour below -- the cap, ordering, corrupt-JSON guards, `_restoreLastRun`,
// `_clearHistory`, the download payload, quota -- had a single test.

function storageCard() {
  localStorage.reset();
  const element = card();
  element._config.entity = "lawn_mower.retention";
  return element;
}

function runEntry(index) {
  return {
    at: new Date(1_700_000_000_000 + index * 1000).toISOString(),
    elapsed_seconds: 10 + index,
    service: "raw_pymammotion_execute_multi_segment",
    waypoints: 1,
    stop_reason: `stop_${index}`,
    failed_segment_index: null,
    segments: [],
    summary: `run ${index}`,
  };
}

function fullResult(index, padBytes = 0) {
  return {
    stop_reason: "target_reached",
    valid: true,
    marker: index,
    padding: "x".repeat(padBytes),
    segments: [
      {
        index: 1,
        passed: true,
        result: {
          stop_reason: "target_reached",
          waypoint_tolerance: 0.15,
          distance: 0.8,
          // The shape `_landingDistance()` actually reads.
          completion_status: { waypoint_distances: [{ distance: 0.09 }] },
        },
      },
    ],
  };
}

test("a stored run keeps its full result, so history is a recovery path", () => {
  const element = storageCard();

  element._saveRunToHistory(runEntry(1), fullResult(1));
  const [entry] = element._loadHistory();

  assert.equal(entry.result.marker, 1);
  // The whole point: landing rows are recoverable from a history entry.
  const rows = element._segmentLandingRows(entry.result);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].landing, 0.09);
  assert.equal(rows[0].tolerance, 0.15);
});

test("history is newest-first and capped at ten entries", () => {
  const element = storageCard();

  for (let i = 1; i <= 13; i += 1) {
    element._saveRunToHistory(runEntry(i), fullResult(i));
  }
  const history = element._loadHistory();

  assert.equal(history.length, 10);
  assert.equal(history[0].stop_reason, "stop_13");
  assert.equal(history[9].stop_reason, "stop_4");
});

test("a run with no result still records its summary", () => {
  // The failed-call path has no result to store; losing the summary too would
  // hide the failure from the history entirely.
  const element = storageCard();

  element._saveRunToHistory(runEntry(1));
  const [entry] = element._loadHistory();

  assert.equal(entry.stop_reason, "stop_1");
  assert.equal(entry.result, undefined);
});

test("over quota, the OLDEST full results are dropped and the operator is told", () => {
  const element = storageCard();
  element._saveRunToHistory(runEntry(1), fullResult(1, 40_000));
  element._saveRunToHistory(runEntry(2), fullResult(2, 40_000));

  // Now clamp the quota so three padded results cannot coexist.
  localStorage._quotaBytes = 100_000;
  const saved = element._saveRunToHistory(runEntry(3), fullResult(3, 40_000));
  const history = element._loadHistory();

  assert.equal(saved, true);
  assert.equal(history.length, 3);
  // Newest keeps its result; the oldest lost it.
  assert.equal(history[0].result.marker, 3);
  assert.equal(history[2].result, undefined);
  assert.equal(history[2].result_dropped, true);
  // Summaries survive stripping -- that is why both bounds exist.
  assert.equal(history[2].stop_reason, "stop_1");
  assert.match(element._storageWarning, /Storage is full/);
  assert.match(
    element._renderHistoryHtml(),
    /summary only \(dropped for space\)/,
  );
  // The warning renders under the STATUS line, not inside the collapsed
  // history panel -- Nudge persists a run without touching history at all.
  element._render = MammotionCustomPathCard.prototype._render;
  assert.equal(element._renderHistoryHtml().includes("Storage is full"), false);
});

test("a storage that refuses every write says so instead of failing silently", () => {
  const element = storageCard();
  localStorage._failWrites = true;

  const saved = element._saveRunToHistory(runEntry(1), fullResult(1));

  assert.equal(saved, false);
  assert.equal(element._loadHistory().length, 0);
  assert.match(element._storageWarning, /Download the run JSON now/);
});

test("a run that failed to persist does NOT read as stored", () => {
  // The bug: `_realRunAt` was stamped BEFORE the write, inside a catch
  // commented "ignore quota failures", so a quota failure left the card
  // believing it had persisted a run it had not. Same shape as the c196b8b1
  // motion-gate bug -- state set on intent, not on success.
  const element = storageCard();
  localStorage._failWrites = true;

  const persisted = element._persistLastRun({ stop_reason: "target_reached" });

  assert.equal(persisted, false);
  assert.ok(!element._realRunAt, "an unwritten run must not carry a timestamp");
  assert.equal(element._runAgeLabel(), "");
  assert.match(element._storageWarning, /Download the run JSON/);
});

test("a run that DID persist restores with its timestamp", () => {
  const element = storageCard();

  assert.equal(
    element._persistLastRun({ stop_reason: "target_reached" }),
    true,
  );
  const stamped = element._realRunAt;

  const reloaded = storageCardKeepingStorage(element._config.entity);
  const restored = reloaded._restoreLastRun();

  assert.equal(restored.stop_reason, "target_reached");
  assert.equal(reloaded._realRunAt, stamped);
});

function storageCardKeepingStorage(entity) {
  const element = card();
  element._config.entity = entity;
  return element;
}

test("clearing history does not orphan the last-run timestamp", () => {
  // It used to remove the run and its history but leave the timestamp key, so
  // the next restore read a time for a run that no longer existed.
  const element = storageCard();
  element._persistLastRun({ stop_reason: "target_reached" });
  element._saveRunToHistory(runEntry(1), fullResult(1));

  element._clearHistory();

  assert.equal(element._loadHistory().length, 0);
  assert.equal(localStorage.getItem(element._lastRunKey()), null);
  assert.equal(localStorage.getItem(element._lastRunAtKey()), null);
  assert.equal(element._realRunAt, null);
  assert.equal(
    storageCardKeepingStorage(element._config.entity)._restoreLastRun(),
    null,
  );
});

test("corrupt stored history reads as empty rather than throwing", () => {
  const element = storageCard();

  localStorage.setItem(element._historyKey(), "{not json");
  assert.deepEqual(element._loadHistory(), []);

  // A non-array is equally corrupt: `.map` on it would throw during render.
  localStorage.setItem(element._historyKey(), '{"runs":[]}');
  assert.deepEqual(element._loadHistory(), []);

  localStorage.setItem(element._lastRunKey(), "{not json");
  assert.equal(element._restoreLastRun(), null);
});

test("the history panel names how many entries carry a full result", () => {
  // "10 runs" without this reads as ten recoverable runs when it may be one.
  const element = storageCard();
  element._saveRunToHistory(runEntry(1));
  element._saveRunToHistory(runEntry(2), fullResult(2));

  const html = element._renderHistoryHtml();

  assert.match(html, /Run history \(2, 1 with full result\)/);
  assert.match(html, /summary only/);
});

test("the downloaded history payload carries the full results", () => {
  // `#download-history` serializes `{card_version, entity, runs: _loadHistory()}`.
  // Before retention those runs were summaries, so the download could not
  // reproduce a single landing table -- it looked like a backup and was not.
  const element = storageCard();
  element._saveRunToHistory(runEntry(1), fullResult(1));
  element._saveRunToHistory(runEntry(2), fullResult(2));

  const payload = {
    card_version: CARD_VERSION,
    entity: element._config.entity ?? null,
    runs: element._loadHistory(),
  };
  const round = JSON.parse(JSON.stringify(payload));

  assert.equal(round.runs.length, 2);
  for (const run of round.runs) {
    assert.equal(element._segmentLandingRows(run.result).length, 1);
  }
});

// --- Deliberate safety-gate overrides ---------------------------------------
//
// Added at the operator's explicit request: every blocker gets a toggle so a
// restriction can be lifted ON PURPOSE rather than by editing a constant and
// redeploying. The load-bearing properties are that overrides are OFF by
// default, that they reset after every run, that a dry run ignores them, and
// that the card sends BACKEND gate names (which are not always the card's own
// blocker codes).

test("overrides are off by default and change no payload", () => {
  const element = card();
  element._waypoints = [{ x: 1.8, y: 1 }];

  assert.equal(element._overrides.size, 0);
  assert.equal(
    element._motionPayload(false).payload.safety_overrides,
    undefined,
  );
  assert.equal(element._overridePanelHtml(element._preflight()), "");
});

test("a toggled override clears its blocker and nothing else", () => {
  const element = card();
  element._waypoints = [{ x: 1.8, y: 1 }];
  element._runtimeState.safety = {
    allowed_for_manual_motion: false,
    blockers: ["rtk_not_precise", "blade_unsafe"],
  };

  const before = element._preflight();
  assert.equal(before.safe, false);
  assert.deepEqual(before.remaining.includes("rtk_not_precise"), true);

  element._toggleOverride("rtk_not_precise");
  const after = element._preflight();

  // The honest full list is unchanged; only what still BLOCKS moves.
  assert.ok(after.blockers.includes("rtk_not_precise"));
  assert.equal(after.remaining.includes("rtk_not_precise"), false);
  assert.deepEqual(after.overridden, ["rtk_not_precise"]);
  // The un-overridden blocker still blocks.
  assert.ok(after.remaining.includes("blade_unsafe"));
  assert.equal(after.safe, false);
});

test("the card sends BACKEND gate names, not its own blocker codes", () => {
  // `blade_unsafe` is the card's code; the backend gate is
  // `mower_reports_blades_off`. Sending the card code would be refused by the
  // schema -- correct fail-closed behaviour, but it would silently drop the
  // override the operator asked for.
  const element = card();
  element._waypoints = [{ x: 1.8, y: 1 }];
  element._toggleOverride("blade_unsafe");
  element._toggleOverride(`real_segment_limit_${MAX_REAL_SEGMENTS}`);

  const names = element._overridePayloadNames();

  assert.ok(names.includes("mower_reports_blades_off"));
  assert.ok(names.includes("real_segment_limit"));
  assert.equal(names.includes("blade_unsafe"), false);
});

test("a dry run ignores overrides on purpose", () => {
  // The dry run is what you use to see the honest verdict. Overriding there
  // would hide exactly what it exists to reveal.
  const element = card();
  element._waypoints = [{ x: 1.8, y: 1 }];
  element._toggleOverride("rtk_not_precise");

  assert.equal(
    element._motionPayload(true).payload.safety_overrides,
    undefined,
  );
  assert.deepEqual(element._motionPayload(false).payload.safety_overrides, [
    "rtk_not_precise",
  ]);
});

test("only FIRING blockers are offered as toggles", () => {
  // A toggle for a gate that is not blocking anything invites arming something
  // for no reason.
  const element = card();
  element._waypoints = [{ x: 1.8, y: 1 }];
  element._runtimeState.safety = {
    allowed_for_manual_motion: false,
    blockers: ["rtk_not_precise"],
  };

  const offered = element
    ._overridableBlockers(element._preflight().blockers)
    .map((item) => item.code);

  assert.ok(offered.includes("rtk_not_precise"));
  assert.equal(offered.includes("blade_unsafe"), false);
});

test("every offered override carries the reason the gate exists", () => {
  // A gate's NAME never says what it was protecting.
  const element = card();
  element._waypoints = [{ x: 1.8, y: 1 }];
  element._runtimeState.safety = {
    allowed_for_manual_motion: false,
    blockers: ["rtk_not_precise", "blade_unsafe", "runtime_not_mowing"],
  };

  for (const item of element._overridableBlockers(
    element._preflight().blockers,
  )) {
    assert.ok(item.why.length > 20, `${item.code} has no rationale`);
    assert.ok(["low", "medium", "high"].includes(item.risk));
  }
});

test("overrides reset after a run, and an unknown code is ignored", () => {
  const element = card();
  element._toggleOverride("rtk_not_precise");
  assert.equal(element._overrides.size, 1);

  element._clearOverrides();
  assert.equal(element._overrides.size, 0);

  element._toggleOverride("not_a_real_blocker");
  assert.equal(element._overrides.size, 0);
});

test("an armed override panel says so loudly", () => {
  const element = card();
  element._waypoints = [{ x: 1.8, y: 1 }];
  element._runtimeState.safety = {
    allowed_for_manual_motion: false,
    blockers: ["rtk_not_precise"],
  };

  const idle = element._overridePanelHtml(element._preflight());
  assert.match(idle, /Override blockers/);
  assert.equal(idle.includes("armed"), false);

  element._toggleOverride("rtk_not_precise");
  const armed = element._overridePanelHtml(element._preflight());
  assert.match(armed, /armed/);
  assert.match(armed, /1 safety gate overridden/);
  assert.match(armed, /reset automatically/);
  assert.match(armed, /recorded in the run JSON/);
});

// ---------------------------------------------------------------------------
// Keep-out zones and the correctable-leg-length bound (beta66 card work)
//
// The polygon below is the REAL trampoline obstacle from the live map --
// `obstacle:1529607395159402290`, the zone a supervised 10.8 m run drove into
// on 2026-08-20 -- not an invented rectangle. beta49 is the precedent: four
// card defects existed only because it had been tested against fixtures rather
// than real `export_map` output.
// ---------------------------------------------------------------------------

const TRAMPOLINE_ZONE = [
  { x: 11.976, y: -2.54 },
  { x: 13.11, y: -2.49 },
  { x: 13.601, y: -2.274 },
  { x: 14.175, y: -1.707 },
  { x: 14.45, y: -0.86 },
  { x: 14.2, y: 0.6 },
  { x: 13.4, y: 1.4 },
  { x: 12.1, y: 1.53 },
  { x: 10.9, y: 0.9 },
  { x: 10.47, y: -0.4 },
  { x: 10.9, y: -1.9 },
];

function cardWithZone() {
  const element = card();
  element._mapData = {
    area_polygons: {},
    keep_out_polygons: { "obstacle:1529607395159402290": TRAMPOLINE_ZONE },
  };
  return element;
}

test("a click inside a real keep-out zone is refused before dispatch", () => {
  const element = cardWithZone();
  element._mapT = {};
  // The zone centroid -- unambiguously inside the trampoline polygon.
  element._svgPointFromEvent = () => ({ x: 12.185, y: -0.76 });
  let previews = 0;
  element._validateAndPreview = () => {
    previews += 1;
  };

  element._onMapClick({ target: {} });

  assert.equal(element._waypoints.length, 0, "the waypoint must not be added");
  assert.equal(previews, 0, "a refused click must not trigger a preview");
  assert.match(element._status, /keep-out zone \(obstacle\)/);
});

test("a click clear of every keep-out is still accepted", () => {
  const element = cardWithZone();
  element._mapT = {};
  element._svgPointFromEvent = () => ({ x: 4.0, y: -5.0 });
  let previews = 0;
  element._validateAndPreview = () => {
    previews += 1;
  };

  element._onMapClick({ target: {} });

  assert.deepEqual(element._waypoints, [{ x: 4.0, y: -5.0 }]);
  assert.equal(previews, 1);
});

test("a map with no keep-out geometry refuses nothing", () => {
  // Absence of geometry must not read as "everywhere is a zone".
  const element = card();
  element._mapData = { area_polygons: {} };
  element._mapT = {};
  element._svgPointFromEvent = () => ({ x: 12.185, y: -0.76 });
  element._validateAndPreview = () => {};

  element._onMapClick({ target: {} });

  assert.equal(element._waypoints.length, 1);
});

test("the keep-out test is PER-POINT, matching the backend's gap", () => {
  // ⚠️ Pinned deliberately, mirroring
  // test_a_leg_that_clips_a_corner_is_not_caught in the backend suite. A leg
  // whose ENDPOINTS straddle a zone is not caught by either side. If segment
  // -level containment ever lands, this test should fail and be rewritten.
  const element = cardWithZone();
  const straddling = [
    { x: 9.0, y: -0.76 },
    { x: 16.0, y: -0.76 },
  ];
  assert.equal(element._keepOutViolations(straddling).length, 0);
});

test("the correctable leg limit is tolerance / sin(floor)", () => {
  const element = card();
  element._config = { ...element._config, waypoint_tolerance: 0.15 };
  const limit = element._correctableLegLimitMetres();
  const expected =
    0.15 / Math.sin((CORRECTABLE_AIM_FLOOR_DEGREES * Math.PI) / 180);
  assert.ok(Math.abs(limit - expected) < 1e-9);
  assert.ok(Math.abs(limit - 0.5796) < 1e-3, `expected ~0.5796, got ${limit}`);
});

test("the card's correctable floor matches the backend constant", () => {
  // services.py derives _MIN_CORRECTABLE_AIM_ERROR_DEGREES as
  // _POST_TURN_ALIGNMENT_TOLERANCE_DEGREES (10) + _REALIGN_DEADBAND_DEGREES (5).
  // Drift here and the card advises against a floor the backend does not use.
  const services = readFileSync(
    "custom_components/mammotion/services.py",
    "utf8",
  );
  const postTurn = services.match(
    /^_POST_TURN_ALIGNMENT_TOLERANCE_DEGREES = ([\d.]+)/m,
  );
  const deadband = services.match(/^_REALIGN_DEADBAND_DEGREES = ([\d.]+)/m);
  assert.ok(postTurn && deadband, "both backend constants must be findable");
  assert.equal(
    CORRECTABLE_AIM_FLOOR_DEGREES,
    Number(postTurn[1]) + Number(deadband[1]),
  );
});

test("a leg over the bound warns but never blocks the run", () => {
  const element = card();
  element._config = { ...element._config, waypoint_tolerance: 0.15 };
  element._waypoints = [{ x: 1, y: 4 }]; // 3 m from the mower at (1,1)
  element._confirmBlades = true;
  element._confirmClear = true;

  const longest = element._longestPlannedLegMetres();
  assert.ok(longest > 2.9 && longest < 3.1, `longest was ${longest}`);

  // The advisory must not CHANGE the level -- whatever the underlying banner
  // says, adding a warning line must leave it exactly as it was.
  const banner = element._readiness();
  assert.equal(
    banner.level,
    element._readinessLevel().level,
    "the advisory must not alter the readiness level",
  );
  const advisory = banner.details.find((line) => line.includes("can protect"));
  assert.ok(advisory, `no advisory in: ${JSON.stringify(banner.details)}`);
  assert.match(advisory, /0\.58 m/);
  assert.match(advisory, /warning, not a blocker/);
});

test("a leg inside the bound produces no advisory", () => {
  const element = card();
  element._config = { ...element._config, waypoint_tolerance: 0.15 };
  element._waypoints = [{ x: 1, y: 1.4 }]; // 0.4 m -- under 0.58
  element._confirmBlades = true;
  element._confirmClear = true;

  const banner = element._readiness();
  assert.equal(
    banner.details.find((line) => line.includes("can protect")),
    undefined,
  );
});

test("the leg advisory is visible while still arming, not only when ready", () => {
  // 🔑 Regression: the advisory was first written inside the "ready" branch,
  // so it stayed hidden through the whole period an operator is choosing where
  // to click. The warning is only useful BEFORE everything is confirmed.
  const element = card();
  element._config = { ...element._config, waypoint_tolerance: 0.15 };
  element._waypoints = [{ x: 1, y: 4 }];
  element._confirmBlades = false;
  element._confirmClear = false;

  const banner = element._readiness();
  assert.notEqual(banner.level, "ready", "precondition: not yet ready");
  assert.ok(
    banner.details.some((line) => line.includes("can protect")),
    `advisory missing while arming: ${JSON.stringify(banner.details)}`,
  );
});
