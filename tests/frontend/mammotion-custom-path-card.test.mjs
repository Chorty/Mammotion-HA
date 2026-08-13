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
globalThis.localStorage = {
  _items: new Map(),
  getItem(key) {
    return this._items.get(key) ?? null;
  },
  setItem(key, value) {
    this._items.set(key, String(value));
  },
  removeItem(key) {
    this._items.delete(key);
  },
};

const {
  LUBA_ACCEPTANCE_PROFILE,
  MAX_REAL_SEGMENTS,
  MAX_WAYPOINTS,
  PROFILE_KEYS,
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
    position: { x: 1, y: 1 },
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

  assert.equal(dry.payload.max_real_segments, MAX_WAYPOINTS);
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
  assert.equal(LUBA_ACCEPTANCE_PROFILE.max_linear_pulse_ceiling, 14);
  assert.equal(payload.max_linear_pulse_ceiling, 14);
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
  assert.equal(payload.max_linear_pulse_ceiling, 14);
});

test("profile label reports acceptance by default and names any override", () => {
  const element = card();

  // Gate 5 re-passed on this profile 2026-08-12, card-driven, 4/4 segments
  // target_reached. The label must no longer say the re-pass is pending.
  assert.match(element._profileLabel(), /LUBA acceptance profile \+ reach/);
  assert.match(element._profileLabel(), /Gate 5 re-pass 2026-08-12/);
  assert.doesNotMatch(element._profileLabel(), /PENDING/);

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
