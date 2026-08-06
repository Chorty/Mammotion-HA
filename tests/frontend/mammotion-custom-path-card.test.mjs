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

test("seven-point dry-run is retained but real payload is capped at two", () => {
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
  assert.equal(payload.waypoint_tolerance, 0.08);
  assert.equal(payload.min_progress_distance, 0.0025);
  assert.equal(payload.calibrated_forward_heading_offset_degrees, 102.4);
  assert.equal(payload.motion_refresh_interval_ms, 200);
  assert.equal(payload.ble_auto_recover, false);
  assert.equal(payload.turn_mode, "vio");
});

test("no loop-to-tolerance ceiling is omitted rather than sent as zero", () => {
  const element = card();
  element._waypoints = [{ x: 2, y: 2 }];

  const { payload } = element._motionPayload(false);

  assert.equal(LUBA_ACCEPTANCE_PROFILE.max_linear_pulse_ceiling, null);
  assert.equal("max_linear_pulse_ceiling" in payload, false);

  element._config.max_linear_pulse_ceiling = 30;
  const opted = element._motionPayload(false).payload;
  assert.equal(opted.max_linear_pulse_ceiling, 30);
});

test("profile label reports acceptance by default and names any override", () => {
  const element = card();

  assert.match(element._profileLabel(), /LUBA acceptance profile/);

  element._config.waypoint_tolerance = 0.15;
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
    // The one profile key README must NOT list: setting it at all re-enables
    // loop-to-tolerance and leaves the accepted profile.
    if (key === "max_linear_pulse_ceiling") {
      assert.equal(
        documented.has(key),
        false,
        "README must leave max_linear_pulse_ceiling unset",
      );
      continue;
    }
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

test("backend blockers and the two-segment limit lock Real Go", () => {
  const element = card();
  element._waypoints = [
    { x: 2, y: 2 },
    { x: 3, y: 3 },
    { x: 4, y: 4 },
  ];
  element._runtimeState.experimental_motion = {
    real_motion_allowed: false,
    blockers: ["pymammotion_backend_unverified"],
  };

  const preflight = element._preflight();

  assert.equal(preflight.safe, false);
  assert.ok(preflight.blockers.includes("real_segment_limit_2"));
  assert.ok(preflight.blockers.includes("pymammotion_backend_unverified"));
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
