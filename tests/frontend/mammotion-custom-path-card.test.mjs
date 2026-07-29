import assert from "node:assert/strict";
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
  MAX_REAL_SEGMENTS,
  MAX_WAYPOINTS,
  MammotionCustomPathCard,
} = await import(
  "../../custom_components/mammotion/www/mammotion-custom-path-card.js"
);

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
