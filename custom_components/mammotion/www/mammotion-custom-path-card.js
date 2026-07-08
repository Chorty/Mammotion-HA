const MAX_WAYPOINTS = 3;

class MammotionCustomPathCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._hass = null;
    this._config = {};
    this._mapData = null;
    this._waypoints = [];
    this._runtimeState = null;
    this._areaHash = "";
    this._mapT = null;
    this._draggingIndex = null;
    this._height = 520;
    this._status = "Load map/runtime, then click up to 3 waypoints to build a path. Movement requires explicit Real Go confirmation.";
    this._validation = null;
    this._dryRun = null;
    this._realRun = null;
    this._loadingMap = false;
    this._loadingRuntime = false;
    this._confirmBladesOff = false;
    this._confirmClearArea = false;
    this._rendered = false;
  }

  setConfig(config) {
    if (!config.entity) {
      throw new Error("entity is required");
    }
    this._config = {
      speed: 0.2,
      blade_mode: "off",
      prefer_ble: true,
      max_turn_commands: 1,
      max_linear_commands: 1,
      sample_delays: [0, 5, 10],
      ...config,
    };
    this._height = Number(this._config.card_height || 520);
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

  _currentPositionPoint() {
    const pos = this._runtimeState?.position || {};
    if (pos.x == null || pos.y == null) {
      return null;
    }
    return {
      x: Number(pos.x),
      y: Number(pos.y),
    };
  }

  _segmentPoints() {
    const start = this._currentPositionPoint();
    if (!start || !this._waypoints.length) {
      return null;
    }
    return [start, ...this._waypoints].map((point) => this._roundedPoint(point));
  }

  _segmentCount() {
    return this._waypoints.length;
  }

  _preflight() {
    const blockers = [];
    const runtime = this._runtimeState || {};
    const safety = runtime.safety || {};
    const start = this._currentPositionPoint();
    if (!start) {
      blockers.push("position_unavailable");
    }
    if (!this._waypoints.length) {
      blockers.push("path_unset");
    }
    if (safety.allowed_for_manual_motion === false) {
      if (Array.isArray(safety.blockers) && safety.blockers.length) {
        blockers.push(...safety.blockers);
      } else {
        blockers.push("runtime_safety_blocked");
      }
    }
    if (!this._validation?.valid) {
      blockers.push("path_validation_failed");
    }
    return {
      safe: blockers.length === 0,
      blockers,
      runtime,
    };
  }

  _runtimePreflightDetails() {
    const runtime = this._runtimeState || {};
    const safety = runtime.safety || {};
    const routeStatus = safety.active_route_status || {};
    const activeTransport = runtime.active_transport ?? "unknown";
    const bladeSafe = safety.blade_safe_for_motion === true;
    const activeMowing = safety.active_mowing_detected === true;
    const chargingNow = String(runtime.charge_state_label || runtime.charge_state || "").toLowerCase().includes("charging");
    const routeBlocks = routeStatus.blocks_motion === true;
    return {
      activeTransport,
      bladeSafeLabel: bladeSafe ? "safe" : "unsafe",
      mowingReadinessLabel: activeMowing ? "blocked (active mowing detected)" : "ready",
      chargingReadinessLabel: chargingNow ? "charging now" : "not charging",
      routeBlockingLabel: routeBlocks
        ? `blocking (${routeStatus.reason || "unknown_reason"})`
        : `clear (${routeStatus.reason || "no_route"})`,
      haState: runtime.ha_state ?? "unknown",
      workMode: runtime.work_mode_label ?? runtime.work_mode ?? "unknown",
      chargeState: runtime.charge_state_label ?? runtime.charge_state ?? "unknown",
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
      const scaleX = Number(svgEl.getAttribute("viewBox")?.split(" ")[2] || rect.width) / rect.width;
      const scaleY = Number(svgEl.getAttribute("viewBox")?.split(" ")[3] || rect.height) / rect.height;
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
    if (event.target?.dataset?.pointIndex != null || !this._mapT) return;
    const point = this._svgPointFromEvent(event);
    if (!point) return;
    if (this._waypoints.length >= MAX_WAYPOINTS) {
      this._status = `Maximum ${MAX_WAYPOINTS} waypoints reached. Remove one before adding another.`;
      this._render();
      return;
    }
    this._waypoints.push(point);
    this._dryRun = null;
    this._realRun = null;
    this._validateAndPreview();
  }

  _onPointDown(event) {
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

  _clearTarget() {
    this._waypoints = [];
    this._validation = null;
    this._dryRun = null;
    this._realRun = null;
    this._status = "Path cleared.";
    this._render();
  }

  _removeLastWaypoint() {
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

  _motionPayload(dryRun) {
    const points = this._segmentPoints();
    if (!points) return null;
    const payload = {
      entity_id: this._config.entity,
      points,
      dry_run: dryRun,
      confirm_blades_off: dryRun ? false : this._confirmBladesOff,
      confirm_clear_area: dryRun ? false : this._confirmClearArea,
      prefer_ble: Boolean(this._config.prefer_ble ?? true),
      max_turn_commands: Number(this._config.max_turn_commands || 1),
      max_linear_commands: Number(this._config.max_linear_commands || 1),
      sample_delays: Array.isArray(this._config.sample_delays)
        ? this._config.sample_delays
        : [0, 5, 10],
    };
    if (this._areaHash) {
      payload.area_hash = String(this._areaHash);
    }
    if (points.length > 2) {
      return {
        service: "raw_pymammotion_execute_multi_segment",
        payload: {
          ...payload,
          max_real_segments: Math.min(points.length - 1, 3),
        },
      };
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
    const lines = [
      `entity_id: ${payload.entity_id}`,
    ];
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
      this._status = "Add at least one waypoint and ensure live mower position is available before dry-run.";
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

  async _runRealGo() {
    const preflight = this._preflight();
    const motion = this._motionPayload(false);
    if (!motion) {
      this._status = "Add at least one waypoint and ensure live mower position is available before Real Go.";
      this._render();
      return;
    }
    if (!this._confirmBladesOff || !this._confirmClearArea) {
      this._status = "Real Go blocked: enable both confirmations first.";
      this._render();
      return;
    }
    if (!preflight.safe) {
      this._status = `Real Go blocked by preflight: ${preflight.blockers.join(", ")}`;
      this._render();
      return;
    }
    const segmentCount = this._segmentCount();
    this._status = `Running guarded Real Go (${segmentCount} segment${segmentCount === 1 ? "" : "s"})…`;
    this._render();
    try {
      this._realRun = await this._callService(motion.service, motion.payload);
      this._status = `Real Go complete: ${this._segmentProgressText(this._realRun)}`;
      await this._loadRuntimeState();
      this._render();
    } catch (err) {
      this._status = `Real Go failed: ${err?.message || err}`;
      this._render();
    }
  }

  async _abortMotion() {
    if (!this._confirmBladesOff || !this._confirmClearArea) {
      this._status = "Abort requires both confirmations enabled.";
      this._render();
      return;
    }
    this._status = "Sending zero-motion stop nudge…";
    this._render();
    try {
      const abortResult = await this._callService("raw_pymammotion_motion_probe", {
        command: "send_movement",
        linear_speed: 0,
        angular_speed: 0,
        prefer_ble: Boolean(this._config.prefer_ble ?? true),
        dry_run: false,
        confirm_blades_off: true,
        confirm_clear_area: true,
        sample_delays: [0],
      });
      const status = abortResult?.command_result?.ok === true ? "ok" : "failed";
      this._status = `Abort result: ${status}`;
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
      (this._mapData?.areas || []).map((area) => [String(area.area_hash), area.name]),
    );
    for (const [hash, points] of Object.entries(polygons)) {
      if (points.length < 2) continue;
      const active = hash === String(this._areaHash);
      const polygon = el("polygon", {
        points: points
          .map((point) => `${mt.toSX(point.x).toFixed(1)},${mt.toSY(point.y).toFixed(1)}`)
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

    const start = this._currentPositionPoint();
    const pathPoints = start ? [start, ...this._waypoints] : [];
    const runResult = this._realRun || this._dryRun;
    const segments = Array.isArray(runResult?.segments) ? runResult.segments : null;

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
            .map((point) => `${mt.toSX(point.x).toFixed(1)},${mt.toSY(point.y).toFixed(1)}`)
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
      circle.addEventListener("pointerdown", (event) => this._onPointDown(event));
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
    const removeDisabled = pathSet ? "" : "disabled";
    const preflight = this._preflight();
    const preflightText = preflight.safe
      ? "Preflight: safe"
      : `Preflight blockers: ${preflight.blockers.join(", ")}`;
    const runtimePanel = this._runtimePreflightDetails();
    const realGoDisabled =
      !pathSet ||
      !this._confirmBladesOff ||
      !this._confirmClearArea ||
      !preflight.safe;
    const segmentCount = this._segmentCount();
    this.shadowRoot.innerHTML = `
      <style>
        ha-card { overflow: hidden; }
        .toolbar { display: flex; gap: 8px; align-items: center; padding: 12px; flex-wrap: wrap; }
        .status { padding: 0 12px 12px; color: var(--secondary-text-color); font-size: 13px; }
        .warnings { padding: 0 12px 12px; color: #f59e0b; font-size: 12px; }
        .waypoint-counter { font-size: 12px; color: var(--secondary-text-color); margin-left: auto; }
        .preflight-panel { margin: 0 12px 12px; padding: 8px 10px; border: 1px solid rgba(127,127,127,0.35); border-radius: 6px; font-size: 12px; color: var(--secondary-text-color); }
        .preflight-panel .title { font-weight: 600; margin-bottom: 6px; color: var(--primary-text-color); }
        .preflight-row { display: flex; justify-content: space-between; gap: 10px; padding: 2px 0; }
        .preflight-row .label { opacity: 0.85; }
        .preflight-row .value { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; text-align: right; }
        details { padding: 0 12px 12px; color: var(--secondary-text-color); font-size: 12px; }
        summary { cursor: pointer; }
        pre { overflow: auto; max-height: 220px; padding: 8px; background: rgba(127,127,127,0.12); border-radius: 4px; }
        svg { display: block; width: 100%; height: ${this._height}px; background: #0d1117; touch-action: none; cursor: crosshair; }
        select, button { font: inherit; }
      </style>
      <ha-card header="Mammotion click/go (guarded segment chain)">
        <div class="toolbar">
          <button id="reload" type="button">Reload map/runtime</button>
          <button id="undo" type="button" ${removeDisabled}>Remove last waypoint</button>
          <button id="clear" type="button">Reset path</button>
          <button id="copy-yaml" type="button" ${removeDisabled}>Copy YAML</button>
          <button id="copy-json" type="button" ${removeDisabled}>Copy JSON</button>
          <button id="copy-dry-run-yaml" type="button" ${removeDisabled}>Copy dry-run YAML</button>
          <button id="dry-run" type="button" ${pathSet ? "" : "disabled"}>Run dry-run</button>
          <button id="real-go" type="button" ${realGoDisabled ? "disabled" : ""}>Real Go</button>
          <button id="abort" type="button">Abort/Stop Nudge</button>
          <label><input id="confirm-blades-off" type="checkbox" ${this._confirmBladesOff ? "checked" : ""}/> confirm blades off</label>
          <label><input id="confirm-clear-area" type="checkbox" ${this._confirmClearArea ? "checked" : ""}/> confirm clear area</label>
          <label>Area
            <select id="area">
              ${areas.map((area) => `<option value="${this._escapeHtml(area.area_hash)}" ${String(area.area_hash) === String(this._areaHash) ? "selected" : ""}>${this._escapeHtml(area.name || area.area_hash)}</option>`).join("")}
            </select>
          </label>
          <span class="waypoint-counter">Waypoints: ${this._waypoints.length} (segments: ${segmentCount})</span>
        </div>
        <svg id="path-map"></svg>
        <div class="status">${this._escapeHtml(this._status)}</div>
        <div class="status">${this._escapeHtml(preflightText)}</div>
        <div class="preflight-panel">
          <div class="title">Runtime preflight details</div>
          <div class="preflight-row"><span class="label">active_transport</span><span class="value">${this._escapeHtml(runtimePanel.activeTransport)}</span></div>
          <div class="preflight-row"><span class="label">blade-safe status</span><span class="value">${this._escapeHtml(runtimePanel.bladeSafeLabel)}</span></div>
          <div class="preflight-row"><span class="label">mowing readiness</span><span class="value">${this._escapeHtml(runtimePanel.mowingReadinessLabel)}</span></div>
          <div class="preflight-row"><span class="label">charging readiness</span><span class="value">${this._escapeHtml(runtimePanel.chargingReadinessLabel)}</span></div>
          <div class="preflight-row"><span class="label">route-blocking status</span><span class="value">${this._escapeHtml(runtimePanel.routeBlockingLabel)}</span></div>
          <div class="preflight-row"><span class="label">ha_state</span><span class="value">${this._escapeHtml(runtimePanel.haState)}</span></div>
          <div class="preflight-row"><span class="label">work_mode</span><span class="value">${this._escapeHtml(runtimePanel.workMode)}</span></div>
          <div class="preflight-row"><span class="label">charge_state</span><span class="value">${this._escapeHtml(runtimePanel.chargeState)}</span></div>
        </div>
        ${(this._validation?.warnings || []).length ? `<div class="warnings">Warnings: ${this._escapeHtml(this._validation.warnings.join(", "))}</div>` : ""}
        ${pathSet ? `<details><summary>Preview service YAML</summary><pre>${this._escapeHtml(this._payloadYaml())}</pre></details>` : ""}
        ${pathSet ? `<details><summary>Dry-run service YAML</summary><pre>${this._escapeHtml(this._dryRunYaml())}</pre></details>` : ""}
        ${this._dryRun ? `<details><summary>Last dry-run result</summary><pre>${this._escapeHtml(JSON.stringify(this._dryRun, null, 2))}</pre></details>` : ""}
        ${this._realRun ? `<details><summary>Last Real Go result</summary><pre>${this._escapeHtml(JSON.stringify(this._realRun, null, 2))}</pre></details>` : ""}
      </ha-card>
    `;
    this._q("#reload")?.addEventListener("click", async () => {
      await this._loadMap();
      await this._loadRuntimeState();
    });
    this._q("#undo")?.addEventListener("click", () => this._removeLastWaypoint());
    this._q("#clear")?.addEventListener("click", () => this._clearTarget());
    this._q("#copy-yaml")?.addEventListener("click", () => this._copyYaml());
    this._q("#copy-json")?.addEventListener("click", () => this._copyJson());
    this._q("#copy-dry-run-yaml")?.addEventListener("click", () => this._copyDryRunYaml());
    this._q("#dry-run")?.addEventListener("click", () => this._runDryRun());
    this._q("#real-go")?.addEventListener("click", () => this._runRealGo());
    this._q("#abort")?.addEventListener("click", () => this._abortMotion());
    this._q("#confirm-blades-off")?.addEventListener("change", (event) => {
      this._confirmBladesOff = Boolean(event.target.checked);
      this._render();
    });
    this._q("#confirm-clear-area")?.addEventListener("change", (event) => {
      this._confirmClearArea = Boolean(event.target.checked);
      this._render();
    });
    this._q("#area")?.addEventListener("change", (event) => {
      this._areaHash = event.target.value;
      this._validateAndPreview();
    });
    const svgEl = this._q("#path-map");
    svgEl?.addEventListener("click", (event) => this._onMapClick(event));
    svgEl?.addEventListener("pointermove", (event) => this._onPointerMove(event));
    svgEl?.addEventListener("pointerup", () => this._onPointerUp());
    svgEl?.addEventListener("pointercancel", () => this._onPointerUp());
    this._renderMap();
  }
}

customElements.define("mammotion-custom-path-card", MammotionCustomPathCard);

window.customCards = window.customCards || [];
window.customCards.push({
  type: "mammotion-custom-path-card",
  name: "Mammotion Click/Go (Guarded)",
  description: "Click up to 3 waypoints to build a guarded segment-chain path, then run dry-run or Real Go.",
});
