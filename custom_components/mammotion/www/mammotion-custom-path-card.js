class MammotionCustomPathCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._hass = null;
    this._config = {};
    this._mapData = null;
    this._points = [];
    this._areaHash = "";
    this._mapT = null;
    this._dragIndex = null;
    this._height = 520;
    this._status = "Load a mower map, then click inside an area to add path points.";
    this._validation = null;
    this._loadingMap = false;
    this._rendered = false;
  }

  setConfig(config) {
    if (!config.entity) {
      throw new Error("entity is required");
    }
    this._config = {
      speed: 0.2,
      blade_mode: "off",
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
        ? "Click inside an area to add points. Drag points to edit."
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
    return Object.values(polygons).flat();
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
    if (!this._hass || this._points.length < 1) {
      this._validation = null;
      this._renderMap();
      return;
    }
    try {
      const data = {
        points: this._points,
        speed: Number(this._config.speed || 0.2),
        blade_mode: "off",
      };
      if (this._areaHash) {
        data.area_hash = this._areaHash;
      }
      this._validation = await this._callService("preview_custom_path", data);
      this._status = this._validation.valid
        ? `Valid path: ${this._validation.point_count} points, distance ${Number(this._validation.distance || 0).toFixed(2)}`
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
    this._points = [...this._points, point];
    this._validateAndPreview();
  }

  _onPointDown(event) {
    event.stopPropagation();
    this._dragIndex = Number(event.target.dataset.pointIndex);
    event.target.setPointerCapture(event.pointerId);
  }

  _onPointerMove(event) {
    if (this._dragIndex == null) return;
    const point = this._svgPointFromEvent(event);
    if (!point) return;
    this._points = this._points.map((existing, index) =>
      index === this._dragIndex ? point : existing,
    );
    this._renderMap();
  }

  _onPointerUp() {
    if (this._dragIndex == null) return;
    this._dragIndex = null;
    this._validateAndPreview();
  }

  _clearPath() {
    this._points = [];
    this._validation = null;
    this._status = "Path cleared.";
    this._render();
  }

  _undoPoint() {
    this._points = this._points.slice(0, -1);
    this._validateAndPreview();
  }

  _roundedPoint(point) {
    return {
      x: Number(point.x.toFixed(3)),
      y: Number(point.y.toFixed(3)),
    };
  }

  _previewPayload() {
    const payload = {
      entity_id: this._config.entity,
      speed: Number(this._config.speed || 0.2),
      blade_mode: "off",
      points: this._points.map((point) => this._roundedPoint(point)),
    };
    if (this._areaHash) {
      payload.area_hash = String(this._areaHash);
    }
    return payload;
  }

  _payloadYaml() {
    const payload = this._previewPayload();
    const lines = [
      `entity_id: ${payload.entity_id}`,
    ];
    if (payload.area_hash) {
      lines.push(`area_hash: "${payload.area_hash}"`);
    }
    lines.push(`speed: ${payload.speed}`);
    lines.push(`blade_mode: "${payload.blade_mode}"`);
    lines.push("points:");
    for (const point of payload.points) {
      lines.push(`  - x: ${point.x}`);
      lines.push(`    y: ${point.y}`);
    }
    return `${lines.join("\n")}\n`;
  }

  _payloadJson() {
    return `${JSON.stringify(this._previewPayload(), null, 2)}\n`;
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
    if (!this._points.length) {
      this._status = "Draw at least one point before copying YAML.";
      this._render();
      return;
    }
    this._copyText(this._payloadYaml(), "YAML");
  }

  _copyJson() {
    if (!this._points.length) {
      this._status = "Draw at least one point before copying JSON.";
      this._render();
      return;
    }
    this._copyText(this._payloadJson(), "JSON");
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

    if (this._points.length >= 2) {
      const path = el("polyline", {
        points: this._points
          .map((point) => `${mt.toSX(point.x).toFixed(1)},${mt.toSY(point.y).toFixed(1)}`)
          .join(" "),
        fill: "none",
        stroke: this._validation?.valid === false ? "#ef4444" : "#22c55e",
        "stroke-width": "4",
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
      });
      svgEl.appendChild(path);
    }

    this._points.forEach((point, index) => {
      const circle = el("circle", {
        cx: mt.toSX(point.x).toFixed(1),
        cy: mt.toSY(point.y).toFixed(1),
        r: index === 0 || index === this._points.length - 1 ? 7 : 5,
        fill: index === 0 ? "#22c55e" : index === this._points.length - 1 ? "#f97316" : "#eab308",
        stroke: "#111827",
        "stroke-width": "2",
        "data-point-index": index,
        cursor: "grab",
      });
      circle.addEventListener("pointerdown", (event) => this._onPointDown(event));
      svgEl.appendChild(circle);
    });
  }

  _render() {
    const areas = this._mapData?.areas || [];
    const undoDisabled = this._points.length ? "" : "disabled";
    this.shadowRoot.innerHTML = `
      <style>
        ha-card { overflow: hidden; }
        .toolbar { display: flex; gap: 8px; align-items: center; padding: 12px; flex-wrap: wrap; }
        .status { padding: 0 12px 12px; color: var(--secondary-text-color); font-size: 13px; }
        .warnings { padding: 0 12px 12px; color: #f59e0b; font-size: 12px; }
        details { padding: 0 12px 12px; color: var(--secondary-text-color); font-size: 12px; }
        summary { cursor: pointer; }
        pre { overflow: auto; max-height: 220px; padding: 8px; background: rgba(127,127,127,0.12); border-radius: 4px; }
        svg { display: block; width: 100%; height: ${this._height}px; background: #0d1117; touch-action: none; cursor: crosshair; }
        select, button { font: inherit; }
      </style>
      <ha-card header="Mammotion custom path preview">
        <div class="toolbar">
          <button id="reload" type="button">Reload map</button>
          <button id="undo" type="button" ${undoDisabled}>Undo point</button>
          <button id="clear" type="button">Clear path</button>
          <button id="copy-yaml" type="button" ${undoDisabled}>Copy YAML</button>
          <button id="copy-json" type="button" ${undoDisabled}>Copy JSON</button>
          <label>Area
            <select id="area">
              ${areas.map((area) => `<option value="${this._escapeHtml(area.area_hash)}" ${String(area.area_hash) === String(this._areaHash) ? "selected" : ""}>${this._escapeHtml(area.name || area.area_hash)}</option>`).join("")}
            </select>
          </label>
        </div>
        <svg id="path-map"></svg>
        <div class="status">${this._escapeHtml(this._status)}</div>
        ${(this._validation?.warnings || []).length ? `<div class="warnings">Warnings: ${this._escapeHtml(this._validation.warnings.join(", "))}</div>` : ""}
        ${this._points.length ? `<details><summary>Preview service YAML</summary><pre>${this._escapeHtml(this._payloadYaml())}</pre></details>` : ""}
      </ha-card>
    `;
    this._q("#reload")?.addEventListener("click", () => this._loadMap());
    this._q("#undo")?.addEventListener("click", () => this._undoPoint());
    this._q("#clear")?.addEventListener("click", () => this._clearPath());
    this._q("#copy-yaml")?.addEventListener("click", () => this._copyYaml());
    this._q("#copy-json")?.addEventListener("click", () => this._copyJson());
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
  name: "Mammotion Custom Path Preview",
  description: "Draw and validate Mammotion mower-map paths without movement.",
});
