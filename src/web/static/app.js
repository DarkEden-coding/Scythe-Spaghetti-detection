"use strict";

const elements = Object.fromEntries(
  [
    "connection-dot", "connection-label", "printer-state", "uptime", "last-frame",
    "next-scan", "system-banner", "system-message", "frame-detail", "camera-stage",
    "camera-empty", "camera-image", "detection-overlay", "frame-stamp", "cadence-fill",
    "overlay-toggle", "verdict", "verdict-title", "verdict-detail", "box-count",
    "box-list", "last-detection", "detected-at", "latest-count", "latest-confidence",
    "metric-count", "metric-confidence", "metric-inference", "metric-interval",
    "pause-button", "resume-button", "debug-detection-button", "acknowledge-button", "control-result",
  ].map((id) => [id, document.getElementById(id)])
);

const svgNamespace = "http://www.w3.org/2000/svg";
let state = null;
let overlayVisible = true;
let lastFrameUrl = "";
let pollTimer = null;
let countdownTimer = null;
let serverClockOffset = 0;

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : `${minutes}m`;
}

function formatTime(timestamp) {
  if (!timestamp) return "—";
  return new Date(timestamp * 1000).toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "—";
}

function setConnection(connection) {
  const labels = {
    online: "MONITOR ONLINE",
    starting: "CONNECTING",
    stopped: "MONITOR STOPPED",
    camera_unavailable: "CAMERA UNAVAILABLE",
    error: "MONITOR ERROR",
    disconnected: "DASHBOARD OFFLINE",
  };
  elements["connection-label"].textContent = labels[connection] || "UNKNOWN";
  elements["connection-dot"].className = "status-dot";
  if (connection === "online") elements["connection-dot"].classList.add("online");
  else if (connection === "camera_unavailable") elements["connection-dot"].classList.add("warning");
  else if (connection !== "starting") elements["connection-dot"].classList.add("error");
}

function renderBanner(payload) {
  const visible = ["error", "camera_unavailable", "stopped", "disconnected"].includes(payload.connection);
  elements["system-banner"].hidden = !visible;
  elements["system-message"].textContent = payload.message || "Scythe is unavailable.";
  elements["camera-image"].classList.toggle("stale", payload.connection === "camera_unavailable");
}

function renderFrame(payload) {
  if (!payload.frame) return;
  elements["camera-empty"].hidden = true;
  elements["camera-image"].hidden = false;
  elements["frame-stamp"].hidden = false;
  elements["frame-stamp"].textContent = `FRAME ${formatTime(payload.frame.captured_at)}`;
  elements["last-frame"].textContent = formatTime(payload.frame.captured_at);
  elements["frame-detail"].textContent = `${payload.frame.width} × ${payload.frame.height} · updated ${formatTime(payload.frame.captured_at)}`;
  elements["camera-image"].alt = payload.current_detection
    ? `Latest printer camera frame with ${payload.current_detection.count} detections`
    : "Latest printer camera frame with no current detections";
  if (payload.frame.url !== lastFrameUrl) {
    lastFrameUrl = payload.frame.url;
    elements["camera-image"].src = payload.frame.url;
  }
}

function svgElement(name, attributes = {}) {
  const node = document.createElementNS(svgNamespace, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function renderOverlay(payload) {
  const overlay = elements["detection-overlay"];
  overlay.replaceChildren();
  const detection = payload.current_detection;
  if (!payload.frame || !detection || !overlayVisible) return;

  overlay.setAttribute("viewBox", `0 0 ${payload.frame.width} ${payload.frame.height}`);
  overlay.setAttribute("preserveAspectRatio", "none");

  detection.boxes.forEach((box, index) => {
    const width = Math.max(1, box.x2 - box.x1);
    const height = Math.max(1, box.y2 - box.y1);
    const labelHeight = Math.max(22, payload.frame.height * 0.035);
    const fontSize = Math.max(13, payload.frame.height * 0.022);
    const labelWidth = Math.max(72, fontSize * 5.4);
    const labelY = Math.max(0, box.y1 - labelHeight);
    const group = svgElement("g");
    group.append(
      svgElement("rect", {
        x: box.x1, y: box.y1, width, height,
        fill: "none", stroke: "#ffbd22", "stroke-width": Math.max(2, payload.frame.width * 0.002),
      }),
      svgElement("rect", {
        x: box.x1, y: labelY, width: labelWidth, height: labelHeight,
        fill: "#ffbd22",
      })
    );
    const label = svgElement("text", {
      x: box.x1 + 6,
      y: labelY + labelHeight * 0.7,
      fill: "#171204",
      "font-family": "system-ui, sans-serif",
      "font-size": fontSize,
      "font-weight": "700",
    });
    label.textContent = `${index + 1} · ${formatPercent(box.confidence)}`;
    group.append(label);
    overlay.append(group);
  });
}

function renderBoxes(detection) {
  const boxes = detection?.boxes || [];
  elements["box-count"].textContent = String(boxes.length);
  elements["box-list"].replaceChildren();
  if (!boxes.length) {
    const empty = document.createElement("p");
    empty.className = "empty-copy";
    empty.textContent = "No detections in the current frame.";
    elements["box-list"].append(empty);
    return;
  }

  boxes.forEach((box, index) => {
    const row = document.createElement("div");
    row.className = "box-row";
    const number = document.createElement("span");
    number.className = "box-index";
    number.textContent = `#${index + 1}`;
    const confidence = document.createElement("span");
    confidence.className = "box-confidence";
    confidence.textContent = formatPercent(box.confidence);
    const area = document.createElement("span");
    area.className = "box-area";
    area.textContent = `${Math.round((box.x2 - box.x1) * (box.y2 - box.y1)).toLocaleString()} px²`;
    row.append(number, confidence, area);
    elements["box-list"].append(row);
  });
}

function renderDetection(payload) {
  const current = payload.current_detection;
  const latest = payload.last_detection;
  elements.verdict.className = "verdict";

  if (current) {
    elements.verdict.classList.add("alert");
    elements["verdict-title"].textContent = current.debug
      ? "DEBUG DETECTION"
      : "SPAGHETTI DETECTED";
    elements["verdict-detail"].textContent = current.debug
      ? "Idle inspection only · no alert sent"
      : current.paused
        ? "Print paused automatically"
        : "Check the printer immediately";
  } else if (payload.frame) {
    elements.verdict.classList.add("clear");
    elements["verdict-title"].textContent = "CLEAR / NO SPAGHETTI";
    elements["verdict-detail"].textContent = payload.printer.detail || "Latest inspection is clear";
  } else {
    elements["verdict-title"].textContent = "WAITING";
    elements["verdict-detail"].textContent = "No inspection result yet";
  }

  renderBoxes(current);
  renderOverlay(payload);
  const count = current?.count || 0;
  elements["metric-count"].textContent = String(count);
  elements["metric-confidence"].textContent = current ? formatPercent(current.max_confidence) : "—";
  elements["metric-inference"].textContent = current ? `${current.duration_seconds.toFixed(2)}s` : inferenceFromDetail(payload.printer.detail);

  elements["last-detection"].hidden = !latest;
  if (latest) {
    elements["detected-at"].textContent = formatTime(latest.detected_at);
    elements["latest-count"].textContent = String(latest.count);
    elements["latest-confidence"].textContent = formatPercent(latest.max_confidence);
  }
}

function inferenceFromDetail(detail) {
  const match = String(detail || "").match(/\((\d+(?:\.\d+)?)s inference\)/);
  return match ? `${match[1]}s` : "—";
}

function renderControls(payload) {
  const printerState = payload.printer.state;
  elements["pause-button"].disabled = printerState !== "printing";
  elements["resume-button"].disabled = printerState !== "paused";
  const debugEnabled = Boolean(payload.debug_detection_enabled);
  elements["debug-detection-button"].disabled = false;
  elements["debug-detection-button"].setAttribute("aria-pressed", String(debugEnabled));
  elements["debug-detection-button"].querySelector("span").textContent =
    `Idle detection: ${debugEnabled ? "On" : "Off"}`;
  elements["acknowledge-button"].hidden = !payload.pending_acknowledgement;
}

function render(payload) {
  state = payload;
  if (Number.isFinite(payload.server_time)) {
    serverClockOffset = payload.server_time - Date.now() / 1000;
  }
  setConnection(payload.connection);
  renderBanner(payload);
  elements["printer-state"].textContent = payload.printer.state;
  elements.uptime.textContent = formatDuration(payload.printer.uptime_seconds);
  elements["metric-interval"].textContent = `${payload.loop_interval}s`;
  renderFrame(payload);
  renderDetection(payload);
  renderControls(payload);
  updateCountdown();
}

function updateCountdown() {
  if (!state?.frame) {
    elements["next-scan"].textContent = "—";
    elements["cadence-fill"].style.width = "0%";
    return;
  }
  const serverNow = Date.now() / 1000 + serverClockOffset;
  const elapsed = serverNow - state.frame.captured_at;
  const remaining = Math.max(0, state.loop_interval - elapsed);
  const progress = Math.min(1, elapsed / state.loop_interval);
  elements["next-scan"].textContent = `${Math.ceil(remaining)}s`;
  elements["cadence-fill"].style.width = `${progress * 100}%`;
}

async function fetchState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
    schedulePoll(2000);
  } catch (error) {
    const disconnected = state || {
      printer: { state: "unknown", detail: "", uptime_seconds: 0 },
      connection: "disconnected",
      message: `Could not reach Scythe: ${error.message}`,
      loop_interval: 30,
      frame: null,
      current_detection: null,
      last_detection: null,
      pending_acknowledgement: false,
      debug_detection_enabled: false,
    };
    disconnected.connection = "disconnected";
    disconnected.message = `Could not reach Scythe: ${error.message}`;
    render(disconnected);
    schedulePoll(5000);
  }
}

function schedulePoll(delay) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(fetchState, delay);
}

async function control(path, button, successMessage, failureMessage, body = null) {
  const buttons = [
    elements["pause-button"], elements["resume-button"],
    elements["debug-detection-button"], elements["acknowledge-button"],
  ];
  buttons.forEach((item) => { item.disabled = true; });
  button.classList.add("busy");
  elements["control-result"].className = "control-result";
  elements["control-result"].textContent = "Sending command…";
  try {
    const headers = { "X-Scythe-Request": "1" };
    if (body) headers["Content-Type"] = "application/json";
    const response = await fetch(path, {
      method: "POST",
      headers,
      body: body ? JSON.stringify(body) : null,
    });
    const result = response.headers.get("content-type")?.includes("application/json")
      ? await response.json()
      : { ok: false };
    if (!response.ok || !result.ok) throw new Error(failureMessage);
    elements["control-result"].textContent = successMessage;
    await fetchState();
  } catch (error) {
    elements["control-result"].className = "control-result error";
    elements["control-result"].textContent = error.message || failureMessage;
    if (state) renderControls(state);
  } finally {
    button.classList.remove("busy");
  }
}

elements["pause-button"].addEventListener("click", () => control(
  "/api/pause", elements["pause-button"], "Printer paused.",
  "The printer was not printing or could not be paused."
));
elements["resume-button"].addEventListener("click", () => control(
  "/api/resume", elements["resume-button"], "Printer resumed.",
  "The printer was not paused or could not be resumed."
));
elements["debug-detection-button"].addEventListener("click", () => {
  const enabled = !Boolean(state?.debug_detection_enabled);
  control(
    "/api/debug-detection", elements["debug-detection-button"],
    `Idle detection ${enabled ? "enabled" : "disabled"}.`,
    "Could not change idle detection mode.", { enabled }
  );
});
elements["acknowledge-button"].addEventListener("click", () => control(
  "/api/acknowledge", elements["acknowledge-button"], "Detection acknowledged. Monitoring can continue.",
  "No detection is waiting for acknowledgement."
));
elements["overlay-toggle"].addEventListener("click", () => {
  overlayVisible = !overlayVisible;
  elements["overlay-toggle"].setAttribute("aria-pressed", String(overlayVisible));
  if (state) renderOverlay(state);
});

countdownTimer = setInterval(updateCountdown, 500);
window.addEventListener("beforeunload", () => clearInterval(countdownTimer));
fetchState();
