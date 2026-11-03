"use strict";

const elements = Object.fromEntries(
  [
    "printer-pill", "printer-state", "system-banner", "system-message",
    "camera-empty", "camera-image", "detection-overlay", "frame-chip", "cadence-fill",
    "overlay-toggle", "verdict", "verdict-eyebrow", "verdict-title", "verdict-detail",
    "verdict-stat", "verdict-confidence", "last-detection", "last-detection-summary",
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

function formatTime(timestamp) {
  if (!timestamp) return "—";
  return new Date(timestamp * 1000).toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

function formatShortTime(timestamp) {
  return new Date(timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "—";
}

function renderStatus(payload) {
  // Monitor problems outrank printer state; otherwise the pill shows the printer.
  const problems = {
    starting: ["Connecting", ""],
    camera_unavailable: ["Camera unavailable", "warn"],
    stopped: ["Monitor stopped", "bad"],
    error: ["Monitor error", "bad"],
    disconnected: ["Offline", "bad"],
  };
  const printerTones = { printing: "good", paused: "warn", error: "bad" };
  const [label, tone] = problems[payload.connection]
    || [payload.printer.state, printerTones[payload.printer.state] || ""];
  elements["printer-state"].textContent = label;
  elements["printer-pill"].className = `status-pill ${tone}`.trim();
}

function renderBanner(payload) {
  const frameAge = payload.frame && Number.isFinite(payload.server_time)
    ? payload.server_time - payload.frame.captured_at
    : 0;
  const staleFrame = frameAge > Math.max(60, payload.loop_interval * 2);
  const unavailable = ["error", "camera_unavailable", "stopped", "disconnected"].includes(payload.connection);
  elements["system-banner"].hidden = !unavailable && !staleFrame;
  elements["system-message"].textContent = staleFrame && !unavailable
    ? `Camera frame is stale (${Math.floor(frameAge)} seconds old).`
    : payload.message || "Scythe is unavailable.";
  elements["camera-image"].classList.toggle("stale", unavailable || staleFrame);
}

function renderFrame(payload) {
  if (!payload.frame) return;
  elements["camera-empty"].hidden = true;
  elements["camera-image"].hidden = false;
  elements["frame-chip"].hidden = false;
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
  overlay.setAttribute("preserveAspectRatio", "xMidYMid meet");

  detection.boxes.forEach((box) => {
    const width = Math.max(1, box.x2 - box.x1);
    const height = Math.max(1, box.y2 - box.y1);
    const labelHeight = Math.max(22, payload.frame.height * 0.035);
    const fontSize = Math.max(13, payload.frame.height * 0.022);
    const labelWidth = Math.max(52, fontSize * 3.4);
    const labelY = Math.max(0, box.y1 - labelHeight - 4);
    const radius = Math.max(3, payload.frame.width * 0.004);
    const group = svgElement("g");
    group.append(
      svgElement("rect", {
        x: box.x1, y: box.y1, width, height, rx: radius,
        fill: "none", stroke: "#ffbd22", "stroke-width": Math.max(2, payload.frame.width * 0.002),
      }),
      svgElement("rect", {
        x: box.x1, y: labelY, width: labelWidth, height: labelHeight, rx: radius,
        fill: "#ffbd22",
      })
    );
    const label = svgElement("text", {
      x: box.x1 + labelWidth / 2,
      y: labelY + labelHeight * 0.7,
      fill: "#171204",
      "text-anchor": "middle",
      "font-family": "system-ui, sans-serif",
      "font-size": fontSize,
      "font-weight": "700",
    });
    label.textContent = formatPercent(box.confidence);
    group.append(label);
    overlay.append(group);
  });
}

function setVerdict(tone, eyebrow, title, detail, confidence = null) {
  elements.verdict.className = `card verdict ${tone}`.trim();
  elements["verdict-eyebrow"].textContent = eyebrow;
  elements["verdict-title"].textContent = title;
  elements["verdict-detail"].textContent = detail;
  elements["verdict-stat"].hidden = confidence === null;
  elements["verdict-confidence"].textContent = confidence === null ? "" : formatPercent(confidence);
}

function renderDetection(payload) {
  const current = payload.current_detection;
  const latest = payload.last_detection;

  if (current) {
    const found = detectionSummary(current);
    const title = `Possible ${(current.label || "spaghetti").toLowerCase()}`;
    if (current.debug) {
      setVerdict("debug", "Idle detection", title, `${found} · no alert sent`, current.max_confidence);
    } else {
      setVerdict("alert", "Failure detected", title,
        current.paused ? `${found} · print paused automatically`
          : `${found} · ${formatTime(current.detected_at)}`,
        current.max_confidence);
    }
  } else if (payload.frame) {
    setVerdict("clear", "All clear", "Looks good", "No failure in the latest scan.");
  } else {
    setVerdict("", "Waiting", "No inspection yet", "Results appear after the first scan.");
  }

  renderOverlay(payload);

  // While an alert is showing, the latest detection is that same alert.
  const showPrevious = Boolean(latest) && !current;
  elements["last-detection"].hidden = !showPrevious;
  if (showPrevious) {
    elements["last-detection-summary"].textContent =
      `${latest.label || "Spaghetti"} · ${formatShortTime(latest.detected_at)} · ${formatPercent(latest.max_confidence)}`;
  }
}

const scoreNames = { any_failure: "any failure", layer_shift: "layer shift" };

// Decision models report a score per question; YOLO reports regions.
function detectionSummary(detection) {
  if (detection.scores?.length) {
    return detection.scores
      .map(({ name, score }) => `${scoreNames[name] || name} ${formatPercent(score)}`)
      .join(" · ");
  }
  return `${detection.count} ${detection.count === 1 ? "region" : "regions"} flagged`;
}

function renderControls(payload) {
  const printerState = payload.printer.state;
  elements["pause-button"].hidden = printerState !== "printing";
  elements["resume-button"].hidden = printerState !== "paused";
  elements["acknowledge-button"].hidden = !payload.pending_acknowledgement;
  [elements["pause-button"], elements["resume-button"], elements["acknowledge-button"],
    elements["debug-detection-button"]].forEach((button) => { button.disabled = false; });
  elements["debug-detection-button"].setAttribute(
    "aria-checked", String(Boolean(payload.debug_detection_enabled))
  );
}

function render(payload) {
  state = payload;
  if (Number.isFinite(payload.server_time)) {
    serverClockOffset = payload.server_time - Date.now() / 1000;
  }
  renderStatus(payload);
  renderBanner(payload);
  renderFrame(payload);
  renderDetection(payload);
  renderControls(payload);
  updateCountdown();
}

function updateCountdown() {
  if (!state?.frame) {
    elements["cadence-fill"].style.width = "0%";
    return;
  }
  const serverNow = Date.now() / 1000 + serverClockOffset;
  const elapsed = serverNow - state.frame.captured_at;
  const remaining = Math.max(0, state.loop_interval - elapsed);
  const progress = Math.min(1, elapsed / state.loop_interval);
  const time = document.createTextNode(`${formatTime(state.frame.captured_at)} `);
  const next = document.createElement("span");
  next.textContent = remaining > 0 ? `· next in ${Math.ceil(remaining)}s` : "· scanning…";
  elements["frame-chip"].replaceChildren(time, next);
  elements["cadence-fill"].style.width = `${progress * 100}%`;
}

async function fetchState() {
  try {
    const response = await fetch("api/state", { cache: "no-store" });
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
  "api/pause", elements["pause-button"], "Printer paused.",
  "The printer was not printing or could not be paused."
));
elements["resume-button"].addEventListener("click", () => control(
  "api/resume", elements["resume-button"], "Printer resumed.",
  "The printer was not paused or could not be resumed."
));
elements["debug-detection-button"].addEventListener("click", () => {
  const enabled = !Boolean(state?.debug_detection_enabled);
  control(
    "api/debug-detection", elements["debug-detection-button"],
    `Idle detection ${enabled ? "enabled" : "disabled"}.`,
    "Could not change idle detection mode.", { enabled }
  );
});
elements["acknowledge-button"].addEventListener("click", () => control(
  "api/acknowledge", elements["acknowledge-button"], "Alert dismissed. Monitoring continues.",
  "No detection is waiting for acknowledgement."
));
elements["overlay-toggle"].addEventListener("click", () => {
  overlayVisible = !overlayVisible;
  elements["overlay-toggle"].setAttribute("aria-pressed", String(overlayVisible));
  if (state) renderOverlay(state);
});

elements["camera-image"].addEventListener("error", () => {
  elements["system-banner"].hidden = false;
  elements["system-message"].textContent = "The latest camera frame could not be displayed.";
  elements["camera-image"].classList.add("stale");
});

countdownTimer = setInterval(updateCountdown, 500);
window.addEventListener("beforeunload", () => clearInterval(countdownTimer));
fetchState();

// -- settings panel --------------------------------------------------------- //

const settingsUi = Object.fromEntries(
  [
    "settings-button", "settings-dialog", "settings-form", "settings-close", "settings-cancel",
    "settings-save", "settings-error", "settings-estimate", "api-key-hint", "model-options",
    "crop-picker", "crop-image", "crop-rect", "crop-empty", "crop-label", "crop-clear", "env-note",
    "discord-token-hint", "restart-banner", "restart-button", "discord-nudge", "discord-nudge-close",
    "discord-guide", "guide-close", "guide-back", "guide-next", "guide-count", "invite-link", "invite-text",
  ].map((id) => [id, document.getElementById(id)])
);
const settingsForm = settingsUi["settings-form"];
let settingsData = null;
let cropBox = null; // [left, top, right, bottom] as fractions, or null for the full frame

const fieldLabels = {
  capture_interval: "capture interval", detection_interval: "detection interval",
  backend: "backend", provider: "provider", base_url: "endpoint", api_key: "API key",
  model: "model", threshold: "alert threshold", crop: "crop", image_max_side: "image size",
  yolo_confidence: "minimum confidence", pause_on_failure: "pause on failure",
  discord_token: "Discord bot token", discord_channel_id: "Discord channel",
  discord_user_id: "Discord user", discord_status_mode: "status messages",
};

function formatMoney(value) {
  if (value === 0) return "free";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

function parseCrop(text) {
  const parts = String(text || "").split(",").map(Number);
  return parts.length === 4 && parts.every(Number.isFinite) ? parts : null;
}

function formatCrop(box) {
  return box ? box.map((value) => value.toFixed(3)).join(",") : "";
}

function renderCrop() {
  const rect = settingsUi["crop-rect"];
  rect.hidden = !cropBox;
  settingsUi["crop-label"].textContent = cropBox
    ? `Cropped to ${Math.round((cropBox[2] - cropBox[0]) * 100)}% × ${Math.round((cropBox[3] - cropBox[1]) * 100)}% of the frame`
    : "Full frame";
  if (!cropBox) return;
  const [left, top, right, bottom] = cropBox;
  Object.assign(rect.style, {
    left: `${left * 100}%`, top: `${top * 100}%`,
    width: `${(right - left) * 100}%`, height: `${(bottom - top) * 100}%`,
  });
}

function formValue(name) {
  const field = settingsForm.elements[name];
  if (field instanceof RadioNodeList) return field.value;
  return field.type === "checkbox" ? field.checked : field.value;
}

function updateSettingsVisibility() {
  const backend = formValue("backend");
  const provider = formValue("provider");
  settingsForm.querySelectorAll("[data-backend]").forEach((node) => {
    node.hidden = node.dataset.backend !== backend;
  });
  settingsForm.querySelectorAll("[data-provider]").forEach((node) => {
    node.hidden = node.dataset.provider !== provider;
  });
  settingsForm.elements.base_url.placeholder = settingsData?.providers?.[provider] || "";
  renderEstimate();
}

function renderEstimate() {
  const box = settingsUi["settings-estimate"];
  if (!settingsData || formValue("backend") !== "decision") return;
  const capture = Number(formValue("capture_interval"));
  const detection = Number(formValue("detection_interval")) || 0;
  if (!(capture > 0)) {
    box.textContent = "Enter a capture interval to see the estimate.";
    return;
  }
  const perHour = 3600 / Math.max(capture, detection);
  const model = formValue("model").trim();
  const saved = settingsData.values;
  let cost = null;
  let source = "";
  if (formValue("provider") !== "openrouter") {
    cost = 0;
  } else if (
    settingsData.estimate?.source === "measured" && model === saved.model
    && saved.backend === "decision" && saved.provider === "openrouter"
  ) {
    cost = settingsData.estimate.cost_per_request;
    source = "from this run's requests";
  } else {
    const known = settingsData.models.find((item) => item.id === model);
    if (known) {
      cost = known.cost_per_request;
      source = "measured in testing at 1240 px";
    }
  }
  const rate = `<b>${Math.round(perHour)}</b> detections per print-hour`;
  box.innerHTML = cost === null
    ? `${rate}. No price data for this model yet; it appears after the first detections.`
    : cost === 0
      ? `${rate}. Self-hosted, so no API cost.`
      : `${rate} ≈ <b>${formatMoney(perHour * cost)}</b> per print-hour (${formatMoney(perHour * cost * 24)} per 24 h of printing). Estimate ${source}.`;
}

function fillSettings(data) {
  settingsData = data;
  const values = data.values;
  settingsUi["model-options"].replaceChildren(...data.models.map((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.label = model.label;
    return option;
  }));
  settingsForm.elements.backend.value = values.backend;
  settingsForm.elements.pause_on_failure.checked = values.pause_on_failure;
  ["provider", "base_url", "model", "threshold", "yolo_confidence", "capture_interval", "detection_interval"]
    .forEach((name) => { settingsForm.elements[name].value = values[name]; });
  settingsForm.elements.api_key.value = "";
  settingsForm.elements.api_key.placeholder = data.api_key_set ? `Saved key ${data.api_key_hint}` : "sk-or-v1-…";
  settingsUi["api-key-hint"].textContent = data.api_key_set
    ? "Leave blank to keep the saved key."
    : "Required for OpenRouter. Stored in settings.json on this machine.";
  ["discord_channel_id", "discord_user_id"].forEach((name) => {
    settingsForm.elements[name].value = values[name] || "";
  });
  settingsForm.elements.discord_status_mode.value = values.discord_status_mode;
  settingsForm.elements.discord_token.value = "";
  settingsForm.elements.discord_token.placeholder = data.discord_token_set
    ? `Saved token ${data.discord_token_hint}` : "Leave empty to use only this dashboard";
  settingsUi["discord-token-hint"].textContent = data.discord_token_set
    ? "Leave blank to keep the saved token." : "";
  settingsUi["restart-banner"].hidden = !data.restart_required;
  settingsUi["discord-nudge"].hidden = data.discord_token_set
    || localStorage.getItem("scythe.discordNudgeDismissed") === "1";
  cropBox = parseCrop(values.crop);
  renderCrop();

  const locked = data.env_locked.map((name) => fieldLabels[name] || name);
  settingsUi["env-note"].hidden = !locked.length;
  settingsUi["env-note"].textContent = locked.length
    ? `Set by environment variables: ${locked.join(", ")}. Changes apply now, but the environment wins after a restart.`
    : "";
  updateSettingsVisibility();
}

async function openSettings() {
  settingsUi["settings-error"].textContent = "";
  try {
    const response = await fetch("api/settings", { cache: "no-store" });
    if (!response.ok) throw new Error(await response.text());
    fillSettings(await response.json());
  } catch (error) {
    settingsUi["settings-error"].textContent = `Could not load settings: ${error.message}`;
  }
  const hasFrame = Boolean(state?.frame);
  settingsUi["crop-image"].hidden = !hasFrame;
  settingsUi["crop-empty"].hidden = hasFrame;
  if (hasFrame) settingsUi["crop-image"].src = state.frame.url;
  settingsUi["settings-dialog"].showModal();
}

function collectSettings() {
  const changes = {
    backend: formValue("backend"),
    pause_on_failure: formValue("pause_on_failure"),
    provider: formValue("provider"),
    base_url: formValue("base_url").trim(),
    model: formValue("model").trim(),
    threshold: formValue("threshold"),
    crop: formatCrop(cropBox),
    yolo_confidence: formValue("yolo_confidence"),
    capture_interval: formValue("capture_interval"),
    detection_interval: formValue("detection_interval") || 0,
  };
  const key = formValue("api_key").trim();
  if (key) changes.api_key = key;
  const token = formValue("discord_token").trim();
  if (token) changes.discord_token = token;
  changes.discord_channel_id = formValue("discord_channel_id").trim() || 0;
  changes.discord_user_id = formValue("discord_user_id").trim() || 0;
  changes.discord_status_mode = formValue("discord_status_mode");
  return changes;
}

async function saveSettings(event) {
  event.preventDefault();
  const save = settingsUi["settings-save"];
  save.disabled = true;
  save.textContent = "Saving…";
  settingsUi["settings-error"].textContent = "";
  try {
    const response = await fetch("api/settings", {
      method: "POST",
      headers: { "X-Scythe-Request": "1", "Content-Type": "application/json" },
      body: JSON.stringify(collectSettings()),
    });
    const result = response.headers.get("content-type")?.includes("application/json")
      ? await response.json()
      : { ok: false, error: await response.text() };
    if (!result.ok) throw new Error(result.error || "Settings were not saved.");
    fillSettings(result.settings);
    settingsUi["settings-dialog"].close();
    elements["control-result"].className = "control-result";
    elements["control-result"].textContent = "Settings saved.";
    fetchState();
  } catch (error) {
    settingsUi["settings-error"].textContent = error.message;
  } finally {
    save.disabled = false;
    save.textContent = "Save";
  }
}

function cropPoint(event) {
  const bounds = settingsUi["crop-image"].getBoundingClientRect();
  return [
    Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
    Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
  ];
}

let cropStart = null;
let cropBefore = null;
settingsUi["crop-picker"].addEventListener("pointerdown", (event) => {
  if (settingsUi["crop-image"].hidden) return;
  settingsUi["crop-picker"].setPointerCapture(event.pointerId);
  cropStart = cropPoint(event);
  cropBefore = cropBox;
});
settingsUi["crop-picker"].addEventListener("pointermove", (event) => {
  if (!cropStart) return;
  const [x, y] = cropPoint(event);
  cropBox = [
    Math.min(cropStart[0], x), Math.min(cropStart[1], y),
    Math.max(cropStart[0], x), Math.max(cropStart[1], y),
  ];
  renderCrop();
});
settingsUi["crop-picker"].addEventListener("pointerup", () => {
  // A click or a sliver is almost certainly a mistake; keep the previous crop.
  if (cropBox && (cropBox[2] - cropBox[0] < 0.05 || cropBox[3] - cropBox[1] < 0.05)) {
    cropBox = cropBefore;
    renderCrop();
  }
  cropStart = null;
});
settingsUi["crop-clear"].addEventListener("click", () => {
  cropBox = null;
  renderCrop();
});

settingsUi["settings-button"].addEventListener("click", openSettings);

// -- Discord guide ---------------------------------------------------------- //

const guideSteps = [...settingsUi["discord-guide"].querySelectorAll(".guide-step")];
let guideStep = 0;

// A bot token's first segment is the bot's user id, which is also the
// application id the invite link needs.
function inviteUrl() {
  const token = settingsForm.elements.discord_token.value.trim();
  try {
    const id = atob(token.split(".")[0].replace(/-/g, "+").replace(/_/g, "/"));
    if (!/^\d{15,22}$/.test(id)) return null;
    return `https://discord.com/oauth2/authorize?client_id=${id}&scope=bot+applications.commands&permissions=117824`;
  } catch {
    return null;
  }
}

function showGuideStep(index) {
  guideStep = Math.max(0, Math.min(guideSteps.length - 1, index));
  guideSteps.forEach((step, i) => { step.hidden = i !== guideStep; });
  settingsUi["guide-count"].textContent = `${guideStep + 1} of ${guideSteps.length}`;
  settingsUi["guide-back"].disabled = guideStep === 0;
  settingsUi["guide-next"].textContent = guideStep === guideSteps.length - 1 ? "Done" : "Next";
  const url = inviteUrl();
  settingsUi["invite-link"].hidden = !url;
  if (url) {
    settingsUi["invite-link"].href = url;
    settingsUi["invite-text"].textContent = "Open the link, pick your server and click Authorize. It already asks for the permissions Scythe needs.";
  }
}

function openGuide() {
  showGuideStep(0);
  settingsUi["discord-guide"].showModal();
}

document.querySelectorAll("[data-open-guide]").forEach((button) => {
  button.addEventListener("click", openGuide);
});
settingsUi["guide-close"].addEventListener("click", () => settingsUi["discord-guide"].close());
settingsUi["guide-back"].addEventListener("click", () => showGuideStep(guideStep - 1));
settingsUi["guide-next"].addEventListener("click", () => {
  if (guideStep === guideSteps.length - 1) {
    settingsUi["discord-guide"].close();
    if (!settingsUi["settings-dialog"].open) openSettings();
  } else {
    showGuideStep(guideStep + 1);
  }
});
settingsUi["discord-nudge-close"].addEventListener("click", () => {
  localStorage.setItem("scythe.discordNudgeDismissed", "1");
  settingsUi["discord-nudge"].hidden = true;
});
settingsUi["restart-button"].addEventListener("click", async () => {
  const button = settingsUi["restart-button"];
  button.disabled = true;
  button.textContent = "Restarting…";
  try {
    await fetch("api/restart", { method: "POST", headers: { "X-Scythe-Request": "1" } });
  } catch { /* the connection drops as the process restarts */ }
  // Reload once the dashboard answers again.
  const poll = setInterval(async () => {
    try {
      if ((await fetch("api/healthz", { cache: "no-store" })).ok) location.reload();
    } catch { /* still restarting */ }
  }, 1500);
  setTimeout(() => clearInterval(poll), 60000);
});
fetch("api/settings", { cache: "no-store" })
  .then((response) => (response.ok ? response.json() : null))
  .then((data) => { if (data) fillSettings(data); })
  .catch(() => {});
settingsUi["settings-close"].addEventListener("click", () => settingsUi["settings-dialog"].close());
settingsUi["settings-cancel"].addEventListener("click", () => settingsUi["settings-dialog"].close());
settingsForm.addEventListener("submit", saveSettings);
settingsForm.addEventListener("input", (event) => {
  if (["backend", "provider"].includes(event.target.name)) updateSettingsVisibility();
  else renderEstimate();
});
settingsForm.addEventListener("change", (event) => {
  if (["backend", "provider"].includes(event.target.name)) updateSettingsVisibility();
});
