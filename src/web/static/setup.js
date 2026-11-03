"use strict";

// First-run setup wizard. Opens by itself until setup is finished, and from
// Settings afterwards. Every step saves as you leave it.

(() => {
  const $ = (id) => document.getElementById(id);
  const dialog = $("setup-dialog");
  const steps = [...dialog.querySelectorAll(".setup-step")];
  const progress = [...$("setup-progress").children];
  const headers = { "X-Scythe-Request": "1", "Content-Type": "application/json" };
  let index = 0;
  let status = null;
  let pollTimer = null;

  async function request(url, body) {
    const response = await fetch(url, body === undefined
      ? { cache: "no-store" }
      : { method: "POST", headers, body: JSON.stringify(body) });
    const data = response.headers.get("content-type")?.includes("application/json")
      ? await response.json() : { ok: response.ok, error: await response.text() };
    if (!response.ok || data.ok === false) throw new Error(data.error || "Something went wrong.");
    return data;
  }

  function stepName() {
    return steps[index].dataset.step;
  }

  function show(next) {
    index = Math.max(0, Math.min(steps.length - 1, next));
    steps.forEach((step, i) => { step.hidden = i !== index; });
    progress.forEach((item, i) => {
      item.classList.toggle("done", i < index);
      item.classList.toggle("current", i === index);
    });
    $("setup-count").textContent = `${index + 1} of ${steps.length}`;
    $("setup-back").disabled = index === 0;
    $("setup-skip").hidden = !["alerts", "remote", "password"].includes(stepName());
    $("setup-next").textContent = stepName() === "done"
      ? (status?.setup_complete ? "Save and restart" : "Start monitoring") : "Next";
    $("setup-error").textContent = "";
    if (stepName() === "remote") refreshSystem();
    if (stepName() === "done") renderSummary();
  }

  // -- printer -------------------------------------------------------------- //

  async function searchPrinter(url = "") {
    const text = $("printer-search-text");
    text.textContent = url ? `Checking ${url}…` : "Looking for Moonraker on this machine…";
    $("setup-webcam").disabled = true;
    try {
      const found = await request("api/setup/printer/find", { url });
      $("setup-printer-url").value = found.url;
      const select = $("setup-webcam");
      select.replaceChildren(...found.webcams.map((name) => new Option(name, name)));
      select.disabled = !found.webcams.length;
      if (found.webcams.includes(status?.printer.webcam)) select.value = status.printer.webcam;
      text.textContent = found.webcams.length
        ? `Found your printer at ${found.url}.`
        : `Found your printer at ${found.url}, but it has no cameras. Add one in Mainsail or Fluidd, then search again.`;
      if (found.webcams.length) testSnapshot();
    } catch (error) {
      text.textContent = error.message;
    }
  }

  function testSnapshot() {
    const url = $("setup-printer-url").value.trim();
    const webcam = $("setup-webcam").value;
    const image = $("setup-snapshot");
    const note = $("setup-snapshot-text");
    if (!webcam) return;
    note.textContent = "Taking a test picture…";
    image.hidden = true;
    image.onload = () => { image.hidden = false; note.textContent = "If this shows your printer, you're good."; };
    image.onerror = () => { note.textContent = `The camera “${webcam}” didn't return a picture. Try another one.`; };
    image.src = `api/setup/printer/snapshot.jpg?${new URLSearchParams({ url, webcam, t: Date.now() })}`;
  }

  // -- system --------------------------------------------------------------- //

  function actionButton(label, action, primary = false) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `button ${primary ? "button-primary" : "button-ghost"}`;
    button.textContent = label;
    button.disabled = !status.system.helper || status.task.running;
    button.addEventListener("click", () => runAction(action));
    return button;
  }

  function renderSystem() {
    const { system, task } = status;
    $("setup-helper-missing").hidden = system.helper;
    $("system-error").textContent = task.error || "";
    const ts = system.tailscale;
    const link = $("tailscale-link");
    const actions = $("tailscale-actions");
    link.hidden = true;
    actions.replaceChildren();
    const busy = task.running && task.action.startsWith("tailscale");
    if (busy) {
      $("tailscale-text").textContent = {
        "tailscale-install": "Installing Tailscale. This can take a few minutes…",
        "tailscale-up": "Starting Tailscale…",
        "tailscale-serve": "Turning on the HTTPS address…",
        "tailscale-serve-off": "Turning off the HTTPS address…",
      }[task.action];
    } else if (!ts.installed) {
      $("tailscale-text").textContent = "Reach Scythe from anywhere, privately, without opening ports. Free for personal use.";
      actions.append(actionButton("Install Tailscale", "tailscale-install", true));
    } else if (ts.state !== "Running") {
      if (ts.auth_url) {
        $("tailscale-text").textContent = "Log in to add this machine to your tailnet. This page updates when you're done.";
        link.hidden = false;
        link.href = ts.auth_url;
        link.className = "button button-primary";
        link.textContent = "Log in to Tailscale";
      } else {
        $("tailscale-text").textContent = "Tailscale is installed but not connected.";
        actions.append(actionButton("Connect", "tailscale-up", true));
      }
    } else if (status.public_url) {
      $("tailscale-text").textContent = "Open Scythe from any device on your tailnet:";
      link.hidden = false;
      link.href = status.public_url;
      link.className = "setup-url";
      link.textContent = status.public_url;
      actions.append(actionButton("Turn off", "tailscale-serve-off"));
    } else {
      $("tailscale-text").textContent = `Connected as ${ts.dns_name}. Turn on an HTTPS address for Scythe. Discord alerts will link to it.`;
      actions.append(actionButton("Turn on HTTPS address", "tailscale-serve", true));
    }

    const ms = system.mainsail;
    const msActions = $("mainsail-actions");
    msActions.replaceChildren();
    $("mainsail-card").hidden = !ms.found;
    if (task.running && task.action.startsWith("mainsail")) {
      $("mainsail-text").textContent = "Updating the web server…";
    } else if (ms.linked) {
      $("mainsail-text").textContent = ms.sidebar
        ? "Scythe is in the Mainsail sidebar and at /scythe/ on your printer's address."
        : "Scythe is at /scythe/ on your printer's address.";
      msActions.append(actionButton("Remove", "mainsail-unlink"));
    } else {
      $("mainsail-text").textContent = "Add Scythe to Mainsail's sidebar, so it opens at /scythe/ on your printer's address. Mainsail updates keep working.";
      msActions.append(actionButton("Add to sidebar", "mainsail-link", true));
    }
  }

  async function refreshSystem() {
    try {
      status = await request("api/setup");
      renderSystem();
    } catch (error) {
      $("system-error").textContent = error.message;
    }
    // Keep polling while something runs or a Tailscale login is pending.
    clearTimeout(pollTimer);
    const waiting = status && (status.task.running
      || (status.system.tailscale.installed && status.system.tailscale.state === "NeedsLogin"));
    if (dialog.open && stepName() === "remote" && waiting) {
      pollTimer = setTimeout(refreshSystem, 2000);
    }
  }

  async function runAction(action) {
    $("system-error").textContent = "";
    try {
      await request("api/setup/system", { action });
    } catch (error) {
      $("system-error").textContent = error.message;
    }
    refreshSystem();
  }

  // -- summary -------------------------------------------------------------- //

  function renderSummary() {
    const items = [
      `Printer: ${$("setup-printer-url").value || status.printer.url} · camera “${$("setup-webcam").value || status.printer.webcam}”`,
      `Detection: ${document.querySelector("[name=setup-backend]:checked")?.value === "decision" ? "AI decision model" : "local model"}`,
      `Discord: ${$("setup-discord-token").value || discordSaved ? "on" : "off (dashboard only)"}`,
      `Remote: ${status.public_url || "this network only"}`,
      `Password: ${status.password_set ? "on" : "off"}`,
    ];
    $("setup-summary").replaceChildren(...items.map((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      return li;
    }));
  }

  // -- saving each step ----------------------------------------------------- //

  let discordSaved = false;

  async function saveStep() {
    switch (stepName()) {
      case "printer": {
        const webcam = $("setup-webcam").value;
        if (!webcam) throw new Error("Find your printer and choose a camera first.");
        await request("api/setup/printer", { url: $("setup-printer-url").value.trim(), webcam });
        break;
      }
      case "detection": {
        const backend = document.querySelector("[name=setup-backend]:checked")?.value || "yolo";
        const changes = { backend };
        const key = $("setup-api-key").value.trim();
        if (key) changes.api_key = key;
        await request("api/settings", changes);
        break;
      }
      case "alerts": {
        const token = $("setup-discord-token").value.trim();
        const channel = $("setup-discord-channel").value.trim();
        const user = $("setup-discord-user").value.trim();
        if (!token && !channel && !user) break;
        const changes = { discord_channel_id: channel || 0, discord_user_id: user || 0 };
        if (token) changes.discord_token = token;
        await request("api/settings", changes);
        discordSaved = true;
        break;
      }
      case "password": {
        const password = $("setup-password").value;
        if (!password) break;
        if (password !== $("setup-password-2").value) throw new Error("The passwords don't match.");
        await request("api/setup/password", { password });
        status.password_set = true;
        break;
      }
      case "done":
        await finish();
        return;
      default:
        break;
    }
  }

  async function finish() {
    const next = $("setup-next");
    next.disabled = true;
    next.textContent = "Restarting…";
    await request("api/setup/finish", {});
    localStorage.setItem("scythe.discordNudgeDismissed", discordSaved ? "1" : "0");
    const poll = setInterval(async () => {
      try {
        if ((await fetch("api/healthz", { cache: "no-store" })).ok) location.reload();
      } catch { /* restarting */ }
    }, 1500);
    setTimeout(() => clearInterval(poll), 90000);
  }

  // -- wiring --------------------------------------------------------------- //

  $("setup-next").addEventListener("click", async () => {
    const button = $("setup-next");
    button.disabled = true;
    try {
      await saveStep();
      if (stepName() !== "done") show(index + 1);
    } catch (error) {
      $("setup-error").textContent = error.message;
    } finally {
      if (stepName() !== "done" || !$("setup-next").textContent.startsWith("Restarting")) {
        button.disabled = false;
      }
    }
  });
  $("setup-back").addEventListener("click", () => show(index - 1));
  $("setup-skip").addEventListener("click", () => show(index + 1));
  $("setup-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("cancel", (event) => {
    if (!status?.setup_complete) event.preventDefault(); // setup can't be dismissed
  });
  $("setup-printer-search").addEventListener("click", () => searchPrinter($("setup-printer-url").value.trim()));
  $("setup-webcam").addEventListener("change", testSnapshot);
  document.querySelectorAll("[name=setup-backend]").forEach((radio) => {
    radio.addEventListener("change", () => {
      $("setup-api-key-field").hidden = radio.value !== "decision" || !radio.checked;
    });
  });
  $("setup-open-guide").addEventListener("click", () => window.scytheOpenGuide?.($("setup-discord-token").value.trim()));
  $("setup-password-remove").addEventListener("click", async () => {
    await request("api/setup/password", { password: "" });
    status.password_set = false;
    renderPassword();
  });

  function renderPassword() {
    $("setup-password-state").textContent = status.password_set
      ? "A password is set. Enter a new one to change it." : "";
    $("setup-password-remove").hidden = !status.password_set;
  }

  async function open() {
    status = await request("api/setup");
    const settings = await request("api/settings").catch(() => null);
    $("setup-close").hidden = !status.setup_complete;
    $("setup-printer-url").value = status.printer.url;
    const backend = settings?.values.backend || "yolo";
    document.querySelectorAll("[name=setup-backend]").forEach((radio) => {
      radio.checked = radio.value === backend;
    });
    $("setup-api-key-field").hidden = backend !== "decision";
    // 32-bit ARM has no ONNX Runtime, so only the decision model works there.
    const local = document.querySelector("[name=setup-backend][value=yolo]");
    local.disabled = !status.local_model_available;
    local.closest(".choice").title = local.disabled
      ? "The local model needs a 64-bit system." : "";
    if (local.disabled && local.checked) {
      document.querySelector("[name=setup-backend][value=decision]").checked = true;
      $("setup-api-key-field").hidden = false;
    }
    $("setup-api-key").placeholder = settings?.api_key_set ? `Saved key ${settings.api_key_hint}` : "sk-or-v1-…";
    discordSaved = Boolean(settings?.discord_token_set);
    $("setup-discord-hint").textContent = discordSaved ? `Saved token ${settings.discord_token_hint}. Leave blank to keep it.` : "";
    $("setup-discord-channel").value = settings?.values.discord_channel_id || "";
    $("setup-discord-user").value = settings?.values.discord_user_id || "";
    renderPassword();
    document.getElementById("discord-nudge").hidden = true;
    show(0);
    dialog.showModal();
    searchPrinter(status.setup_complete ? status.printer.url : "");
  }

  $("rerun-setup").addEventListener("click", () => {
    $("settings-dialog").close();
    open();
  });
  $("logout-button").addEventListener("click", async () => {
    await request("api/logout", {});
    location.href = "login";
  });

  request("api/setup").then((data) => {
    $("logout-button").hidden = !data.password_set;
    if (!data.setup_complete) open();
  }).catch(() => {});
})();
