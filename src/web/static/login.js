"use strict";

document.getElementById("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const error = document.getElementById("login-error");
  error.textContent = "";
  const response = await fetch("api/login", {
    method: "POST",
    headers: { "X-Scythe-Request": "1", "Content-Type": "application/json" },
    body: JSON.stringify({ password: document.getElementById("login-password").value }),
  });
  if (response.ok) {
    location.href = "./";
  } else {
    error.textContent = (await response.json().catch(() => ({}))).error || "Could not log in.";
  }
});
