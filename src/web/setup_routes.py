"""HTTP routes for first-run setup and the optional dashboard password."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib.util
import logging
import secrets
from time import time
from typing import Any

from aiohttp import web

from src.config import ConfigError, save_overrides
from src.settings_control import SettingsControl
from src.setup_service import (
    SetupError,
    System,
    check_password,
    find_printer,
    hash_password,
    snapshot_jpeg,
)

log = logging.getLogger(__name__)

COOKIE = "scythe_session"
SESSION_SECONDS = 30 * 24 * 3600
#: Reachable without logging in.
PUBLIC_PATHS = frozenset(
    {"/login", "/login.js", "/style.css", "/logo.svg", "/api/login", "/api/healthz"}
)
SYSTEM_ACTIONS = frozenset(
    {
        "tailscale-install",
        "tailscale-up",
        "tailscale-serve",
        "tailscale-serve-off",
        "mainsail-link",
        "mainsail-unlink",
    }
)


async def _json(request: web.Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except ValueError as exc:
        raise web.HTTPBadRequest(text="Expected a JSON request body.") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="Expected a JSON object.")
    return payload


def _error(message: str, status: int = 400) -> web.Response:
    return web.json_response({"ok": False, "error": message}, status=status)


class SetupRoutes:
    """Setup wizard API plus login. ``restart`` replaces the process."""

    def __init__(self, control: SettingsControl, system: System | None = None,
                 restart: Any = None) -> None:
        self._control = control
        self._system = system or System()
        self._restart = restart
        #: The running or last system action, for the wizard to poll.
        self._task: dict[str, Any] = {"action": "", "running": False, "error": ""}

    @property
    def _settings(self):
        return self._control.settings

    def routes(self) -> list[web.RouteDef]:
        return [
            web.post("/api/login", self._login),
            web.post("/api/logout", self._logout),
            web.get("/api/setup", self._status),
            web.post("/api/setup/printer/find", self._find_printer),
            web.get("/api/setup/printer/snapshot.jpg", self._snapshot),
            web.post("/api/setup/printer", self._save_printer),
            web.post("/api/setup/system", self._system_action),
            web.post("/api/setup/password", self._set_password),
            web.post("/api/setup/finish", self._finish),
        ]

    # -- sessions ---------------------------------------------------------- #

    def _sign(self, expiry: int) -> str:
        secret = self._settings.web.session_secret.encode()
        mac = hmac.new(secret, str(expiry).encode(), hashlib.sha256).hexdigest()
        return f"{expiry}.{mac}"

    def is_authenticated(self, request: web.Request) -> bool:
        web_settings = self._settings.web
        if not web_settings.password_hash:
            return True
        cookie = request.cookies.get(COOKIE, "")
        expiry, _, _mac = cookie.partition(".")
        if not expiry.isdigit() or int(expiry) < time():
            return False
        return hmac.compare_digest(cookie, self._sign(int(expiry)))

    def _set_session(self, response: web.Response, request: web.Request) -> None:
        expiry = int(time()) + SESSION_SECONDS
        response.set_cookie(
            COOKIE,
            self._sign(expiry),
            max_age=SESSION_SECONDS,
            httponly=True,
            samesite="Strict",
            secure=request.secure,
        )

    @web.middleware
    async def auth_middleware(self, request: web.Request, handler: Any):
        if request.path in PUBLIC_PATHS or self.is_authenticated(request):
            return await handler(request)
        if request.path.startswith("/api/"):
            return _error("Log in first.", status=401)
        # Relative, so it also works under Mainsail's /scythe/ path.
        raise web.HTTPFound("login")

    async def _login(self, request: web.Request) -> web.Response:
        payload = await _json(request)
        password = str(payload.get("password", ""))
        stored = self._settings.web.password_hash
        if not stored:
            return web.json_response({"ok": True})
        ok = await asyncio.to_thread(check_password, password, stored)
        if not ok:
            await asyncio.sleep(1)  # slow down guessing
            return _error("Wrong password.", status=401)
        response = web.json_response({"ok": True})
        self._set_session(response, request)
        return response

    async def _logout(self, _request: web.Request) -> web.Response:
        response = web.json_response({"ok": True})
        response.del_cookie(COOKIE)
        return response

    # -- status ------------------------------------------------------------ #

    async def _status(self, _request: web.Request) -> web.Response:
        settings = self._settings
        system = await asyncio.to_thread(self._system.status, settings.web.port)
        return web.json_response(
            {
                "setup_complete": settings.setup_complete,
                "printer": {
                    "url": settings.printer.url,
                    "webcam": settings.printer.webcam_name,
                },
                "password_set": bool(settings.web.password_hash),
                "public_url": settings.web.public_url,
                "system": system,
                "local_model_available": importlib.util.find_spec("onnxruntime")
                is not None,
                "task": self._task,
            },
            headers={"Cache-Control": "no-store"},
        )

    async def _save(self, changes: dict[str, Any]) -> None:
        await asyncio.to_thread(save_overrides, changes, self._control.config_path)

    # -- printer ----------------------------------------------------------- #

    async def _find_printer(self, request: web.Request) -> web.Response:
        payload = await _json(request)
        url = str(payload.get("url", "")).strip()
        try:
            found = await asyncio.to_thread(find_printer, url)
        except SetupError as exc:
            return _error(str(exc))
        return web.json_response({"ok": True, **found})

    async def _snapshot(self, request: web.Request) -> web.Response:
        url = request.query.get("url", "")
        webcam = request.query.get("webcam", "")
        try:
            jpeg = await asyncio.to_thread(snapshot_jpeg, url, webcam)
        except SetupError as exc:
            return _error(str(exc))
        return web.Response(body=jpeg, content_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

    async def _save_printer(self, request: web.Request) -> web.Response:
        payload = await _json(request)
        url = str(payload.get("url", "")).strip()
        webcam = str(payload.get("webcam", "")).strip()
        if not webcam:
            return _error("Choose a camera.")
        printer = self._settings.printer
        old = (printer.url, printer.webcam_name)
        printer.url, printer.webcam_name = url, webcam
        try:
            printer.validate()
        except ConfigError as exc:
            printer.url, printer.webcam_name = old
            return _error(str(exc))
        await self._save({"printer": {"url": url, "webcam_name": webcam}})
        if self._settings.setup_complete and old != (url, webcam):
            self._control.restart_required = True
        return web.json_response({"ok": True})

    # -- system ------------------------------------------------------------ #

    async def _system_action(self, request: web.Request) -> web.Response:
        payload = await _json(request)
        action = payload.get("action")
        if action not in SYSTEM_ACTIONS:
            return _error("Unknown action.")
        if self._task["running"]:
            return _error("Another change is still running.", status=409)
        self._task = {"action": action, "running": True, "error": ""}
        # Installs can take minutes, longer than a proxy waits for a response,
        # so the wizard polls /api/setup for the result.
        asyncio.get_running_loop().create_task(self._run_action(action))
        return web.json_response({"ok": True, "task": self._task})

    async def _run_action(self, action: str) -> None:
        port = self._settings.web.port
        system = self._system
        try:
            if action == "mainsail-link":
                await asyncio.to_thread(system.link_mainsail, port)
            elif action == "mainsail-unlink":
                await asyncio.to_thread(system.unlink_mainsail)
            elif action == "tailscale-serve":
                # Behind the Mainsail link, serve nginx so Mainsail and Scythe
                # share the HTTPS address; otherwise serve Scythe directly.
                linked = (await asyncio.to_thread(system.mainsail))["linked"]
                await asyncio.to_thread(
                    system.run_helper, action, "80" if linked else str(port)
                )
                status = await asyncio.to_thread(system.status, port)
                await self._set_public_url(status["tailnet_url"])
            elif action == "tailscale-serve-off":
                await asyncio.to_thread(system.run_helper, action)
                await self._set_public_url("")
            else:
                await asyncio.to_thread(system.run_helper, action)
            self._task = {"action": action, "running": False, "error": ""}
        except (SetupError, OSError) as exc:
            log.warning("Setup action %s failed: %s", action, exc)
            self._task = {"action": action, "running": False, "error": str(exc)}

    async def _set_public_url(self, url: str) -> None:
        self._settings.web.public_url = url
        await self._save({"web": {"public_url": url}})

    # -- password and finish ----------------------------------------------- #

    async def _set_password(self, request: web.Request) -> web.Response:
        payload = await _json(request)
        password = str(payload.get("password", ""))
        web_settings = self._settings.web
        if password and len(password) < 8:
            return _error("Use at least 8 characters.")
        if not password:
            web_settings.password_hash = ""
            await self._save({"web": {"password_hash": ""}})
            return web.json_response({"ok": True, "password_set": False})
        web_settings.password_hash = await asyncio.to_thread(hash_password, password)
        # A new secret also logs out every other browser.
        web_settings.session_secret = secrets.token_hex(32)
        await self._save(
            {
                "web": {
                    "password_hash": web_settings.password_hash,
                    "session_secret": web_settings.session_secret,
                }
            }
        )
        response = web.json_response({"ok": True, "password_set": True})
        self._set_session(response, request)
        return response

    async def _finish(self, _request: web.Request) -> web.Response:
        self._settings.setup_complete = True
        await self._save({"setup_complete": True})
        if self._restart is not None:
            log.info("Setup finished; restarting to start monitoring.")
            asyncio.get_running_loop().call_later(0.5, self._restart)
        return web.json_response({"ok": True})


__all__ = ["SetupRoutes"]
