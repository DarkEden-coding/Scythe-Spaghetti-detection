"""Backend for the dashboard's first-run setup.

Finds the printer, tests the camera, and drives the optional system changes:
Tailscale and the Mainsail sidebar link. Anything that needs root goes through
the fixed-action helper that install.sh puts in /usr/local/lib/scythe, which
the Scythe user may run with sudo and nothing else.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import shutil
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from src.config import PrinterSettings
from src.errors import ScytheError

log = logging.getLogger(__name__)

HELPER = Path("/usr/local/lib/scythe/scythe-helper")
NGINX_SITES = (
    Path("/etc/nginx/sites-available/mainsail"),
    Path("/etc/nginx/sites-available/fluidd"),
)
#: Mainsail reads custom sidebar links from here.
NAVI_FILE = Path.home() / "printer_data" / "config" / ".theme" / "navi.json"
NAVI_ENTRY = {"title": "Scythe", "href": "/scythe/", "target": "_self", "position": 85}

#: Where Moonraker usually answers on the printer's own host: through the
#: Mainsail/Fluidd nginx first, then directly.
PRINTER_CANDIDATES = (
    "http://localhost/",
    "http://localhost:7125/",
    "http://127.0.0.1:7125/",
)

PBKDF2_ROUNDS = 200_000


class SetupError(ScytheError):
    """A setup step failed; the message is shown to the user."""


# --------------------------------------------------------------------------- #
# Printer
# --------------------------------------------------------------------------- #


def _client(url: str, webcam: str = ""):
    from src.printer.moonraker import MoonrakerClient

    return MoonrakerClient(
        PrinterSettings(url=url, webcam_name=webcam, request_timeout=4, max_retries=0)
    )


def list_webcams(url: str) -> list[str]:
    """Names of the cameras Moonraker knows, or raise if it can't be reached."""
    if not url.startswith(("http://", "https://")):
        raise SetupError("The address must start with http:// or https://.")
    with _client(url) as client:
        try:
            payload = client._get_json("server/webcams/list")
            webcams = payload["result"]["webcams"]
        except Exception as exc:  # noqa: BLE001 - any failure means "not Moonraker"
            raise SetupError(f"No Moonraker answered at {url}.") from exc
    return [str(cam.get("name")) for cam in webcams if cam.get("name")]


def find_printer(url: str = "") -> dict[str, Any]:
    """Try ``url`` or the usual local addresses; return the first that answers."""
    for candidate in [url] if url else PRINTER_CANDIDATES:
        try:
            return {"url": candidate, "webcams": list_webcams(candidate)}
        except SetupError:
            continue
    if url:
        raise SetupError(f"No Moonraker answered at {url}. Check the address.")
    raise SetupError(
        "Couldn't find Moonraker on this machine. Enter your printer's address, "
        "for example http://mainsailos.local/."
    )


def snapshot_jpeg(url: str, webcam: str) -> bytes:
    """One test frame from the chosen camera, as JPEG."""
    from src.errors import WebcamNotFound

    with _client(url, webcam) as client:
        try:
            image = client.get_snapshot()
        except WebcamNotFound as exc:
            raise SetupError(str(exc)) from exc
    if image is None:
        raise SetupError(f"The camera {webcam!r} didn't return an image.")
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Password
# --------------------------------------------------------------------------- #


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt),
                                 PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _scheme, rounds, salt, expected = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt), int(rounds)
        )
    except ValueError:
        return False
    return hmac.compare_digest(digest.hex(), expected)


# --------------------------------------------------------------------------- #
# System: root helper, Tailscale, Mainsail
# --------------------------------------------------------------------------- #


@dataclass
class System:
    """Probes and changes outside Scythe's own directory.

    Paths are attributes so tests can point them at a temporary directory.
    """

    helper: Path = HELPER
    nginx_sites: tuple[Path, ...] = NGINX_SITES
    navi_file: Path = NAVI_FILE

    # -- helper ------------------------------------------------------------ #

    def helper_available(self) -> bool:
        if not self.helper.exists() or shutil.which("sudo") is None:
            return False
        check = subprocess.run(
            ["sudo", "-n", "-l", str(self.helper)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return check.returncode == 0

    def run_helper(self, *args: str, timeout: float = 300) -> str:
        if not self.helper_available():
            raise SetupError(
                "Scythe isn't allowed to make system changes. Re-run the installer "
                "to set that up, or follow the manual steps in docs/."
            )
        result = subprocess.run(
            ["sudo", "-n", str(self.helper), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise SetupError(result.stderr.strip() or f"{args[0]} failed.")
        return result.stdout.strip()

    # -- Tailscale --------------------------------------------------------- #

    def tailscale(self) -> dict[str, Any]:
        """Install/login state and this machine's tailnet name."""
        if shutil.which("tailscale") is None:
            return {"installed": False, "state": "", "auth_url": "", "dns_name": ""}
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        try:
            status = json.loads(result.stdout)
        except ValueError:
            status = {}
        return {
            "installed": True,
            "state": status.get("BackendState", "Stopped"),
            "auth_url": status.get("AuthURL", ""),
            "dns_name": (status.get("Self") or {}).get("DNSName", "").rstrip("."),
        }

    # -- Mainsail ---------------------------------------------------------- #

    def nginx_site(self) -> Path | None:
        return next((site for site in self.nginx_sites if site.exists()), None)

    def mainsail(self) -> dict[str, Any]:
        site = self.nginx_site()
        linked = False
        if site is not None:
            try:
                linked = "location /scythe/" in site.read_text(encoding="utf-8")
            except OSError:
                linked = False
        return {
            "found": site is not None,
            "frontend": site.name if site else "",
            "linked": linked,
            "sidebar": self._navi_has_entry(),
        }

    def _read_navi(self) -> list[dict[str, Any]]:
        try:
            entries = json.loads(self.navi_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return entries if isinstance(entries, list) else []

    def _navi_has_entry(self) -> bool:
        return any(e.get("href") == NAVI_ENTRY["href"] for e in self._read_navi()
                   if isinstance(e, dict))

    def link_mainsail(self, port: int) -> None:
        self.run_helper("mainsail-link", str(port))
        if self.navi_file.parent.parent.exists():
            entries = [e for e in self._read_navi()
                       if not (isinstance(e, dict) and e.get("href") == NAVI_ENTRY["href"])]
            self.navi_file.parent.mkdir(exist_ok=True)
            self.navi_file.write_text(json.dumps([*entries, NAVI_ENTRY], indent=2) + "\n")

    def unlink_mainsail(self) -> None:
        self.run_helper("mainsail-unlink")
        entries = [e for e in self._read_navi()
                   if not (isinstance(e, dict) and e.get("href") == NAVI_ENTRY["href"])]
        if self.navi_file.exists():
            if entries:
                self.navi_file.write_text(json.dumps(entries, indent=2) + "\n")
            else:
                self.navi_file.unlink()

    def status(self, port: int) -> dict[str, Any]:
        tailscale = self.tailscale()
        mainsail = self.mainsail()
        tailnet_url = ""
        if tailscale["dns_name"]:
            # With the Mainsail link, HTTPS serves Mainsail and Scythe lives at
            # /scythe/; otherwise HTTPS serves Scythe directly.
            tailnet_url = f"https://{tailscale['dns_name']}/" + (
                "scythe/" if mainsail["linked"] else ""
            )
        return {
            "helper": self.helper_available(),
            "tailscale": tailscale,
            "tailnet_url": tailnet_url,
            "mainsail": mainsail,
            "port": port,
            "user": os.environ.get("USER", ""),
        }


__all__ = [
    "SetupError",
    "System",
    "check_password",
    "find_printer",
    "hash_password",
    "list_webcams",
    "snapshot_jpeg",
]
