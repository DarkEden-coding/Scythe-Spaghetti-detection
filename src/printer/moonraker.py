"""Moonraker HTTP client.

Constructing the client performs no I/O. The webcam snapshot URL is discovered
on first use and cached, so a printer that is rebooting at startup delays the
first frame instead of killing the process at import time.
"""

from __future__ import annotations

import logging
import secrets
import threading
from io import BytesIO
from urllib.parse import urljoin

import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import PrinterSettings
from src.errors import PrinterUnavailable, WebcamNotFound
from src.printer.base import PrintState

log = logging.getLogger(__name__)

#: Idempotent reads are retried; POSTs (pause) deliberately are not.
_RETRY_METHODS = frozenset({"GET", "HEAD"})


class MoonrakerClient:
    """Talks to a Moonraker instance over HTTP.

    Thread-safe for the access pattern the monitor uses: snapshot-URL discovery
    is guarded by a lock, and :class:`requests.Session` is safe for concurrent
    requests once configured.
    """

    def __init__(
        self, settings: PrinterSettings, session: requests.Session | None = None
    ):
        self._settings = settings
        self._base_url = settings.base_url
        self._timeout = settings.request_timeout
        self._snapshot_url: str | None = None
        self._lock = threading.Lock()
        self._owns_session = session is None
        self._session = session or self._build_session(settings)

    # -- construction ------------------------------------------------------ #

    @staticmethod
    def _build_session(settings: PrinterSettings) -> requests.Session:
        """A session with connection reuse and bounded retries."""
        session = requests.Session()
        retry = Retry(
            total=settings.max_retries,
            connect=settings.max_retries,
            read=settings.max_retries,
            status=settings.max_retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=_RETRY_METHODS,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def __enter__(self) -> MoonrakerClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    # -- plumbing ---------------------------------------------------------- #

    def _url(self, path: str) -> str:
        return urljoin(self._base_url, path.lstrip("/"))

    def _get(self, path: str) -> requests.Response:
        url = self._url(path)
        try:
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise PrinterUnavailable(f"GET {url} failed: {exc}") from exc
        return response

    def _get_json(self, path: str) -> dict:
        response = self._get(path)
        try:
            payload = response.json()
        except ValueError as exc:
            raise PrinterUnavailable(
                f"GET {self._url(path)} returned non-JSON content"
            ) from exc
        if not isinstance(payload, dict):
            raise PrinterUnavailable(f"GET {self._url(path)} returned unexpected JSON")
        return payload

    # -- webcam ------------------------------------------------------------ #

    def _resolve_snapshot_url(self) -> str:
        """Look up the configured webcam's snapshot path, once."""
        payload = self._get_json("server/webcams/list")
        try:
            webcams = payload["result"]["webcams"]
        except (KeyError, TypeError) as exc:
            raise PrinterUnavailable(
                "Moonraker webcam list had an unexpected shape; is the "
                "printer URL correct?"
            ) from exc

        wanted = self._settings.webcam_name
        for webcam in webcams:
            if webcam.get("name") == wanted:
                url = webcam.get("snapshot_url")
                if not url:
                    raise PrinterUnavailable(
                        f"Webcam {wanted!r} has no snapshot_url configured."
                    )
                log.info("Resolved webcam %r to %s", wanted, url)
                return url

        raise WebcamNotFound(wanted, [str(w.get("name")) for w in webcams])

    @property
    def snapshot_url(self) -> str:
        """The cached snapshot URL, discovering it on first access."""
        with self._lock:
            if self._snapshot_url is None:
                self._snapshot_url = self._resolve_snapshot_url()
            return self._snapshot_url

    def get_snapshot(self) -> Image.Image | None:
        """Fetch the current camera frame.

        Returns ``None`` for transient failures (network blip, camera busy) so
        the monitor can log and carry on. A misconfigured webcam name raises
        :class:`WebcamNotFound`, since retrying will never fix it.
        """
        try:
            url = self.snapshot_url
        except WebcamNotFound:
            raise
        except PrinterUnavailable as exc:
            log.warning("Could not resolve webcam URL: %s", exc)
            return None

        separator = "&" if "?" in url else "?"
        fresh_url = f"{url}{separator}_scythe={secrets.token_hex(8)}"
        try:
            response = self._get(fresh_url)
        except PrinterUnavailable as exc:
            log.warning("Snapshot request failed: %s", exc)
            # The snapshot URL may have changed (camera re-added); rediscover.
            with self._lock:
                self._snapshot_url = None
            return None

        try:
            image = Image.open(BytesIO(response.content))
            image.load()  # force decode now, inside our error handling
        except (UnidentifiedImageError, OSError) as exc:
            log.warning("Snapshot was not a decodable image: %s", exc)
            return None
        return image

    # -- print state ------------------------------------------------------- #

    def get_state(self) -> PrintState:
        """Return the current print state, or ``UNKNOWN`` if unreadable."""
        try:
            payload = self._get_json("printer/objects/query?print_stats")
            raw = payload["result"]["status"]["print_stats"]["state"]
        except PrinterUnavailable as exc:
            log.warning("Could not read print state: %s", exc)
            return PrintState.UNKNOWN
        except (KeyError, TypeError) as exc:
            log.warning("Unexpected print_stats payload: %s", exc)
            return PrintState.UNKNOWN
        return PrintState.parse(raw)

    def is_printing(self) -> bool:
        return self.get_state().is_active

    def pause(self) -> bool:
        """Pause the running print.

        Returns ``False`` if the printer was not printing or the request
        failed — the caller decides how loudly to complain.
        """
        if not self.is_printing():
            log.info("Pause requested but printer is not printing; skipping.")
            return False
        return self._post_action("printer/print/pause", "pause")

    def resume(self) -> bool:
        """Resume the print only when Moonraker reports it as paused."""
        if self.get_state() is not PrintState.PAUSED:
            log.info("Resume requested but printer is not paused; skipping.")
            return False
        return self._post_action("printer/print/resume", "resume")

    def _post_action(self, path: str, action: str) -> bool:
        """Issue one non-retried printer action."""
        url = self._url(path)
        try:
            response = self._session.post(url, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            log.error("Failed to %s print: %s", action, exc)
            return False
        log.warning("Print %sd.", action)
        return True
