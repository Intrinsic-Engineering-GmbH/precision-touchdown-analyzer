"""Pushing the public scoreboard to the relay on the Raspberry Pi.

The judge PC's address changes from day to day, the Pi's does not, so the
analyzer calls the Pi and never the other way round: every few seconds it
PUTs what the public board shows - the page, the board of the newest
session, each session's board and its ranking PDF - to the relay's push port
(ptp-relay/README.md). The relay keeps the files and serves them to the
internet; it asks the analyzer for nothing, and the analyzer can stay on
127.0.0.1.

The newest session's board goes every time, changed or not: its age on the
relay is how the page there tells that the analyzer has stopped.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from touchdown_analyzer import config

if TYPE_CHECKING:
    from touchdown_analyzer.control.review import ReviewService

log = logging.getLogger(__name__)

URL_KEY = "RELAY_URL"  # in .env: the push port, e.g. http://192.168.0.118:5054
TOKEN_KEY = "RELAY_TOKEN"  # in .env: PUSH_TOKEN of the relay's .env
PUSH_EVERY_S = 3.0
# A relay that restarted has lost its copies (they live in memory there),
# so everything goes again now and then, not only what changed.
EVERYTHING_EVERY_S = 60.0
TIMEOUT_S = 5.0
PAGE = Path(__file__).parent / "static" / "board.html"
# The names the relay takes (its nginx config): no slash, no leading dot.
SAFE_SESSION = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")
TYPES = {".json": "application/json", ".html": "text/html", ".pdf": "application/pdf"}


class RelayError(Exception):
    """The relay did not take a file."""


# (path on the relay, what to send - None deletes it -, its fingerprint)
Upload = tuple[str, bytes | Path | None, str]


class RelayPusher:
    """Keeps the relay's copy of the public board up to date."""

    def __init__(
        self,
        review: ReviewService,
        url: str,
        token: str,
        *,
        every_s: float = PUSH_EVERY_S,
        page: Path = PAGE,
    ) -> None:
        self.review = review
        self.url = url.strip().rstrip("/")
        self.token = token.strip()
        self.every_s = every_s
        self.page = page
        self._sent: dict[str, str] = {}  # what the relay has: path -> fingerprint
        self._everything_at = float("-inf")
        self.ok = False
        self.error = ""
        self.last_push = ""

    @classmethod
    def from_env(cls, review: ReviewService, env_file: Path | None = None) -> RelayPusher | None:
        """The relay named in ``.env`` (or the environment); ``None`` without one."""
        url = config.saved_value(URL_KEY, env_file)
        if not url:
            return None
        return cls(review, url, config.saved_value(TOKEN_KEY, env_file) or "")

    def status(self) -> dict[str, Any]:
        return {"url": self.url, "ok": self.ok, "last_push": self.last_push, "error": self.error}

    # -- what goes -------------------------------------------------------------

    def plan(self, *, everything: bool) -> list[Upload]:
        """The files to send now: the newest board always, the rest when changed."""
        uploads: list[Upload] = []
        live = self.review.public_board(None)
        body = _json(live)
        uploads.append(("board.json", body, _digest(body)))
        if everything:
            page = self.page.read_bytes()
            uploads.append(("index.html", page, _digest(page)))
        sessions = [s for s in self.review.sessions_with_results() if SAFE_SESSION.match(s)]
        for session in sessions if everything else sessions[:1]:
            board = live if session == live["session"] else self.review.public_board(session)
            body = _json(board)
            self._add(uploads, f"sessions/{session}.json", body, _digest(body), everything)
            files = self.review.export(session)
            if files is not None and files[1].is_file():
                pdf = files[1]
                stat = pdf.stat()
                mark = f"{stat.st_mtime_ns}:{stat.st_size}"
                self._add(uploads, f"pdf/{session}.pdf", pdf, mark, everything)
                if session == live["session"]:
                    self._add(uploads, "ranking.pdf", pdf, mark, everything)
        if everything:
            # sessions deleted on the judge PC go from the relay too
            for path in list(self._sent):
                name = path.partition("/")[2].rpartition(".")[0]
                if path.startswith(("sessions/", "pdf/")) and name not in sessions:
                    uploads.append((path, None, ""))
        return uploads

    def _add(
        self, uploads: list[Upload], path: str, body: bytes | Path, mark: str, everything: bool
    ) -> None:
        if everything or self._sent.get(path) != mark:
            uploads.append((path, body, mark))

    # -- sending ------------------------------------------------------------------

    def send(self, uploads: list[Upload]) -> None:
        """PUT (or DELETE) each file; stops at the first the relay does not take."""
        for path, body, mark in uploads:
            data = body.read_bytes() if isinstance(body, Path) else body
            request = urllib.request.Request(  # noqa: S310 - the relay the user named
                f"{self.url}/{urllib.parse.quote(path)}",
                data=data,
                method="DELETE" if data is None else "PUT",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": TYPES.get(Path(path).suffix, "application/octet-stream"),
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT_S):  # noqa: S310
                    pass
            except urllib.error.HTTPError as exc:
                if data is None and exc.code == 404:
                    pass  # already gone
                elif exc.code in (401, 403):
                    raise RelayError(f"the relay refused the token ({TOKEN_KEY})") from exc
                else:
                    raise RelayError(f"the relay answered {exc.code} for {path}") from exc
            except (urllib.error.URLError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                raise RelayError(f"relay not reachable at {self.url}: {reason}") from exc
            if data is None:
                self._sent.pop(path, None)
            else:
                self._sent[path] = mark

    # -- the loop --------------------------------------------------------------------

    async def push_once(self) -> None:
        now = time.monotonic()
        everything = now - self._everything_at >= EVERYTHING_EVERY_S
        try:
            # on the event loop, like the public endpoints that read the same stores
            uploads = self.plan(everything=everything)
            await asyncio.to_thread(self.send, uploads)
        except Exception as exc:  # noqa: BLE001 - the loop must survive anything
            error = str(exc) if isinstance(exc, RelayError) else f"{type(exc).__name__}: {exc}"
            if error != self.error:
                log.warning("public board: %s", error)
            self.ok, self.error = False, error
            self._everything_at = float("-inf")  # the relay may have lost its copies
            return
        if not self.ok:
            log.warning("public board: pushing to %s", self.url)
        self.ok, self.error = True, ""
        self.last_push = datetime.now(UTC).isoformat(timespec="seconds")
        if everything:
            self._everything_at = now

    async def run(self) -> None:
        while True:
            await self.push_once()
            await asyncio.sleep(self.every_s)


def _json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha1(data, usedforsecurity=False).hexdigest()
