"""The landing record and its per-session store (docs/design.md 5).

A session's results live in one ``landings.json`` next to its clips and
overlays. JSON rather than SQLite: a flying day is a few dozen landings, the
file is readable by anyone who opens it, and it round-trips through git-free
backups without tooling. Every write replaces the file atomically.

The automatic result and the judge's decision are kept apart on purpose:
``longitudinal_m`` is what the estimator found, ``confirmed_*`` is what was
signed off, and ``history`` records each change with a timestamp, so the
number that was scored can always be traced back to the number that was
measured.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STORE_NAME = "landings.json"

# status: what the judge has done about it
PENDING = "pending"
CONFIRMED = "confirmed"
REJECTED = "rejected"

# outcome: what the estimator found
MEASURED = "measured"  # a number
SHORT = "short"  # touched down before the window: bound '< x'
LONG = "long"  # still airborne when it left the window: bound '> x'
ON_GROUND = "on_ground"  # rolling for the whole track (taxi, rollout, take-off run)
DEPARTURE = "departure"  # lifted off in view
AIRBORNE = "airborne"  # flew through without touching
UNSEEN = "unseen"  # low and level throughout: the instant is the judge's to pick
UNKNOWN = "unknown"


@dataclass(slots=True)
class TrackPoint:
    """One frame of a landing's track, enough to redraw it in the browser."""

    segment: str
    frame: int
    t: float
    u: float
    v: float
    world_x: float
    world_y: float
    clipped: bool
    gap_px: float | None = None  # lit ground between tyre and shadow, full-res px
    reach_px: float | None = None  # dark rows under the tyre before lit ground
    tyre_seen: bool = True  # the tyre itself was found here (else interpolated)


@dataclass(slots=True)
class Landing:
    id: str
    session: str
    created_utc: str
    kind: str  # landing | departure | pass
    outcome: str
    status: str = PENDING

    # the event
    touchdown_utc: str | None = None
    segment: str | None = None
    frame: int | None = None
    subframe: float | None = None
    fps: float = 60.0
    direction: int = 1  # +1 = travelling towards +x of the calibration frame

    # the measurement (calibration frame, already signed for direction)
    longitudinal_m: float | None = None
    lateral_m: float | None = None
    uncertainty_m: float | None = None
    bound_m: float | None = None  # for short / long: the bound, direction-signed
    speed_mps: float | None = None
    image_x: float | None = None
    image_y: float | None = None

    # how it was measured
    method: str = ""  # shadow | depth | none
    fit: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    calibration: dict[str, Any] = field(default_factory=dict)
    track: list[TrackPoint] = field(default_factory=list)
    first_utc: str | None = None
    # The whole pass as [segment, frame] at each end: every track of it, not
    # only the one measured. The judge's frame bar spans this.
    pass_first: list[Any] | None = None
    pass_last: list[Any] | None = None
    last_utc: str | None = None

    # who
    registration: str = ""
    competition_number: str = ""
    aircraft_type: str = ""
    identified_by: str = ""  # ogn | judge | ""
    ogn: dict[str, Any] | None = None
    pilot: str = ""  # entered by the judge; the key of the ranking

    # artefacts
    clip_path: str | None = None
    overlay_path: str | None = None

    # the judge
    confirmed_utc: str | None = None
    confirmed_longitudinal_m: float | None = None
    confirmed_frame: int | None = None
    confirmed_segment: str | None = None  # None: the contact segment
    note: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    # -- derived ----------------------------------------------------------

    @property
    def scored_longitudinal_m(self) -> float | None:
        if self.confirmed_longitudinal_m is not None:
            return self.confirmed_longitudinal_m
        return self.longitudinal_m

    def label(self) -> str:
        """The number, or the bound, as the judge reads it."""
        if self.outcome == MEASURED and self.scored_longitudinal_m is not None:
            return f"{self.scored_longitudinal_m:+.1f} m"
        if self.outcome == SHORT and self.bound_m is not None:
            return f"< {self.bound_m:+.0f} m"
        if self.outcome == LONG and self.bound_m is not None:
            return f"> {self.bound_m:+.0f} m"
        return {
            ON_GROUND: "rolling",
            DEPARTURE: "take-off",
            AIRBORNE: "fly-through",
            UNSEEN: "pick frame",
        }.get(self.outcome, "?")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label"] = self.label()
        payload["scored_longitudinal_m"] = self.scored_longitudinal_m
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Landing:
        data = dict(payload)
        data.pop("label", None)
        data.pop("scored_longitudinal_m", None)
        data["track"] = [TrackPoint(**p) for p in data.get("track", [])]
        known = {f for f in cls.__dataclass_fields__}  # tolerate newer files
        return cls(**{k: v for k, v in data.items() if k in known})


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


class LandingStore:
    """All landings of one session, kept in memory and mirrored to disk."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / STORE_NAME
        self._lock = threading.RLock()
        self._landings: dict[str, Landing] = {}
        self._mtime: float | None = None
        self._load()

    def _load(self) -> None:
        self._landings.clear()
        self._mtime = self._stamp()
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return
        for item in payload.get("landings", []):
            try:
                landing = Landing.from_dict(item)
            except TypeError:
                continue
            self._landings[landing.id] = landing

    def _stamp(self) -> float | None:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return None

    def _refresh(self) -> None:
        """Pick up a file rewritten by another process.

        The control server and a ``touchdown-analyzer analyze`` run in a
        terminal both hold this store in memory; whichever wrote last is
        the truth, so a change on disk replaces what is held here rather
        than being clobbered by the next save.
        """
        if self._stamp() != self._mtime:
            self._load()

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "written_utc": now_utc(),
            "landings": [x.to_dict() for x in self._sorted()],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
        self._mtime = self._stamp()

    def _sorted(self) -> list[Landing]:
        return sorted(
            self._landings.values(),
            key=lambda x: (x.touchdown_utc or x.first_utc or "", x.id),
        )

    # -- queries ----------------------------------------------------------

    def all(self) -> list[Landing]:
        with self._lock:
            self._refresh()
            return self._sorted()

    def get(self, landing_id: str) -> Landing | None:
        with self._lock:
            self._refresh()
            return self._landings.get(landing_id)

    def next_id(self) -> str:
        with self._lock:
            self._refresh()
            return f"L{len(self._landings) + 1:04d}"

    def has_track(self, segment: str, first_frame: int, last_frame: int) -> bool:
        """Whether a track covering these frames is already stored.

        Re-analysing a segment must not duplicate its landings.
        """
        with self._lock:
            self._refresh()
            for landing in self._landings.values():
                if not landing.track:
                    continue
                first, last = landing.track[0], landing.track[-1]
                if first.segment == segment and abs(first.frame - first_frame) <= 3:
                    return True
                if last.segment == segment and abs(last.frame - last_frame) <= 3:
                    return True
        return False

    # -- mutations --------------------------------------------------------

    def add(self, landing: Landing) -> Landing:
        with self._lock:
            self._refresh()
            if not landing.id:
                landing.id = self.next_id()
            self._landings[landing.id] = landing
            self._save()
        return landing

    def update(self, landing: Landing, action: str, **detail: Any) -> Landing:
        with self._lock:
            self._refresh()
            landing.history.append({"utc": now_utc(), "action": action, **detail})
            self._landings[landing.id] = landing
            self._save()
        return landing

    def remove_segment(self, segment: str) -> int:
        """Drop everything that came from ``segment`` (before re-analysing it)."""
        with self._lock:
            doomed = [
                x.id for x in self._landings.values() if x.track and x.track[0].segment == segment
            ]
            for landing_id in doomed:
                landing = self._landings.pop(landing_id)
                for path in (landing.clip_path, landing.overlay_path):
                    if path:
                        Path(path).unlink(missing_ok=True)
            if doomed:
                self._save()
            return len(doomed)

    def clear(self) -> None:
        """Forget every landing and delete the clips and overlays they own."""
        with self._lock:
            for landing in self._landings.values():
                for path in (landing.clip_path, landing.overlay_path):
                    if path:
                        Path(path).unlink(missing_ok=True)
            self._landings.clear()
            self._save()
