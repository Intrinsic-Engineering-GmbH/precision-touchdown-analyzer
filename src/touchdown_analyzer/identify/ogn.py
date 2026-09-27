"""Open Glider Network: who was over the field when the wheel touched.

Two sources, both public, both best-effort:

* **Live positions** from ``live.glidernet.org``, polled while a session is
  running and appended to ``ogn_fixes.jsonl`` in the session directory.
  A landing is then matched to the aircraft whose fix nearest the touchdown
  instant was low and close to the target line. GNSS accuracy is useless
  for the *measurement* but ample for *identification* (design 3.4).
* **The OGN FlightBook** (``flightbook.glidernet.org``), which turns OGN
  tracks into take-off / landing minutes per aircraft and airfield - the
  same data the club's flight log is built on. Fetched on demand for a
  session day; matched on landing *or take-off* time, which also settles
  whether a track past the camera was a landing or an aerotow leaving. The
  KTrax logbook (``ktrax.kisstech.ch``) is the fallback for fields the
  FlightBook does not list.

Both need an OGN receiver within range of the field; the judge names every
aircraft by hand where there is none.

Stdlib only, so the control server carries no extra dependency for it.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FIXES_NAME = "ogn_fixes.jsonl"
LOGBOOK_NAME = "ogn_logbook.json"
CONFIG_NAME = "ogn.json"

LIVE_URL = "https://live.glidernet.org/lxml.php"
FLIGHTBOOK_URL = "https://flightbook.glidernet.org/api/logbook"
LOGBOOK_URL = "https://ktrax.kisstech.ch/backend/logbook"

POLL_S = 5.0
# A fix counts for a landing if it is within this many seconds of the
# touchdown, this close to the target line, and this low above field level.
MATCH_WINDOW_S = 90.0
MATCH_RADIUS_M = 1500.0
MATCH_MAX_AGL_M = 150.0
LOGBOOK_WINDOW_S = 240.0


@dataclass(slots=True)
class Field:
    """The airfield: where to look, and how far around it."""

    airfield: str = ""  # ICAO code, for the logbook
    name: str = ""  # for people; "Bellechasse"
    lat: float = 0.0
    lon: float = 0.0
    elevation_m: float = 0.0
    radius_km: float = 3.0
    enabled: bool = False
    timezone_offset_h: float = 2.0  # what the logbook query reports times in

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(north, south, east, west) around the field."""
        dlat = self.radius_km / 111.0
        dlon = self.radius_km / (111.0 * max(0.1, math.cos(math.radians(self.lat))))
        return self.lat + dlat, self.lat - dlat, self.lon + dlon, self.lon - dlon

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_field(config_dir: Path) -> Field:
    path = config_dir / CONFIG_NAME
    if not path.is_file():
        return Field()
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        known = set(Field.__dataclass_fields__)
        return Field(**{k: v for k, v in payload.items() if k in known})
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return Field()


def save_field(config_dir: Path, field_: Field) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / CONFIG_NAME
    path.write_text(json.dumps(field_.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path


@dataclass(slots=True)
class Fix:
    """One OGN position report."""

    utc: str
    flarm_id: str
    registration: str
    competition_number: str
    lat: float
    lon: float
    alt_m: float
    track_deg: float
    speed_kmh: float
    climb_ms: float
    aircraft_type: int
    receiver: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance; the field is a few km across, so this is plenty."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------
# live feed
# --------------------------------------------------------------------------


def _get(url: str, timeout: float = 15.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "touchdown-analyzer"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
        return str(response.read().decode("utf-8", errors="replace"))


def parse_live(xml: str, now: datetime) -> list[Fix]:
    """Parse the ``<m a="..."/>`` markers of the live page.

    Field order (undocumented, stable for years): lat, lon, competition
    number, registration, altitude m, time hh:mm:ss, age s, track, speed
    km/h, climb m/s, aircraft type, receiver, -, flarm id.
    """
    fixes: list[Fix] = []
    for chunk in xml.split('<m a="')[1:]:
        raw = chunk.split('"', 1)[0]
        parts = raw.split(",")
        if len(parts) < 14:
            continue
        try:
            age = float(parts[6] or 0)
            fixes.append(
                Fix(
                    utc=(now - timedelta(seconds=age)).isoformat(timespec="seconds"),
                    flarm_id=parts[13].strip(),
                    registration=parts[3].strip(),
                    competition_number=parts[2].strip().lstrip("_"),
                    lat=float(parts[0]),
                    lon=float(parts[1]),
                    alt_m=float(parts[4] or 0),
                    track_deg=float(parts[7] or 0),
                    speed_kmh=float(parts[8] or 0),
                    climb_ms=float(parts[9] or 0),
                    aircraft_type=int(float(parts[10] or 0)),
                    receiver=parts[11].strip(),
                )
            )
        except ValueError:
            continue
    return fixes


def fetch_live(field_: Field, *, timeout: float = 15.0) -> list[Fix]:
    north, south, east, west = field_.bbox
    query = urllib.parse.urlencode({"a": 0, "b": north, "c": south, "d": east, "e": west})
    return parse_live(_get(f"{LIVE_URL}?{query}", timeout), datetime.now(UTC))


class Poller:
    """Logs live fixes around the field into the session directory."""

    def __init__(self, field_: Field, session_dir: Path, *, interval_s: float = POLL_S) -> None:
        self.field = field_
        self.path = session_dir / FIXES_NAME
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen: set[tuple[str, str]] = set()
        self.fixes_logged = 0
        self.last_error = ""
        self.last_poll_utc: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while not self._stop.is_set():
            try:
                fixes = fetch_live(self.field)
                self.last_error = ""
            except (urllib.error.URLError, OSError, ValueError) as exc:
                self.last_error = str(exc)
                fixes = []
            self.last_poll_utc = datetime.now(UTC).isoformat(timespec="seconds")
            new = [f for f in fixes if (f.flarm_id, f.utc) not in self._seen]
            if new:
                with open(self.path, "a", encoding="utf-8") as fh:
                    for fix in new:
                        self._seen.add((fix.flarm_id, fix.utc))
                        fh.write(json.dumps(fix.as_dict()) + "\n")
                self.fixes_logged += len(new)
            if self._stop.wait(self.interval_s):
                break

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "fixes_logged": self.fixes_logged,
            "last_poll_utc": self.last_poll_utc,
            "error": self.last_error,
        }


def load_fixes(session_dir: Path) -> list[Fix]:
    path = session_dir / FIXES_NAME
    if not path.is_file():
        return []
    fixes = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                fixes.append(Fix(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return fixes


# --------------------------------------------------------------------------
# logbook
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Sortie:
    registration: str
    competition_number: str
    aircraft_type: str
    takeoff_utc: str | None
    landing_utc: str | None
    landing_place: str
    raw: dict[str, Any] = field(default_factory=dict)


def _sortie_time(day: str, hhmm: str, offset_h: float) -> str | None:
    if not hhmm or "X" in hhmm.upper():
        return None  # masked for unsubscribed airfields
    try:
        local = datetime.strptime(f"{day} {hhmm}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    return (local - timedelta(hours=offset_h)).replace(tzinfo=UTC).isoformat()


def parse_logbook(payload: dict[str, Any], offset_h: float) -> list[Sortie]:
    sorties = []
    for item in payload.get("sorties", []):
        ldg = item.get("ldg") or {}
        tkof = item.get("tkof") or {}
        sorties.append(
            Sortie(
                registration=str(item.get("cs", "")).strip(),
                competition_number=str(item.get("cn", "")).strip().strip("-"),
                aircraft_type=str(item.get("actype", "")).strip(),
                takeoff_utc=_sortie_time(
                    item.get("date_tkof", item.get("date", "")), tkof.get("time", ""), offset_h
                ),
                landing_utc=_sortie_time(
                    item.get("date_ldg", item.get("date", "")), ldg.get("time", ""), offset_h
                ),
                landing_place=str(ldg.get("loc", "")),
                raw=item,
            )
        )
    return sorties


def _stamp(raw: Any) -> str | None:
    """Epoch seconds -> ISO UTC; a flight still in the air has none."""
    if not raw:
        return None
    return datetime.fromtimestamp(int(raw), tz=UTC).isoformat()


def parse_flightbook(payload: dict[str, Any]) -> list[Sortie]:
    """The OGN FlightBook's day at one airfield: devices, and flights per device.

    ``start_tsp`` / ``stop_tsp`` are epoch seconds (minute resolution); a
    flight still in the air has no stop. ``towing`` marks the tug's flights.
    """
    devices = payload.get("devices") or []
    code = str((payload.get("airfield") or {}).get("code") or payload.get("code") or "")
    sorties = []
    for flight in payload.get("flights") or []:
        try:
            device = devices[int(flight.get("device", -1))]
        except (IndexError, TypeError, ValueError):
            device = {}

        sorties.append(
            Sortie(
                registration=str(device.get("registration") or "").strip(),
                competition_number=str(device.get("competition") or "").strip(),
                aircraft_type=str(device.get("aircraft") or "").strip()
                + (" (tug)" if flight.get("towing") else ""),
                takeoff_utc=_stamp(flight.get("start_tsp")),
                landing_utc=_stamp(flight.get("stop_tsp")),
                landing_place=code,
                raw={"flarm_id": device.get("address", ""), "towing": bool(flight.get("towing"))},
            )
        )
    return sorties


def fetch_flightbook(field_: Field, day: date, *, timeout: float = 30.0) -> list[Sortie]:
    url = f"{FLIGHTBOOK_URL}/{urllib.parse.quote(field_.airfield)}/{day.isoformat()}"
    return parse_flightbook(json.loads(_get(url, timeout)))


def parse_airfield(payload: dict[str, Any], previous: Field | None = None) -> Field | None:
    """The airfield block of a FlightBook day: code, name, position, elevation, zone.

    ``tz_offset`` reads like ``CEST+0200``. Settings that FlightBook does not
    know (radius, enabled) are carried over from ``previous``.
    """
    block = payload.get("airfield") or {}
    latlng = block.get("latlng") or []
    if not block.get("code") or len(latlng) != 2:
        return None
    base = previous or Field()
    offset = base.timezone_offset_h
    tz = str(block.get("time_info", {}).get("tz_offset") or "")
    if len(tz) >= 5 and tz[-5] in "+-" and tz[-4:].isdigit():
        offset = (1 if tz[-5] == "+" else -1) * (int(tz[-4:-2]) + int(tz[-2:]) / 60)
    return Field(
        airfield=str(block["code"]).upper(),
        name=str(block.get("name") or ""),
        lat=float(latlng[0]),
        lon=float(latlng[1]),
        elevation_m=float(block.get("elevation") or 0.0),
        radius_km=base.radius_km,
        enabled=base.enabled,
        timezone_offset_h=offset,
    )


def fetch_airfield(
    code: str, previous: Field | None = None, *, timeout: float = 15.0
) -> Field | None:
    """Look an ICAO code up on the OGN FlightBook. None if it is not known there."""
    code = code.strip().upper()
    if not code:
        return None
    url = f"{FLIGHTBOOK_URL}/{urllib.parse.quote(code)}/{datetime.now(tz=UTC).date().isoformat()}"
    try:
        payload = json.loads(_get(url, timeout))
    except (OSError, ValueError) as exc:
        log.warning("airfield lookup failed for %s: %s", code, exc)
        raise
    return parse_airfield(payload, previous)


def fetch_logbook(field_: Field, day: date, *, timeout: float = 30.0) -> list[Sortie]:
    """The day's sorties: the OGN FlightBook first, KTrax if it has none."""
    try:
        sorties = fetch_flightbook(field_, day, timeout=timeout)
    except (OSError, ValueError, KeyError) as exc:
        log.warning("flightbook fetch failed for %s: %s", field_.airfield, exc)
        sorties = []
    if sorties:
        return sorties
    return fetch_ktrax(field_, day, timeout=timeout)


def fetch_ktrax(field_: Field, day: date, *, timeout: float = 30.0) -> list[Sortie]:
    query = urllib.parse.urlencode(
        {
            "db": "sortie",
            "query_type": "ap",
            "tz": field_.timezone_offset_h,
            "id": field_.airfield,
            "dbeg": day.isoformat(),
            "dend": day.isoformat(),
        }
    )
    payload = json.loads(_get(f"{LOGBOOK_URL}?{query}", timeout))
    return parse_logbook(payload, field_.timezone_offset_h)


def save_logbook(session_dir: Path, sorties: list[Sortie]) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / LOGBOOK_NAME
    path.write_text(
        json.dumps(
            {"fetched_utc": datetime.now(UTC).isoformat(), "sorties": [asdict(s) for s in sorties]},
            indent=1,
        ),
        encoding="utf-8",
    )
    return path


def load_logbook(session_dir: Path) -> list[Sortie]:
    path = session_dir / LOGBOOK_NAME
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return [Sortie(**s) for s in payload.get("sorties", [])]
    except (OSError, json.JSONDecodeError, TypeError):
        return []


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Match:
    source: str  # live | logbook
    flarm_id: str
    registration: str
    competition_number: str
    aircraft_type: str
    dt_s: float  # fix / logbook time minus touchdown
    distance_m: float | None
    agl_m: float | None
    probability: float
    candidates: int
    event: str = "landing"  # landing | takeoff | fix
    # Every aircraft that fitted, best first, so the judge can pick another.
    options: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def match_fixes(touchdown: datetime, fixes: list[Fix], field_: Field) -> Match | None:
    """The aircraft whose track was at the field at the touchdown instant."""
    best: dict[str, tuple[float, Fix, float, float]] = {}
    for fix in fixes:
        try:
            when = datetime.fromisoformat(fix.utc)
        except ValueError:
            continue
        dt = (when - touchdown).total_seconds()
        if abs(dt) > MATCH_WINDOW_S:
            continue
        dist = distance_m(fix.lat, fix.lon, field_.lat, field_.lon)
        agl = fix.alt_m - field_.elevation_m
        if dist > MATCH_RADIUS_M or agl > MATCH_MAX_AGL_M:
            continue
        score = abs(dt) / MATCH_WINDOW_S + dist / MATCH_RADIUS_M
        current = best.get(fix.flarm_id)
        if current is None or score < current[0]:
            best[fix.flarm_id] = (score, fix, dist, agl)
    if not best:
        return None
    ranked = sorted(best.values(), key=lambda b: b[0])
    score, fix, dist, agl = ranked[0]
    # One candidate: confident. Two close ones: not so much.
    runner_up = ranked[1][0] if len(ranked) > 1 else None
    probability = max(0.05, 1.0 - score / 2)
    if runner_up is not None:
        probability *= min(1.0, (runner_up - score) / max(runner_up, 1e-6) + 0.5)
    return Match(
        source="live",
        flarm_id=fix.flarm_id,
        registration=fix.registration if fix.registration != fix.flarm_id else "",
        competition_number=fix.competition_number,
        aircraft_type="",
        dt_s=(datetime.fromisoformat(fix.utc) - touchdown).total_seconds(),
        distance_m=dist,
        agl_m=agl,
        probability=round(probability, 2),
        candidates=len(ranked),
        event="fix",
        options=[
            {
                "flarm_id": f.flarm_id,
                "registration": f.registration if f.registration != f.flarm_id else "",
                "competition_number": f.competition_number,
                "aircraft_type": "",
                "dt_s": (datetime.fromisoformat(f.utc) - touchdown).total_seconds(),
                "distance_m": d,
                "event": "fix",
            }
            for _, f, d, _ in ranked
        ],
    )


def match_logbook(touchdown: datetime, sorties: list[Sortie], field_: Field) -> Match | None:
    """The logbook event nearest the instant: a landing, or a take-off.

    Take-offs are matched too, on purpose: an aerotow passing the camera
    looks much like a landing to the estimator, and the logbook is what
    says which it was. ``event`` carries the answer.
    """
    candidates = []
    for sortie in sorties:
        if (
            sortie.landing_place
            and field_.airfield
            and sortie.landing_place.upper() != field_.airfield.upper()
        ):
            continue
        for event, when in (("landing", sortie.landing_utc), ("takeoff", sortie.takeoff_utc)):
            if not when:
                continue
            try:
                dt = (datetime.fromisoformat(when) - touchdown).total_seconds()
            except ValueError:
                continue
            if abs(dt) <= LOGBOOK_WINDOW_S:
                candidates.append((abs(dt), dt, event, sortie))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    _, dt, event, sortie = candidates[0]
    probability = max(0.05, 1.0 - abs(dt) / LOGBOOK_WINDOW_S)
    # An aerotow is two aircraft at the same minute; the runner-up decides
    # how sure to be, not just how many there were.
    if len(candidates) > 1:
        runner_up = candidates[1][0]
        probability *= 0.5 if runner_up - abs(dt) < 60 else 0.8
    return Match(
        source="logbook",
        flarm_id=str(sortie.raw.get("flarm_id", "")),
        registration=sortie.registration,
        competition_number=sortie.competition_number,
        aircraft_type=sortie.aircraft_type,
        dt_s=dt,
        distance_m=None,
        agl_m=None,
        probability=round(probability, 2),
        candidates=len(candidates),
        event=event,
        options=[
            {
                "flarm_id": str(s.raw.get("flarm_id", "")),
                "registration": s.registration,
                "competition_number": s.competition_number,
                "aircraft_type": s.aircraft_type,
                "dt_s": d,
                "distance_m": None,
                "event": e,
            }
            for _, d, e, s in candidates
        ],
    )


def identify(touchdown_utc: str, session_dir: Path, field_: Field) -> Match | None:
    """Best available identification for a touchdown instant."""
    touchdown = datetime.fromisoformat(touchdown_utc)
    found = match_fixes(touchdown, load_fixes(session_dir), field_)
    if found is None:
        found = match_logbook(touchdown, load_logbook(session_dir), field_)
    return found
