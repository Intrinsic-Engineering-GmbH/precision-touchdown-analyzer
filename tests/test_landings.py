"""The landing store, clip naming, OGN parsing / matching and the review API."""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from touchdown_analyzer.capture import ffmpeg as ff
from touchdown_analyzer.capture import segments as segments_mod
from touchdown_analyzer.clips import cutter
from touchdown_analyzer.control.app import create_app
from touchdown_analyzer.control.relay import RelayPusher
from touchdown_analyzer.control.review import ReviewService
from touchdown_analyzer.control.service import CaptureService
from touchdown_analyzer.identify import ogn
from touchdown_analyzer.store import landings as store_mod
from touchdown_analyzer.store.landings import Landing, LandingStore, TrackPoint

TD = "2026-09-13T11:58:37.500000+00:00"


def landing(**overrides) -> Landing:
    track = [
        TrackPoint(
            segment="s.mp4",
            frame=80 + i,
            t=i / 60,
            u=100 + 25 * i,
            v=730,
            world_x=-10 + 0.4 * i,
            world_y=3.0,
            clipped=False,
            reach_px=40 - i,
        )
        for i in range(40)
    ]
    base = dict(
        id="L0001",
        session="2026-09-13",
        created_utc=store_mod.now_utc(),
        kind="landing",
        outcome=store_mod.MEASURED,
        touchdown_utc=TD,
        segment="s.mp4",
        frame=100,
        subframe=100.3,
        direction=1,
        longitudinal_m=-1.9,
        lateral_m=3.0,
        uncertainty_m=0.4,
        speed_mps=24.0,
        image_x=600.0,
        image_y=730.0,
        track=track,
    )
    base.update(overrides)
    return Landing(**base)


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------


def test_store_round_trips_and_labels(tmp_path: Path) -> None:
    store = LandingStore(tmp_path)
    store.add(landing())
    store.add(landing(id="L0002", outcome=store_mod.SHORT, bound_m=-17.4, longitudinal_m=None))
    store.add(landing(id="L0003", outcome=store_mod.LONG, bound_m=18.2, longitudinal_m=None))

    again = LandingStore(tmp_path)
    ids = [x.id for x in again.all()]
    assert ids == ["L0001", "L0002", "L0003"]
    assert again.get("L0001").label() == "-1.9 m"
    assert again.get("L0002").label() == "< -17 m"
    assert again.get("L0003").label() == "> +18 m"
    assert again.next_id() == "L0004"
    assert again.has_track("s.mp4", 81, 200)
    assert not again.has_track("other.mp4", 81, 200)


def test_store_update_keeps_history_and_clear_removes_artefacts(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    store = LandingStore(tmp_path)
    x = store.add(landing(clip_path=str(clip)))
    x.registration = "HB-3213"
    store.update(x, "edited", registration_before="")
    assert json.loads(store.path.read_text())["landings"][0]["history"][0]["action"] == "edited"
    store.clear()
    assert not clip.exists()
    assert store.all() == []


def test_store_picks_up_a_rewrite_by_another_process(tmp_path: Path) -> None:
    mine = LandingStore(tmp_path)
    mine.add(landing())
    other = LandingStore(tmp_path)  # a second process holding the same file
    other.add(landing(id="L0002", outcome=store_mod.LONG, bound_m=18.0, longitudinal_m=None))
    # the first store sees the newcomer instead of overwriting it
    assert [x.id for x in mine.all()] == ["L0001", "L0002"]
    assert mine.next_id() == "L0003"


def test_confirmed_value_wins_over_measured() -> None:
    x = landing(confirmed_longitudinal_m=-3.2)
    assert x.scored_longitudinal_m == -3.2
    assert x.label() == "-3.2 m"


# --------------------------------------------------------------------------
# clips
# --------------------------------------------------------------------------


def test_clip_name_uses_touchdown_time_and_registration() -> None:
    when = datetime(2026, 7, 18, 14, 32, 7)
    assert cutter.clip_name(when, "", 7) == "2026-07-18_14-32-07_UNKNOWN-007.mp4"
    assert cutter.clip_name(when, "hb-3213", 7) == "2026-07-18_14-32-07_HB-3213.mp4"
    assert cutter.clip_name(when, "../x", 7) == "2026-07-18_14-32-07_..X.mp4".replace("..X", "..X")


def test_clip_window_is_pre_and_post_roll() -> None:
    when = datetime(2026, 7, 18, 14, 32, 7, tzinfo=UTC)
    start, end = cutter.window(when)
    assert when - start == timedelta(seconds=cutter.PRE_ROLL_S)
    assert end - when == timedelta(seconds=cutter.POST_ROLL_S)


def test_cut_refuses_when_no_segment_covers_the_window(tmp_path: Path) -> None:
    when = datetime(2026, 7, 18, 14, 32, 7, tzinfo=UTC)
    piece = cutter.Piece(
        tmp_path / "a.mp4", when + timedelta(minutes=5), when + timedelta(minutes=6)
    )
    with pytest.raises(cutter.ClipError):
        cutter.cut([piece], when, when + timedelta(seconds=8), "ffmpeg", tmp_path / "out.mp4")


# --------------------------------------------------------------------------
# OGN
# --------------------------------------------------------------------------

LIVE_XML = """<?xml version="1.0"?><markers>
<m a="46.977000,7.127500,DKU,HB-3380,450,11:58:30,7,90,95,-1.2,1,LSTB1,0,DD1234"/>
<m a="46.951351,7.430420,_16,11f4f816,1567,11:52:22,33,0,0,0.0,7,NAVITER3,0,11f4f816"/>
</markers>"""


def test_live_feed_parses_and_matches_the_low_close_aircraft() -> None:
    now = datetime.fromisoformat(TD) + timedelta(seconds=7)
    fixes = ogn.parse_live(LIVE_XML, now)
    assert len(fixes) == 2
    assert fixes[0].registration == "HB-3380" and fixes[0].competition_number == "DKU"
    assert fixes[0].utc.startswith("2026-09-13T11:58:37")

    field = ogn.Field(airfield="LSTB", lat=46.9769, lon=7.1269, elevation_m=434, enabled=True)
    match = ogn.match_fixes(datetime.fromisoformat(TD), fixes, field)
    assert match is not None
    assert match.registration == "HB-3380"
    assert match.candidates == 1  # the high, far one is not a candidate
    assert match.probability > 0.8
    assert [o["registration"] for o in match.options] == ["HB-3380"]


def test_every_live_candidate_is_offered_best_first() -> None:
    now = datetime.fromisoformat(TD) + timedelta(seconds=7)
    second = '<m a="46.977200,7.127900,F7,HB-3213,452,11:58:33,7,90,95,-1.0,1,LSTB1,0,DD5678"/>'
    fixes = ogn.parse_live(LIVE_XML.replace("</markers>", second + "</markers>"), now)
    field = ogn.Field(airfield="LSTB", lat=46.9769, lon=7.1269, elevation_m=434, enabled=True)
    match = ogn.match_fixes(datetime.fromisoformat(TD), fixes, field)
    assert match is not None and match.candidates == 2
    assert [o["registration"] for o in match.options] == ["HB-3380", "HB-3213"]
    assert match.options[1]["competition_number"] == "F7"


def test_logbook_parses_masked_and_real_times() -> None:
    payload = {
        "sorties": [
            {
                "cs": "HB-3213",
                "cn": "F7",
                "actype": "LS4",
                "date": "2026-09-13",
                "tkof": {"time": "12:10", "loc": "LSTB"},
                "ldg": {"time": "13:58", "loc": "LSTB"},
            },
            {
                "cs": "HBXXXX",
                "cn": "-",
                "actype": "?",
                "date": "2026-09-13",
                "tkof": {"time": "10:XX", "loc": "LSTB"},
                "ldg": {"time": "11:XX", "loc": "LSTB"},
            },
        ]
    }
    sorties = ogn.parse_logbook(payload, 2.0)
    assert sorties[0].landing_utc == "2026-09-13T11:58:00+00:00"
    assert sorties[1].landing_utc is None
    field = ogn.Field(airfield="LSTB")
    match = ogn.match_logbook(datetime.fromisoformat(TD), sorties, field)
    assert match is not None and match.registration == "HB-3213" and match.source == "logbook"


FLIGHTBOOK = {
    "code": "LSTB",
    "airfield": {"code": "LSTB"},
    "devices": [
        {
            "address": "4B5134",
            "aircraft": "LS-8 18",
            "competition": "F7",
            "registration": "HB-3213",
        },
        {"address": "4B26C8", "aircraft": "PA-18", "competition": "RW", "registration": "HB-ORW"},
    ],
    "flights": [
        # HB-3213 lands 11:58 UTC (the epoch of TD rounded down to the minute)
        {"device": 0, "start_tsp": 1789300200, "stop_tsp": 1789300680, "towing": False},
        # the tug takes off at 11:57 and is still up
        {"device": 1, "start_tsp": 1789300620, "stop_tsp": None, "towing": True},
    ],
}


def test_flightbook_parses_devices_and_flights() -> None:
    sorties = ogn.parse_flightbook(FLIGHTBOOK)
    assert [s.registration for s in sorties] == ["HB-3213", "HB-ORW"]
    assert sorties[0].landing_utc == "2026-09-13T11:58:00+00:00"
    assert sorties[1].landing_utc is None and sorties[1].aircraft_type == "PA-18 (tug)"
    assert sorties[0].raw["flarm_id"] == "4B5134"


def test_logbook_matches_a_landing_or_a_takeoff_and_says_which() -> None:
    sorties = ogn.parse_flightbook(FLIGHTBOOK)
    field = ogn.Field(airfield="LSTB")
    landing = ogn.match_logbook(datetime.fromisoformat(TD), sorties, field)
    assert landing is not None and landing.registration == "HB-3213" and landing.event == "landing"
    takeoff = ogn.match_logbook(datetime.fromisoformat("2026-09-13T11:57:10+00:00"), sorties, field)
    assert takeoff is not None and takeoff.registration == "HB-ORW" and takeoff.event == "takeoff"
    # both events within a minute of each other: less sure
    assert takeoff.probability < landing.probability or takeoff.candidates > 1
    # and the judge gets both to choose from, each saying which event it was
    offered = {(o["registration"], o["event"]) for o in takeoff.options}
    assert {("HB-ORW", "takeoff"), ("HB-3213", "landing")} <= offered
    assert len(takeoff.options) == takeoff.candidates


def test_field_config_round_trip(tmp_path: Path) -> None:
    field = ogn.Field(airfield="LSTB", name="Bellechasse", lat=46.9769, lon=7.1269, enabled=True)
    ogn.save_field(tmp_path, field)
    assert ogn.load_field(tmp_path) == field
    assert ogn.load_field(tmp_path / "missing") == ogn.Field()
    # a config written before the name existed still loads
    (tmp_path / ogn.CONFIG_NAME).write_text(json.dumps({"airfield": "LSTB", "lat": 46.9}))
    assert ogn.load_field(tmp_path) == ogn.Field(airfield="LSTB", lat=46.9)


AIRFIELD_DAY = {
    "airfield": {
        "code": "lstb",
        "country": "CH",
        "elevation": 432,
        "latlng": [46.97932, 7.1328],
        "name": "Bellechasse",
        "time_info": {"tz_name": "Europe/Zurich", "tz_offset": "CEST+0200"},
    },
    "code": "LSTB",
    "devices": [],
    "flights": [],
}


def test_airfield_lookup_fills_position_and_keeps_local_settings() -> None:
    previous = ogn.Field(airfield="LSTB", radius_km=5.0, enabled=True, timezone_offset_h=9.0)
    found = ogn.parse_airfield(AIRFIELD_DAY, previous)
    assert found == ogn.Field(
        airfield="LSTB",
        name="Bellechasse",
        lat=46.97932,
        lon=7.1328,
        elevation_m=432.0,
        radius_km=5.0,
        enabled=True,
        timezone_offset_h=2.0,
    )
    # unknown zone format: the previous offset stays
    odd = json.loads(json.dumps(AIRFIELD_DAY))
    odd["airfield"]["time_info"]["tz_offset"] = "?"
    assert ogn.parse_airfield(odd, previous).timezone_offset_h == 9.0  # type: ignore[union-attr]
    west = json.loads(json.dumps(AIRFIELD_DAY))
    west["airfield"]["time_info"]["tz_offset"] = "PDT-0730"
    assert ogn.parse_airfield(west).timezone_offset_h == -7.5  # type: ignore[union-attr]
    # unknown field: FlightBook answers without coordinates
    assert ogn.parse_airfield({"airfield": {"code": "XXXX"}}) is None
    assert ogn.parse_airfield({}) is None


def test_airfield_fetch_uses_todays_flightbook_page(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = []

    def fake_get(url: str, timeout: float) -> str:
        urls.append(url)
        return json.dumps(AIRFIELD_DAY)

    monkeypatch.setattr(ogn, "_get", fake_get)
    found = ogn.fetch_airfield(" lstb ")
    assert found is not None and found.name == "Bellechasse"
    assert urls[0].startswith(f"{ogn.FLIGHTBOOK_URL}/LSTB/20")
    assert ogn.fetch_airfield("") is None and len(urls) == 1


# --------------------------------------------------------------------------
# review API
# --------------------------------------------------------------------------


@pytest.fixture
def review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReviewService:
    monkeypatch.setattr(ff, "find_tool", lambda name, override=None: name)
    capture = CaptureService(tmp_path / "raw", config_dir=tmp_path / "config")
    review = ReviewService(capture, out_root=tmp_path / "landings")
    store = review.store("2026-09-13")
    clip = tmp_path / "landings" / "2026-09-13" / "2026-09-13_13-58-37_UNKNOWN-001.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"mp4")
    store.add(landing(clip_path=str(clip)))
    store.add(landing(id="L0002", kind="pass", outcome=store_mod.ON_GROUND, longitudinal_m=None))
    return review


@pytest.fixture
def client(review: ReviewService) -> TestClient:
    return TestClient(create_app(review.capture, review))


def test_listing_counts_only_landings_as_pending(client: TestClient) -> None:
    assert client.get("/api/landings").json() == ["2026-09-13"]
    summary = client.get("/api/landings/2026-09-13").json()
    assert summary["count"] == 2 and summary["pending"] == 1
    assert summary["landings"][0]["label"] == "-1.9 m"


def test_confirm_renames_the_clip_and_records_the_judge(client: TestClient) -> None:
    r = client.post("/api/landings/2026-09-13/L0001/confirm", json={"registration": "hb-3213"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "confirmed"
    assert body["registration"] == "HB-3213" and body["identified_by"] == "judge"
    assert body["clip_path"].endswith("2026-09-13_13-58-37_HB-3213.mp4")
    assert Path(body["clip_path"]).is_file()
    assert [h["action"] for h in body["history"]] == ["renamed", "confirmed"]


OPTIONS = [
    {
        "registration": "HB-3380",
        "competition_number": "DKU",
        "aircraft_type": "Discus",
        "flarm_id": "DD1234",
        "dt_s": 2.0,
        "distance_m": 60.0,
        "event": "fix",
    },
    {
        "registration": "HB-3213",
        "competition_number": "F7",
        "aircraft_type": "LS4",
        "flarm_id": "DD5678",
        "dt_s": 5.0,
        "distance_m": 90.0,
        "event": "fix",
    },
]


@pytest.fixture
def ogn_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A landing OGN identified as HB-3380, with HB-3213 as the runner-up."""
    monkeypatch.setattr(ff, "find_tool", lambda name, override=None: name)
    capture = CaptureService(tmp_path / "raw", config_dir=tmp_path / "config")
    review = ReviewService(capture, out_root=tmp_path / "landings")
    review.store("2026-09-13").add(
        landing(
            registration="HB-3380",
            competition_number="DKU",
            aircraft_type="Discus",
            identified_by="ogn",
            ogn={
                **OPTIONS[0],
                "source": "live",
                "agl_m": 5.0,
                "probability": 0.5,
                "candidates": 2,
                "options": OPTIONS,
            },
        )
    )
    return TestClient(create_app(capture, review))


def test_judge_picks_another_ogn_candidate(ogn_client: TestClient) -> None:
    r = ogn_client.post(
        "/api/landings/2026-09-13/L0001/edit",
        json={
            "registration": "HB-3213",
            "competition_number": "F7",
            "aircraft_type": "LS4",
            "source": "ogn",
        },
    )
    body = r.json()
    assert (body["registration"], body["identified_by"]) == ("HB-3213", "ogn")
    assert (body["competition_number"], body["aircraft_type"]) == ("F7", "LS4")


def test_own_registration_drops_the_ogn_aircraft_details(ogn_client: TestClient) -> None:
    body = ogn_client.post(
        "/api/landings/2026-09-13/L0001/edit", json={"registration": "hb-1111"}
    ).json()
    assert (body["registration"], body["identified_by"]) == ("HB-1111", "judge")
    assert (body["competition_number"], body["aircraft_type"]) == ("", "")
    # typing a registration OGN did see brings that aircraft's details along
    body = ogn_client.post(
        "/api/landings/2026-09-13/L0001/confirm", json={"registration": "hb-3213"}
    ).json()
    assert (body["competition_number"], body["aircraft_type"]) == ("F7", "LS4")


def test_saving_an_unchanged_registration_keeps_its_source(ogn_client: TestClient) -> None:
    body = ogn_client.post(
        "/api/landings/2026-09-13/L0001/edit", json={"registration": "HB-3380", "note": "ok"}
    ).json()
    assert body["identified_by"] == "ogn" and body["aircraft_type"] == "Discus"
    assert (
        ogn_client.post(
            "/api/landings/2026-09-13/L0001/edit", json={"registration": "X", "source": "radio"}
        ).status_code
        == 409
    )


def test_pilot_is_the_judges_entry_and_survives_a_reload(client: TestClient) -> None:
    r = client.post("/api/landings/2026-09-13/L0001/edit", json={"pilot": "  Anna Muster "})
    assert r.status_code == 200 and r.json()["pilot"] == "Anna Muster"
    assert r.json()["history"][-1]["pilot_before"] == ""
    # confirming with a name sets it; confirming without one keeps it
    r = client.post("/api/landings/2026-09-13/L0001/confirm", json={"pilot": "Beat Beispiel"})
    assert r.json()["pilot"] == "Beat Beispiel"
    client.post("/api/landings/2026-09-13/L0001/reopen")
    r = client.post("/api/landings/2026-09-13/L0001/confirm", json={})
    assert r.json()["pilot"] == "Beat Beispiel"
    # the store reads it back; a file written before the field existed loads too
    assert client.get("/api/landings/2026-09-13/L0001").json()["pilot"] == "Beat Beispiel"
    assert Landing.from_dict({**client.get("/api/landings/2026-09-13/L0002").json()}).pilot == ""


def test_judge_can_move_the_contact_frame(client: TestClient) -> None:
    r = client.post("/api/landings/2026-09-13/L0001/edit", json={"frame": 90})
    body = r.json()
    assert body["confirmed_frame"] == 90
    # track point 90 is index 10: world_x = -10 + 0.4 * 10
    assert body["confirmed_longitudinal_m"] == pytest.approx(-6.0)
    assert body["label"] == "-6.0 m"


def test_judge_cannot_score_a_frame_the_tracker_did_not_follow(client: TestClient) -> None:
    # the frame bar reaches past the track; the nearest tracked frame must not be scored instead
    r = client.post("/api/landings/2026-09-13/L0001/edit", json={"frame": 200})
    assert r.status_code == 409 and "frames 80-119" in r.json()["detail"]
    assert client.get("/api/landings/2026-09-13/L0001").json()["confirmed_frame"] is None


def test_judge_clicks_the_wheel_on_a_frame_the_tracker_missed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # no calibration saved: a click cannot be measured
    click = {"frame": 200, "image_x": 700.0, "image_y": 650.0}
    r = client.post("/api/landings/2026-09-13/L0001/edit", json=click)
    assert r.status_code == 409 and "calibration" in r.json()["detail"]

    monkeypatch.setattr(
        CaptureService, "measure", lambda self, x, y: {"world_x": 2.5, "world_y": 1.0}
    )
    body = client.post("/api/landings/2026-09-13/L0001/edit", json=click).json()
    assert body["confirmed_frame"] == 200 and body["confirmed_segment"] is None
    assert body["confirmed_longitudinal_m"] == pytest.approx(2.5)
    assert (body["image_x"], body["image_y"]) == (700.0, 650.0)
    assert body["outcome"] == "measured" and body["history"][-1]["clicked"] is True

    # a frame in another segment needs that segment to exist
    elsewhere = {**click, "segment": "nope.mp4"}
    r = client.post("/api/landings/2026-09-13/L0001/edit", json=elsewhere)
    assert r.status_code == 409 and "no segment" in r.json()["detail"]


def test_a_clicked_frame_in_the_next_segment_is_kept_with_its_segment(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = tmp_path / "raw" / "2026-09-13"
    session_dir.mkdir(parents=True)
    segments_mod.write_index(
        session_dir,
        [
            segments_mod.Segment("s.mp4", "2026-09-13T11:58:30+00:00", 10.0, 600, 60.0, 1),
            segments_mod.Segment("t.mp4", "2026-09-13T11:58:40+00:00", 10.0, 600, 60.0, 1),
        ],
    )
    monkeypatch.setattr(
        CaptureService, "measure", lambda self, x, y: {"world_x": -1.5, "world_y": 0.0}
    )
    body = client.post(
        "/api/landings/2026-09-13/L0001/edit",
        json={"frame": 30, "segment": "t.mp4", "image_x": 900.0, "image_y": 640.0},
    ).json()
    assert (body["confirmed_segment"], body["confirmed_frame"]) == ("t.mp4", 30)
    assert body["touchdown_utc"].startswith("2026-09-13T11:58:40.5")
    # back to the automatic touchpoint drops the segment too
    body = client.post("/api/landings/2026-09-13/L0001/edit", json={"reset_frame": True}).json()
    assert body["confirmed_segment"] is None and body["confirmed_frame"] is None


def _set_pass(
    tmp_path: Path, first: list | None, last: list | None, *, ruler: tuple[int, int] | None = None
) -> None:
    """Give L0001 a pass extent, as the analysis records it, and put only the
    track frames in ``ruler`` over the measuring window (none: all off it)."""
    path = tmp_path / "landings" / "2026-09-13" / "landings.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload["landings"]:
        if entry["id"] == "L0001":
            entry["pass_first"], entry["pass_last"] = first, last
            for point in entry["track"]:
                if ruler is None or not ruler[0] <= point["frame"] <= ruler[1]:
                    point["world_x"] = 30.0  # beyond the 19.4 m window
    before = path.stat().st_mtime_ns
    path.write_text(json.dumps(payload), encoding="utf-8")
    # The store notices a rewrite by its time; Windows can give two writes a
    # few milliseconds apart the same one.
    later = max(path.stat().st_mtime_ns, before + 1_000_000)
    os.utime(path, ns=(later, later))


def test_frame_bar_spans_the_wheel_over_the_ruler_across_segments(
    client: TestClient, tmp_path: Path
) -> None:
    session_dir = tmp_path / "raw" / "2026-09-13"
    session_dir.mkdir(parents=True)
    segments_mod.write_index(
        session_dir,
        [
            segments_mod.Segment("r.mp4", "2026-09-13T11:58:20+00:00", 10.0, 600, 60.0, 1),
            segments_mod.Segment("s.mp4", "2026-09-13T11:58:30+00:00", 10.0, 600, 60.0, 1),
            segments_mod.Segment("t.mp4", "2026-09-13T11:58:40+00:00", 10.0, 600, 60.0, 1),
            # another recording, half an hour later
            segments_mod.Segment("u.mp4", "2026-09-13T12:30:00+00:00", 10.0, 600, 60.0, 1),
        ],
    )

    def bar() -> list[tuple[str, int, int]]:
        spans = client.get("/api/landings/2026-09-13/L0001/timeline").json()
        return [(s["segment"], s["first"], s["last"]) for s in spans]

    # the wheel over the ruler for all its tracked frames 80-119: from 1 s
    # (60 frames) before it came over it to 1 s after it left
    assert bar() == [("s.mp4", 20, 179)]
    # over the ruler for part of the track only: that part, whatever the pass
    _set_pass(tmp_path, ["s.mp4", 10], ["s.mp4", 300], ruler=(90, 109))
    assert bar() == [("s.mp4", 30, 169)]
    # never over it: the whole pass; over a segment end it runs on into the
    # next file, by frame count
    _set_pass(tmp_path, ["s.mp4", 580], ["t.mp4", 10])
    assert bar() == [("s.mp4", 520, 599), ("t.mp4", 0, 70)]
    # and the lead reaches back into the previous one
    _set_pass(tmp_path, ["s.mp4", 20], ["s.mp4", 90])
    assert bar() == [("r.mp4", 560, 599), ("s.mp4", 0, 150)]
    # at the very start of the recording it stops there
    _set_pass(tmp_path, ["r.mp4", 10], ["r.mp4", 50])
    assert bar() == [("r.mp4", 0, 110)]
    # and at the end of one, it does not run on into the next recording
    _set_pass(tmp_path, ["t.mp4", 540], ["t.mp4", 590])
    assert bar() == [("t.mp4", 480, 599)]


def test_judge_can_go_back_to_the_automatic_touchpoint(client: TestClient) -> None:
    client.post("/api/landings/2026-09-13/L0001/edit", json={"frame": 90})
    body = client.post("/api/landings/2026-09-13/L0001/edit", json={"reset_frame": True}).json()
    assert body["confirmed_frame"] is None and body["confirmed_longitudinal_m"] is None
    assert body["label"] == "-1.9 m"
    # back on the automatic anchor pixel (track index 20 = frame 100)
    assert body["image_x"] == pytest.approx(100 + 25 * 20)


def test_outcome_can_be_corrected_to_a_bound(client: TestClient) -> None:
    body = client.post("/api/landings/2026-09-13/L0001/edit", json={"outcome": "long"}).json()
    assert body["outcome"] == "long" and body["label"].startswith("> ")
    assert body["bound_m"] == pytest.approx(-10 + 0.4 * 39)


def test_reject_and_reopen(client: TestClient) -> None:
    assert (
        client.post("/api/landings/2026-09-13/L0001/reject", json={"note": "bird"}).json()["status"]
        == "rejected"
    )
    assert client.post("/api/landings/2026-09-13/L0001/reopen").json()["status"] == "pending"
    assert client.get("/api/landings/2026-09-13/L0009").status_code == 409


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def test_scoring_rules_deduct_short_and_long_differently(tmp_path: Path) -> None:
    from touchdown_analyzer.store import scoring

    rules = scoring.ScoringRules(max_points=100, short_per_m=5, long_per_m=2, min_points=0)
    assert rules.score(0.0, "measured") == 100
    assert rules.score(-4.0, "measured") == 80  # short: 5 pts/m
    assert rules.score(4.0, "measured") == 92  # long: 2 pts/m
    assert rules.score(-30.0, "measured") == 0  # never below the floor
    assert rules.score(None, "short") == 0 and rules.score(None, "long") == 0
    # not a touchdown in view: nothing earned
    for outcome in ("departure", "airborne", "on_ground"):
        assert rules.score(None, outcome) == 0
    assert rules.score(None, "unseen") is None  # instant still to be picked
    rules.decimals = 1
    assert rules.score(-1.25, "measured") == 93.8
    scoring.save(tmp_path, rules)
    assert scoring.load(tmp_path) == rules
    assert scoring.load(tmp_path / "missing") == scoring.ScoringRules()


def test_target_line_has_a_width_that_earns_full_points(tmp_path: Path) -> None:
    from touchdown_analyzer.store import scoring

    # a 2 m wide line: everything within +/-1 m of its centre is a bullseye
    rules = scoring.ScoringRules(
        max_points=100, target_width_m=2.0, short_per_m=5, long_per_m=2, min_points=0
    )
    assert rules.target_half_width_m == 1.0
    assert rules.score(0.0, "measured") == 100
    assert rules.score(-1.0, "measured") == 100  # on the short edge of the line
    assert rules.score(1.0, "measured") == 100  # on the long edge
    # the deduction is measured from the edge, not the centre
    assert rules.score(-4.0, "measured") == 85  # 5 pts/m over 3.0 m of miss
    assert rules.score(4.0, "measured") == 94  # 2 pts/m over 3.0 m of miss
    assert rules.miss_m(-1.0) == 0.0 and rules.miss_m(-4.0) == 3.0
    assert rules.score(-30.0, "measured") == 0  # the floor still holds

    # width 0 is the old infinitely thin line, unchanged
    thin = scoring.ScoringRules(max_points=100, short_per_m=5, long_per_m=2, min_points=0)
    assert thin.target_width_m == 0.0
    assert thin.score(-4.0, "measured") == 80

    scoring.save(tmp_path, rules)
    assert scoring.load(tmp_path) == rules


def test_scoring_config_without_a_width_still_loads(tmp_path: Path) -> None:
    from touchdown_analyzer.store import scoring

    # a scoring.json written before the width existed
    (tmp_path / scoring.CONFIG_NAME).write_text(
        '{"max_points": 100, "short_per_m": 5, "long_per_m": 2}', encoding="utf-8"
    )
    rules = scoring.load(tmp_path)
    assert rules.target_width_m == 0.0
    assert rules.score(-4.0, "measured") == 80


def test_scores_come_with_the_landings_and_rules_can_be_changed(client: TestClient) -> None:
    body = client.get("/api/landings/2026-09-13/L0001").json()
    # -1.9 m short under the default rules: 100 - 5 * 1.9 = 90.5, rounded to 0 decimals
    assert body["score"] == 90
    r = client.post(
        "/api/scoring",
        json={"max_points": 1000, "short_per_m": 50, "long_per_m": 20, "decimals": 0},
    )
    assert r.status_code == 200 and r.json()["max_points"] == 1000
    assert client.get("/api/scoring").json()["short_per_m"] == 50
    assert client.get("/api/landings/2026-09-13/L0001").json()["score"] == 905
    assert client.get("/api/landings/2026-09-13/L0002").json()["score"] == 0  # rolling
    assert client.post("/api/scoring", json={"max_points": -1}).status_code == 422
    assert client.post("/api/scoring", json={"target_width_m": -1}).status_code == 422
    # a 4 m wide line puts L0001 (-1.9 m) inside it: full points
    r = client.post(
        "/api/scoring",
        json={"max_points": 100, "target_width_m": 4.0, "short_per_m": 5, "long_per_m": 2},
    )
    assert r.status_code == 200 and r.json()["target_width_m"] == 4.0
    assert client.get("/api/landings/2026-09-13/L0001").json()["score"] == 100
    assert client.get("/scoring").status_code == 200


# --------------------------------------------------------------------------
# the ranking as files
# --------------------------------------------------------------------------


def scored(**overrides) -> dict:
    base = {
        "id": "L0001",
        "kind": "landing",
        "status": "confirmed",
        "outcome": "measured",
        "touchdown_utc": TD,
        "pilot": "",
        "registration": "",
        "competition_number": "",
        "aircraft_type": "",
        "scored_longitudinal_m": -1.9,
        "label": "-1.9 m",
        "score": 90,
    }
    base.update(overrides)
    return base


def test_ranking_groups_and_orders_like_the_board() -> None:
    from touchdown_analyzer.store import ranking

    result = ranking.rank(
        [
            scored(id="L0001", pilot="anna", registration="HB-3213", score=90),
            scored(id="L0002", pilot="Beat", score=100, touchdown_utc="2026-09-13T12:10:00+00:00"),
            # same pilot, later spelling wins, aircraft collected
            scored(
                id="L0003",
                pilot="Anna",
                registration="HB-1234",
                score=95,
                touchdown_utc="2026-09-13T12:20:00+00:00",
            ),
            # no pilot: the aircraft stands in
            scored(
                id="L0004",
                registration="D-KXYZ",
                score=100,
                touchdown_utc="2026-09-13T12:30:00+00:00",
            ),
            scored(id="L0005", status="pending", registration="HB-9999", score=None),
            scored(id="L0006", status="rejected", pilot="Nobody"),
            scored(id="L0007", kind="pass", outcome="on_ground", status="confirmed", score=None),
        ]
    )
    assert [(g.name, g.total, len(g.landings)) for g in result.ranked] == [
        ("Anna", 185, 2),
        ("Beat", 100, 1),
        ("D-KXYZ", 100, 1),
    ]
    assert result.ranked[0].aircraft == ["HB-3213", "HB-1234"]
    assert [x["id"] for x in result.pending] == ["L0005"]
    assert result.confirmed == 4


def test_ranking_by_the_mean_of_each_pilots_landings() -> None:
    from touchdown_analyzer.store import ranking, scoring

    landings = [
        scored(id="L0001", pilot="Anna", score=90),
        scored(id="L0002", pilot="Anna", score=95, touchdown_utc="2026-09-13T12:20:00+00:00"),
        scored(id="L0003", pilot="Anna", score=80, touchdown_utc="2026-09-13T12:40:00+00:00"),
        scored(id="L0004", pilot="Beat", score=100, touchdown_utc="2026-09-13T12:10:00+00:00"),
        scored(id="L0005", pilot="Cla", score=87, touchdown_utc="2026-09-13T12:50:00+00:00"),
        scored(id="L0006", pilot="Cla", score=88, touchdown_utc="2026-09-13T13:00:00+00:00"),
    ]
    added = ranking.rank(landings, scoring.ScoringRules(aggregate=scoring.SUM))
    assert [(g.name, g.total) for g in added.ranked] == [
        ("Anna", 265),
        ("Cla", 175),
        ("Beat", 100),
    ]
    # the mean is rounded like a score: 87.5 -> 88 ties with Anna's 88.33 -> 88,
    # and the tie goes to the pilot with more landings
    mean = ranking.rank(landings, scoring.ScoringRules(aggregate=scoring.MEAN))
    assert [(g.name, g.total) for g in mean.ranked] == [
        ("Beat", 100),
        ("Anna", 88),
        ("Cla", 88),
    ]
    one = ranking.rank(landings, scoring.ScoringRules(aggregate=scoring.MEAN, decimals=1))
    assert [(g.name, g.total) for g in one.ranked] == [
        ("Beat", 100),
        ("Anna", 88.3),
        ("Cla", 87.5),
    ]


def test_ranking_mode_is_saved_and_reaches_the_files(client: TestClient) -> None:
    r = client.post("/api/scoring", json={"aggregate": "mean"})
    assert r.status_code == 200 and r.json()["aggregate"] == "mean"
    assert client.get("/api/scoring").json()["aggregate"] == "mean"
    assert client.get("/api/public/board").json()["rules"]["aggregate"] == "mean"
    assert client.post("/api/scoring", json={"aggregate": "median"}).status_code == 422
    import io
    import zipfile

    r = client.get("/api/landings/2026-09-13/ranking.xlsx")
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        sheet = zf.read("xl/worksheets/sheet1.xml").decode()
    assert ">Mean<" in sheet and ">Total<" not in sheet


def test_ranking_files_are_written_and_kept_current(client: TestClient, tmp_path: Path) -> None:
    import zipfile

    out = tmp_path / "landings" / "2026-09-13"
    xlsx, pdf = out / "ranking.xlsx", out / "ranking.pdf"
    # the first read of the summary writes them; nothing is confirmed yet
    client.get("/api/landings/2026-09-13")
    assert xlsx.is_file() and pdf.is_file()
    assert pdf.read_bytes().startswith(b"%PDF-1.4") and pdf.read_bytes().rstrip().endswith(b"%%EOF")

    # the judge's edits show up in the files at once
    client.post(
        "/api/landings/2026-09-13/L0001/confirm",
        json={"pilot": "Anna Muster", "registration": "HB-3213"},
    )
    with zipfile.ZipFile(xlsx) as zf:
        sheet = zf.read("xl/worksheets/sheet1.xml").decode()
        assert "Anna Muster" in sheet and "HB-3213" in sheet
        assert zf.testzip() is None
        assert "Landings" in zf.read("xl/workbook.xml").decode()
    assert b"Anna Muster" in pdf.read_bytes() and b"(90)" in pdf.read_bytes()

    # a change of the rules is picked up on the next read
    client.post("/api/scoring", json={"max_points": 1000, "short_per_m": 50, "long_per_m": 20})
    client.get("/api/landings/2026-09-13")
    assert b"(905)" in pdf.read_bytes()

    # ... and nothing is rewritten while nothing changes
    before = (xlsx.stat().st_mtime_ns, pdf.stat().st_mtime_ns)
    client.get("/api/landings/2026-09-13")
    assert (xlsx.stat().st_mtime_ns, pdf.stat().st_mtime_ns) == before

    # downloads
    r = client.get("/api/landings/2026-09-13/ranking.xlsx")
    assert r.status_code == 200 and r.content[:2] == b"PK"
    assert r.headers["content-type"].startswith("application/vnd.openxmlformats")
    assert "ranking-2026-09-13.xlsx" in r.headers["content-disposition"]
    r = client.get("/api/landings/2026-09-13/ranking.pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert client.get("/api/landings/2099-01-01/ranking.pdf").status_code == 404


def test_pdf_pages_break_and_odd_characters_survive(tmp_path: Path) -> None:
    import zipfile

    from touchdown_analyzer.store import ranking, scoring

    # e-acute is in the PDF's WinAnsi font encoding, the CJK character is not
    many = [scored(id=f"L{i:04d}", pilot=f"Pilot {i} é中", score=i % 100) for i in range(1, 121)]
    xlsx, pdf = ranking.export(tmp_path, "2026-09-13", many, scoring.ScoringRules())
    data = pdf.read_bytes()
    pages = data.count(b"/Type /Page ")
    assert pages > 1 and b"page 1 of %d" % pages in data and b"/Count %d" % pages in data
    assert b"Pilot 120 \xe9?" in data
    with zipfile.ZipFile(xlsx) as zf:
        assert "Pilot 120 é中" in zf.read("xl/worksheets/sheet2.xml").decode("utf-8")


def test_board_page_is_served(client: TestClient) -> None:
    r = client.get("/board")
    assert r.status_code == 200 and "Results board" in r.text
    # the board reads what the judge's page reads: the landings with a score
    first = client.get("/api/landings/2026-09-13").json()["landings"][0]
    assert {"status", "kind", "score", "label", "scored_longitudinal_m"} <= set(first)


def test_public_board_shows_only_what_the_board_draws(client: TestClient) -> None:
    from touchdown_analyzer.control.review import PUBLIC_LANDING_FIELDS

    r = client.get("/public")
    assert r.status_code == 200 and "Results board" in r.text
    board = client.get("/api/public/board").json()
    # the newest session, the rules the board needs, the landings - nothing else
    assert board["session"] == "2026-09-13"
    assert set(board["rules"]) == {"name", "max_points", "decimals", "aggregate"}
    # the rolling aircraft (a pass) is not on the board
    assert [x["id"] for x in board["landings"]] == ["L0001"]
    only = board["landings"][0]
    assert set(only) == set(PUBLIC_LANDING_FIELDS)
    assert only["score"] == 90 and only["label"] == "-1.9 m"
    # no tracks, file paths or notes
    text = r.text + client.get("/api/public/board").text
    assert "track" not in board["landings"][0] and "clip_path" not in text
    # a session is looked up, never joined onto a path
    assert client.get("/api/public/board", params={"session": "2026-09-13"}).status_code == 200
    for bad in ("2099-01-01", "..", "../raw", "2026-09-13/../x"):
        assert client.get("/api/public/board", params={"session": bad}).status_code == 404
        assert client.get("/api/public/ranking.pdf", params={"session": bad}).status_code == 404
    pdf = client.get("/api/public/ranking.pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"


def test_public_board_before_any_results(tmp_path: Path) -> None:
    capture = CaptureService(tmp_path / "raw", config_dir=tmp_path / "config")
    review = ReviewService(capture, out_root=tmp_path / "landings")
    empty = TestClient(create_app(capture, review))
    assert empty.get("/api/public/board").json()["landings"] == []
    assert empty.get("/api/public/ranking.pdf").status_code == 404


class _Relay(http.server.BaseHTTPRequestHandler):
    """The relay's push port, as nginx runs it: PUT / DELETE with the token."""

    files: dict[str, bytes]
    seen: list[tuple[str, str]]

    def _allowed(self) -> bool:
        if self.headers.get("Authorization") == "Bearer secret":
            return True
        self.send_response(401)
        self.end_headers()
        return False

    def do_PUT(self) -> None:  # noqa: N802 - http.server's naming
        if self._allowed():
            self.seen.append(("PUT", self.path))
            self.files[self.path] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(201)
            self.end_headers()

    def do_DELETE(self) -> None:  # noqa: N802
        if self._allowed():
            self.seen.append(("DELETE", self.path))
            self.send_response(204 if self.files.pop(self.path, None) is not None else 404)
            self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def relay_server() -> Iterator[tuple[str, dict[str, bytes], list[tuple[str, str]]]]:
    files: dict[str, bytes] = {}
    seen: list[tuple[str, str]] = []
    handler = type("Relay", (_Relay,), {"files": files, "seen": seen})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", files, seen
    server.shutdown()
    server.server_close()


def test_public_board_is_pushed_to_the_relay(review: ReviewService, relay_server: Any) -> None:
    url, files, seen = relay_server
    pusher = RelayPusher(review, url + "/", "secret")
    asyncio.run(pusher.push_once())
    assert pusher.status()["ok"], pusher.error
    # the first push: everything the public board needs
    assert set(files) == {
        "/board.json",
        "/index.html",
        "/sessions/2026-09-13.json",
        "/pdf/2026-09-13.pdf",
        "/ranking.pdf",
    }
    public = TestClient(create_app(review.capture, review)).get("/api/public/board").json()
    assert json.loads(files["/board.json"]) == public
    assert json.loads(files["/sessions/2026-09-13.json"]) == public
    assert b"Results board" in files["/index.html"]
    assert files["/ranking.pdf"].startswith(b"%PDF")
    assert files["/ranking.pdf"] == files["/pdf/2026-09-13.pdf"]
    # then the board alone, changed or not: that it keeps coming says the
    # analyzer runs
    seen.clear()
    asyncio.run(pusher.push_once())
    assert seen == [("PUT", "/board.json")]
    # a change of the judge's goes with the next push, the PDF with it
    seen.clear()
    review.confirm("2026-09-13", "L0001", registration="HB-3213", pilot="Anna")
    asyncio.run(pusher.push_once())
    assert {path for _, path in seen} == {
        "/board.json",
        "/sessions/2026-09-13.json",
        "/pdf/2026-09-13.pdf",
        "/ranking.pdf",
    }
    assert json.loads(files["/board.json"])["landings"][0]["pilot"] == "Anna"
    # a session deleted on the judge PC goes from the relay with the next full push
    files["/sessions/2026-09-01.json"] = b"{}"
    pusher._sent["sessions/2026-09-01.json"] = "x"  # noqa: SLF001 - pushed earlier
    pusher._everything_at = float("-inf")  # noqa: SLF001
    asyncio.run(pusher.push_once())
    assert "/sessions/2026-09-01.json" not in files
    assert ("DELETE", "/sessions/2026-09-01.json") in seen


def test_relay_that_refuses_or_is_away(review: ReviewService, relay_server: Any) -> None:
    url, files, _ = relay_server
    wrong = RelayPusher(review, url, "guess")
    asyncio.run(wrong.push_once())
    assert not wrong.ok and "token" in wrong.error and not files
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    away = RelayPusher(review, f"http://127.0.0.1:{port}", "secret")
    asyncio.run(away.push_once())
    assert not away.ok and "not reachable" in away.error
    # once it answers again, everything goes - it may have lost its copies
    away.url = url
    asyncio.run(away.push_once())
    assert away.ok and "/index.html" in files


def test_relay_from_env_and_in_the_status(
    review: ReviewService, relay_server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, files, _ = relay_server
    monkeypatch.delenv("RELAY_URL", raising=False)
    monkeypatch.delenv("RELAY_TOKEN", raising=False)
    env = tmp_path / ".env"
    assert RelayPusher.from_env(review, env) is None
    env.write_text(f"CAMERA_URL=rtsp://x\nRELAY_URL={url}\nRELAY_TOKEN=secret\n", encoding="utf-8")
    pusher = RelayPusher.from_env(review, env)
    assert pusher is not None and pusher.token == "secret"
    pusher.every_s = 0.05
    with TestClient(create_app(review.capture, review, pusher)) as client:
        for _ in range(100):
            if pusher.ok:
                break
            time.sleep(0.05)
        status = client.get("/api/status").json()["relay"]
    assert status["url"] == url and status["ok"] and status["last_push"]
    assert "/board.json" in files


def test_analysis_refuses_without_calibration(client: TestClient) -> None:
    r = client.post("/api/analysis/start", json={"session": "2026-09-13"})
    assert r.status_code == 409
    assert "calibrat" in r.json()["detail"] or "session" in r.json()["detail"]
    status = client.get("/api/analysis/status").json()
    assert status["running"] is False
