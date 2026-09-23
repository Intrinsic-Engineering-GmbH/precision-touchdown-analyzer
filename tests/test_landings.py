"""The landing store, clip naming, OGN parsing / matching and the review API."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from touchdown_analyzer.capture import ffmpeg as ff
from touchdown_analyzer.clips import cutter
from touchdown_analyzer.control.app import create_app
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
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(ff, "find_tool", lambda name, override=None: name)
    capture = CaptureService(tmp_path / "raw", config_dir=tmp_path / "config")
    review = ReviewService(capture, out_root=tmp_path / "landings")
    store = review.store("2026-09-13")
    clip = tmp_path / "landings" / "2026-09-13" / "2026-09-13_13-58-37_UNKNOWN-001.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"mp4")
    store.add(landing(clip_path=str(clip)))
    store.add(landing(id="L0002", kind="pass", outcome=store_mod.ON_GROUND, longitudinal_m=None))
    return TestClient(create_app(capture, review))


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
    assert rules.score(1.0, "departure") is None and rules.score(None, "unseen") is None
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
    assert (
        client.get("/api/landings/2026-09-13/L0002").json()["score"] is None
    )  # rolling, not scored
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


def test_analysis_refuses_without_calibration(client: TestClient) -> None:
    r = client.post("/api/analysis/start", json={"session": "2026-09-13"})
    assert r.status_code == 409
    assert "calibrat" in r.json()["detail"] or "session" in r.json()["detail"]
    status = client.get("/api/analysis/status").json()
    assert status["running"] is False
