from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from touchdown_analyzer.capture import ffmpeg as ff
from touchdown_analyzer.capture import probe, recorder, segments
from touchdown_analyzer.config import RecorderConfig, is_network_source, redact

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "rtsp://user:secret@10.0.0.5/axis-media/media.amp",
            "rtsp://<redacted>@10.0.0.5/axis-media/media.amp",
        ),
        ("rtsp://10.0.0.5/stream", "rtsp://10.0.0.5/stream"),
        ("D:/footage/day.mp4", "D:/footage/day.mp4"),
    ],
)
def test_redact_strips_credentials(source: str, expected: str) -> None:
    assert redact(source) == expected


def test_session_dir_and_pattern() -> None:
    cfg = RecorderConfig(source="rtsp://cam/stream", session="2026-07-18", root=Path("raw"))
    assert cfg.session_dir == Path("raw/2026-07-18")
    assert cfg.segment_pattern.name == "%Y-%m-%d_%H-%M-%S.mp4"


def test_is_network_source() -> None:
    assert is_network_source("rtsp://cam/stream")
    assert not is_network_source("C:/video.mp4")


# --------------------------------------------------------------------------
# ffmpeg helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rate", "expected"),
    [("60/1", 60.0), ("30000/1001", pytest.approx(29.97, abs=0.01)), ("0/0", None), (None, None)],
)
def test_parse_rate(rate: str | None, expected: float | None) -> None:
    assert ff.parse_rate(rate) == expected


def test_frame_times_derivations() -> None:
    times = ff.FrameTimes(pts=[0.0, 0.5, 1.0], keyframes=[0, 2])
    assert times.intervals == pytest.approx([0.5, 0.5])
    assert times.gop_lengths == [2]


# --------------------------------------------------------------------------
# recorder command construction
# --------------------------------------------------------------------------


def test_command_copies_without_retiming() -> None:
    """Camera PTS must survive: the sub-frame touchdown fit depends on it."""
    cfg = RecorderConfig(source="D:/footage/day.mp4", session="s")
    cmd = recorder.build_command(cfg)

    assert "-use_wallclock_as_timestamps" not in cmd, "would bake network jitter into PTS"
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert cmd[cmd.index("-segment_time") + 1] == "10"
    assert "-strftime" in cmd and "-an" in cmd


def test_command_adds_rtsp_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ff, "socket_timeout_flag", lambda _: "-timeout")
    cfg = RecorderConfig(source="rtsp://cam/stream", session="s", rtsp_transport="tcp")
    cmd = recorder.build_command(cfg)

    assert cmd[cmd.index("-rtsp_transport") + 1] == "tcp"
    assert "-timeout" in cmd


def test_gap_is_appended_to_both_the_stats_and_the_log(tmp_path: Path) -> None:
    stats = recorder.RunStats(started_utc="2026-07-18T12:00:00+00:00")
    since = datetime.now(UTC) - timedelta(seconds=12)
    gaps_path = tmp_path / "gaps.jsonl"

    recorder._record_gap(since, stats, gaps_path)
    recorder._record_gap(since, stats, gaps_path)

    assert len(stats.gaps) == 2
    assert stats.gaps[0]["seconds"] == pytest.approx(12.0, abs=1.0)
    written = [json.loads(line) for line in gaps_path.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 2


def test_manifest_records_offset_and_hides_credentials(tmp_path: Path) -> None:
    cfg = RecorderConfig(
        source="rtsp://user:secret@cam/stream", session="2026-07-18", root=tmp_path
    )
    cfg.session_dir.mkdir(parents=True)
    manifest = json.loads(recorder.write_manifest(cfg, None).read_text(encoding="utf-8"))

    assert "secret" not in json.dumps(manifest)
    assert "utc_offset_seconds" in manifest
    assert manifest["session"] == "2026-07-18"


def test_manifest_stream_is_filled_in_later(tmp_path: Path) -> None:
    cfg = RecorderConfig(source="rtsp://cam/stream", session="2026-07-18", root=tmp_path)
    cfg.session_dir.mkdir(parents=True)
    path = recorder.write_manifest(cfg, None)
    info = ff.StreamInfo(1920, 1080, "h264", "yuv420p", 60.0, 59.94, None, 10.0, 600, {})
    recorder.update_manifest_stream(path, info)

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["stream"]["width"] == 1920
    assert manifest["stream"]["avg_fps"] == 59.94
    assert manifest["session"] == "2026-07-18"


def test_first_closed_segment_waits_for_the_next_one(tmp_path: Path) -> None:
    old = tmp_path / "2026-07-18_10-00-00.mp4"
    old.touch()
    existing = {old}
    first = tmp_path / "2026-07-18_11-00-00.mp4"
    first.touch()
    assert recorder._first_closed_segment(tmp_path, existing) is None  # still being written
    (tmp_path / "2026-07-18_11-00-10.mp4").touch()
    assert recorder._first_closed_segment(tmp_path, existing) == first


def test_live_probe_has_a_socket_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stalled camera must fail ffprobe quickly, not hold it until killed."""
    seen: list[list[str]] = []

    def fake(ffprobe: str, args: list[str], timeout: float) -> dict:
        seen.append(args)
        return {"streams": [{"width": 1920, "height": 1080}]}

    monkeypatch.setattr(ff, "_run_ffprobe", fake)
    monkeypatch.setattr(ff, "major_version", lambda _: 9)
    ff.probe_stream("rtsp://cam/stream", "ffprobe")
    assert seen[0][seen[0].index("-timeout") + 1] == "5000000"


# --------------------------------------------------------------------------
# segment index
# --------------------------------------------------------------------------


def _segment(name: str, start: datetime, duration: float = 10.0) -> segments.Segment:
    return segments.Segment(
        name=name,
        start_utc=start.isoformat(),
        duration_s=duration,
        nb_frames=int(duration * 60),
        fps=60.0,
        size_bytes=1024,
    )


def test_parse_start_applies_recorder_timezone() -> None:
    tz = timezone(timedelta(hours=2))
    parsed = segments.parse_start("2026-07-18_14-32-07.mp4", tz)
    assert parsed == datetime(2026, 7, 18, 12, 32, 7, tzinfo=UTC)


def _manifest(session_dir: Path, offset_seconds: int, *, bom: bool = False) -> None:
    text = json.dumps({"schema_version": 1, "utc_offset_seconds": offset_seconds})
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / recorder.MANIFEST_NAME).write_text(
        text, encoding="utf-8-sig" if bom else "utf-8"
    )


def test_manifest_offset_is_applied(tmp_path: Path) -> None:
    _manifest(tmp_path, 7200)
    assert segments._utc_offset(tmp_path).utcoffset(None) == timedelta(hours=2)


def test_manifest_with_a_byte_order_mark_still_parses(tmp_path: Path) -> None:
    """PowerShell's `Set-Content -Encoding utf8` writes a BOM.

    Plain utf-8 JSON parsing rejects it, and the old fallback to UTC put every
    timestamp out by the local offset with nothing to show for it.
    """
    _manifest(tmp_path, 7200, bom=True)
    assert segments._utc_offset(tmp_path).utcoffset(None) == timedelta(hours=2)


def test_unreadable_manifest_warns_rather_than_failing_quietly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / recorder.MANIFEST_NAME).write_text("not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert segments._utc_offset(tmp_path) == UTC
    assert "may be out by the recorder's local offset" in caplog.text


def test_missing_manifest_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert segments._utc_offset(tmp_path) == UTC
    assert "assuming segment names are UTC" in caplog.text


def test_parse_start_rejects_other_filenames() -> None:
    assert segments.parse_start("session.json", UTC) is None


def test_resolve_locates_frame_within_segment() -> None:
    start = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
    index = [_segment("a.mp4", start), _segment("b.mp4", start + timedelta(seconds=10))]

    location = segments.resolve(index, start + timedelta(seconds=12.5))
    assert location is not None
    assert location.segment.name == "b.mp4"
    assert location.offset_s == pytest.approx(2.5)
    assert location.frame_index == 150


def test_resolve_returns_none_inside_a_gap() -> None:
    start = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
    index = [_segment("a.mp4", start), _segment("c.mp4", start + timedelta(seconds=30))]
    assert segments.resolve(index, start + timedelta(seconds=15)) is None


def test_resolve_requires_aware_datetime() -> None:
    with pytest.raises(ValueError):
        segments.resolve([], datetime(2026, 7, 18, 12, 0, 0))


def _clip(index: list[segments.Segment], touchdown: datetime) -> list[str]:
    """The default -3 s / +5 s touchdown window (docs/design.md 3.3)."""
    covering = segments.window(
        index, touchdown - timedelta(seconds=3), touchdown + timedelta(seconds=5)
    )
    return [s.name for s in covering]


def test_window_within_one_segment() -> None:
    start = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
    index = [_segment(f"{i}.mp4", start + timedelta(seconds=10 * i)) for i in range(6)]
    assert _clip(index, start + timedelta(seconds=25)) == ["2.mp4"]


def test_window_spans_segment_boundaries() -> None:
    """A touchdown near a cut must still pull every covering segment."""
    start = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
    index = [_segment(f"{i}.mp4", start + timedelta(seconds=10 * i)) for i in range(6)]

    assert _clip(index, start + timedelta(seconds=28)) == ["2.mp4", "3.mp4"]
    assert _clip(index, start + timedelta(seconds=31)) == ["2.mp4", "3.mp4"]
    assert _clip(index, start + timedelta(seconds=39)) == ["3.mp4", "4.mp4"]


def test_report_flags_gaps_and_fps_outliers() -> None:
    start = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
    slow = _segment("c.mp4", start + timedelta(seconds=40))
    slow.fps = 50.0
    index = [_segment("a.mp4", start), _segment("b.mp4", start + timedelta(seconds=10)), slow]

    result = segments.report("s", index, target_fps=60.0)
    assert len(result.gaps) == 1
    assert result.gaps[0]["after"] == "b.mp4"
    assert result.gaps[0]["seconds"] == pytest.approx(20.0)
    assert result.fps_outliers == ["c.mp4 (50.0 fps)"]
    assert result.coverage == pytest.approx(30.0 / 50.0)


def test_overlapping_segments_never_exceed_full_coverage() -> None:
    """Content can outlive the next filename when the two clocks disagree."""
    start = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
    index = [
        _segment("a.mp4", start, duration=4.0),  # only 2 s before b.mp4 starts
        _segment("b.mp4", start + timedelta(seconds=2), duration=10.0),
    ]

    result = segments.report("s", index, target_fps=60.0)
    assert result.coverage <= 1.0
    assert result.gaps == []
    assert len(result.overlaps) == 1
    assert result.overlaps[0]["seconds"] == pytest.approx(2.0)


def test_resolve_prefers_the_later_segment_when_they_overlap() -> None:
    """An instant after the next segment began belongs to that segment."""
    start = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
    index = [
        _segment("a.mp4", start, duration=4.0),
        _segment("b.mp4", start + timedelta(seconds=2), duration=10.0),
    ]

    location = segments.resolve(index, start + timedelta(seconds=3))
    assert location is not None
    assert location.segment.name == "b.mp4"
    assert location.offset_s == pytest.approx(1.0)


def test_report_tolerates_keyframe_drift() -> None:
    start = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
    index = [_segment("a.mp4", start, 9.2), _segment("b.mp4", start + timedelta(seconds=10))]
    assert segments.report("s", index).gaps == []


def test_index_round_trip(tmp_path: Path) -> None:
    start = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
    written = [_segment("a.mp4", start), _segment("b.mp4", start + timedelta(seconds=10))]
    segments.write_index(tmp_path, written)

    assert [s.name for s in segments.load_index(tmp_path)] == ["a.mp4", "b.mp4"]


def test_load_index_without_a_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        segments.load_index(tmp_path)


# --------------------------------------------------------------------------
# pre-flight checks
# --------------------------------------------------------------------------


def _stream(**overrides: object) -> ff.StreamInfo:
    defaults = dict(
        width=1920,
        height=1080,
        codec="h264",
        pix_fmt="yuv420p",
        nominal_fps=60.0,
        avg_fps=60.0,
        bit_rate_bps=14_000_000,
        duration_s=30.0,
        nb_frames=1800,
        raw={},
    )
    return ff.StreamInfo(**{**defaults, **overrides})  # type: ignore[arg-type]


def _status(checks: list[probe.Check], name: str) -> str:
    return next(c.status for c in checks if c.name == name)


def test_healthy_stream_passes() -> None:
    checks = probe._check_stream(_stream(), 60.0)
    assert probe.worst_status(checks) == probe.PASS


def test_fifty_fps_fails_with_the_power_line_hint() -> None:
    checks = probe._check_stream(_stream(nominal_fps=50.0), 60.0)
    fps = next(c for c in checks if c.name == "declared fps")
    assert fps.status == probe.FAIL
    assert "50 Hz" in fps.remedy


def test_regular_frames_pass() -> None:
    pts = [i / 60 for i in range(1800)]
    times = ff.FrameTimes(pts=pts, keyframes=list(range(0, 1800, 60)))
    checks = probe._check_frames(times, 30.0, 60.0)

    assert _status(checks, "measured fps") == probe.PASS
    assert _status(checks, "frame spacing") == probe.PASS
    assert _status(checks, "GOP") == probe.PASS


def test_zipstream_dropping_frames_fails() -> None:
    """Half the frames missing in a static scene - the classic Zipstream trap."""
    pts = [i / 30 for i in range(900)]
    times = ff.FrameTimes(pts=pts, keyframes=list(range(0, 900, 30)))
    checks = probe._check_frames(times, 30.0, 60.0)

    measured = next(c for c in checks if c.name == "measured fps")
    assert measured.status == probe.FAIL
    assert "Zipstream" in measured.remedy


def test_variable_frame_spacing_fails() -> None:
    pts, t = [], 0.0
    for i in range(600):
        pts.append(t)
        t += 1 / 60 if i % 3 else 1 / 10
    times = ff.FrameTimes(pts=pts, keyframes=[0])
    assert _status(probe._check_frames(times, pts[-1], 60.0), "frame spacing") == probe.FAIL


def test_dynamic_gop_fails() -> None:
    pts = [i / 60 for i in range(1800)]
    times = ff.FrameTimes(pts=pts, keyframes=[0, 60, 200, 260, 500])
    gop = next(c for c in probe._check_frames(times, 30.0, 60.0) if c.name == "GOP")
    assert gop.status == probe.FAIL
    assert "fixed GOP" in gop.remedy


def test_long_but_regular_gop_warns() -> None:
    """A 250-frame GOP is stable, but puts clip cuts ~4 s off the touchdown."""
    gop = probe._check_gop([250, 250, 250], 60.0)
    assert gop.status == probe.WARN
    assert "4.2 s" in gop.remedy


def test_gop_at_the_target_passes() -> None:
    assert probe._check_gop([60, 60, 60], 60.0).status == probe.PASS


def test_slight_gop_jitter_is_tolerated() -> None:
    assert probe._check_gop([60, 61, 59], 60.0).status == probe.PASS


def test_report_lists_remedies() -> None:
    checks = [probe.Check("declared fps", probe.FAIL, "50.00", "Switch to 60 Hz.")]
    rendered = probe.format_report(checks)
    assert "FAIL" in rendered
    assert "Switch to 60 Hz." in rendered
    assert "1 failed" in rendered
