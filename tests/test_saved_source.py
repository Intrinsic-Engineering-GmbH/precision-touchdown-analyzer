from __future__ import annotations

from pathlib import Path

import pytest

from touchdown_analyzer import config
from touchdown_analyzer.cli import build_parser, main, resolve_source
from touchdown_analyzer.control.service import CaptureService, ServiceError

URL = "rtsp://root:s3cret@192.168.200.189/axis-media/media.amp"


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_nothing_saved_by_default(_isolated_env: Path) -> None:
    assert config.saved_source() is None


def test_remember_then_recall(_isolated_env: Path) -> None:
    path = config.remember_source(URL)
    assert path == _isolated_env
    assert config.saved_source() == URL
    assert _isolated_env.read_text(encoding="utf-8") == f"CAMERA_URL={URL}\n"


def test_remember_keeps_other_lines_and_replaces_in_place(_isolated_env: Path) -> None:
    _isolated_env.write_text(
        "# comment\nOTHER=1\nCAMERA_URL=rtsp://old\nMORE=2\n", encoding="utf-8"
    )
    config.remember_source(URL)
    assert _isolated_env.read_text(encoding="utf-8") == (
        f"# comment\nOTHER=1\nCAMERA_URL={URL}\nMORE=2\n"
    )


def test_remember_fills_in_the_example_placeholder(_isolated_env: Path) -> None:
    _isolated_env.write_text("# CAMERA_URL=rtsp://example\n", encoding="utf-8")
    config.remember_source(URL)
    assert _isolated_env.read_text(encoding="utf-8") == f"CAMERA_URL={URL}\n"


def test_quoted_and_bom_values_are_read(_isolated_env: Path) -> None:
    _isolated_env.write_text(f'CAMERA_URL="{URL}"\n', encoding="utf-8-sig")
    assert config.saved_source() == URL


def test_environment_variable_wins(_isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config.remember_source("rtsp://from-file")
    monkeypatch.setenv(config.SOURCE_KEY, "rtsp://from-env")
    assert config.saved_source() == "rtsp://from-env"


def test_empty_value_counts_as_unset(_isolated_env: Path) -> None:
    _isolated_env.write_text("CAMERA_URL=\n", encoding="utf-8")
    assert config.saved_source() is None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_source_is_optional_on_probe_and_record() -> None:
    parser = build_parser()
    assert parser.parse_args(["probe"]).source is None
    assert parser.parse_args(["record"]).source is None


def test_resolve_remembers_an_explicit_source(capsys: pytest.CaptureFixture[str]) -> None:
    assert resolve_source(URL) == URL
    assert config.saved_source() == URL
    out = capsys.readouterr().out
    assert "remembered" in out
    assert "s3cret" not in out


def test_resolve_falls_back_to_the_saved_source(capsys: pytest.CaptureFixture[str]) -> None:
    config.remember_source(URL)
    assert resolve_source(None) == URL
    out = capsys.readouterr().out
    assert "using saved source" in out
    assert "s3cret" not in out  # printed redacted


def test_resolve_with_nothing_saved_explains_what_to_do() -> None:
    with pytest.raises(SystemExit, match="no --source given"):
        resolve_source(None)


def test_probe_without_any_source_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["probe"])


# --------------------------------------------------------------------------
# service
# --------------------------------------------------------------------------


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CaptureService:
    from touchdown_analyzer.capture import ffmpeg as ff

    monkeypatch.setattr(ff, "find_tool", lambda name, override=None: name)
    return CaptureService(tmp_path / "raw", config_dir=tmp_path / "config")


def test_blank_source_uses_the_saved_one(service: CaptureService) -> None:
    config.remember_source(URL)
    assert service.resolve_source("") == URL
    assert service.resolve_source("   ") == URL


def test_blank_source_with_nothing_saved_is_refused(service: CaptureService) -> None:
    with pytest.raises(ServiceError, match="none saved"):
        service.resolve_source("")


def test_remember_flag_saves_from_the_ui(service: CaptureService) -> None:
    service.resolve_source(URL, remember=True)
    assert config.saved_source() == URL


def test_without_the_flag_nothing_is_written(service: CaptureService, _isolated_env: Path) -> None:
    service.resolve_source(URL)
    assert not _isolated_env.exists()


def test_status_advertises_the_saved_source_redacted(service: CaptureService) -> None:
    assert service.status()["saved_source"] is None
    config.remember_source(URL)
    shown = service.status()["saved_source"]
    assert shown is not None
    assert "192.168.200.189" in shown
    assert "s3cret" not in shown


def test_probe_job_never_carries_the_password(
    service: CaptureService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ProbeJob.source is returned by /api/status, so it must be redacted."""
    from touchdown_analyzer.capture import probe as probe_mod

    seen: dict[str, str] = {}

    def fake_run(source: str, *a, **k):
        seen["source"] = source
        return []

    monkeypatch.setattr(probe_mod, "run", fake_run)
    job = service.start_probe(URL, seconds=1)
    service._probe_thread.join(2.0)  # type: ignore[union-attr]

    assert seen["source"] == URL  # the real URL reached ffmpeg
    assert "s3cret" not in job.source  # but not the status payload
    assert "s3cret" not in str(service.probe_status())


def test_redacted_saved_source_maps_back(service: CaptureService, _isolated_env: Path) -> None:
    # The UI prefills the field with the redacted URL from /api/status.
    config.remember_source(URL)
    shown = service.status()["saved_source"]
    assert service.resolve_source(shown, remember=True) == URL
    assert config.saved_source() == URL
