"""A site's settings as one .pta file: export, import, and what is refused."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from touchdown_analyzer import config, settings_file


def site(home: Path) -> Path:
    """A data home with every kind of setting in it."""
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "calibration.json").write_text('{"residual_m": 0.05}', encoding="utf-8")
    (cfg / "calibration_frame.jpg").write_bytes(b"\xff\xd8jpeg")
    (cfg / "ogn.json").write_text('{"airfield": "LSTB"}', encoding="utf-8")
    (cfg / "scoring.json").write_text('{"name": "Club rules"}', encoding="utf-8")
    (cfg / "folders.json").write_text('{"raw": "D:\\\\raw"}', encoding="utf-8")
    (home / ".env").write_text(
        "CAMERA_URL=rtsp://judge:secret@cam/stream\n"
        "RELAY_URL=http://192.168.0.118:5054\n"
        "RELAY_TOKEN=abc123\n"
        "TOUCHDOWN_ANALYZER_HOME=/somewhere\n",
        encoding="utf-8",
    )
    return home


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in settings_file.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_settings_travel_to_another_machine(tmp_path: Path) -> None:
    source = site(tmp_path / "field")
    pta = tmp_path / "LSTB.pta"
    names = settings_file.export_settings(source, pta)
    assert "config/calibration.json" in names and "settings.env: RELAY_TOKEN" in names
    with zipfile.ZipFile(pta) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        env = zf.read("settings.env").decode()
    assert manifest["format"] == "pta-settings" and manifest["version"] == 1
    # the site's .env settings, not the machine's
    assert "RELAY_TOKEN=abc123" in env and "TOUCHDOWN_ANALYZER_HOME" not in env

    target = tmp_path / "laptop"
    (target / "config").mkdir(parents=True)
    (target / "config" / "scoring.json").write_text('{"name": "old"}', encoding="utf-8")
    (target / "config" / "local.json").write_text("{}", encoding="utf-8")
    (target / ".env").write_text("CAMERA_URL=rtsp://old\nOTHER=kept\n", encoding="utf-8")
    changed, backup = settings_file.import_settings(target, pta)

    assert "config/scoring.json" in changed and ".env: CAMERA_URL" in changed
    for name in ("calibration.json", "calibration_frame.jpg", "ogn.json", "scoring.json"):
        assert (target / "config" / name).read_bytes() == (source / "config" / name).read_bytes()
    # what the archive does not have stays
    assert (target / "config" / "local.json").is_file()
    env_file = target / ".env"
    assert config.saved_value("CAMERA_URL", env_file) == "rtsp://judge:secret@cam/stream"
    assert config.saved_value("RELAY_TOKEN", env_file) == "abc123"
    assert config.saved_value("OTHER", env_file) == "kept"
    # the settings from before, as a file to import back
    files, env_before = settings_file.read_settings(backup)
    assert backup.parent == target / "backups"
    assert files["scoring.json"] == b'{"name": "old"}' and env_before == {
        "CAMERA_URL": "rtsp://old"
    }


def test_export_of_a_fresh_install(tmp_path: Path) -> None:
    pta = tmp_path / "empty.pta"
    assert settings_file.export_settings(tmp_path / "home", pta) == []
    assert settings_file.read_settings(pta) == ({}, {})


@pytest.mark.parametrize(
    "entry",
    ["../evil.json", "config/../evil.json", "config/sub/x.json", "config/.hidden", "raw/x.mp4"],
)
def test_import_refuses_anything_but_plain_config_files(tmp_path: Path, entry: str) -> None:
    pta = tmp_path / "bad.pta"
    with zipfile.ZipFile(pta, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": "pta-settings", "version": 1}))
        zf.writestr(entry, "x")
    home = tmp_path / "home"
    with pytest.raises(settings_file.SettingsFileError, match="unexpected entry"):
        settings_file.import_settings(home, pta)
    assert not (home / "config").exists() and not (tmp_path / "evil.json").exists()


def test_import_refuses_what_is_not_a_settings_file(tmp_path: Path) -> None:
    text = tmp_path / "notes.pta"
    text.write_text("hello", encoding="utf-8")
    with pytest.raises(settings_file.SettingsFileError, match="not a settings file"):
        settings_file.read_settings(text)
    other = tmp_path / "other.pta"
    with zipfile.ZipFile(other, "w") as zf:
        zf.writestr("config/ogn.json", "{}")
    with pytest.raises(settings_file.SettingsFileError, match="no manifest"):
        settings_file.read_settings(other)
    newer = tmp_path / "newer.pta"
    with zipfile.ZipFile(newer, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": "pta-settings", "version": 99}))
    with pytest.raises(settings_file.SettingsFileError, match="newer program"):
        settings_file.read_settings(newer)
