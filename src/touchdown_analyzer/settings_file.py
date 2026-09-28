"""A site's settings in one file: ``*.pta``, a zip.

Everything that makes one airfield's installation what it is, to carry it to
another machine or keep it: the files in ``config/`` - calibration and its
still, airfield (OGN), scoring rules, folders - and the settings kept in
``.env``: the camera URL and the relay's address and token. Recordings and
results are not settings and stay out.

The file carries the camera password and the relay token - keep it like a
password.

Layout of the zip::

    manifest.json        format, version, the program's version, when, what
    config/<name>        each file of config/ (top level only)
    settings.env         CAMERA_URL=..., RELAY_URL=..., RELAY_TOKEN=...

Importing replaces the files the archive has and leaves the others alone;
the settings as they were go to ``backups/settings-before-import-<time>.pta``
first, so an import can be undone by importing that.
"""

from __future__ import annotations

import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from touchdown_analyzer import __version__, config
from touchdown_analyzer.control import relay

FORMAT = "pta-settings"
FORMAT_VERSION = 1
SUFFIX = ".pta"
MANIFEST = "manifest.json"
ENV_MEMBER = "settings.env"
CONFIG_DIR = "config"
BACKUP_DIR = "backups"
# The .env settings that belong to the site. Nothing else of .env is taken
# from an archive (TOUCHDOWN_ANALYZER_HOME and the like stay the machine's).
ENV_KEYS = (config.SOURCE_KEY, relay.URL_KEY, relay.TOKEN_KEY)
# a plain file name: no directories, no leading dot
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")
MAX_BYTES = 200 * 1024 * 1024  # all members together, unpacked


class SettingsFileError(Exception):
    """Not a settings file this program can read."""


def export_settings(home: Path, dest: Path) -> list[str]:
    """Write the site's settings to ``dest`` (a .pta file); returns what went in."""
    names: list[str] = []
    config_dir = home / CONFIG_DIR
    files = sorted(p for p in (config_dir.iterdir() if config_dir.is_dir() else []) if p.is_file())
    env = {key: value for key in ENV_KEYS if (value := config.saved_value(key, home / ".env"))}
    tmp = dest.with_name(dest.name + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if SAFE_NAME.match(path.name):
                zf.write(path, f"{CONFIG_DIR}/{path.name}")
                names.append(f"{CONFIG_DIR}/{path.name}")
        if env:
            zf.writestr(ENV_MEMBER, "".join(f"{k}={v}\n" for k, v in env.items()))
            names += [f"{ENV_MEMBER}: {key}" for key in env]
        manifest = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "program": __version__,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "contents": names,
        }
        zf.writestr(MANIFEST, json.dumps(manifest, indent=2) + "\n")
    tmp.replace(dest)
    return names


def read_settings(src: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    """What a .pta file holds: config files by name, and the .env settings.

    Checked before anything is written: a zip with the manifest, plain file
    names only, a sane size.
    """
    try:
        zf = zipfile.ZipFile(src)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SettingsFileError(f"{src.name} is not a settings file: {exc}") from exc
    with zf:
        try:
            manifest = json.loads(zf.read(MANIFEST))
        except (KeyError, ValueError) as exc:
            raise SettingsFileError(f"{src.name} has no {MANIFEST}") from exc
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
            raise SettingsFileError(f"{src.name} is not a {FORMAT} file")
        if int(manifest.get("version") or 0) > FORMAT_VERSION:
            raise SettingsFileError(
                f"{src.name} was written by a newer program ({manifest.get('program')})"
            )
        if sum(info.file_size for info in zf.infolist()) > MAX_BYTES:
            raise SettingsFileError(f"{src.name} is too large for a settings file")
        files: dict[str, bytes] = {}
        env: dict[str, str] = {}
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or name == MANIFEST:
                continue
            if name == ENV_MEMBER:
                for line in zf.read(info).decode("utf-8-sig").splitlines():
                    key, _, value = line.strip().partition("=")
                    if key.strip() in ENV_KEYS:
                        env[key.strip()] = value.strip()
                continue
            folder, _, base = name.partition("/")
            if folder != CONFIG_DIR or not SAFE_NAME.match(base):
                raise SettingsFileError(f"{src.name}: unexpected entry {name!r}")
            files[base] = zf.read(info)
    return files, env


def import_settings(home: Path, src: Path) -> tuple[list[str], Path]:
    """Take the settings of ``src`` over; returns what changed and the backup."""
    files, env = read_settings(src)
    backups = home / BACKUP_DIR
    backups.mkdir(parents=True, exist_ok=True)
    backup = backups / f"settings-before-import-{datetime.now():%Y-%m-%d_%H-%M-%S}{SUFFIX}"
    export_settings(home, backup)
    config_dir = home / CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    changed: list[str] = []
    for name, data in files.items():
        tmp = config_dir / f".{name}.part"
        tmp.write_bytes(data)
        tmp.replace(config_dir / name)
        changed.append(f"{CONFIG_DIR}/{name}")
    for key, value in env.items():
        config.remember_value(key, value, home / ".env")
        changed.append(f".env: {key}")
    return changed, backup
