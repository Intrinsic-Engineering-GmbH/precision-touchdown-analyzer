from __future__ import annotations

import pytest

from touchdown_analyzer import __version__
from touchdown_analyzer.capture import ffmpeg as ff
from touchdown_analyzer.cli import EXIT_OK, EXIT_TOOLING, build_parser, main


def test_version_is_set() -> None:
    assert __version__


def test_main_without_a_command_prints_help() -> None:
    assert main([]) == EXIT_OK


@pytest.mark.parametrize("command", ["record", "probe", "index"])
def test_commands_are_registered(command: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([command, "--help"])


def test_record_defaults_to_todays_session() -> None:
    args = build_parser().parse_args(["record", "--source", "rtsp://cam/stream"])
    assert args.session
    assert args.segment_seconds == 10


def test_missing_ffmpeg_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def absent(name: str, override: str | None = None) -> str:
        raise ff.FfmpegNotFound(f"{name} not found on PATH.")

    monkeypatch.setattr(ff, "find_tool", absent)
    assert main(["probe", "--source", "rtsp://cam/stream"]) == EXIT_TOOLING


def test_serve_defaults_avoid_the_reserved_port() -> None:
    """8000 is reserved by http.sys on many Windows machines."""
    args = build_parser().parse_args(["serve"])
    assert args.port != 8000
    assert args.host == "127.0.0.1"
    assert args.results is None  # beside the root, as the analyse command does


def test_launcher_folders_default_under_data_and_round_trip(tmp_path) -> None:
    from touchdown_analyzer import launcher

    assert launcher.load_folders(tmp_path) == (
        tmp_path / "data" / "raw",
        tmp_path / "data" / "landings",
    )
    raw, results = tmp_path / "D" / "video", tmp_path / "E" / "results"
    launcher.save_folders(tmp_path, raw, results)
    assert launcher.load_folders(tmp_path) == (raw, results)
    command = launcher.server_command("127.0.0.1", 8080, raw, results)
    assert command[command.index("--root") + 1] == str(raw)
    assert command[command.index("--results") + 1] == str(results)
    args = build_parser().parse_args(command[command.index("serve") :])
    assert (args.root, args.results) == (raw, results)


def test_bind_problem_detects_a_taken_port() -> None:
    import socket

    from touchdown_analyzer.cli import bind_problem

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        assert bind_problem("127.0.0.1", port) != ""
    finally:
        holder.close()


def test_bind_problem_accepts_a_free_port() -> None:
    import socket

    from touchdown_analyzer.cli import bind_problem

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    assert bind_problem("127.0.0.1", port) == ""


def test_serve_reports_a_blocked_port(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("touchdown_analyzer.cli.bind_problem", lambda h, p: "permission denied")
    assert main(["serve", "--port", "8000"]) == EXIT_TOOLING
    assert "cannot serve on" in capsys.readouterr().out


def test_index_of_a_missing_session(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ff, "find_tool", lambda name, override=None: name)
    assert main(["index", "--session", "nope", "--root", str(tmp_path)]) == EXIT_TOOLING
