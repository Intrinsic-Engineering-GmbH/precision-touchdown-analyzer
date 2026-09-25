"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import threading
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from touchdown_analyzer import __version__
from touchdown_analyzer.capture import ffmpeg as ff
from touchdown_analyzer.capture import probe as probe_mod
from touchdown_analyzer.capture import recorder as recorder_mod
from touchdown_analyzer.capture import segments as segments_mod
from touchdown_analyzer.config import (
    SOURCE_KEY,
    TARGET_FPS,
    RecorderConfig,
    redact,
    remember_source,
    saved_source,
)

EXIT_OK = 0
EXIT_TOOLING = 2
EXIT_DISK = 3
EXIT_CHECKS_FAILED = 4

SOURCE_HELP = "RTSP URL or video file. Remembered in .env; omit it next time to reuse the last one"

DEFAULT_ROOT = Path("data/raw")
# Not 8000: Windows reserves it on many machines (http.sys / Hyper-V ranges),
# where binding fails with a bare WinError 10013.
DEFAULT_PORT = 8080


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="touchdown-analyzer",
        description="Analyze glider landings: segment clips and measure touchdown displacement.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--ffmpeg", help="path to the ffmpeg binary")
    parser.add_argument("--ffprobe", help="path to the ffprobe binary")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    check = sub.add_parser(
        "probe",
        help="pre-flight camera check",
        description="Record a short sample and verify the camera against docs/design.md 2.4.",
    )
    check.add_argument("--source", help=SOURCE_HELP)
    check.add_argument("--seconds", type=float, default=30.0, help="sample length (default: 30)")
    check.add_argument("--target-fps", type=float, default=TARGET_FPS)
    check.add_argument("--rtsp-transport", default="tcp", choices=["tcp", "udp"])
    check.add_argument(
        "--keep-sample", type=Path, help="write the sample here instead of a tempdir"
    )
    check.set_defaults(func=cmd_probe)

    rec = sub.add_parser(
        "record",
        help="record continuously into fixed-length segments",
        description="Record until interrupted, restarting ffmpeg across failures.",
    )
    rec.add_argument("--source", help=SOURCE_HELP)
    rec.add_argument("--session", default=date.today().isoformat(), help="session name")
    rec.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    rec.add_argument("--segment-seconds", type=int, default=10)
    rec.add_argument("--target-fps", type=float, default=TARGET_FPS)
    rec.add_argument("--rtsp-transport", default="tcp", choices=["tcp", "udp"])
    rec.add_argument("--min-free-gb", type=float, default=20.0, help="stop below this free space")
    rec.add_argument(
        "--duration",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="stop after this long (default: run until interrupted)",
    )
    rec.set_defaults(func=cmd_record)

    idx = sub.add_parser(
        "index",
        help="build the segment index for a recorded session",
        description="ffprobe every segment and report continuity. Safe to re-run.",
    )
    idx.add_argument("--session", required=True)
    idx.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    idx.add_argument("--target-fps", type=float, default=TARGET_FPS)
    idx.add_argument("--hash", action="store_true", help="also record a sha256 per segment")
    idx.set_defaults(func=cmd_index)

    ana = sub.add_parser(
        "analyze",
        help="find the landings in a recorded session and measure them",
        description="Track every aircraft through the raw segments, find the touchdown "
        "and its offset from the target line, and cut a clip per landing.",
    )
    ana.add_argument("--session", required=True)
    ana.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ana.add_argument(
        "--out",
        type=Path,
        help="where landings.json, clips and overlays go (default: <root>/../landings/<session>)",
    )
    ana.add_argument("--calibration", type=Path, default=Path("config/calibration.json"))
    ana.add_argument("--segment", action="append", help="analyse only this segment (repeatable)")
    ana.add_argument("--no-clips", action="store_true", help="skip cutting clips")
    ana.add_argument(
        "--fresh", action="store_true", help="discard the session's earlier results first"
    )
    ana.set_defaults(func=cmd_analyze)

    web = sub.add_parser(
        "serve",
        help="browser UI to control captures",
        description="Start, stop and monitor recordings from a browser.",
    )
    web.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    web.add_argument(
        "--results",
        type=Path,
        help="where landings.json, clips and overlays go (default: <root>/../landings)",
    )
    web.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices")
    web.add_argument("--port", type=int, default=DEFAULT_PORT)
    web.set_defaults(func=cmd_serve)

    gui = sub.add_parser(
        "launcher",
        help="control window: start / stop the server, watch the services",
        description="A small desktop window that runs the web server and shows what the "
        "recorder, the analysis worker and the OGN poller are doing.",
    )
    gui.add_argument("--autostart", action="store_true", help="start the server right away")
    gui.set_defaults(func=cmd_launcher)

    return parser


def cmd_launcher(args: argparse.Namespace) -> int:
    from touchdown_analyzer import launcher

    if args.autostart and "--autostart" not in sys.argv:
        sys.argv.append("--autostart")
    return launcher.run()


def resolve_source(given: str | None) -> str:
    """The source on the command line, else the one saved in .env.

    A URL given explicitly is saved for next time - the camera password is
    the tedious part to retype, and .env is git-ignored for exactly this.
    """
    if given and given.strip():
        source = given.strip()
        if source != saved_source():
            path = remember_source(source)
            print(f"(remembered as {SOURCE_KEY} in {path}; omit --source next time)")
        return source
    saved = saved_source()
    if saved:
        print(f"using saved source {redact(saved)}")
        return saved
    raise SystemExit(
        f"no --source given and no {SOURCE_KEY} saved yet. Pass --source once, "
        "or add it to .env (see .env.example)."
    )


def cmd_probe(args: argparse.Namespace) -> int:
    args.source = resolve_source(args.source)
    ffmpeg = ff.find_tool("ffmpeg", args.ffmpeg)
    ffprobe = ff.find_tool("ffprobe", args.ffprobe)

    print(f"Probing {redact(args.source)} for {args.seconds:.0f}s ...\n")
    checks = probe_mod.run(
        args.source,
        ffmpeg,
        ffprobe,
        seconds=args.seconds,
        target_fps=args.target_fps,
        rtsp_transport=args.rtsp_transport,
        keep_sample=args.keep_sample,
    )
    print(probe_mod.format_report(checks))
    return EXIT_CHECKS_FAILED if probe_mod.worst_status(checks) == probe_mod.FAIL else EXIT_OK


def cmd_record(args: argparse.Namespace) -> int:
    config = RecorderConfig(
        source=resolve_source(args.source),
        session=args.session,
        root=args.root,
        segment_seconds=args.segment_seconds,
        target_fps=args.target_fps,
        rtsp_transport=args.rtsp_transport,
        min_free_gb=args.min_free_gb,
        duration_s=args.duration,
        ffmpeg=ff.find_tool("ffmpeg", args.ffmpeg),
        ffprobe=ff.find_tool("ffprobe", args.ffprobe),
    )

    stop = threading.Event()

    def request_stop(*_: object) -> None:
        if not stop.is_set():
            print("\nstopping - finalising the current segment ...")
            stop.set()

    signal.signal(signal.SIGINT, request_stop)
    with_term = getattr(signal, "SIGTERM", None)
    if with_term is not None:
        signal.signal(with_term, request_stop)

    print(f"Recording {config.safe_source}")
    print(f"  -> {config.session_dir}  ({config.segment_seconds}s segments)")
    print("Press Ctrl+C to stop.\n")

    try:
        stats = recorder_mod.record(config, stop)
    except recorder_mod.DiskFull as exc:
        print(f"\nrecording stopped: {exc}")
        return EXIT_DISK

    print(f"\n{recorder_mod.summarise(stats)}")
    print(f"Next: touchdown-analyzer index --session {config.session} --root {config.root}")
    return EXIT_OK


def cmd_index(args: argparse.Namespace) -> int:
    ffprobe = ff.find_tool("ffprobe", args.ffprobe)
    session_dir = args.root / args.session
    if not session_dir.is_dir():
        print(f"no such session directory: {session_dir}")
        return EXIT_TOOLING

    segments = segments_mod.build_index(session_dir, ffprobe, hash_files=args.hash)
    report = segments_mod.report(args.session, segments, target_fps=args.target_fps)

    print(f"session {report.session}: {report.segments} segments")
    if not segments:
        print("  no readable segments found")
        return EXIT_TOOLING

    print(f"  from      {report.first_utc}")
    print(f"  to        {report.last_utc}")
    print(f"  recorded  {report.recorded_s / 60:.1f} min of a {report.span_s / 60:.1f} min span")
    print(f"  coverage  {report.coverage:.2%}")

    if report.gaps:
        print(f"  gaps      {len(report.gaps)}")
        for gap in report.gaps[:10]:
            print(f"    {gap['seconds']:8.1f}s after {gap['after']}")
        if len(report.gaps) > 10:
            print(f"    ... and {len(report.gaps) - 10} more")
    else:
        print("  gaps      none")

    if report.overlaps:
        print(f"  overlaps  {len(report.overlaps)} (camera and recorder clocks disagree)")
        for overlap in report.overlaps[:10]:
            print(f"    {overlap['seconds']:8.1f}s at {overlap['after']}")

    if report.fps_outliers:
        print(f"  fps       {len(report.fps_outliers)} segments off target")
        for outlier in report.fps_outliers[:10]:
            print(f"    {outlier}")

    print(f"\nindex written to {session_dir / segments_mod.INDEX_NAME}")
    return EXIT_OK


def landings_dir(root: Path, session: str) -> Path:
    """``data/raw`` -> ``data/landings/<session>``: results beside the raw footage."""
    return root.parent / "landings" / session


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        from touchdown_analyzer.analysis import pipeline
    except ImportError as exc:
        print(f"\nanalysis needs OpenCV: {exc}")
        print('  pip install -e ".[analysis]"')
        return EXIT_TOOLING
    from touchdown_analyzer.calibration import homography as hg
    from touchdown_analyzer.store.landings import LandingStore

    session_dir = args.root / args.session
    if not session_dir.is_dir():
        print(f"no such session directory: {session_dir}")
        return EXIT_TOOLING
    if not args.calibration.is_file():
        print(f"no calibration at {args.calibration}; calibrate first (touchdown-analyzer serve)")
        return EXIT_TOOLING
    calibration = hg.load(args.calibration)

    ffmpeg = None if args.no_clips else ff.find_tool("ffmpeg", args.ffmpeg)
    ffprobe = ff.find_tool("ffprobe", args.ffprobe)
    try:
        index = segments_mod.load_index(session_dir)
    except FileNotFoundError:
        index = segments_mod.build_index(session_dir, ffprobe)
    refs = pipeline.segment_refs(session_dir, index)
    if args.segment:
        wanted = set(args.segment)
        refs = [r for r in refs if r.name in wanted]
    if not refs:
        print("nothing to analyse")
        return EXIT_TOOLING

    out_dir = args.out or landings_dir(args.root, args.session)
    store = LandingStore(out_dir)
    if args.fresh:
        store.clear()
    analyzer = pipeline.Analyzer(
        args.session,
        calibration,
        store,
        out_dir=out_dir,
        ffmpeg=ffmpeg,
        cut_clips=not args.no_clips,
    )
    analyzer.add_pieces(pipeline.segment_refs(session_dir, index))

    print(f"analysing {len(refs)} segment(s) of {args.session}")
    print(
        f"calibration residual {calibration.residual_m:.2f} m"
        + ("" if calibration.acceptable else "  (above target - numbers will be rough)")
    )

    def found(landing) -> None:  # noqa: ANN001
        store.add(landing)
        when = (landing.touchdown_utc or "")[11:19]
        print(
            f"  {landing.id}  {when} UTC  {landing.label():>10}  "
            f"{landing.outcome:<10} {'->' if landing.direction > 0 else '<-'}"
            f"{'  ' + '; '.join(landing.flags) if landing.flags else ''}"
        )

    for ref in refs:
        print(f"{ref.name} ...", flush=True)
        analyzer.run_segment(ref, on_landing=found)
    analyzer.finish(on_landing=found)

    results = store.all()
    print(f"\n{len(results)} track(s) recorded in {store.path}")
    return EXIT_OK


def bind_problem(host: str, port: int) -> str:
    """Why ``host:port`` cannot be bound, or ``""`` if it can.

    Checked up front because uvicorn reports a bind failure as a bare
    ``WinError 10013`` and exits, which says nothing about the usual cause on
    Windows: a port reserved by http.sys or a Hyper-V dynamic range.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # No SO_REUSEADDR: on Windows it would let this bind succeed even
        # when another process already holds the port.
        probe.bind((host, port))
    except OSError as exc:
        return exc.strerror or str(exc)
    finally:
        probe.close()
    return ""


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from touchdown_analyzer.control.app import serve
    except ImportError as exc:
        print(f"\nthe control UI needs fastapi and uvicorn: {exc}")
        print('  pip install -e ".[ui]"')
        return EXIT_TOOLING

    problem = bind_problem(args.host, args.port)
    if problem:
        print(f"\ncannot serve on {args.host}:{args.port} - {problem}")
        print("Another program may hold that port, or Windows has reserved it.")
        print("  netsh interface ipv4 show excludedportrange protocol=tcp")
        print(f"Pick another one, for example:  touchdown-analyzer serve --port {args.port + 1}")
        return EXIT_TOOLING

    args.root.mkdir(parents=True, exist_ok=True)
    shown = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    # VS Code's serverReadyAction matches this line to open a browser, so it
    # has to reach the terminal before uvicorn blocks the thread.
    print(
        f"Capture control on http://{shown}:{args.port}  (sessions under {args.root})",
        flush=True,
    )
    if args.host == "0.0.0.0":  # noqa: S104 - deliberate, and called out to the user
        print("Reachable from other devices on this network. There is no login;")
        print("only do this on a network you trust.")
    print("Press Ctrl+C to stop.\n")

    serve(
        args.root,
        args.host,
        args.port,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        results=args.results,
    )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_OK

    try:
        return int(args.func(args))
    except ff.FfmpegNotFound as exc:
        print(f"\n{exc}")
        return EXIT_TOOLING
    except ff.ProbeError as exc:
        print(f"\ncould not read the source: {exc}")
        return EXIT_TOOLING
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
