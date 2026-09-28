"""The control window: start and stop the server, watch the services.

A small tkinter application - tkinter ships with Python on Windows and
with ``python3-tk`` on Debian, so this adds no dependency. It runs the web
server as a child process (the same program with ``serve``), tails its log,
polls its API for what the recorder, the analysis worker and the OGN
poller are doing, and opens the browser on it.

Installed, this is what the Start-menu entry runs. From a checkout:
``touchdown-analyzer launcher``.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

from touchdown_analyzer import __version__, config, paths, settings_file
from touchdown_analyzer.control import relay

DEFAULT_PORT = int(os.environ.get("TOUCHDOWN_ANALYZER_PORT", "8080") or 8080)
POLL_MS = 2000
# The gliding badge of the web pages, rendered by packaging/make_icon.py --static.
ICONS = Path(__file__).parent / "control" / "static"
LOG_LINES = 400
FOLDERS_NAME = "folders.json"  # in config/: where recordings and results go


def load_folders(home: Path) -> tuple[Path, Path]:
    """The recordings (raw) and results folders: as saved, else under data/.

    Relative paths are taken from the data home.
    """
    raw, results = home / "data" / "raw", home / "data" / "landings"
    try:
        saved = json.loads((home / "config" / FOLDERS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return raw, results
    if not isinstance(saved, dict):
        return raw, results
    if saved.get("raw"):
        raw = home / str(saved["raw"])
    if saved.get("results"):
        results = home / str(saved["results"])
    return raw, results


def save_folders(home: Path, raw: Path, results: Path) -> Path:
    path = home / "config" / FOLDERS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"raw": str(raw), "results": str(results)}, indent=2) + "\n", encoding="utf-8"
    )
    return path


def server_command(host: str, port: int, root: Path, results: Path) -> list[str]:
    """How to start the server as a child of this program."""
    args = ["serve", "--host", host, "--port", str(port), "--root", str(root)]
    args += ["--results", str(results)]
    if paths.frozen():
        return [sys.executable, *args]
    return [sys.executable, "-m", "touchdown_analyzer", *args]


def fetch(url: str, timeout: float = 1.5) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - localhost
            return dict(json.loads(response.read().decode("utf-8")))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def post(url: str, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any] | None:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - localhost
            return dict(json.loads(response.read().decode("utf-8")))
    except (urllib.error.URLError, OSError, ValueError):
        return None


class ServerProcess:
    """The server child: start, stop, and its output as it comes."""

    def __init__(self, home: Path, log_path: Path) -> None:
        self.home = home
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None
        self.lines: queue.Queue[str] = queue.Queue()
        self.port = DEFAULT_PORT
        self.host = "127.0.0.1"

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, host: str, port: int, raw: Path, results: Path) -> None:
        if self.running:
            return
        self.host, self.port = host, port
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env[paths.ENV_HOME] = str(self.home)
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(  # noqa: S603 - our own program
            server_command(host, port, raw, results),
            cwd=self.home,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=creation,
        )
        threading.Thread(target=self._pump, args=(self.process,), daemon=True).start()

    def _pump(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stdout is not None
        with open(self.log_path, "a", encoding="utf-8") as log:
            for raw in process.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                log.write(line + "\n")
                log.flush()
                self.lines.put(line)
        code = process.wait()
        self.lines.put(f"[server exited with code {code}]")

    def stop(self, timeout: float = 8.0) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout)

    @property
    def url(self) -> str:
        shown = "localhost" if self.host in ("127.0.0.1", "0.0.0.0") else self.host
        return f"http://{shown}:{self.port}"


def run() -> int:
    """Open the control window. Returns the exit code."""
    import tkinter as tk
    from tkinter import ttk

    home = paths.enter_data_home()
    log_dir = home / "logs"
    log_dir.mkdir(exist_ok=True)
    server = ServerProcess(home, log_dir / f"server-{datetime.now():%Y-%m-%d}.log")

    if sys.platform == "win32" and not paths.frozen():
        # Run through python.exe, the taskbar would show Python's icon;
        # an own app id makes it take the window's.
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(paths.APP_NAME)

    root = tk.Tk()
    root.title(f"{paths.APP_TITLE} - control")
    root.minsize(640, 520)
    logo_image: tk.PhotoImage | None = None  # a local of run(): lives as long as the window
    try:
        root.iconphoto(
            True, *(tk.PhotoImage(file=str(ICONS / f"icon-{s}.png")) for s in (256, 48, 32, 16))
        )
        logo_image = tk.PhotoImage(file=str(ICONS / "icon-36.png"))
    except tk.TclError:
        pass

    style = ttk.Style(root)
    with_theme = "vista" if sys.platform == "win32" else "clam"
    if with_theme in style.theme_names():
        style.theme_use(with_theme)
    style.configure("Title.TLabel", font=("Segoe UI", 13, "bold"))
    style.configure("Muted.TLabel", foreground="#6b7480")
    style.configure("Ok.TLabel", foreground="#127a4a", font=("Segoe UI", 10, "bold"))
    style.configure("Bad.TLabel", foreground="#b3261e", font=("Segoe UI", 10, "bold"))
    style.configure("Warn.TLabel", foreground="#9a6100", font=("Segoe UI", 10, "bold"))

    outer = ttk.Frame(root, padding=14)
    outer.pack(fill="both", expand=True)

    header = ttk.Frame(outer)
    header.grid(row=0, column=0, rowspan=2, sticky="w", pady=(0, 10))
    if logo_image is not None:
        ttk.Label(header, image=logo_image).grid(row=0, column=0, rowspan=2, padx=(0, 10))
    ttk.Label(header, text=paths.APP_TITLE, style="Title.TLabel").grid(row=0, column=1, sticky="w")
    ttk.Label(header, text=f"v{__version__} · {home}", style="Muted.TLabel").grid(
        row=1, column=1, sticky="w"
    )

    # -- server ---------------------------------------------------------------
    box = ttk.LabelFrame(outer, text="Web server", padding=10)
    box.grid(row=2, column=0, sticky="ew")
    box.columnconfigure(5, weight=1)
    state = ttk.Label(box, text="stopped", style="Bad.TLabel")
    state.grid(row=0, column=0, sticky="w")
    ttk.Label(box, text="port").grid(row=0, column=1, padx=(16, 4))
    port_var = tk.StringVar(value=str(DEFAULT_PORT))
    ttk.Entry(box, textvariable=port_var, width=6).grid(row=0, column=2)
    lan_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(box, text="reachable on the network (no login)", variable=lan_var).grid(
        row=0, column=3, padx=(12, 0)
    )
    url_label = ttk.Label(box, text="", style="Muted.TLabel")
    url_label.grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 0))
    buttons = ttk.Frame(box)
    buttons.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(8, 0))
    start_btn = ttk.Button(buttons, text="Start")
    stop_btn = ttk.Button(buttons, text="Stop", state="disabled")
    open_btn = ttk.Button(buttons, text="Open in browser", state="disabled")
    start_btn.pack(side="left")
    stop_btn.pack(side="left", padx=6)
    open_btn.pack(side="left")
    # the site's settings as one .pta file (settings_file.py)
    import_btn = ttk.Button(buttons, text="Import settings…")
    export_btn = ttk.Button(buttons, text="Export settings…")
    import_btn.pack(side="right")
    export_btn.pack(side="right", padx=6)

    # -- folders ------------------------------------------------------------------
    # Where the recordings and the results go. Read when the server starts,
    # so they are locked while it runs.
    from tkinter import filedialog

    raw_dir, results_dir = load_folders(home)
    fb = ttk.LabelFrame(outer, text="Folders", padding=10)
    fb.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    fb.columnconfigure(1, weight=1)
    folder_vars: dict[str, tk.StringVar] = {}
    folder_widgets: list[ttk.Entry | ttk.Button] = []
    for i, (key, title, initial) in enumerate(
        (("raw", "Recordings (raw)", raw_dir), ("results", "Results", results_dir))
    ):
        ttk.Label(fb, text=title, width=16).grid(row=i, column=0, sticky="w", pady=1)
        var = tk.StringVar(value=str(initial))
        entry = ttk.Entry(fb, textvariable=var)
        entry.grid(row=i, column=1, sticky="ew", padx=(0, 6), pady=1)

        def browse(var: tk.StringVar = var, title: str = title) -> None:
            chosen = filedialog.askdirectory(
                parent=root, initialdir=var.get(), mustexist=False, title=title
            )
            if chosen:
                var.set(str(Path(chosen)))

        button = ttk.Button(fb, text="Browse…", command=browse)
        button.grid(row=i, column=2, pady=1)
        folder_vars[key] = var
        folder_widgets += [entry, button]
    ttk.Label(
        fb,
        text="Raw video is large - choose a disk with room for it. Used when the server starts.",
        style="Muted.TLabel",
    ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

    # -- airfield ---------------------------------------------------------------
    # Which field the OGN logbook and live feed are read for. The ICAO code
    # is enough: "Look up" asks the OGN FlightBook for the rest.
    from touchdown_analyzer.identify import ogn

    field = ogn.load_field(home / "config")
    af = ttk.LabelFrame(outer, text="Airfield (OGN identification)", padding=10)
    af.grid(row=4, column=0, sticky="ew", pady=(10, 0))
    line1 = ttk.Frame(af)
    line1.pack(fill="x")
    ttk.Label(line1, text="ICAO").pack(side="left")
    icao_var = tk.StringVar(value=field.airfield)
    icao_entry = ttk.Entry(line1, textvariable=icao_var, width=7)
    icao_entry.pack(side="left", padx=(4, 6))
    lookup_btn = ttk.Button(line1, text="Look up")
    lookup_btn.pack(side="left")
    name_label = ttk.Label(line1, text=field.name, style="Muted.TLabel")
    name_label.pack(side="left", padx=(10, 0))
    line2 = ttk.Frame(af)
    line2.pack(fill="x", pady=(6, 0))
    coords: dict[str, tk.StringVar] = {}
    for key, title, width in (
        ("lat", "lat", 10),
        ("lon", "lon", 10),
        ("elevation_m", "elev m", 6),
        ("radius_km", "radius km", 5),
    ):
        ttk.Label(line2, text=title).pack(side="left")
        coords[key] = tk.StringVar(value=f"{getattr(field, key):g}")
        ttk.Entry(line2, textvariable=coords[key], width=width).pack(side="left", padx=(4, 12))
    line3 = ttk.Frame(af)
    line3.pack(fill="x", pady=(8, 0))
    ogn_enabled_var = tk.BooleanVar(value=field.enabled)
    ttk.Checkbutton(line3, text="use OGN", variable=ogn_enabled_var).pack(side="left")
    save_field_btn = ttk.Button(line3, text="Save")
    save_field_btn.pack(side="left", padx=(12, 0))
    field_status = ttk.Label(line3, text="", style="Muted.TLabel")
    field_status.pack(side="left", padx=(10, 0))

    # -- public scoreboard ----------------------------------------------------------
    # The board goes to the relay on the Raspberry Pi (ptp-relay/README.md):
    # the analyzer pushes, the Pi never calls in, so this PC may be anywhere
    # on the network. Kept in .env - the token is a password - and read when
    # the server starts.
    env_file = home / config.ENV_FILE
    pb = ttk.LabelFrame(outer, text="Public scoreboard (relay)", padding=10)
    pb.grid(row=5, column=0, sticky="ew", pady=(10, 0))
    pb.columnconfigure(1, weight=1)
    relay_url_var = tk.StringVar(value=config.saved_value(relay.URL_KEY, env_file) or "")
    relay_token_var = tk.StringVar(value=config.saved_value(relay.TOKEN_KEY, env_file) or "")
    ttk.Label(pb, text="push to", width=16).grid(row=0, column=0, sticky="w")
    relay_url_entry = ttk.Entry(pb, textvariable=relay_url_var)
    relay_url_entry.grid(row=0, column=1, sticky="ew", padx=(0, 12))
    ttk.Label(pb, text="token").grid(row=0, column=2, padx=(0, 4))
    relay_token_entry = ttk.Entry(pb, textvariable=relay_token_var, width=22, show="•")
    relay_token_entry.grid(row=0, column=3)
    ttk.Label(
        pb,
        text="The relay's push port, e.g. http://192.168.0.118:5054, and PUSH_TOKEN of its .env."
        " Empty: off. Used when the server starts.",
        style="Muted.TLabel",
    ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

    # -- services -------------------------------------------------------------
    svc = ttk.LabelFrame(outer, text="Services", padding=10)
    svc.grid(row=6, column=0, sticky="ew", pady=(10, 0))
    svc.columnconfigure(1, weight=1)
    rows: dict[str, tuple[ttk.Label, ttk.Label]] = {}
    for i, (key, title) in enumerate(
        (
            ("recorder", "Recorder"),
            ("analysis", "Analysis worker"),
            ("ogn", "OGN"),
            ("calibration", "Calibration"),
            ("relay", "Public board"),
            ("tools", "ffmpeg"),
        )
    ):
        ttk.Label(svc, text=title, width=16).grid(row=i, column=0, sticky="w", pady=1)
        value = ttk.Label(svc, text="—", style="Muted.TLabel")
        value.grid(row=i, column=1, sticky="w")
        badge = ttk.Label(svc, text="", style="Muted.TLabel")
        badge.grid(row=i, column=2, sticky="e", padx=(8, 0))
        rows[key] = (value, badge)

    # -- log --------------------------------------------------------------------
    logbox = ttk.LabelFrame(outer, text="Server log", padding=6)
    logbox.grid(row=7, column=0, sticky="nsew", pady=(10, 0))
    outer.rowconfigure(7, weight=1)
    outer.columnconfigure(0, weight=1)
    text = tk.Text(logbox, height=12, wrap="none", font=("Consolas", 9), state="disabled")
    scroll = ttk.Scrollbar(logbox, command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    text.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    def log(line: str) -> None:
        text.configure(state="normal")
        text.insert("end", line + "\n")
        count = int(text.index("end-1c").split(".")[0])
        if count > LOG_LINES:
            text.delete("1.0", f"{count - LOG_LINES}.0")
        text.see("end")
        text.configure(state="disabled")

    # -- tools check (once) -------------------------------------------------------
    from touchdown_analyzer.capture import ffmpeg as ff

    try:
        ffmpeg = ff.find_tool("ffmpeg")
        rows["tools"][0].configure(text=ffmpeg)
        rows["tools"][1].configure(text="found", style="Ok.TLabel")
    except ff.FfmpegNotFound:
        rows["tools"][0].configure(
            text="not found - winget install Gyan.FFmpeg (Windows), apt install ffmpeg"
        )
        rows["tools"][1].configure(text="missing", style="Bad.TLabel")

    # -- actions ----------------------------------------------------------------
    def set_running(running: bool) -> None:
        state.configure(
            text="running" if running else "stopped", style="Ok.TLabel" if running else "Bad.TLabel"
        )
        start_btn.configure(state="disabled" if running else "normal")
        stop_btn.configure(state="normal" if running else "disabled")
        open_btn.configure(state="normal" if running else "disabled")
        url_label.configure(text=server.url if running else "")
        for widget in [*folder_widgets, relay_url_entry, relay_token_entry, import_btn]:
            widget.configure(state="disabled" if running else "normal")

    def read_folders() -> tuple[Path, Path] | None:
        chosen = []
        for key, title in (("raw", "recordings"), ("results", "results")):
            text = folder_vars[key].get().strip()
            if not text:
                log(f"[choose a {title} folder first]")
                return None
            folder = home / Path(text).expanduser()
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                log(f"[cannot use {folder} for the {title}: {exc}]")
                return None
            chosen.append(folder)
        return chosen[0], chosen[1]

    def start() -> None:
        try:
            port = int(port_var.get())
        except ValueError:
            log("[port must be a number]")
            return
        host = "0.0.0.0" if lan_var.get() else "127.0.0.1"  # noqa: S104 - the user asked
        folders = read_folders()
        if folders is None:
            return
        raw, results = folders
        try:
            save_folders(home, raw, results)
        except OSError as exc:
            log(f"[could not save the folders: {exc}]")
        try:
            for key, var in ((relay.URL_KEY, relay_url_var), (relay.TOKEN_KEY, relay_token_var)):
                if var.get().strip() != (config.saved_value(key, env_file) or ""):
                    config.remember_value(key, var.get(), env_file)
        except OSError as exc:
            log(f"[could not save the relay settings: {exc}]")
        log(f"[starting server on {host}:{port}, recordings in {raw}, results in {results}]")
        try:
            server.start(host, port, raw, results)
        except OSError as exc:
            log(f"[could not start: {exc}]")
            return
        set_running(True)

    def stop() -> None:
        log("[stopping server]")
        server.stop()
        set_running(False)

    def open_browser() -> None:
        webbrowser.open(server.url + "/landings")

    start_btn.configure(command=start)
    stop_btn.configure(command=stop)
    open_btn.configure(command=open_browser)

    def show_field(found: ogn.Field) -> None:
        icao_var.set(found.airfield)
        name_label.configure(text=found.name)
        for key, var in coords.items():
            var.set(f"{getattr(found, key):g}")
        ogn_enabled_var.set(found.enabled)

    def read_field() -> ogn.Field | None:
        try:
            return ogn.Field(
                airfield=icao_var.get().strip().upper(),
                name=name_label.cget("text"),
                lat=float(coords["lat"].get()),
                lon=float(coords["lon"].get()),
                elevation_m=float(coords["elevation_m"].get()),
                radius_km=float(coords["radius_km"].get()),
                enabled=ogn_enabled_var.get(),
                timezone_offset_h=field.timezone_offset_h,
            )
        except ValueError:
            field_status.configure(text="lat, lon, elevation and radius must be numbers")
            return None

    def lookup() -> None:
        code = icao_var.get().strip().upper()
        if not code:
            field_status.configure(text="enter the ICAO code first (LSTB)")
            return
        lookup_btn.configure(state="disabled")
        field_status.configure(text=f"asking the OGN FlightBook about {code}…")
        current = read_field() or field

        def work() -> None:
            try:
                found = ogn.fetch_airfield(code, current)
                message = "" if found else f"{code} is not known to OGN"
            except (OSError, ValueError) as exc:
                found, message = None, f"lookup failed: {exc}"
            root.after(0, done, found, message)

        def done(found: ogn.Field | None, message: str) -> None:
            lookup_btn.configure(state="normal")
            if found:
                show_field(found)
                message = (
                    f"{found.name} · {found.lat:.4f}, {found.lon:.4f}"
                    f" · {found.elevation_m:.0f} m - press Save"
                )
            field_status.configure(text=message)

        threading.Thread(target=work, daemon=True).start()

    def save_field() -> None:
        nonlocal field
        new = read_field()
        if new is None:
            return
        if new.enabled and not new.lat and not new.lon:
            field_status.configure(text="look the field up (or enter lat/lon) before enabling OGN")
            return
        field = new
        path = ogn.save_field(home / "config", field)
        message = f"saved to {path.name}"
        if server.running:
            if post(server.url + "/api/ogn", field.as_dict()) is None:
                message += " - the running server did not take it; restart it"
            else:
                message += " and handed to the running server"
        field_status.configure(text=message)
        log(
            f"[airfield {field.airfield or '-'} {'enabled' if field.enabled else 'off'}: {message}]"
        )

    lookup_btn.configure(command=lookup)
    icao_entry.bind("<Return>", lambda _event: lookup())
    save_field_btn.configure(command=save_field)

    # -- settings file ------------------------------------------------------------
    from tkinter import messagebox

    def export_settings() -> None:
        name = f"PTA-{field.airfield or 'site'}-{datetime.now():%Y-%m-%d}{settings_file.SUFFIX}"
        chosen = filedialog.asksaveasfilename(
            parent=root,
            title="Export settings",
            initialfile=name,
            defaultextension=settings_file.SUFFIX,
            filetypes=[("PTA settings", f"*{settings_file.SUFFIX}")],
        )
        if not chosen:
            return
        try:
            names = settings_file.export_settings(home, Path(chosen))
        except OSError as exc:
            log(f"[could not export the settings: {exc}]")
            return
        log(f"[settings exported to {chosen}: {', '.join(names) or 'nothing set yet'}]")
        log("[it holds the camera password and the relay token - keep it safe]")

    def import_settings() -> None:
        nonlocal field
        chosen = filedialog.askopenfilename(
            parent=root,
            title="Import settings",
            filetypes=[("PTA settings", f"*{settings_file.SUFFIX}"), ("All files", "*.*")],
        )
        if not chosen:
            return
        try:
            files, env = settings_file.read_settings(Path(chosen))
        except settings_file.SettingsFileError as exc:
            messagebox.showerror("Import settings", str(exc), parent=root)
            return
        listed = "\n".join([*(f"  config/{n}" for n in files), *(f"  {k}" for k in env)])
        if not messagebox.askyesno(
            "Import settings",
            f"Replace these settings with those from {Path(chosen).name}?\n\n{listed}\n\n"
            f"The current ones are kept in the folder {settings_file.BACKUP_DIR} first.",
            parent=root,
        ):
            return
        try:
            changed, backup = settings_file.import_settings(home, Path(chosen))
        except (OSError, settings_file.SettingsFileError) as exc:
            log(f"[could not import the settings: {exc}]")
            return
        # show what is now in effect
        raw, results = load_folders(home)
        folder_vars["raw"].set(str(raw))
        folder_vars["results"].set(str(results))
        field = ogn.load_field(home / "config")
        show_field(field)
        relay_url_var.set(config.saved_value(relay.URL_KEY, env_file) or "")
        relay_token_var.set(config.saved_value(relay.TOKEN_KEY, env_file) or "")
        field_status.configure(text="")
        log(f"[settings imported from {chosen}: {', '.join(changed)}]")
        log(f"[the settings before are in {backup}]")

    export_btn.configure(command=export_settings)
    import_btn.configure(command=import_settings)

    def fmt(label: ttk.Label, badge: ttk.Label, value: str, ok: bool | None) -> None:
        label.configure(text=value)
        badge.configure(
            text="" if ok is None else ("active" if ok else "idle"),
            style="Ok.TLabel" if ok else "Muted.TLabel",
        )

    def poll() -> None:
        while True:
            try:
                log(server.lines.get_nowait())
            except queue.Empty:
                break
        running = server.running
        if running != (stop_btn.instate(["!disabled"])):
            set_running(running)
        if running:
            status = fetch(server.url + "/api/status")
            analysis = fetch(server.url + "/api/analysis/status")
            if status:
                if status.get("recording"):
                    fmt(
                        *rows["recorder"],
                        f"recording {status.get('session')} · {status.get('fps', 0):.1f} fps"
                        f" · {status.get('segments', 0)} segments"
                        f" · {status.get('free_gb', 0):.0f} GB free",
                        True,
                    )
                else:
                    fmt(*rows["recorder"], f"idle · {status.get('free_gb', 0):.0f} GB free", False)
            if analysis:
                if not analysis.get("available"):
                    fmt(*rows["analysis"], "unavailable (OpenCV missing)", None)
                elif analysis.get("running"):
                    fmt(
                        *rows["analysis"],
                        f"{analysis.get('stage')} {analysis.get('session') or ''}"
                        f" · {analysis.get('done', 0)} done, {analysis.get('queue', 0)} queued"
                        + (
                            f" · last {analysis.get('last_landing')}"
                            if analysis.get("last_landing")
                            else ""
                        ),
                        True,
                    )
                else:
                    fmt(*rows["analysis"], "idle", False)
                ogn_info = analysis.get("ogn") or {}
                poller = ogn_info.get("poller") or {}
                field_info = ogn_info.get("field") or {}
                if poller.get("running"):
                    logged = poller.get("fixes_logged", 0)
                    fmt(*rows["ogn"], f"{field_info.get('airfield')} · {logged} fixes logged", True)
                else:
                    airfield = field_info.get("airfield") or "no airfield"
                    mode = "enabled" if field_info.get("enabled") else "off"
                    fmt(*rows["ogn"], f"{airfield} · {mode}", False)
            calibration = fetch(server.url + "/api/calibration")
            cal = (calibration or {}).get("calibration")
            if cal:
                fmt(
                    *rows["calibration"],
                    f"residual {cal['residual_m']:.2f} m"
                    + ("" if cal.get("acceptable") else " (rough - re-survey)"),
                    None,
                )
            elif calibration is not None:
                fmt(*rows["calibration"], "none saved - calibrate first", None)
            if status:
                pushed = status.get("relay")
                if not pushed:
                    fmt(*rows["relay"], "off - no relay set", None)
                elif pushed.get("ok"):
                    when = datetime.fromisoformat(pushed["last_push"]).astimezone()
                    fmt(*rows["relay"], f"{pushed['url']} · pushed {when:%H:%M:%S}", True)
                elif not pushed.get("error"):
                    fmt(*rows["relay"], f"{pushed['url']} · starting", None)
                else:
                    rows["relay"][0].configure(text=pushed["error"])
                    rows["relay"][1].configure(text="failing", style="Bad.TLabel")
            if analysis is None and status is None:
                fmt(*rows["recorder"], "server not answering yet", None)
        else:
            for key in ("recorder", "analysis", "ogn", "relay"):
                fmt(*rows[key], "— (server stopped)", None)
        root.after(POLL_MS, poll)

    def on_close() -> None:
        if server.running:
            server.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    set_running(False)
    log(f"[{paths.APP_TITLE} v{__version__} - data directory {home}]")
    if "--autostart" in sys.argv:
        start()
    root.after(300, poll)
    root.mainloop()
    return 0


def main() -> int:
    """GUI entry point: no arguments opens the window; arguments run the CLI.

    One executable serves both, so the installed program can start itself
    as the server child (``PTA.exe serve ...``).
    """
    args = [a for a in sys.argv[1:] if a != "--autostart"]
    if paths.frozen() and (sys.stdout is None or sys.stderr is None):
        # A windowed program has no console; anything printed would raise
        # or vanish. Keep it in a log file in the data directory instead.
        home = paths.enter_data_home()
        (home / "logs").mkdir(exist_ok=True)
        stream = open(  # noqa: SIM115 - lives for the whole process
            home / "logs" / f"{paths.APP_NAME}-{datetime.now():%Y-%m-%d}.log",
            "a",
            encoding="utf-8",
            buffering=1,
        )
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
        if os.environ.get("TDA_DEBUG_HANG"):
            # Where is it stuck? A stack dump into the log after N seconds.
            import faulthandler

            faulthandler.dump_traceback_later(
                float(os.environ["TDA_DEBUG_HANG"]), repeat=True, file=stream
            )
    if args and args[0] == "--uninstall":
        from touchdown_analyzer import winstall

        return winstall.uninstall_main(args[1:])
    if args:
        from touchdown_analyzer.cli import main as cli_main

        if paths.frozen():
            paths.enter_data_home()
        return cli_main(args)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
