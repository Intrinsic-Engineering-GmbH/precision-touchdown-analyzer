"""The Windows setup program: a small wizard around winstall.install().

Built as a single executable (packaging/windows/setup.spec) that carries
the application folder as app.zip. Installs machine-wide to Program Files
by default - the wizard runs without privileges and asks for
administrator rights (the UAC prompt) only when *Install* is pressed and
the chosen folder needs them; it then runs a second, elevated copy of
itself in silent mode and waits for it. A folder under the user's
profile installs without the prompt, per user. Where the recordings and
results go is chosen in the control window, not here.

    PTA-Setup.exe                     the wizard
    PTA-Setup.exe /S                  silent, defaults (asks for elevation)
    PTA-Setup.exe /S /D=C:\\path      silent, that program folder
    PTA-Setup.exe /S /DATA=D:\\path   silent, that data folder
    PTA-Setup.exe /S /NODESKTOP /NOMENU
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

from touchdown_analyzer import __version__, paths, winstall


def bundle_path() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return base / "app.zip"


@dataclass(frozen=True)
class Options:
    install_dir: Path
    data_dir: Path
    desktop: bool = True
    start_menu: bool = True
    silent: bool = False
    log: Path | None = None  # where a silent run leaves its error message

    def argv(self) -> list[str]:
        """The silent command line that reproduces these options."""
        args = ["/S", f"/D={self.install_dir}", f"/DATA={self.data_dir}"]
        if not self.desktop:
            args.append("/NODESKTOP")
        if not self.start_menu:
            args.append("/NOMENU")
        if self.log:
            args.append(f"/LOG={self.log}")
        return args


def parse(argv: list[str]) -> Options:
    options = Options(winstall.default_install_dir(), winstall.default_data_dir())
    for arg in argv:
        upper = arg.upper()
        if upper == "/S":
            options = replace(options, silent=True)
        elif upper.startswith("/D="):
            options = replace(options, install_dir=Path(arg[3:].strip('"')))
        elif upper.startswith("/DATA="):
            options = replace(options, data_dir=Path(arg[6:].strip('"')))
        elif upper.startswith("/LOG="):
            options = replace(options, log=Path(arg[5:].strip('"')))
        elif upper == "/NODESKTOP":
            options = replace(options, desktop=False)
        elif upper == "/NOMENU":
            options = replace(options, start_menu=False)
    return options


def needs_elevation(options: Options) -> bool:
    if winstall.is_admin():
        return False
    return not (winstall.writable(options.install_dir) and winstall.writable(options.data_dir))


def install_elevated(options: Options) -> Path:
    """Run the silent install through UAC; raise with its message if it fails."""
    log = Path(tempfile.gettempdir()) / f"{paths.APP_NAME}-Setup.log"
    log.unlink(missing_ok=True)
    code = winstall.run_elevated(replace(options, silent=True, log=log).argv())
    if code != 0:
        message = ""
        if log.is_file():
            message = log.read_text(encoding="utf-8", errors="replace").strip()
        raise RuntimeError(message or f"the elevated setup exited with code {code}")
    return options.install_dir / winstall.EXE_NAME


def silent_install(options: Options) -> int:
    if needs_elevation(options):
        try:
            install_elevated(options)
        except (PermissionError, RuntimeError, OSError):
            return 1
        return 0
    try:
        winstall.install(
            bundle_path(),
            options.install_dir,
            options.data_dir,
            desktop=options.desktop,
            start_menu=options.start_menu,
        )
    except Exception as exc:  # noqa: BLE001 - reported to the parent through the log
        if options.log:
            options.log.write_text(
                f"{exc}\n\n{traceback.format_exc()}", encoding="utf-8", errors="replace"
            )
        return 1
    return 0


def wizard() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title(f"{paths.APP_TITLE} {__version__} - Setup")
    root.resizable(False, False)
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except tk.TclError:
        pass

    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)
    ttk.Label(frame, text=paths.APP_TITLE, font=("Segoe UI", 14, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w"
    )
    ttk.Label(
        frame,
        text="Video-based landing measurement for glider spot-landing competitions.\n"
        "Installing under Program Files makes the program available to every user of this "
        "computer; Windows will ask for administrator rights when you press Install. "
        "The folders for recordings and results are chosen in the control window.",
        wraplength=500,
        justify="left",
    ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 14))

    def folder_row(row: int, label: str, initial: Path, hint: str) -> tk.StringVar:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=(6, 0))
        var = tk.StringVar(value=str(initial))
        ttk.Entry(frame, textvariable=var, width=52).grid(
            row=row, column=1, sticky="ew", padx=6, pady=(6, 0)
        )

        def browse() -> None:
            chosen = filedialog.askdirectory(initialdir=var.get(), mustexist=False)
            if chosen:
                chosen_path = Path(chosen)
                if chosen_path.name != paths.APP_NAME:
                    chosen_path = chosen_path / paths.APP_NAME
                var.set(str(chosen_path))

        ttk.Button(frame, text="Browse\u2026", command=browse).grid(row=row, column=2, pady=(6, 0))
        ttk.Label(frame, text=hint, foreground="#6b7480", wraplength=500, justify="left").grid(
            row=row + 1, column=1, columnspan=2, sticky="w", padx=6
        )
        return var

    dest_var = folder_row(2, "Program folder", winstall.default_install_dir(), "")

    desktop_var = tk.BooleanVar(value=True)
    menu_var = tk.BooleanVar(value=True)
    launch_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frame, text="Desktop shortcut", variable=desktop_var).grid(
        row=6, column=0, columnspan=3, sticky="w", pady=(12, 0)
    )
    ttk.Checkbutton(frame, text="Start menu entry", variable=menu_var).grid(
        row=7, column=0, columnspan=3, sticky="w"
    )
    ttk.Checkbutton(frame, text="Open the control window when finished", variable=launch_var).grid(
        row=8, column=0, columnspan=3, sticky="w"
    )

    progress = ttk.Progressbar(frame, mode="determinate", length=500)
    progress.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(16, 2))
    status = ttk.Label(frame, text="", foreground="#6b7480")
    status.grid(row=10, column=0, columnspan=3, sticky="w")

    buttons = ttk.Frame(frame)
    buttons.grid(row=11, column=0, columnspan=3, sticky="e", pady=(14, 0))
    install_btn = ttk.Button(buttons, text="Install")
    cancel_btn = ttk.Button(buttons, text="Cancel", command=root.destroy)
    install_btn.pack(side="left", padx=(0, 6))
    cancel_btn.pack(side="left")

    def on_progress(done: int, total: int, name: str) -> None:
        root.after(
            0, lambda: (progress.configure(maximum=total, value=done), status.configure(text=name))
        )

    def finish(exe: Path | None, error: str) -> None:
        progress.stop()
        progress.configure(mode="determinate")
        if error:
            messagebox.showerror(paths.APP_TITLE, f"Installation failed:\n{error}")
            install_btn.configure(state="normal")
            cancel_btn.configure(state="normal")
            status.configure(text="")
            return
        progress.configure(maximum=1, value=1)
        status.configure(text=f"Installed to {exe.parent if exe else ''}")
        if launch_var.get() and exe is not None:
            subprocess.Popen([str(exe)], cwd=str(exe.parent))  # noqa: S603 - the program just installed
        messagebox.showinfo(paths.APP_TITLE, "Installation complete.")
        root.destroy()

    def run_install() -> None:
        options = Options(
            Path(dest_var.get().strip()),
            winstall.default_data_dir(),
            desktop=desktop_var.get(),
            start_menu=menu_var.get(),
        )
        if not options.install_dir.is_absolute():
            messagebox.showerror(paths.APP_TITLE, "The program folder must be a full path.")
            return
        install_btn.configure(state="disabled")
        cancel_btn.configure(state="disabled")
        elevate = needs_elevation(options)
        if elevate:
            progress.configure(mode="indeterminate")
            progress.start(12)
            status.configure(text="Waiting for administrator approval, then installing\u2026")

        def work() -> None:
            try:
                if elevate:
                    exe = install_elevated(options)
                else:
                    exe = winstall.install(
                        bundle_path(),
                        options.install_dir,
                        options.data_dir,
                        desktop=options.desktop,
                        start_menu=options.start_menu,
                        progress=on_progress,
                    )
                root.after(0, finish, exe, "")
            except Exception as exc:  # noqa: BLE001 - shown to the user
                root.after(0, finish, None, str(exc))

        threading.Thread(target=work, daemon=True).start()

    install_btn.configure(command=run_install)
    root.mainloop()
    return 0


def main() -> int:
    options = parse(sys.argv[1:])
    if options.silent:
        return silent_install(options)
    return wizard()


if __name__ == "__main__":
    raise SystemExit(main())
