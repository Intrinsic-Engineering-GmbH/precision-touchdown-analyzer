"""Live preview of the camera as an MJPEG stream.

Browsers cannot play RTSP, so ffmpeg decodes the stream, shrinks and decimates
it, and hands over a run of JPEGs; the web layer serves those as
``multipart/x-mixed-replace``, which a bare ``<img>`` tag plays with no
player and no script. This is a viewfinder for aiming the camera and checking
the strip is in frame - it is deliberately small and slow so it never competes
with the recorder for bandwidth or CPU.

A second RTSP session on the camera is fine; Axis cameras serve several. What
must not happen is a preview that keeps running after the browser has gone,
so the reader stops ffmpeg as soon as nobody has taken a frame for a while.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from collections.abc import Iterator

from touchdown_analyzer.capture import ffmpeg as ff
from touchdown_analyzer.config import is_network_source

BOUNDARY = "touchdown-frame"
PREVIEW_FPS = 8
PREVIEW_WIDTH = 960
JPEG_QUALITY = "6"  # -q:v; 2 is best, 31 worst

# ffmpeg is killed once no frame has been consumed for this long.
IDLE_TIMEOUT_S = 10.0
FRAME_WAIT_S = 15.0
QUIT_WAIT_S = 3.0  # for ffmpeg to tear the RTSP session down after 'q'

_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"


class PreviewError(RuntimeError):
    """The preview could not be started."""


def build_command(
    source: str,
    ffmpeg: str,
    *,
    fps: int = PREVIEW_FPS,
    width: int = PREVIEW_WIDTH,
    rtsp_transport: str = "tcp",
) -> list[str]:
    # stdin stays open: it is how close() asks ffmpeg to quit ('q').
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if source.startswith(("rtsp://", "rtsps://")):
        cmd += ["-rtsp_transport", rtsp_transport]
    if is_network_source(source):
        cmd += [ff.socket_timeout_flag(ffmpeg), "5000000"]
    else:
        # A file would otherwise be decoded at full speed and end; play it at
        # its own rate and loop it, so a recording can stand in for the camera.
        cmd += ["-re", "-stream_loop", "-1"]
    cmd += [
        "-i",
        source,
        "-an",
        "-vf",
        f"fps={fps},scale={width}:-2",
        "-c:v",
        "mjpeg",
        "-q:v",
        JPEG_QUALITY,
        "-f",
        "mjpeg",
        "pipe:1",
    ]
    return cmd


class Preview:
    """One ffmpeg process, one consumer, latest-frame-wins."""

    def __init__(self, cmd: list[str]) -> None:
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=ff.NO_WINDOW,
            )
        except OSError as exc:
            raise PreviewError(f"could not start ffmpeg: {exc}") from exc

        self._latest: bytes | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._last_taken = time.monotonic()
        self._error = ""
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        # Independent of the reader, which may be blocked in read() on a
        # source that has stalled: killing the process is what unblocks it.
        threading.Thread(target=self._watch_idle, daemon=True).start()

    # -- producer ---------------------------------------------------------

    def _watch_idle(self) -> None:
        while self._proc.poll() is None:
            if time.monotonic() - self._last_taken > IDLE_TIMEOUT_S:
                self._stop_process()  # the browser went away
                return
            time.sleep(0.5)

    def _read(self) -> None:
        assert self._proc.stdout is not None
        buffer = bytearray()
        try:
            while True:
                chunk = self._proc.stdout.read(65536)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    start = buffer.find(_SOI)
                    if start < 0:
                        buffer.clear()
                        break
                    end = buffer.find(_EOI, start + 2)
                    if end < 0:
                        if start:
                            del buffer[:start]
                        break
                    frame = bytes(buffer[start : end + 2])
                    del buffer[: end + 2]
                    with self._cond:
                        self._latest = frame
                        self._seq += 1
                        self._cond.notify_all()
        finally:
            self._stop_process()
            with self._cond:
                if self._proc.returncode not in (None, 0, -9, 137) and self._proc.stderr:
                    self._error = self._proc.stderr.read().decode("utf-8", "replace").strip()
                self._cond.notify_all()

    def _stop_process(self) -> None:
        """Ask ffmpeg to quit, and kill it only if it will not.

        A killed ffmpeg never sends the RTSP TEARDOWN, so the camera keeps
        the session until its own timeout; a few of those from preview
        restarts and the camera refuses the recorder's next connection.
        """
        if self._proc.poll() is not None:
            return
        with contextlib.suppress(OSError, ValueError):
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.write(b"q\n")
                self._proc.stdin.flush()
        try:
            self._proc.wait(timeout=QUIT_WAIT_S)
            return
        except subprocess.TimeoutExpired:
            pass
        self._proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.wait(timeout=5)

    # -- consumer ---------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._reader.is_alive()

    @property
    def error(self) -> str:
        return self._error

    def _wait_for_new(self, seen: int) -> None:
        """Block (holding the condition) until a frame newer than ``seen``."""
        self._cond.wait_for(lambda: self._seq != seen or not self.alive, FRAME_WAIT_S)

    def frames(self) -> Iterator[bytes]:
        """Yield each new frame as one multipart part; stops when ffmpeg does."""
        seen = 0
        try:
            while True:
                with self._cond:
                    self._wait_for_new(seen)
                    if self._seq == seen:
                        if not self.alive:
                            return
                        continue  # timed out but ffmpeg is still going
                    frame = self._latest
                    seen = self._seq
                    self._last_taken = time.monotonic()
                if frame is None:
                    continue
                yield (
                    (
                        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(frame)}\r\n\r\n"
                    ).encode()
                    + frame
                    + b"\r\n"
                )
        finally:
            self.close()

    def close(self) -> None:
        self._stop_process()


def open_preview(
    source: str,
    ffmpeg: str,
    *,
    fps: int = PREVIEW_FPS,
    width: int = PREVIEW_WIDTH,
    rtsp_transport: str = "tcp",
    first_frame_timeout: float = 12.0,
) -> Preview:
    """Start ffmpeg and wait for the first frame, so a bad URL fails fast."""
    preview = Preview(
        build_command(source, ffmpeg, fps=fps, width=width, rtsp_transport=rtsp_transport)
    )
    deadline = time.monotonic() + first_frame_timeout
    with preview._cond:
        while preview._seq == 0 and preview.alive and time.monotonic() < deadline:
            preview._cond.wait(0.25)
    if preview._seq == 0:
        preview.close()
        raise PreviewError(preview.error or "no video arrived from the source")
    return preview
