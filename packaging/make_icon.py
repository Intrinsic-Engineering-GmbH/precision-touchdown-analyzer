"""Render the gliding-badge logo to icon.png and icon.ico.

The same drawing as the page header's SVG (blue disc, white ring, three
gulls), rasterised with OpenCV so the build needs no extra tool. The .ico
holds PNG-compressed images, which Windows accepts since Vista.

    python packaging/make_icon.py packaging/out
    python packaging/make_icon.py --static    the PNGs the control window uses

The --static ones are committed (src/touchdown_analyzer/control/static/),
so a checkout shows the badge without building anything first.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import cv2
import numpy as np

BLUE = (147, 58, 31)  # BGR of #1f3a93
WHITE = (255, 255, 255)

# The gull outline as cubic Bezier segments (from the SVG path).
GULL = [
    ((-36, 2), (-26, -9), (-13, -11), (-4, -4)),
    ((-4, -4), (-2, -3), (2, -3), (4, -4)),
    ((4, -4), (13, -11), (26, -9), (36, 2)),
    ((36, 2), (26, -3), (14, -1), (6, 3)),
    ((6, 3), (4.5, 7), (0, 9.5), (-3.5, 8.5)),
    ((-3.5, 8.5), (-6, 7.5), (-6, 4.5), (-6, 3)),
    ((-6, 3), (-14, -1), (-26, -3), (-36, 2)),
]


def bezier(p0, p1, p2, p3, n=24):  # type: ignore[no-untyped-def]
    t = np.linspace(0, 1, n)[:, None]
    p = np.array([p0, p1, p2, p3], dtype=float)
    return (
        (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t**2 * p[2] + t**3 * p[3]
    )


def gull_polygon(cx: float, cy: float, scale: float) -> np.ndarray:
    pts = np.vstack([bezier(*seg) for seg in GULL])
    return (pts * scale + (cx, cy)).astype(np.float64)


def render(size: int) -> np.ndarray:
    """BGRA image of the badge at ``size`` px, drawn 4x and downscaled."""
    big = size * 4
    unit = big / 40.0  # the SVG's 40-unit box
    image = np.zeros((big, big, 4), dtype=np.uint8)
    centre = (int(20 * unit), int(20 * unit))
    cv2.circle(image, centre, int(19.5 * unit), (*WHITE, 255), -1, cv2.LINE_AA)
    cv2.circle(image, centre, int(18.2 * unit), (*BLUE, 255), -1, cv2.LINE_AA)
    cv2.circle(image, centre, int(15.8 * unit), (*WHITE, 255), max(1, int(1.1 * unit)), cv2.LINE_AA)
    for cx, cy, scale in ((20.5, 12.5, 0.30), (20, 20.3, 0.34), (20, 28, 0.34)):
        poly = gull_polygon(cx * unit, cy * unit, scale * unit)
        cv2.fillPoly(
            image, [np.round(poly * 256).astype(np.int32)], (*WHITE, 255), cv2.LINE_AA, shift=8
        )
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def write_ico(path: Path, images: list[np.ndarray]) -> None:
    """An .ico whose entries are PNG streams."""
    blobs = [cv2.imencode(".png", im)[1].tobytes() for im in images]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for im, blob in zip(images, blobs, strict=True):
        side = im.shape[0]
        entries += struct.pack(
            "<BBBBHHII",
            side if side < 256 else 0,
            side if side < 256 else 0,
            0,
            0,
            1,
            32,
            len(blob),
            offset,
        )
        offset += len(blob)
    path.write_bytes(header + entries + b"".join(blobs))


STATIC = Path(__file__).resolve().parents[1] / "src" / "touchdown_analyzer" / "control" / "static"
STATIC_SIZES = (16, 32, 36, 48, 256)  # window icon sizes, and 36 for the header as on the pages


def write_static(static_dir: Path = STATIC) -> None:
    for size in STATIC_SIZES:
        cv2.imwrite(str(static_dir / f"icon-{size}.png"), render(size))
    print("icon-*.png written to", static_dir)


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "icon.png"), render(256))
    write_ico(out_dir / "icon.ico", [render(s) for s in (16, 24, 32, 48, 64, 128, 256)])
    print("icon.png and icon.ico written to", out_dir)


if __name__ == "__main__":
    if sys.argv[1:] == ["--static"]:
        write_static()
    else:
        main(Path(sys.argv[1] if len(sys.argv) > 1 else "packaging/out"))
