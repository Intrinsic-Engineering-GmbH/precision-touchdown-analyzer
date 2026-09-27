"""The front wheel and its lowest point, per frame and over a track (design 4.2).

Everything downstream maps the contact pixel through the ground-plane
homography, so it has to be the bottom of the tyre - a point on the fuselage
0.8 m up is off by metres through the same mapping, and a point in the
*shadow* is off by whatever the sun decides.

Colour cannot tell the tyre from the shadow in this footage: the camera
renders a hard shadow on the strip as the same saturated near-black blue as
the rubber and the dark paint. Geometry and time can:

* **The airframe** is what is clearly *lighter* than the background - white
  paint. No shadow ever is. Its extent is the size of the aircraft and says
  where its nose is.
* **The aircraft's lowest point** in every column is the airframe plus what
  hangs from it - fairing, gear, hub, tyre - in an unbroken run straight
  down. A shadow lying on the ground has lit grass between it and the
  aircraft and is not part of that run.
* **The front wheel** (:func:`wheel_track`) is the narrow bulge of that
  outline in the front part of the aircraft. It sits at a fixed fraction of
  the length behind the nose, which the whole track votes on; a shadow
  touching the aircraft is broad and slides along it as the height changes,
  so it gets no majority. The column then follows the aircraft's own smooth
  motion, and the lowest point under it goes through an anti-jump filter.

The shadow is still the most precise timing cue there is (design 4.2), read
at the wheel by :func:`locate`: the rows of lit ground between the tyre and
its shadow close as the wheel comes down and stay closed from the instant of
contact.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from touchdown_analyzer.analysis.detect import Blob
from touchdown_analyzer.calibration import homography as hg

# Margin around the coarse blob box, full-resolution px, so that a shadow
# hanging below the wheel is inside the region that is kept.
ROI_MARGIN = 40
# Changed at all: summed absolute BGR difference against the background.
FG_DIFF = 45.0
# Airframe: this much lighter than the background (white paint on grass and
# on the strip), judged on images smoothed by HALO_SIGMA - the camera
# sharpens, and the bright rim it draws along every dark edge would read as
# paint otherwise.
AIRFRAME_RATIO = 1.12
HALO_SIGMA = 1.5
# Components of the airframe mask this much smaller than the largest are
# specks of bright ground, not aircraft.
MINOR_FRACTION = 0.1
# Shortest airframe worth measuring, px.
MIN_LENGTH_PX = 60.0
# Hanging parts differ from the ground by this much, smoothed, summed over
# B, G and R; a run of them may skip RUN_GAP_PX rows, and reaches at most
# HANG_FRACTION of the aircraft's length below the airframe (a taildragger's
# gear leg is the longest).
SOLID_DIFF = 60.0
# An end of the airframe is its last this fraction of the length.
END_FRACTION = 0.03
RUN_GAP_PX = 1
HANG_FRACTION = 0.22

_K2 = np.ones((2, 2), np.uint8)
_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


@dataclass(slots=True)
class View:
    """What one frame shows of one aircraft, kept until its track is complete."""

    x0: int
    y0: int
    ratio: np.ndarray  # float16, frame luminance / background luminance
    tail: float  # image x extent of the airframe
    nose: float  # (``nose`` > ``tail``; which end leads is decided per track)
    cu: float  # image x of the airframe's centroid
    left_v: float  # image y of the airframe at its left / right end
    right_v: float
    at_left: bool  # the region reaches the left / right edge of the image
    at_right: bool
    air_bottom: np.ndarray  # per region column: the airframe's lowest row, -1 if none
    lowest: np.ndarray  # per region column: the aircraft's lowest row, -1 if none

    @property
    def length(self) -> float:
        return self.nose - self.tail


@dataclass(slots=True)
class ContactPoint:
    """Where the wheel is in this frame, in image and ground coordinates."""

    u: float  # image x, full-resolution pixels
    v: float  # image y
    world_x: float  # metres along the strip, ground-plane assumption
    world_y: float  # metres across the strip


def extract(frame: np.ndarray, background: np.ndarray, blob: Blob) -> View | None:
    """Airframe, outline and the ratio image around ``blob``."""
    height, width = frame.shape[:2]
    x0 = max(0, blob.x - ROI_MARGIN)
    y0 = max(0, blob.y - ROI_MARGIN)
    x1 = min(width, blob.x + blob.w + ROI_MARGIN)
    y1 = min(height, blob.y + blob.h + ROI_MARGIN)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    roi = frame[y0:y1, x0:x1].astype(np.float32)
    back = background[y0:y1, x0:x1].astype(np.float32)
    ratio = roi.mean(axis=2) / (back.mean(axis=2) + 1.0)
    # Only what the detector saw move: ground texture elsewhere in the box
    # differs from the (blurred) background model too.
    moving = _moving(blob, x0, y0, ratio.shape) & (np.abs(roi - back).sum(axis=2) > FG_DIFF)

    soft_roi = cv2.GaussianBlur(roi, (0, 0), HALO_SIGMA)
    soft_back = cv2.GaussianBlur(back, (0, 0), HALO_SIGMA)
    soft_ratio = soft_roi.mean(axis=2) / (soft_back.mean(axis=2) + 1.0)
    light = cv2.morphologyEx(
        (moving & (soft_ratio >= AIRFRAME_RATIO)).astype(np.uint8), cv2.MORPH_OPEN, _K3
    )
    aircraft = _keep_major(cv2.morphologyEx(light, cv2.MORPH_CLOSE, _K9, iterations=2))
    if not aircraft.any():
        return None
    ys, xs = np.nonzero(aircraft)
    tail, nose = float(xs.min() + x0), float(xs.max() + x0 + 1)
    if nose - tail < MIN_LENGTH_PX:
        return None
    # the row of each end: where the airframe is, over its last few columns
    tip = max(2.0, END_FRACTION * (nose - tail))
    left_v = float(ys[xs <= xs.min() + tip].mean() + y0)
    right_v = float(ys[xs >= xs.max() - tip].mean() + y0)

    solid = moving & (np.abs(soft_roi - soft_back).sum(axis=2) > SOLID_DIFF)
    air_bottom, lowest = _outline(aircraft, solid, nose - tail)
    return View(
        x0=x0,
        y0=y0,
        ratio=ratio.astype(np.float16),
        tail=tail,
        nose=nose,
        cu=float(xs.mean() + x0),
        left_v=left_v,
        right_v=right_v,
        at_left=x0 == 0,
        at_right=x1 == width,
        air_bottom=air_bottom,
        lowest=lowest,
    )


def _moving(blob: Blob, x0: int, y0: int, shape: tuple[int, ...]) -> np.ndarray:
    """The detector's own foreground, at full resolution in the region."""
    out = np.zeros(shape[:2], np.uint8)
    if blob.mask.size == 0:
        return np.ones(shape[:2], dtype=bool)
    mask = cv2.resize(blob.mask.astype(np.uint8), (blob.w, blob.h), interpolation=cv2.INTER_NEAREST)
    bx, by = blob.x - x0, blob.y - y0
    h = min(mask.shape[0], out.shape[0] - by)
    w = min(mask.shape[1], out.shape[1] - bx)
    out[by : by + h, bx : bx + w] = mask[:h, :w]
    return np.asarray(cv2.dilate(out, _K9) > 0)


def _keep_major(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros(mask.shape, bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(count, dtype=bool)
    keep[1:] = areas >= MINOR_FRACTION * areas.max()
    return np.asarray(keep[labels])


def _outline(
    aircraft: np.ndarray, solid: np.ndarray, length: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per column: the airframe's lowest row, and the aircraft's lowest row.

    The aircraft reaches below its airframe by whatever hangs from it in an
    unbroken run of ``solid`` pixels straight down, at most ``HANG_FRACTION``
    of its length. -1 where a column has no airframe.
    """
    rows_n, cols_n = aircraft.shape
    rows = np.arange(rows_n)[:, None]
    has_air = aircraft.any(axis=0)
    air_bottom = np.where(has_air, (rows * aircraft).max(axis=0), -1)
    lowest = air_bottom.copy()
    alive = has_air.copy()
    missed = np.zeros(cols_n, dtype=np.int32)
    columns = np.arange(cols_n)
    for step in range(1, int(round(HANG_FRACTION * length)) + 1):
        r = air_bottom + step
        inside = alive & (r < rows_n)
        hit = np.zeros(cols_n, dtype=bool)
        hit[inside] = solid[r[inside], columns[inside]]
        lowest = np.where(hit, r, lowest)
        missed = np.where(hit, 0, missed + 1)
        alive = inside & (missed <= RUN_GAP_PX)
        if not alive.any():
            break
    return air_bottom.astype(np.int32), lowest.astype(np.int32)


# --------------------------------------------------------------------------
# the wheel over a track
# --------------------------------------------------------------------------

# Where along the aircraft the front (main) wheel is looked for, as a fraction
# of the length back from the nose - under the cockpit or the wing root on a
# glider and a taildragger alike. Shadows touching the nose, the tail wheel
# and the skid are outside.
FRONT_MIN = 0.15
FRONT_MAX = 0.55
FRONT_TYPICAL = 0.25
# The outline a wheel bulges from: this fraction of the length either side.
# A frame votes for the wheel's place when its bulge is this deep and at most
# BULGE_WIDTH of the length wide - a wheel standing clear, not a shadow.
PROTRUSION_FRACTION = 0.1
BULGE_MIN_PX = 3.0
BULGE_WIDTH = 0.07
# How far from where the nose puts it the wheel's own bulge is sought, and
# the half width, as fractions of the length, of the column read for the
# lowest point.
REFINE_FRACTION = 0.08
WINDOW_FRACTION = 0.015
# Play in how far the wheel hangs: the gear gives, the pixels jitter.
HANG_PLAY_PX = 2.0
# The wheel column over a track is a smooth function of time; a reading
# further than this from that curve is replaced by it.
WHEEL_OUTLIER_PX = 12.0
# The anti-jump filter (_robust_local): a robust line through this many
# frames either side of each; frames with the aircraft cut off at the edge
# count this much; a reading within JUMP_PX of the path is "seen".
SMOOTH_HALF_FRAMES = 8
EDGE_WEIGHT = 0.05
ROBUST_ITERATIONS = 4
JUMP_PX = 2.5


def _smooth(
    t: np.ndarray, raw: np.ndarray, good: np.ndarray, outlier_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """(fitted curve, kept readings): a low-order polynomial in time through
    the good readings with outliers thrown out."""
    if good.sum() < 3:
        fill = float(np.median(raw[good])) if good.any() else 0.0
        return np.where(good, raw, fill), good.copy()
    t0 = t - t[0]
    keep = good.copy()
    coef = np.polyfit(t0[keep], raw[keep], 1)
    for _ in range(3):
        order = 2 if keep.sum() >= 8 else 1
        coef = np.polyfit(t0[keep], raw[keep], order)
        residual = raw - np.polyval(coef, t0)
        scale = max(outlier_px, 3 * 1.4826 * float(np.median(np.abs(residual[keep]))))
        refined = good & (np.abs(residual) <= scale)
        if refined.sum() < 3 or np.array_equal(refined, keep):
            break
        keep = refined
    fitted = np.polyval(coef, t0)
    return fitted, keep & (np.abs(raw - fitted) <= outlier_px)


@dataclass(slots=True)
class WheelTrack:
    """The front wheel through a track: one point per frame."""

    u: np.ndarray
    v: np.ndarray  # lowest point of the wheel; -1 where there was none
    seen: np.ndarray  # bool: something hung below the airframe at the wheel
    length_px: float = 0.0  # the aircraft, nose to tail
    fraction: float = 0.0  # where the wheel sits, back from the nose
    hang_px: float = 0.0  # how far the wheel reaches below the airframe


def wheel_track(views: list[View], t: np.ndarray, usable: np.ndarray) -> WheelTrack:
    """The front wheel's lowest point in every frame of a track.

    ``usable``: the frames that show the whole aircraft (not cut off at the
    edge of the picture).
    """
    n = len(views)
    t = np.asarray(t, dtype=float)
    usable = np.asarray(usable, dtype=bool)
    good = usable if usable.sum() >= 3 else np.ones(n, dtype=bool)
    lengths = np.array([vw.length for vw in views])
    length = float(np.median(lengths[good]))
    fit_cu, _ = _smooth(t, np.array([vw.cu for vw in views]), good, 0.05 * length)
    heading = 1.0 if fit_cu[-1] >= fit_cu[0] else -1.0
    noses = np.array([_nose(vw, heading, length) for vw in views])
    bulges = [_protrusion(vw.lowest, int(round(PROTRUSION_FRACTION * length))) for vw in views]

    # where along the aircraft the wheel is: the frames' vote
    fractions = np.full(n, np.nan)
    for i, vw in enumerate(views):
        frac = (noses[i] - (vw.x0 + np.arange(vw.lowest.size) + 0.5)) * heading / length
        front = (frac >= FRONT_MIN) & (frac <= FRONT_MAX) & (vw.lowest >= 0)
        if front.any():
            k = int(np.flatnonzero(front)[np.argmax(bulges[i][front])])
            if _narrow(bulges[i], k, BULGE_WIDTH * length):
                fractions[i] = frac[k]
    have = np.isfinite(fractions) & good
    if not have.any():
        have = np.isfinite(fractions)
    fraction = float(np.median(fractions[have])) if have.any() else FRONT_TYPICAL

    # the column: rigid with the nose, whose reading jitters, so smoothed;
    # then pulled onto the wheel's own bulge nearby and smoothed again, as
    # the length's reading drifts when a propeller or a wingtip comes and goes
    predicted, _ = _smooth(t, noses - heading * fraction * length, good, WHEEL_OUTLIER_PX)
    reach = max(3, int(round(REFINE_FRACTION * length)))
    found = np.full(n, np.nan)
    for i, vw in enumerate(views):
        c = int(round(predicted[i] - vw.x0))
        lo, hi = max(0, c - reach), min(vw.lowest.size, c + reach + 1)
        if hi - lo >= 3:
            k = int(np.argmax(bulges[i][lo:hi]))
            if bulges[i][lo + k] >= BULGE_MIN_PX:
                found[i] = vw.x0 + lo + k + 0.5
    # where the aircraft is cut off at the edge its bulges are half a wheel
    # or a gear leg: the column there is carried on from the whole frames
    have_u = np.isfinite(found) & good
    if have_u.sum() < 3:
        have_u = np.isfinite(found)
    if have_u.sum() >= 3:
        offset = float(np.median(found[have_u] - predicted[have_u]))
        u, _ = _smooth(t, np.where(have_u, found, predicted + offset), have_u, WHEEL_OUTLIER_PX)
    else:
        u = predicted

    # the lowest point under it, and how far below the airframe that is
    half = max(2, int(round(WINDOW_FRACTION * length)))
    v = np.full(n, np.nan)
    base = np.full(n, np.nan)
    clear = np.zeros(n, dtype=bool)
    seen = np.zeros(n, dtype=bool)
    for i, vw in enumerate(views):
        c = int(round(u[i] - vw.x0))
        lo, hi = max(0, c - half), min(vw.lowest.size, c + half + 1)
        air = vw.air_bottom[lo:hi] if hi > lo else vw.air_bottom[:0]
        if not (air >= 0).any():
            continue
        window = vw.lowest[lo:hi]
        k = lo + int(np.argmax(window))
        v[i] = float(window.max()) + 1.0 + vw.y0
        base[i] = float(np.median(air[air >= 0])) + 1.0 + vw.y0
        seen[i] = bool((window > air).any())
        clear[i] = _narrow(bulges[i], k, BULGE_WIDTH * length)
    ok = np.isfinite(v)
    if not ok.any():
        return WheelTrack(u=u, v=np.full(n, -1.0), seen=seen, length_px=length, fraction=fraction)

    # The wheel hangs a fixed distance below the airframe. Where it stands
    # clear that distance can be read; a shadow touching the tyre only ever
    # makes it look longer, so it is capped there.
    hang = v - base
    clear &= ok & good
    if clear.sum() >= 3:
        hang_px = float(np.median(hang[clear]))
        v = np.where(ok, np.minimum(v, base + hang_px + max(HANG_PLAY_PX, 0.1 * hang_px)), v)
    else:
        hang_px = float(np.nanmedian(hang))

    # The reference is the nose - the airframe's foremost point in the
    # direction of flight, which no shadow ever reaches. Its path is smooth:
    # every point a robust local line through its neighbours, frames where
    # it is cut off at the edge counting little.
    leads = np.array([not (vw.at_right if heading > 0 else vw.at_left) for vw in views])
    nose_weight = np.where(leads, 1.0, EDGE_WEIGHT)
    nose_v = np.array([vw.right_v if heading > 0 else vw.left_v for vw in views])
    nose_u = _robust_local(t, noses, nose_weight, SMOOTH_HALF_FRAMES)
    nose_v = _robust_local(t, nose_v, nose_weight, SMOOTH_HALF_FRAMES)

    # The wheel is rigid on the aircraft: its offset from the nose, read in
    # the frames where the wheel stands clear, places it in every frame.
    sel = clear & leads
    if sel.sum() < 3:
        sel = ok & good & leads
    if sel.sum() < 3:
        sel = ok
    du = float(np.median(u[sel] - nose_u[sel]))
    dv = float(np.median(v[sel] - nose_v[sel]))
    wheel_u, wheel_v = nose_u + du, nose_v + dv
    return WheelTrack(
        u=wheel_u,
        v=wheel_v,
        seen=seen & ok & (np.abs(np.where(ok, v, wheel_v) - wheel_v) <= JUMP_PX),
        length_px=length,
        fraction=fraction,
        hang_px=hang_px,
    )


def _nose(view: View, heading: float, length: float) -> float:
    """The leading end of the airframe; from the trailing end while it is cut off."""
    if heading > 0:
        return view.nose if not view.at_right else view.tail + length
    return view.tail if not view.at_left else view.nose - length


def _protrusion(lowest: np.ndarray, half: int) -> np.ndarray:
    """How far each column reaches below the outline around it.

    A wheel - with its gear or fairing - is a narrow bulge under the belly;
    a shadow that touches the aircraft is broad, and lifts the surrounding
    median with it.
    """
    filled = lowest.astype(float)
    if (filled < 0).all():
        return np.zeros_like(filled)
    filled[filled < 0] = np.min(filled[filled >= 0])
    half = max(2, half)
    padded = np.pad(filled, half, mode="edge")
    around = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)
    return np.asarray(filled - around)


def _narrow(bulge: np.ndarray, k: int, width: float) -> bool:
    """The bulge at ``k`` is deep enough and no wider than ``width``."""
    if bulge[k] < BULGE_MIN_PX:
        return False
    out = bulge > max(BULGE_MIN_PX, 0.5 * bulge[k])
    lo = hi = k
    while lo > 0 and out[lo - 1]:
        lo -= 1
    while hi < out.size - 1 and out[hi + 1]:
        hi += 1
    return hi - lo + 1 <= width


def _robust_local(t: np.ndarray, y: np.ndarray, weight: np.ndarray, half: int) -> np.ndarray:
    """Each value replaced by a weighted straight line through its neighbours.

    ``half`` frames either side, tricube-weighted by distance, times
    ``weight``; then iterated with Tukey's biweight on the residuals, so a
    reading that leaps away from the others - one frame or a run of them -
    loses its say. A bend in the path, such as the touchdown, is kept; a step
    is not.
    """
    n = len(y)
    if n == 0 or not (weight > 0).any():
        return y.astype(float)
    # Where the neighbours hardly count (the aircraft still cut off at the
    # edge) the window widens until it holds enough frames that do: the line
    # is carried in from where the view is whole.
    need = min(float(weight.sum()), half + 1.0)
    spans = []
    for i in range(n):
        h = half
        while weight[max(0, i - h) : i + h + 1].sum() < need and (i - h > 0 or i + h < n - 1):
            h += 1
        spans.append(h)
    robust = np.ones(n)
    fitted = y.astype(float).copy()
    for _ in range(ROBUST_ITERATIONS):
        for i in range(n):
            h = spans[i]
            lo, hi = max(0, i - h), min(n, i + h + 1)
            tricube = (1 - (np.abs(np.arange(lo, hi) - i) / (h + 1)) ** 3) ** 3
            w = weight[lo:hi] * robust[lo:hi] * tricube
            if w.sum() <= 1e-9:
                continue
            dt = t[lo:hi] - t[i]
            sw, st, stt = w.sum(), (w * dt).sum(), (w * dt * dt).sum()
            sy, sty = (w * y[lo:hi]).sum(), (w * dt * y[lo:hi]).sum()
            det = sw * stt - st * st
            # the line's value at t[i]; a level where the neighbours all sit at t[i]
            fitted[i] = (stt * sy - st * sty) / det if abs(det) > 1e-12 else sy / sw
        residual = np.abs(y - fitted)
        used = weight > 0
        sigma = max(JUMP_PX / 2, 1.4826 * float(np.median(residual[used])))
        cut = 4.685 * sigma
        robust = np.where(residual < cut, (1 - (residual / cut) ** 2) ** 2, 0.0)
    return fitted


# --------------------------------------------------------------------------
# one frame at the wheel
# --------------------------------------------------------------------------

# Columns either side of the wheel read for the shadow, and how far below the
# tyre a shadow still counts as its shadow.
GAP_HALF_WIDTH = 10
GAP_MAX_PX = 200


def locate(
    view: View, u: float, v: float, matrix: np.ndarray | list[list[float]]
) -> tuple[ContactPoint, float | None, float | None] | None:
    """(contact point, tyre gap, shadow reach) of one frame at wheel ``(u, v)``.

    ``None`` when ``(u, v)`` is not inside the region of this frame.
    """
    col = u - view.x0
    row = int(round(v - view.y0))
    rows, cols = view.ratio.shape
    if not (0 <= col < cols and 0 <= row < rows):
        return None
    world_x, world_y = hg.project(matrix, u, v)
    lo = max(0, int(col) - GAP_HALF_WIDTH)
    hi = min(cols, int(col) + GAP_HALF_WIDTH + 1)
    reaches, gaps = [], []
    for c in range(lo, hi):
        below = view.ratio[row:, c].astype(np.float32)
        r = _column_reach(below)
        if r is not None:
            reaches.append(r)
        g = _column_gap(below)
        if g is not None:
            gaps.append(g)
    need = max(3, (hi - lo) // 3)
    return (
        ContactPoint(u=u, v=v, world_x=world_x, world_y=world_y),
        float(np.median(gaps)) if len(gaps) >= need else None,
        float(np.median(reaches)) if len(reaches) >= need else None,
    )


# Below the tyre, luminance relative to the background: sunlit ground is
# brighter than LIT; tyre, umbra and penumbra are all darker. Two lit rows
# in a row end the dark run, so one bright speck of grass cannot.
LIT = 0.82


def _column_reach(ratio: np.ndarray) -> float | None:
    """Rows of dark under the tyre's bottom of one column before lit ground.

    ``None`` when no lit ground is found within reach - something else dark
    lies under the wheel.
    """
    n = min(len(ratio), GAP_MAX_PX)
    for i in range(n):
        if ratio[i] > LIT and (i + 1 >= n or ratio[i + 1] > LIT):
            return float(i)
    return None


# Directly under the tyre the ground is in penumbra, not full sun: lit
# enough to tell from the umbra of the shadow, which is what the gap is
# measured against. The tyre's own bottom edge is blurred over a few rows;
# those are skipped, not counted.
GAP_LIT = 0.60
GAP_BLUR_PX = 3


def _column_gap(ratio: np.ndarray) -> float | None:
    """Rows of lit ground between the bottom of the tyre and its shadow.

    ``ratio`` starts at the tyre's bottom row. ``None`` when no shadow
    follows within reach (nothing to close a gap against). Zero when the
    shadow touches the tyre.
    """
    n = min(len(ratio), GAP_MAX_PX)
    i = 0
    while i < n and ratio[i] < GAP_LIT and i < GAP_BLUR_PX:
        i += 1
    lit = 0
    while i < n and ratio[i] >= GAP_LIT:
        lit += 1
        i += 1
    if i >= n:
        return None  # never reached shadow
    return float(lit)
