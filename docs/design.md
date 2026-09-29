# Design

Measure the displacement between a glider's first main-wheel touchdown and the
target line of a spot-landing competition, from one fixed camera that records
continuously; store each landing as a clip; let a judge confirm; show a ranking.

Decisions:

- **One AXIS P1485-LE**, side-on, 75 m from the strip, mast > 8 m. It films the
  **touchdown window only** (±19 m around the target line, ~1.5 s per landing).
  Landings outside the window are bounded (`< −19 m` / `> +19 m`), not measured.
- **Identification by OGN, confirmed by the judge.** No OCR: a 30 cm character
  is ~15 px at this distance.
- **Purpose: club training and competitions.** Target accuracy ~0.3 m; the
  automatic result and the judge's decision are stored side by side.

---

## 1. Architecture

Recorder and analysis are separate processes sharing a folder, so analysis can
never cost the recorder a frame. Everything downstream reads the raw segments,
so a day is re-processable offline.

```
┌──────────────┐   10 s mp4 segments   ┌──────────────────────────────┐
│  recorder    │ ────────────────────► │  analysis worker             │
│ (ffmpeg)     │   data/raw/<session>/ │  detect → track → measure    │
└──────────────┘                       └──────────────────────────────┘
        │                                        │
        │                                        ├─► <time>_<REG>.mp4          (clip)
        │                                        ├─► landings.json             (records)
        └────────── same files ──────────────────┴─► <time>_<REG>_overlay.jpg  (proof)
                                                 │
                                          ┌──────┴───────┐
                                          │  control UI  │ FastAPI: capture, calibration,
                                          │  (judge)     │ frames, landings, scoring
                                          └──────┬───────┘ confirm / reject / pilot name
                                                 │ confirmed landings + scoring rules
                                          ┌──────┴───────┐
                                          │  ranking     │ /board: one row per pilot,
                                          │  display     │ total points, refresh 5 s
                                          └──────────────┘ ranking.xlsx / ranking.pdf
```

### Module layout

```
src/touchdown_analyzer/
    cli.py               argparse: probe | record | index | analyze | serve | launcher
    launcher.py          tkinter control window: server, services, OGN airfield, log
    config.py            RecorderConfig, camera targets (1080p60, GOP 60), .env
    paths.py             data directory of an installed copy
    winstall.py          Windows install / uninstall
    capture/             stdlib only
        ffmpeg.py        locate and drive ffmpeg / ffprobe
        recorder.py      ffmpeg -f segment -c copy, watchdog, disk guard, session.json
        segments.py      segments.jsonl: file ↔ wall clock ↔ PTS, gap detection
        probe.py         pre-flight camera check against the targets
        preview.py       MJPEG live view
        frames.py        single frames from a source or a segment
    calibration/
        homography.py    clicked markers → ground-plane homography (DLT), residual
    analysis/            needs OpenCV
        detect.py        MOG2 blobs, nearest-neighbour tracker
        contact.py       airframe, front wheel and its lowest point, shadow reach
        touchdown.py     shadow-reach and apparent-depth fits → contact instant
        pipeline.py      segments in, Landing records out
        overlay.py       contact frame with the geometry drawn on
        worker.py        background thread: analyse a session, follow a recording
    clips/cutter.py      stdlib only: ffmpeg stream-copy cut, clip naming
    identify/ogn.py      stdlib only: OGN live fixes, FlightBook / KTrax logbook
    store/               stdlib only
        landings.py      Landing record, per-session JSON store
        scoring.py       ScoringRules, config/scoring.json
        ranking.py       ranking.xlsx / ranking.pdf without libraries
    control/
        app.py           FastAPI routes
        service.py       recorder, probe, preview, OGN feed supervision
        review.py        analysis worker, confirm / reject / edit, ranking export
        static/*.html    index, calibrate, viewer, landings, scoring, board
```

### Stack

Python 3.12+, ffmpeg on `PATH`. Core dependency: NumPy. Optional: `ui`
(FastAPI, uvicorn), `analysis` (opencv-python-headless 4.x). `capture/`,
`clips/`, `identify/` and `store/` are stdlib-only so the recorder cannot fail
to import at the airfield. Results are JSON, not a database.

Packaging: PyInstaller + setup wizard on Windows (bundles ffmpeg); a `.deb` with
a venv under `/opt/pta` and a systemd service on Debian.

---

## 2. Camera and site

### 2.1 AXIS P1485-LE at 75 m

| | |
|---|---|
| Sensor | 1/2.8" CMOS, 1920 × 1080, 50/60 fps, rolling shutter |
| Lens | 10.8–28.2 mm varifocal, HFOV 29°–11° |
| Covered width at 10.8 mm | 38.8 m = ±19.4 m around the target line |
| Scale at the target line | 49.5 px/m (20 mm/px) |
| Travel per frame at 60 fps | 0.42 m at 25 m/s |
| Motion blur at 1/1000 s | 25 mm ≈ 1.2 px |

`python tools/site_geometry.py --distance 75 --mast 8` recomputes these.

### 2.2 The measurement window

±19.4 m (`WINDOW_HALF_M` in `pipeline.py`). A glider is in frame ~1.5 s
(~93 frames), enough for the fits, which need a few frames on each side of the
contact. Consequences: the clip is a fixed window, the tracker has no approach
phase to wait for, identification cannot come from the image.

Landings that touch down outside the window are reported as a bound
(`short` / `long`), a legitimate result in a spot-landing competition.

Wider window if ever needed: a second body (P1465-LE 3–9 mm at HFOV ≈ 40°,
28 mm/px, ±27 m); nothing downstream of the calibration changes.

### 2.3 Mast height: > 8 m

Height barely affects the longitudinal number but sets the cross-strip
resolution and what a vertical pixel error costs longitudinally:

| Mast | Elevation | Cross-strip | Longitudinal per 1 px vertical |
|------|-----------|-------------|--------------------------------|
| 3 m | 2.3° | 0.52 m/px | 0.13 m |
| **> 8 m** | **> 6.1°** | **< 0.19 m/px** | **< 0.05 m** |
| 12 m | 9.1° | 0.13 m/px | 0.03 m |

Tilt down ~6°, ground line about two thirds down the image. Mount on the side
that never looks into a low sun (in Switzerland: looking north).

### 2.4 Camera configuration

`touchdown-analyzer probe` checks resolution, fps and GOP; the rest is set once
in the camera:

- 1080p, **60 fps** (60 Hz power line mode). WDR off, it caps the frame rate.
- Max shutter 1/1000 s, exposure zone on the strip.
- **Zipstream off**, no dynamic FPS or GOP. Fixed GOP 60, capped VBR 12–16 Mbit/s
  (~5–7 GB/h).
- Stabilisation, barrel correction, defog **off**. Focus and zoom locked,
  autofocus off; re-zooming means re-calibrating.
- Day mode forced, IR off. NTP time. Fixed IP, wired PoE.

Rolling shutter: the wheel occupies a narrow band of rows, so the timing bias
is milliseconds and largely cancels in the fits.

---

## 3. Feature 1 — landing segmentation and clips

### 3.1 Detection (`analysis/detect.py`)

MOG2 on a half-scale frame (history 120 frames, threshold 24, shadow flag off),
morphology, contours, blobs ≥ 600 px². Tracking is nearest-neighbour
association against a constant-velocity prediction (gate 220 px, 5 missed
frames), one `Track` per moving object; a glider rarely shares the window with
anything, so no Kalman/SORT.

A track is analysed if it has ≥ 12 frames, ≥ 4000 px² at its largest and spans
≥ 250 px. The tracker survives segment boundaries.

### 3.2 Outcomes (`analysis/pipeline.py`, `store/landings.py`)

No state machine; each finished track gets one outcome:

| Outcome | Meaning | Label |
|---------|---------|-------|
| `measured` | contact instant found in the window | `+1.3 m` |
| `short` | already rolling at frame entry | `< −19 m` |
| `long` | still airborne at frame exit | `> +19 m` |
| `unseen` | low and level throughout; judge picks the frame | `pick frame` |
| `on_ground` | rolling for the whole track (taxi, rollout) | `rolling` |
| `departure` | lifted off in view | `take-off` |
| `airborne` | flew through without touching | `fly-through` |

`kind` follows: landing / departure / pass. A "ground run" more than 15 m off
the strip axis (`PLAUSIBLE_LATERAL_M`) is reclassified as `airborne`. A
calibration residual above target is flagged on every landing.

### 3.3 Clips (`clips/cutter.py`)

From **1 s before** the wheel comes over the measuring window (the ruler,
±19.4 m) **to 1 s after** it leaves it; the judge's frame bar spans the same.
Without a track, a fixed −3 s / +5 s around the contact. Stream copy from the
covering raw segments: fast, lossless, at
most a GOP (1 s) early. Frame-exact work always goes through the raw segments.

Name: `YYYY-MM-DD_HH-MM-SS_<REG>.mp4`, timestamp = touchdown; `UNKNOWN-<seq>`
until identified, renamed on confirmation (history keeps the old name).
Overlay: same stem, `_overlay.jpg`.

Retention: raw segments a couple of weeks (re-processing), clips and results
indefinitely. The disk guard stops recording below 20 GB free.

### 3.4 Identification (`identify/ogn.py`)

1. **OGN live positions** (`live.glidernet.org`), polled every 5 s during a
   session into `ogn_fixes.jsonl`. A landing matches the aircraft whose fix is
   within 30 s of the touchdown, within 1.5 km of the target line and below
   150 m AGL.
2. **OGN FlightBook** logbook (`flightbook.glidernet.org`, KTrax as fallback),
   fetched on demand for the day: registration, type, and the landing/take-off
   minute, which also settles landing versus aerotow departure.
3. **Judge.** Names or corrects the aircraft and enters the pilot; the pilot is
   suggested from the aircraft's previous landing.

`identified_by` records which. The airfield (ICAO, position, elevation, time
zone) is in `config/ogn.json`, looked up from the FlightBook.

---

## 4. Feature 2 — touchdown displacement

### 4.1 Calibration (`calibration/homography.py`)

Survey ≥ 4 markers with a tape (the UI asks for three pairs, one on each strip
edge); click them in a still; solve. World frame: origin on the target line at
the strip centre, x along the strip (positive beyond the line in landing
direction), y lateral (positive right, seen from behind).

Solved by normalised DLT least squares over all markers, no RANSAC: a misclick
shows as one large residual instead of being silently discarded. RMS residual
is the quality number, **target 0.10 m** (`TARGET_RESIDUAL_M`). Saved to
`config/calibration.json`; every landing carries a copy.

A plane homography is exact for the wheel contact point, which is *on* the
ground plane at contact. Anything above the ground maps too far from the camera
by `height / tan(elevation)` — metres per decimetre at mast elevations. Hence
everything hinges on finding the tyre, not the fuselage or the shadow.

No lens undistortion and no drift check: the lens is not wide-angle and the
camera sits on a fixed mast.

### 4.2 Contact instant (`analysis/contact.py`, `analysis/touchdown.py`)

Per frame, inside the tracked box at full resolution, against MOG2's
background image brought to the frame's exposure:

1. **Airframe**: what is clearly *lighter* than the background (white paint),
   judged on lightly smoothed images so the camera's sharpening rims along dark
   edges do not count. No shadow is ever lighter. Its extent is the aircraft's
   size and its leading end the nose.
2. **Outline**: per column, the airframe plus what hangs from it (fairing,
   gear, hub, tyre) in an unbroken run straight down. A shadow on the ground
   has lit grass between it and the aircraft and is not part of it.

Over the track:

3. **Front wheel column**: the narrow bulge of the outline 15–55 % of the
   length behind the nose. Its place along the aircraft is the median over the
   frames where it stands clear (a touching shadow is broad, and slides along
   the aircraft as the height changes); the column follows the aircraft's
   smooth motion, pulled onto the bulge frame by frame.
4. **Lowest point, from the nose**: the reference is the airframe's foremost
   point in the direction of flight, which no shadow reaches. Its path is
   smoothed: each point a robust local line through ±8 frames (tricube
   weights, Tukey biweight on the residuals), so a jump - one frame or a run -
   loses its say while the bend at touchdown stays; frames with the nose cut
   off at the edge count 5 %, and at the start of a pass the window widens
   until whole frames carry the line in. The wheel's offset from the nose is
   read once, as the median over the frames where the wheel stands clear
   (bottom of the outline at the wheel column, capped at how far it hangs
   below the airframe), and places the wheel in every frame - rigid on the
   aircraft. Against the judge's clicks the row is within ~3 px on average
   (8.8 px reading the outline itself). The line is drawn and analysed to
   where the wheel leaves the picture, also while the aircraft is cut off.
5. **Shadow reach**: rows of dark under the tyre before sunlit ground begins.

Two independent cues, each a corner fit on a per-frame series at sub-frame
resolution (8 sub-steps):

- **Shadow reach.** Shrinks linearly with wheel height and stops changing at
  contact; the floor is free (it depends on the sun). Needs no calibration; the
  primary cue in sunshine. Needs ≥ 5 px change and 4 frames each side.
- **Apparent depth.** The contact pixel mapped through the homography as if on
  the ground: the across-strip coordinate `Y(t)` falls until contact, then
  follows the ground run. Fitted as a hinge with free slope on both legs (so a
  mediocre calibration does not break it); the airborne leg must drop ≥ 1 m and
  ≥ 4 ground-run RMS, and the hinge must beat a single line by 20 % RMS. A
  single-line trend steeper than 3 m/s of apparent depth reads as height
  changing.

Combination: within 4 frames the cues are reported together; up to 30 frames
apart they bracket the contact, the estimate is the middle and the spread the
uncertainty; further apart they contradict and the landing is flagged. Without
a shadow the depth cue alone decides, and the landing / take-off / level-pass
ambiguity goes to the judge (`unseen`) and the logbook.

`method` records `shadow`, `depth` or `none`; `fit` keeps both fits; `flags`
the caveats.

### 4.3 Metres

Contact pixel → homography → `(x, y)` in metres, signed for the direction of
travel (both landing directions, one calibration). Stored: longitudinal,
lateral, uncertainty, frame, sub-frame, speed, direction, image point, and the
per-frame track (`TrackPoint`) so the browser can redraw it. The overlay JPG
draws the target line, the contact and the measured distance on the contact
frame.

### 4.4 Error budget

P1485-LE, 75 m, 8 m mast, 60 fps, 0.10 m calibration:

| Source | Contribution |
|--------|--------------|
| Timing, raw frame rate | 0.42 m (25 m/s ÷ 60 fps) |
| Timing after sub-frame fit | **0.08–0.15 m** |
| Contact row / column (2 px) | 0.04 m |
| Vertical mis-detection (1 px) → longitudinal | 0.05 m |
| Homography + survey | 0.05–0.15 m |
| Heat shimmer | 0.05–0.2 m on hot afternoons |

≈ 0.2 m RSS, dominated by timing; at 30 fps ≈ 0.4 m. Holds for a landing with a
real sink rate; a greaser has no sharp corner and the judge's scrubber decides
the last metres.

### 4.5 Validation

- Hand-mark the contact frame on the Frames page (`/api/annotations`); compare
  bias and σ per session.
- Markers every 5 m along the strip edge; check the mapping reproduces the
  spacing.
- Roll a wheel to a tape-measured point: static truth for the geometry alone.

### 4.6 What the first footage changed (13 Sept 2026, Bellechasse)

Ten clips from a low tripod, six-marker calibration, residual 2.25 m:

- The lowest silhouette pixel is the shadow, not the wheel → the aircraft/shadow
  split and the tyre tracking of §4.2.
- The ground run is not level in calibrated coordinates (2 m drift over the
  window) → free slopes on both hinge legs; landing and take-off then look
  alike to the depth cue, the shadow or the logbook settles it.
- With a high sun there is no lit gap between glider and shadow, only a shadow
  reach that settles at a non-zero floor → corner fit with free floor.
- A flat flare has no sharp corner: on HB-3213 the two cues were 21 frames
  apart, so the estimate is the middle and the spread (±6 m there) the
  uncertainty. With the tyre tracked as an object the result is −3.2 m, the
  frame a 3× zoom picks.

### 4.7 What the second footage changed (26 Sept 2026, strip and tug)

- The camera's auto-exposure darkens the frame by up to 20 % as a white
  aircraft comes in, right at the touchdown → every frame is scaled to the
  background's exposure (median ratio) before anything is compared.
- A hard shadow on the pink strip is saturated blue-black, as are the tyre and
  dark paint: colour cannot separate them → §4.2 works on lightness, geometry
  and time instead.
- A glider's thin tail boom drops out of the mask → blobs side by side are one
  aircraft; one pass is one event.
- Against the judge's five clicks the wheel point is now within 21 px across
  and 15 px down, typically ~10 (it was on the shadow, 20–60 px off), and it
  no longer jumps (95th-percentile frame-to-frame jerk 4–10 px, was 45–240).
  The contact *instant* is only slightly better: on the nine corrected
  landings 25 frames off on average (median 21), was 28 (25). The shadow here
  falls beside the wheel, so the gap below the tyre rarely reads, and the
  depth cue cannot tell a landing from a take-off when the ground run drifts.
- Overcast: no shadow, and with this calibration "rolling" and "floating a
  hand's width up" are indistinguishable (HB-1827) → `unseen`, the judge picks
  the frame. A 0.10 m calibration and the 8 m mast are what let the depth cue
  carry an overcast afternoon alone.

---

## 5. Data model (`store/landings.py`)

One `landings.json` per session in `data/landings/<session>/`, written
atomically; the file is the archive. Per landing:

```
id, session, created_utc, kind, outcome, status        pending | confirmed | rejected
touchdown_utc, segment, frame, subframe, fps, direction
longitudinal_m, lateral_m, uncertainty_m, bound_m, speed_mps, image_x/y
method, fit, flags, calibration, track[], first_utc, last_utc
registration, competition_number, aircraft_type, identified_by, ogn, pilot
clip_path, overlay_path
confirmed_utc, confirmed_longitudinal_m, confirmed_frame, note, history[]
```

The estimator's number and the judge's number are separate fields;
`scored_longitudinal_m` prefers the confirmed one. `history` records every
change with a timestamp.

Session directory (`data/raw/<session>/`): 10 s segments, `session.json`,
`segments.jsonl`, `gaps.jsonl`, `recorder.log`, `ogn_fixes.jsonl`,
`ogn_logbook.json`. Config: `config/calibration.json`, `scoring.json`,
`ogn.json`; camera URL in `.env` (git-ignored, carries the password).

Scoring (`store/scoring.py`): max points on the line, a deduction per metre
short and per metre long, a floor, out-of-range points, decimals; edited on the
Scoring page. Only `measured`, `short` and `long` are scored.

---

## 6. Control UI (`control/`)

FastAPI on localhost:8080 (`--host 0.0.0.0` for the field WiFi, no login).

| Page | Does |
|------|------|
| Capture (`/`) | probe, start/stop recording, live preview, disk and fps status |
| Calibration | grab a still, click markers, solve, residual |
| Frames (`/viewer`) | step a segment frame by frame, hand-mark a touchdown |
| Landings | every event with offset, overlay, scrubber, magnifier, OGN proposal, pilot; *Confirm / Reject / Use this frame*; analyse a session or follow the recording |
| Scoring | the rules and the scale drawn out |
| Board (`/board`) | read-only ranking for a big screen: per pilot the confirmed landings, offsets, points, total; unverified beside; refresh 5 s |

Confirming renames the clip and overlay; any change to what the board shows
rewrites `ranking.xlsx` / `ranking.pdf` in the session directory. The launcher
(tkinter) starts the server and shows the services and the log.

---

## 7. Milestones

| # | Deliverable | Done when |
|---|-------------|-----------|
| M0 | Recording rig + a full flying day of footage | 20–50 landings archived, camera geometry settled |
| M1 | Calibration tool | residual < 0.1 m against tape-measured points |
| M2 | Feature 1 | every landing of a test day is cut correctly, no missed landings |
| M3 | Feature 2 | measured vs. hand-marked: bias < 0.1 m, σ < 0.3 m |
| M4 | Identification + review UI | a session is reviewable in < 5 min |
| M5 | Season hardening | a whole day unattended, disk housekeeping, crash recovery |

Project testing has achieved M1; M2 is pending.
