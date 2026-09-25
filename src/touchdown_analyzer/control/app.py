"""FastAPI app for controlling captures from a browser.

Bind to 0.0.0.0 and the page is reachable from a phone on the field WiFi,
which is the point: the recording machine is usually in a shed.

There is no authentication. Serve it on a trusted network only, or leave the
default host (127.0.0.1) alone — the source field accepts an RTSP URL with
credentials in it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from touchdown_analyzer import __version__
from touchdown_analyzer.capture.preview import BOUNDARY
from touchdown_analyzer.control.review import ReviewService
from touchdown_analyzer.control.service import CaptureService, ServiceError

STATIC = Path(__file__).parent / "static"


class StartRequest(BaseModel):
    source: str = ""  # empty = the saved camera URL
    remember: bool = False
    session: str
    segment_seconds: int = Field(default=10, ge=1, le=600)
    target_fps: float = Field(default=60.0, gt=0, le=1000)
    rtsp_transport: str = Field(default="tcp", pattern="^(tcp|udp)$")
    min_free_gb: float = Field(default=20.0, ge=0)
    duration_s: float = Field(default=0.0, ge=0)


class ProbeRequest(BaseModel):
    source: str = ""
    remember: bool = False
    seconds: float = Field(default=30.0, gt=0, le=600)
    target_fps: float = Field(default=60.0, gt=0, le=1000)
    rtsp_transport: str = Field(default="tcp", pattern="^(tcp|udp)$")


class FrameRequest(BaseModel):
    source: str = ""
    remember: bool = False
    rtsp_transport: str = Field(default="tcp", pattern="^(tcp|udp)$")


class MarkerModel(BaseModel):
    image_x: float
    image_y: float
    world_x: float
    world_y: float
    label: str = ""


class SolveRequest(BaseModel):
    markers: list[MarkerModel] = Field(min_length=1, max_length=64)
    image_size: tuple[int, int] | None = None
    source: str = ""
    notes: str = ""
    save: bool = False


class WindowRequest(BaseModel):
    session: str
    segment: str
    frame: int = Field(default=0, ge=0)
    # Frames extracted either side of ``frame``. The default suits stepping;
    # looping a whole landing asks for enough to hold it in one window.
    half: int = Field(default=30, ge=5, le=180)


class MeasureRequest(BaseModel):
    image_x: float
    image_y: float


class AnnotationRequest(BaseModel):
    session: str
    segment: str
    frame: int = Field(ge=0)
    image_x: float
    image_y: float
    aircraft: str = ""
    note: str = ""


class AnalysisRequest(BaseModel):
    session: str
    follow: bool = False  # keep watching for new segments while recording
    fresh: bool = False  # discard the session's earlier results first


class ConfirmRequest(BaseModel):
    registration: str = ""
    pilot: str = ""
    note: str = ""


class RejectRequest(BaseModel):
    note: str = ""


class EditRequest(BaseModel):
    registration: str | None = None
    pilot: str | None = None
    competition_number: str | None = None
    aircraft_type: str | None = None
    outcome: str | None = None
    frame: int | None = Field(default=None, ge=0)
    reset_frame: bool = False  # drop the judge's frame, back to the automatic one
    note: str | None = None


class ScoringRequest(BaseModel):
    max_points: float = Field(default=100.0, gt=0)
    target_width_m: float = Field(default=0.0, ge=0)
    short_per_m: float = Field(default=5.0, ge=0)
    long_per_m: float = Field(default=2.0, ge=0)
    min_points: float = 0.0
    out_of_range_points: float = 0.0
    decimals: int = Field(default=0, ge=0, le=3)
    name: str = "Club rules"


class FieldRequest(BaseModel):
    airfield: str = ""
    name: str = ""
    lat: float = 0.0
    lon: float = 0.0
    elevation_m: float = 0.0
    radius_km: float = Field(default=3.0, gt=0, le=50)
    enabled: bool = False
    timezone_offset_h: float = 2.0


def create_app(service: CaptureService, review: ReviewService | None = None) -> FastAPI:
    app = FastAPI(title="touchdown-analyzer capture control", version=__version__)
    review = review or ReviewService(service)

    @app.exception_handler(ServiceError)
    async def _service_error(_: Any, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        payload = service.status()
        payload["version"] = __version__
        payload["probe"] = service.probe_status()
        return payload

    @app.post("/api/record/start")
    async def start(request: StartRequest) -> dict[str, Any]:
        try:
            service.start(
                request.source,
                request.session,
                segment_seconds=request.segment_seconds,
                target_fps=request.target_fps,
                rtsp_transport=request.rtsp_transport,
                min_free_gb=request.min_free_gb,
                duration_s=request.duration_s,
                remember=request.remember,
            )
        except ServiceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        # OGN positions are only useful if they were logged while it happened.
        review.start_poller(request.session)
        return service.status()

    @app.post("/api/record/stop")
    async def stop() -> dict[str, Any]:
        try:
            service.stop()
        except ServiceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        review.stop_poller()
        return service.status()

    # -- landings: analysis, results, the judge -----------------------------

    @app.get("/landings", include_in_schema=False)
    async def landings_page() -> FileResponse:
        return FileResponse(STATIC / "landings.html")

    @app.get("/api/analysis/status")
    async def analysis_status() -> dict[str, Any]:
        return review.analysis_status()

    @app.post("/api/analysis/start")
    async def analysis_start(request: AnalysisRequest) -> dict[str, Any]:
        return review.start_analysis(request.session, follow=request.follow, fresh=request.fresh)

    @app.post("/api/analysis/stop")
    async def analysis_stop() -> dict[str, Any]:
        return review.stop_analysis()

    @app.get("/api/landings")
    async def landing_sessions() -> list[str]:
        return review.sessions_with_results()

    @app.get("/api/landings/{session}")
    async def landings(session: str) -> dict[str, Any]:
        return review.summary(session)

    # before /{landing_id}, which would otherwise take "ranking.xlsx" for an id
    @app.get("/api/landings/{session}/ranking.xlsx", include_in_schema=False)
    async def ranking_xlsx(session: str) -> FileResponse:
        return _ranking_file(
            session, 0, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    @app.get("/api/landings/{session}/ranking.pdf", include_in_schema=False)
    async def ranking_pdf(session: str) -> FileResponse:
        return _ranking_file(session, 1, "application/pdf")

    def _ranking_file(session: str, which: int, media_type: str) -> FileResponse:
        """The ranking as a file, brought up to date first if it is behind."""
        paths = review.export(session)
        if paths is None:
            raise HTTPException(status_code=404, detail=f"no results for {session}")
        path = paths[which]
        return FileResponse(
            path,
            media_type=media_type,
            filename=f"ranking-{session}{path.suffix}",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/landings/{session}/{landing_id}")
    async def landing(session: str, landing_id: str) -> dict[str, Any]:
        return review.scored(review.landing(session, landing_id))

    @app.post("/api/landings/{session}/{landing_id}/confirm")
    async def landing_confirm(
        session: str, landing_id: str, request: ConfirmRequest
    ) -> dict[str, Any]:
        return review.scored(
            review.confirm(
                session,
                landing_id,
                registration=request.registration,
                pilot=request.pilot,
                note=request.note,
            )
        )

    @app.post("/api/landings/{session}/{landing_id}/reject")
    async def landing_reject(
        session: str, landing_id: str, request: RejectRequest
    ) -> dict[str, Any]:
        return review.scored(review.reject(session, landing_id, note=request.note))

    @app.post("/api/landings/{session}/{landing_id}/reopen")
    async def landing_reopen(session: str, landing_id: str) -> dict[str, Any]:
        return review.scored(review.reopen(session, landing_id))

    @app.post("/api/landings/{session}/{landing_id}/edit")
    async def landing_edit(session: str, landing_id: str, request: EditRequest) -> dict[str, Any]:
        edited = review.edit(
            session,
            landing_id,
            registration=request.registration,
            pilot=request.pilot,
            competition_number=request.competition_number,
            aircraft_type=request.aircraft_type,
            outcome=request.outcome,
            frame=request.frame,
            reset_frame=request.reset_frame,
            note=request.note,
        )
        return review.scored(edited)

    @app.get("/api/landings/{session}/{landing_id}/overlay.jpg", include_in_schema=False)
    async def landing_overlay(session: str, landing_id: str) -> FileResponse:
        try:
            path = review.artefact(session, landing_id, "overlay")
        except ServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.get("/api/landings/{session}/{landing_id}/clip.mp4", include_in_schema=False)
    async def landing_clip(session: str, landing_id: str) -> FileResponse:
        try:
            path = review.artefact(session, landing_id, "clip")
        except ServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type="video/mp4")

    @app.get("/scoring", include_in_schema=False)
    async def scoring_page() -> FileResponse:
        return FileResponse(STATIC / "scoring.html")

    @app.get("/board", include_in_schema=False)
    async def board_page() -> FileResponse:
        """The results for the big screen: read-only, refreshes itself."""
        return FileResponse(STATIC / "board.html")

    @app.get("/api/scoring")
    async def scoring_rules() -> dict[str, Any]:
        return review.rules.as_dict()

    @app.post("/api/scoring")
    async def scoring_save(request: ScoringRequest) -> dict[str, Any]:
        return review.save_rules(request.model_dump())

    @app.get("/api/ogn")
    async def ogn_status() -> dict[str, Any]:
        return review.ogn_status()

    @app.post("/api/ogn")
    async def ogn_save(request: FieldRequest) -> dict[str, Any]:
        return review.save_field(request.model_dump())

    @app.post("/api/ogn/logbook/{session}")
    async def ogn_logbook(session: str) -> dict[str, Any]:
        return review.fetch_logbook(session)

    @app.post("/api/probe")
    async def probe(request: ProbeRequest) -> dict[str, Any]:
        try:
            job = service.start_probe(
                request.source,
                seconds=request.seconds,
                target_fps=request.target_fps,
                rtsp_transport=request.rtsp_transport,
                remember=request.remember,
            )
        except ServiceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job.as_dict()

    @app.get("/api/preview.mjpeg", include_in_schema=False)
    async def preview(
        source: str = "",
        fps: int = Query(default=8, ge=1, le=30),
        width: int = Query(default=960, ge=160, le=1920),
        rtsp_transport: str = Query(default="tcp", pattern="^(tcp|udp)$"),
    ) -> StreamingResponse:
        try:
            stream = service.open_preview(
                source, fps=fps, width=width, rtsp_transport=rtsp_transport
            )
        except ServiceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return StreamingResponse(
            stream.frames(),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/calibration", include_in_schema=False)
    async def calibration_page() -> FileResponse:
        return FileResponse(STATIC / "calibrate.html")

    @app.post("/api/calibration/frame")
    async def calibration_frame(request: FrameRequest) -> dict[str, Any]:
        try:
            return service.grab_calibration_frame(
                request.source, rtsp_transport=request.rtsp_transport, remember=request.remember
            )
        except ServiceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/calibration/frame.jpg", include_in_schema=False)
    async def calibration_image() -> FileResponse:
        path = service.calibration_frame_path
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no calibration frame grabbed yet")
        # Never cached: the same URL serves a new still after every grab.
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.post("/api/calibration/solve")
    async def calibration_solve(request: SolveRequest) -> dict[str, Any]:
        try:
            return service.solve_calibration(
                [marker.model_dump() for marker in request.markers],
                image_size=request.image_size,
                source=request.source,
                notes=request.notes,
                save=request.save,
            )
        except ServiceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/calibration")
    async def calibration_current() -> dict[str, Any]:
        return {"calibration": service.load_calibration()}

    @app.get("/viewer", include_in_schema=False)
    async def viewer_page() -> FileResponse:
        return FileResponse(STATIC / "viewer.html")

    @app.get("/api/sessions/{session}/segments")
    async def session_segments(session: str) -> list[dict[str, Any]]:
        try:
            return service.segments_of(session)
        except ServiceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/viewer/window")
    async def viewer_window(request: WindowRequest) -> dict[str, Any]:
        try:
            return service.frame_window(
                request.session, request.segment, request.frame, half=request.half
            )
        except ServiceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/viewer/frame/{frame}.jpg", include_in_schema=False)
    async def viewer_frame(frame: int) -> FileResponse:
        try:
            path = service.frame_image(frame)
        except ServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.post("/api/viewer/measure")
    async def viewer_measure(request: MeasureRequest) -> dict[str, Any]:
        return {"measurement": service.measure(request.image_x, request.image_y)}

    @app.post("/api/annotations")
    async def add_annotation(request: AnnotationRequest) -> dict[str, Any]:
        try:
            return service.annotate(
                request.session,
                request.segment,
                request.frame,
                image_x=request.image_x,
                image_y=request.image_y,
                aircraft=request.aircraft,
                note=request.note,
            )
        except ServiceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/annotations/{session}")
    async def list_annotations(session: str) -> list[dict[str, Any]]:
        return service.annotations(session)

    @app.get("/api/sessions")
    async def sessions() -> list[dict[str, Any]]:
        return service.sessions()

    @app.get("/api/sessions/{session}")
    async def session_report(session: str) -> dict[str, Any]:
        try:
            return service.session_report(session)
        except ServiceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


def serve(
    root: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    results: Path | None = None,
) -> None:
    """Run the control UI until interrupted."""
    import uvicorn

    service = CaptureService(root, ffmpeg=ffmpeg, ffprobe=ffprobe)
    review = ReviewService(service, out_root=results)
    uvicorn.run(create_app(service, review), host=host, port=port, log_level="warning")
