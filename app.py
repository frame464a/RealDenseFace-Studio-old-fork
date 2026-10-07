"""RealDenseFace Studio: live face tracking, recording and geometry export.

Run:  python app.py
"""
from __future__ import annotations

import common.fix_chumpy  # noqa: F401  (must import before FLAME model load)

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QProcess, QProcessEnvironment, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPalette, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
TAKES_DIR = ROOT / "output" / "takes"
LOG_PATH = ROOT / "output" / "app.log"

VIEW_MODES = ["Overlay", "Mesh", "Camera", "Side by side"]
RESOLUTIONS = {"640 x 480": (640, 480), "1280 x 720": (1280, 720), "1920 x 1080": (1920, 1080)}
EXPORT_FORMATS = [
    ("USD  (.usdc)", "usd", "Animated mesh + camera in one file. Blender, Houdini, Maya, Unreal, C4D."),
    ("Alembic (.abc)", "abc", "Animated mesh + camera in one file. Maya, Houdini, Blender, C4D, Nuke, Unreal."),
    ("OBJ + PC2 point cache", "pc2", "Base mesh + point cache. Blender Mesh Cache modifier, 3ds Max, C4D, Houdini."),
    ("OBJ sequence", "objseq", "One .obj per frame. Works everywhere, large on disk."),
    ("Video (.mp4)", "mp4", "Rendered video of the tracked mesh, for review or sharing."),
]
VIDEO_STYLES = [("Mesh over footage", "overlay"), ("Mesh only", "mesh"), ("Footage | mesh side by side", "side_by_side")]
UNITS = [("Meters", 1.0), ("Centimeters", 100.0), ("Millimeters", 1000.0)]


def list_cameras() -> list[str]:
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception:
        return [f"Camera {i}" for i in range(3)]


# --------------------------------------------------------------------------------------
# Live tracking worker
# --------------------------------------------------------------------------------------
class LiveWorker(QThread):
    frame_ready = Signal(QImage, dict)
    status = Signal(str)
    loaded = Signal()
    take_saved = Signal(str)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.commands: queue.Queue = queue.Queue()
        self.view_mode = 0
        self.mirror = True
        self.fov_y_deg = 40.0
        self.fast_start = False
        self._running = True

    # Commands are executed inside the worker thread.
    def send(self, name: str, *args) -> None:
        self.commands.put((name, args))

    def stop(self) -> None:
        self._running = False
        self.wait(5000)

    def _load(self) -> None:
        import torch

        from tracker import (
            FLAMEConfig,
            FaceBoxConfig,
            OnlineGNOptimizer,
            OnlineGNOptimizerConfig,
            RealDenseFaceInferenceConfig,
            RealDenseFaceInferencer,
            load_fitting_config,
        )
        from visualization import Visualizer

        import torch._dynamo

        torch.set_grad_enabled(False)
        torch._dynamo.config.suppress_errors = True  # fall back to eager mode if compilation fails on this machine
        self.status.emit("Loading models… (first start can take a few minutes)")
        flame_config = FLAMEConfig()
        self.inferencer = RealDenseFaceInferencer(
            RealDenseFaceInferenceConfig(
                model_config_path="configs/model/vits.yaml",
                model_weights_path="weights/realdenseface/vits.pth",
                flame=flame_config,
                compile_model=not self.fast_start,
                output_type="torch",
            ),
            FaceBoxConfig(),
        )
        self.optimizer = OnlineGNOptimizer(
            load_fitting_config("configs/fitting/video_online.yaml", OnlineGNOptimizerConfig, flame=flame_config, device="cuda")
        )
        self.visualizer = Visualizer(flame_assets_path=flame_config.flame_assets_path, device="cuda")

        # Warm up the solver / renderer on a demo frame so the first live frame is not slow.
        self.status.emit("Warming up tracker…")
        cap = cv2.VideoCapture("assets/demo_video.mp4")
        ok, frame = cap.read()
        cap.release()
        if ok:
            frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._build_camera(frame.shape[1], frame.shape[0])
            for i in range(3):
                vertices = self._track(frame, i)
                if vertices is not None:
                    self._render(frame, vertices)
            self.camera = None
            self._reset_tracking()

    def run(self) -> None:
        try:
            self._run()
        except Exception:
            self.error.emit(traceback.format_exc())

    def _open_source(self, kind: str, value, resolution: tuple[int, int]):
        if self.cap is not None:
            self.cap.release()
        if kind == "camera":
            cap = cv2.VideoCapture(int(value), cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_FPS, 60)
            self.source_fps = None
        else:
            cap = cv2.VideoCapture(str(value))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.source_fps = fps if fps and fps > 0 else 30.0
        if not cap.isOpened():
            self.status.emit(f"Could not open {kind}: {value}")
            self.cap = None
            return
        self.cap = cap
        self.source_kind = kind
        self.source_name = Path(str(value)).stem if kind == "video" else f"cam{value}"
        self.camera = None  # rebuilt for the new frame size
        self._reset_tracking()

    def _build_camera(self, w: int, h: int) -> None:
        from track_video_online import build_camera

        self.camera = build_camera(w, h, self.fov_y_deg, 1.0, 0.01, 100.0)
        self.optimizer.refresh_camera_matrices(self.camera)

    def _reset_tracking(self) -> None:
        self.optimizer.reset()
        self.face_bbox = None
        self.has_face = False

    def _run(self) -> None:
        self.cap = None
        self.camera = None
        self.take = None
        self.face_bbox = None
        self.has_face = False
        self._load()
        self.loaded.emit()

        fps_smooth = 0.0
        t_prev = time.perf_counter()
        frame_index = 0
        while self._running:
            self._handle_commands()
            if self.cap is None:
                time.sleep(0.05)
                continue

            loop_start = time.perf_counter()
            ok, frame_bgr = self.cap.read()
            if not ok:
                if self.source_kind == "video":  # loop video files
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._reset_tracking()
                    continue
                self.status.emit("Camera stopped delivering frames.")
                time.sleep(0.2)
                continue
            if self.mirror and self.source_kind == "camera":
                frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            h, w = frame_rgb.shape[:2]
            if self.camera is None or self.camera.width != w or self.camera.height != h:
                self._build_camera(w, h)

            vertices = self._track(frame_rgb, frame_index)
            frame_index += 1

            if self.take is not None and vertices is not None:
                self.take.add(vertices, self._last_x, frame_bgr, loop_start)

            display = self._render(frame_rgb, vertices)
            now = time.perf_counter()
            fps_smooth = 0.9 * fps_smooth + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            image = QImage(display.data, display.shape[1], display.shape[0], display.strides[0], QImage.Format_RGB888).copy()
            self.frame_ready.emit(
                image,
                {
                    "fps": fps_smooth,
                    "face": vertices is not None,
                    "recording": self.take is not None,
                    "rec_frames": 0 if self.take is None else len(self.take.vertices),
                    "rec_seconds": 0.0 if self.take is None else self.take.duration(),
                },
            )

            if self.source_kind == "video" and self.source_fps:
                remaining = 1.0 / self.source_fps - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

        if self.take is not None:
            self._stop_recording()
        if self.cap is not None:
            self.cap.release()

    def _track(self, frame_rgb: np.ndarray, frame_index: int):
        h, w = frame_rgb.shape[:2]
        # (Re)acquire the face with the detector; also re-check periodically so we notice when it leaves.
        if self.face_bbox is None or frame_index % 15 == 0:
            detected = self.inferencer._detect_face_bbox(frame_rgb)
            if detected is None:
                self.face_bbox = None
                self.has_face = False
                return None
            if self.face_bbox is None:
                self.face_bbox = self.inferencer.run(frame_rgb, detected).face_bbox

        inference = self.inferencer.run(frame_rgb, self.face_bbox)
        x = self.optimizer.track(inference)
        self.face_bbox = inference.face_bbox
        bw = self.face_bbox[2] - self.face_bbox[0]
        bh = self.face_bbox[3] - self.face_bbox[1]
        if bw < 20 or bh < 20 or bw > w * 1.5:
            self.face_bbox = None
        self.has_face = True
        self._last_x = x
        vertices, _ = self.optimizer.decode(x)
        return vertices

    def _render(self, frame_rgb: np.ndarray, vertices) -> np.ndarray:
        mode = VIEW_MODES[self.view_mode]
        if vertices is None or mode == "Camera":
            if mode == "Mesh" and vertices is None:
                return np.zeros_like(frame_rgb)
            return frame_rgb
        overlay, shading = self.visualizer.vis_flame_shading_overlay(
            self.camera, frame_rgb, vertices, alpha=0.55, return_shading=True, shading_background="black"
        )
        if mode == "Overlay":
            return overlay
        if mode == "Mesh":
            return shading
        half = (frame_rgb.shape[1] // 2) & ~1
        left = cv2.resize(frame_rgb, (half, frame_rgb.shape[0] // 2))
        right = cv2.resize(shading, (half, frame_rgb.shape[0] // 2))
        return np.ascontiguousarray(np.concatenate([left, right], axis=1))

    def _handle_commands(self) -> None:
        while True:
            try:
                name, args = self.commands.get_nowait()
            except queue.Empty:
                return
            if name == "source":
                self._open_source(*args)
            elif name == "close_source":
                if self.cap is not None:
                    self.cap.release()
                self.cap = None
            elif name == "reset":
                self._reset_tracking()
            elif name == "fov":
                self.fov_y_deg = float(args[0])
                self.camera = None
                self._reset_tracking()
            elif name == "record":
                self._start_recording(*args)
            elif name == "stop_record":
                self._stop_recording()

    def _start_recording(self, save_video: bool) -> None:
        if self.take is None and self.cap is not None:
            self.take = Take(save_video)
            self.status.emit("Recording…")

    def _stop_recording(self) -> None:
        take, self.take = self.take, None
        if take is None:
            return
        self.status.emit("Saving take…")
        path = take.save(
            identity=self.optimizer.identity,
            fov_y_deg=self.fov_y_deg,
            width=self.camera.width,
            height=self.camera.height,
            source=self.source_name,
            fps_hint=self.source_fps,
        )
        if path is None:
            self.status.emit("Take was empty (no face tracked) and was discarded.")
        else:
            self.status.emit(f"Saved {path.parent.name}")
            self.take_saved.emit(str(path))


class Take:
    def __init__(self, save_video: bool) -> None:
        self.vertices: list[np.ndarray] = []
        self.xs: list[np.ndarray] = []
        self.times: list[float] = []
        self.jpegs: list[bytes] | None = [] if save_video else None

    def duration(self) -> float:
        return 0.0 if len(self.times) < 2 else self.times[-1] - self.times[0]

    def add(self, vertices, x, frame_bgr: np.ndarray, t: float) -> None:
        self.vertices.append(vertices.detach().cpu().numpy().astype(np.float32))
        self.xs.append(x.detach().reshape(-1).cpu().numpy().astype(np.float32))
        self.times.append(t)
        if self.jpegs is not None:
            self.jpegs.append(cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())

    def save(self, identity, fov_y_deg: float, width: int, height: int, source: str, fps_hint: float | None) -> Path | None:
        if len(self.vertices) < 2:
            return None
        times = np.asarray(self.times, dtype=np.float64)
        fps = fps_hint or float((len(times) - 1) / max(times[-1] - times[0], 1e-6))
        TAKES_DIR.mkdir(parents=True, exist_ok=True)
        take_dir = TAKES_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{source}"
        take_dir.mkdir()
        path = take_dir / "take.npz"
        np.savez(
            path,
            vertices=np.stack(self.vertices),
            identity=identity.detach().cpu().numpy().astype(np.float32),
            x=np.stack(self.xs),
            timestamps=times - times[0],
            fps=np.float32(fps),
            camera_fov_y=np.float32(fov_y_deg),
            camera_rot=np.diag([1.0, -1.0, -1.0]).astype(np.float32),
            camera_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            image_width=np.int32(width),
            image_height=np.int32(height),
            num_expressions=np.int32(self.xs[0].shape[0] - 18),
        )
        if self.jpegs:
            writer = cv2.VideoWriter(str(take_dir / "camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            for buf in self.jpegs:
                writer.write(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR))
            writer.release()
        (take_dir / "info.json").write_text(
            json.dumps({"frames": len(self.vertices), "fps": fps, "duration": float(times[-1] - times[0]), "source": source}),
            encoding="utf-8",
        )
        return path


# --------------------------------------------------------------------------------------
# Export worker
# --------------------------------------------------------------------------------------
class ExportWorker(QThread):
    done = Signal(str)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, kwargs: dict) -> None:
        super().__init__()
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            import torch

            torch.set_grad_enabled(False)
            from export_animation import export, export_video

            kwargs = dict(self.kwargs)
            if kwargs.pop("fmt") == "mp4":
                result = export_video(
                    kwargs["input_npz"], kwargs["output"], kwargs["style"],
                    progress=lambda i, n: self.progress.emit(i, n),
                )
            else:
                kwargs.pop("style")
                result = export(fmt=self.kwargs["fmt"], **kwargs)
            self.done.emit(str(result))
        except Exception:
            self.failed.emit(traceback.format_exc())


# --------------------------------------------------------------------------------------
# UI widgets
# --------------------------------------------------------------------------------------
class VideoView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.image: QImage | None = None
        self.message = "Starting…"
        self.recording = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(480, 320)

    def set_image(self, image: QImage) -> None:
        self.image = image
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#0d0f12"))
        if self.image is not None:
            iw, ih = self.image.width(), self.image.height()
            scale = min(self.width() / iw, self.height() / ih)
            w, h = iw * scale, ih * scale
            target = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
            p.drawImage(target, self.image)
            if self.recording:
                pen = p.pen()
                pen.setColor(QColor("#ff3b3b"))
                pen.setWidth(4)
                p.setPen(pen)
                p.drawRect(target.adjusted(2, 2, -2, -2))
        if self.message:
            p.setPen(QColor("#e8e8e8"))
            f = QFont(self.font())
            f.setPointSize(13)
            p.setFont(f)
            box = self.rect().adjusted(0, 0, 0, -20)
            p.fillRect(QRectF(0, self.height() / 2 - 30, self.width(), 60), QColor(0, 0, 0, 150))
            p.drawText(box.adjusted(0, 20, 0, 0), Qt.AlignCenter, self.message)
        p.end()


def section(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.setSpacing(8)
    return box, layout


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RealDenseFace Studio")
        self.resize(1500, 900)
        self.export_worker: ExportWorker | None = None
        self.process: QProcess | None = None
        self.models_ready = False

        self.view = VideoView()
        self.status_label = QLabel()
        self.fps_label = QLabel()
        self.face_label = QLabel()
        for lbl in (self.fps_label, self.face_label):
            lbl.setObjectName("pill")

        # ---- Source -----------------------------------------------------------------
        source_box, source_layout = section("1  Source")
        self.camera_combo = QComboBox()
        self.cameras = list_cameras()
        self.camera_combo.addItems(self.cameras)
        self.res_combo = QComboBox()
        self.res_combo.addItems(list(RESOLUTIONS))
        self.res_combo.setCurrentText("1280 x 720")
        self.mirror_check = QCheckBox("Mirror (selfie view)")
        self.mirror_check.setChecked(True)
        self.start_cam_btn = QPushButton("▶  Start webcam")
        self.start_cam_btn.setObjectName("primary")
        self.open_video_btn = QPushButton("🎞  Open video file…")
        form = QFormLayout()
        form.addRow("Camera", self.camera_combo)
        form.addRow("Resolution", self.res_combo)
        source_layout.addLayout(form)
        source_layout.addWidget(self.mirror_check)
        row = QHBoxLayout()
        row.addWidget(self.start_cam_btn)
        row.addWidget(self.open_video_btn)
        source_layout.addLayout(row)

        # ---- View / tracking --------------------------------------------------------
        view_box, view_layout = section("2  View")
        self.view_buttons: list[QToolButton] = []
        row = QHBoxLayout()
        row.setSpacing(4)
        for i, name in enumerate(VIEW_MODES):
            b = QToolButton()
            b.setText(name)
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setChecked(i == 0)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(lambda _=False, i=i: self.set_view_mode(i))
            self.view_buttons.append(b)
            row.addWidget(b)
        view_layout.addLayout(row)
        row = QHBoxLayout()
        self.reset_btn = QPushButton("↺  Reset tracking")
        self.reset_btn.setToolTip("Re-detect the face and re-estimate the head shape from scratch.")
        row.addWidget(self.reset_btn)
        view_layout.addLayout(row)
        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(10, 100)
        self.fov_spin.setValue(40)
        self.fov_spin.setSuffix(" °")
        self.fov_spin.setToolTip("Vertical field of view of the camera. Most webcams are 35–50°.")
        form = QFormLayout()
        form.addRow("Camera FOV", self.fov_spin)
        view_layout.addLayout(form)

        # ---- Record -----------------------------------------------------------------
        rec_box, rec_layout = section("3  Record")
        self.rec_btn = QPushButton("●  Start recording")
        self.rec_btn.setObjectName("record")
        self.rec_btn.setMinimumHeight(54)
        self.rec_btn.setEnabled(False)
        self.rec_info = QLabel("Shortcuts: Space record · R reset · 1–4 view")
        self.rec_info.setObjectName("hint")
        self.save_video_check = QCheckBox("Also save camera footage with the take")
        self.save_video_check.setChecked(True)
        rec_layout.addWidget(self.rec_btn)
        rec_layout.addWidget(self.rec_info)
        rec_layout.addWidget(self.save_video_check)
        self.process_btn = QPushButton("⚙  Process a video file (best quality)…")
        self.process_btn.setToolTip(
            "Tracks a whole video offline with the larger model and a global solve.\n"
            "More accurate than live tracking. The result appears in the takes list."
        )
        self.process_progress = QProgressBar()
        self.process_progress.setVisible(False)
        self.process_label = QLabel()
        self.process_label.setObjectName("hint")
        self.process_label.setVisible(False)
        rec_layout.addWidget(self.process_btn)
        rec_layout.addWidget(self.process_label)
        rec_layout.addWidget(self.process_progress)

        # ---- Takes / export -----------------------------------------------------------
        exp_box, exp_layout = section("4  Takes && export")
        self.takes_list = QListWidget()
        self.takes_list.setMinimumHeight(150)
        exp_layout.addWidget(self.takes_list)
        self.format_combo = QComboBox()
        for i, (label, _, tip) in enumerate(EXPORT_FORMATS):
            self.format_combo.addItem(label)
            self.format_combo.setItemData(i, tip, Qt.ToolTipRole)
        self.units_combo = QComboBox()
        for label, _ in UNITS:
            self.units_combo.addItem(label)
        self.head_only_check = QCheckBox("Head-only (remove head rotation/position)")
        self.camera_check = QCheckBox("Include tracking camera")
        self.camera_check.setChecked(True)
        self.style_combo = QComboBox()
        for label, _ in VIDEO_STYLES:
            self.style_combo.addItem(label)
        self.export_form = QFormLayout()
        self.export_form.addRow("Format", self.format_combo)
        self.export_form.addRow("Units", self.units_combo)
        self.export_form.addRow("Style", self.style_combo)
        exp_layout.addLayout(self.export_form)
        exp_layout.addWidget(self.head_only_check)
        exp_layout.addWidget(self.camera_check)
        self.export_progress = QProgressBar()
        self.export_progress.setVisible(False)
        exp_layout.addWidget(self.export_progress)
        self.format_combo.currentIndexChanged.connect(self.update_export_options)
        row = QHBoxLayout()
        self.export_btn = QPushButton("⬇  Export…")
        self.export_btn.setObjectName("primary")
        self.folder_btn = QPushButton("📂  Show")
        self.delete_btn = QPushButton("🗑")
        self.delete_btn.setToolTip("Delete take")
        self.delete_btn.setFixedWidth(44)
        row.addWidget(self.export_btn, 2)
        row.addWidget(self.folder_btn, 1)
        row.addWidget(self.delete_btn)
        exp_layout.addLayout(row)

        # ---- Layout -----------------------------------------------------------------
        side = QWidget()
        side.setObjectName("side")
        side.setFixedWidth(400)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(12)
        title = QLabel("RealDenseFace Studio")
        title.setObjectName("title")
        side_layout.addWidget(title)
        for box in (source_box, view_box, rec_box, exp_box):
            side_layout.addWidget(box)
        side_layout.addStretch(1)

        status_bar = QFrame()
        status_bar.setObjectName("statusbar")
        sb = QHBoxLayout(status_bar)
        sb.setContentsMargins(12, 6, 12, 6)
        sb.addWidget(self.status_label, 1)
        sb.addWidget(self.face_label)
        sb.addWidget(self.fps_label)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.view, 1)
        left_layout.addWidget(status_bar)

        central = QWidget()
        main = QHBoxLayout(central)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        main.addWidget(left, 1)
        main.addWidget(side)
        self.setCentralWidget(central)

        # ---- Signals ----------------------------------------------------------------
        self.worker = LiveWorker()
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.set_status)
        self.worker.loaded.connect(self.on_loaded)
        self.worker.take_saved.connect(self.on_take_saved)
        self.worker.error.connect(self.on_worker_error)
        self.start_cam_btn.clicked.connect(self.start_camera)
        self.open_video_btn.clicked.connect(self.open_video)
        self.mirror_check.toggled.connect(lambda v: setattr(self.worker, "mirror", v))
        self.reset_btn.clicked.connect(lambda: self.worker.send("reset"))
        self.fov_spin.editingFinished.connect(lambda: self.worker.send("fov", self.fov_spin.value()))
        self.rec_btn.clicked.connect(self.toggle_record)
        self.process_btn.clicked.connect(self.process_video)
        self.export_btn.clicked.connect(self.export_take)
        self.folder_btn.clicked.connect(self.show_take_folder)
        self.delete_btn.clicked.connect(self.delete_take)
        self.takes_list.currentRowChanged.connect(lambda _: self.update_buttons())
        self.takes_list.itemDoubleClicked.connect(lambda _: self.export_take())

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_record)
        QShortcut(QKeySequence(Qt.Key_R), self, activated=lambda: self.worker.send("reset"))
        for i in range(len(VIEW_MODES)):
            QShortcut(QKeySequence(str(i + 1)), self, activated=lambda i=i: self.set_view_mode(i))

        self.recording = False
        self.source_active = False
        self.refresh_takes()
        self.update_export_options()
        self.update_buttons()
        self.set_status("Loading models… (first start can take a few minutes)")
        self.view.message = "Loading models…\nFirst start can take a few minutes, later starts are faster."
        self.worker.start()

    # ---- live --------------------------------------------------------------------
    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        if not self.source_active and self.models_ready:
            self.view.message = text
            self.view.update()

    def on_loaded(self) -> None:
        self.models_ready = True
        self.set_status("Ready. Pick a camera and press Start webcam, or open a video file.")
        self.view.message = "Ready\nPress “Start webcam” or open a video file"
        self.view.update()
        self.update_buttons()

    def on_frame(self, image: QImage, stats: dict) -> None:
        if self.view.message and stats["face"]:
            self.view.message = ""
        elif not stats["face"]:
            self.view.message = "Looking for a face…"
        self.view.recording = stats["recording"]
        self.view.set_image(image)
        self.fps_label.setText(f"{stats['fps']:.0f} fps")
        self.face_label.setText("● face tracked" if stats["face"] else "○ no face")
        self.face_label.setProperty("ok", stats["face"])
        self.face_label.style().polish(self.face_label)
        if stats["recording"] and self.recording:
            s = stats["rec_seconds"]
            self.rec_info.setText(f"REC  {int(s // 60):02d}:{s % 60:05.2f}   ·   {stats['rec_frames']} frames")

    def start_camera(self) -> None:
        if self.recording:
            self.toggle_record()
        idx = self.camera_combo.currentIndex()
        self.worker.mirror = self.mirror_check.isChecked()
        self.worker.send("source", "camera", idx, RESOLUTIONS[self.res_combo.currentText()])
        self.source_active = True
        self.view.message = "Opening camera…"
        self.set_status(f"Live: {self.camera_combo.currentText()}")
        self.update_buttons()

    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video for live preview", "", "Video (*.mp4 *.mov *.avi *.mkv *.webm *.m4v);;All files (*)"
        )
        if not path:
            return
        if self.recording:
            self.toggle_record()
        self.worker.send("source", "video", path, None)
        self.source_active = True
        self.view.message = "Opening video…"
        self.set_status(
            f"Playing {Path(path).name} with live tracking. Record a section, or use "
            "“Process a video file” for best quality."
        )
        self.update_buttons()

    def set_view_mode(self, i: int) -> None:
        self.worker.view_mode = i
        self.view_buttons[i].setChecked(True)

    def toggle_record(self) -> None:
        if not self.models_ready or not self.source_active:
            return
        self.recording = not self.recording
        if self.recording:
            self.worker.send("record", self.save_video_check.isChecked())
            self.rec_btn.setText("■  Stop recording")
            self.rec_btn.setProperty("active", True)
        else:
            self.worker.send("stop_record")
            self.rec_btn.setText("●  Start recording")
            self.rec_btn.setProperty("active", False)
            self.rec_info.setText("Shortcuts: Space record · R reset · 1–4 view")
        self.rec_btn.style().polish(self.rec_btn)

    def on_take_saved(self, path: str) -> None:
        self.refresh_takes(select=Path(path).parent.name)

    def on_worker_error(self, tb: str) -> None:
        print(tb, flush=True)
        QMessageBox.critical(self, "Tracking error", tb[-3000:])

    # ---- takes ---------------------------------------------------------------------
    def refresh_takes(self, select: str | None = None) -> None:
        self.takes_list.clear()
        if TAKES_DIR.exists():
            for d in sorted(TAKES_DIR.iterdir(), reverse=True):
                npz = d / "take.npz"
                if not npz.exists():
                    continue
                try:
                    info = json.loads((d / "info.json").read_text(encoding="utf-8"))
                    desc = f"{info['frames']} frames · {info['duration']:.1f}s · {info['fps']:.0f} fps"
                    if info.get("offline"):
                        desc += " · best quality"
                except Exception:
                    desc = ""
                stamp = d.name[:15]
                try:
                    when = datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime("%b %d  %H:%M:%S")
                except ValueError:
                    when = stamp
                source = d.name[16:]
                item = QListWidgetItem(f"{when}   {source}\n{desc}")
                item.setData(Qt.UserRole, str(npz))
                self.takes_list.addItem(item)
                if select == d.name:
                    self.takes_list.setCurrentItem(item)
        if self.takes_list.currentRow() < 0 and self.takes_list.count():
            self.takes_list.setCurrentRow(0)
        if not self.takes_list.count():
            item = QListWidgetItem("No takes yet. Record one or process a video.")
            item.setFlags(Qt.NoItemFlags)
            self.takes_list.addItem(item)
        self.update_buttons()

    def current_take(self) -> Path | None:
        item = self.takes_list.currentItem()
        if item is None or item.data(Qt.UserRole) is None:
            return None
        return Path(item.data(Qt.UserRole))

    def update_buttons(self) -> None:
        has_take = self.current_take() is not None
        busy_export = self.export_worker is not None and self.export_worker.isRunning()
        self.export_btn.setEnabled(has_take and not busy_export)
        self.folder_btn.setEnabled(has_take)
        self.delete_btn.setEnabled(has_take)
        self.rec_btn.setEnabled(self.models_ready and self.source_active)
        self.start_cam_btn.setEnabled(self.models_ready)
        self.open_video_btn.setEnabled(self.models_ready)
        self.process_btn.setEnabled(self.process is None)

    def update_export_options(self) -> None:
        fmt = EXPORT_FORMATS[self.format_combo.currentIndex()][1]
        is_video = fmt == "mp4"
        self.export_form.setRowVisible(self.units_combo, not is_video)
        self.export_form.setRowVisible(self.style_combo, is_video)
        self.head_only_check.setVisible(not is_video)
        self.camera_check.setVisible(fmt in ("usd", "abc"))

    def show_take_folder(self) -> None:
        take = self.current_take()
        if take:
            subprocess.Popen(["explorer", str(take.parent)])

    def delete_take(self) -> None:
        take = self.current_take()
        if take is None:
            return
        if QMessageBox.question(self, "Delete take", f"Delete {take.parent.name} and its exports?") == QMessageBox.Yes:
            shutil.rmtree(take.parent, ignore_errors=True)
            self.refresh_takes()

    def export_take(self) -> None:
        take = self.current_take()
        if take is None:
            return
        fmt = EXPORT_FORMATS[self.format_combo.currentIndex()][1]  # (label, key, tooltip)
        scale = UNITS[self.units_combo.currentIndex()][1]
        if fmt == "usd":
            path, _ = QFileDialog.getSaveFileName(self, "Export USD", str(take.parent / "face.usdc"), "USD (*.usdc *.usda *.usd)")
        elif fmt == "abc":
            path, _ = QFileDialog.getSaveFileName(self, "Export Alembic", str(take.parent / "face.abc"), "Alembic (*.abc)")
        elif fmt == "pc2":
            path, _ = QFileDialog.getSaveFileName(self, "Export OBJ + PC2", str(take.parent / "face.pc2"), "Point cache (*.pc2)")
        elif fmt == "mp4":
            style = VIDEO_STYLES[self.style_combo.currentIndex()][1]
            path, _ = QFileDialog.getSaveFileName(self, "Export video", str(take.parent / f"face_{style}.mp4"), "Video (*.mp4)")
            from export_animation import find_footage

            if path and style != "mesh" and find_footage(take) is None:
                QMessageBox.warning(
                    self, "No footage",
                    "This take has no camera footage (it was recorded with “Also save camera footage” off).\n"
                    "Use the “Mesh only” style instead.",
                )
                return
        else:
            path = QFileDialog.getExistingDirectory(self, "Folder for OBJ sequence", str(take.parent))
            if path:
                path = str(Path(path) / "obj_sequence")
        if not path:
            return
        self.export_worker = ExportWorker(
            dict(
                input_npz=take,
                output=path,
                fmt=fmt,
                head_space=self.head_only_check.isChecked(),
                scale=scale,
                with_camera=self.camera_check.isChecked(),
                style=VIDEO_STYLES[self.style_combo.currentIndex()][1],
            )
        )
        self.export_worker.progress.connect(self.on_export_progress)
        self.export_worker.done.connect(self.on_export_done)
        self.export_worker.failed.connect(self.on_export_failed)
        self.export_worker.finished.connect(self.update_buttons)
        self.export_btn.setText("Exporting…")
        self.update_buttons()
        self.export_worker.start()

    def on_export_progress(self, i: int, n: int) -> None:
        self.export_progress.setVisible(True)
        self.export_progress.setRange(0, n)
        self.export_progress.setValue(i)

    def on_export_done(self, path: str) -> None:
        self.export_progress.setVisible(False)
        self.export_btn.setText("⬇  Export…")
        self.set_status(f"Exported {path}")
        box = QMessageBox(self)
        box.setWindowTitle("Export finished")
        box.setText(f"Saved:\n{path}")
        show = box.addButton("Show in Explorer", QMessageBox.AcceptRole)
        box.addButton("OK", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() == show:
            subprocess.Popen(["explorer", "/select,", str(Path(path))])

    def on_export_failed(self, tb: str) -> None:
        self.export_progress.setVisible(False)
        self.export_btn.setText("⬇  Export…")
        print(tb, flush=True)
        QMessageBox.critical(self, "Export failed", tb[-3000:])

    # ---- offline processing ----------------------------------------------------------
    def process_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Process video (best quality)", "", "Video (*.mp4 *.mov *.avi *.mkv *.webm *.m4v);;All files (*)"
        )
        if not path:
            return
        TAKES_DIR.mkdir(parents=True, exist_ok=True)
        self.offline_dir = TAKES_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{Path(path).stem}"
        self.offline_dir.mkdir()
        self.offline_video = Path(path)
        self.process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        try:
            import torch

            if torch.cuda.device_count() > 1:  # keep live tracking on GPU 0, process on GPU 1
                env.insert("CUDA_VISIBLE_DEVICES", "1")
        except Exception:
            pass
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.on_process_output)
        self.process.finished.connect(self.on_process_finished)
        self.process_log = ""
        self.process_label.setText(f"Processing {self.offline_video.name}… starting")
        self.process_label.setVisible(True)
        self.process_progress.setVisible(True)
        self.process_progress.setRange(0, 0)
        self.update_buttons()
        self.process.start(
            sys.executable,
            [
                "track_video_offline.py",
                "--input", str(self.offline_video),
                "--output_npz", str(self.offline_dir / "take.npz"),
                "--output_mp4", str(self.offline_dir / "preview.mp4"),
            ],
        )

    def on_process_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        self.process_log += text
        for line in reversed(re.split(r"[\r\n]+", text)):
            m = re.search(r"^(.*?):\s*(\d+)%\|", line.strip())
            if m:
                stage = m.group(1).replace("[GN] ", "").strip()
                self.process_progress.setRange(0, 100)
                self.process_progress.setValue(int(m.group(2)))
                self.process_label.setText(f"Processing {self.offline_video.name}: {stage}")
                break

    def on_process_finished(self, code: int, _status) -> None:
        self.process_progress.setVisible(False)
        npz = self.offline_dir / "take.npz"
        if code == 0 and npz.exists():
            data = np.load(npz)
            frames = int(data["x"].shape[0])
            fps = float(data["fps"]) if "fps" in data else 25.0
            (self.offline_dir / "info.json").write_text(
                json.dumps(
                    {"frames": frames, "fps": fps, "duration": frames / fps, "source": self.offline_video.name, "offline": True,
                     "source_path": str(self.offline_video)}
                ),
                encoding="utf-8",
            )
            self.process_label.setText(f"Done: {self.offline_video.name} ({frames} frames). Ready to export.")
            self.refresh_takes(select=self.offline_dir.name)
        else:
            shutil.rmtree(self.offline_dir, ignore_errors=True)
            self.process_label.setText("Processing failed. See output/app.log.")
            print(self.process_log, flush=True)
            QMessageBox.critical(self, "Processing failed", self.process_log[-3000:])
        self.process = None
        self.update_buttons()

    def closeEvent(self, event) -> None:
        if self.recording:
            self.toggle_record()
        if self.process is not None:
            self.process.kill()
        self.worker.stop()
        super().closeEvent(event)


STYLE = """
QWidget { font-family: 'Segoe UI'; font-size: 10pt; color: #e6e6e6; }
QMainWindow, #side { background: #16181d; }
#title { font-size: 15pt; font-weight: 600; padding: 2px 0 6px 2px; }
QGroupBox { border: 1px solid #2a2e36; border-radius: 10px; margin-top: 14px; padding: 12px 10px 10px 10px;
            background: #1c1f25; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #9aa4b2; }
QPushButton, QToolButton { background: #2a2f38; border: 1px solid #353b46; border-radius: 7px; padding: 7px 10px; }
QPushButton:hover, QToolButton:hover { background: #333a45; }
QPushButton:disabled { color: #666; background: #22262d; }
QToolButton:checked { background: #3d6cf0; border-color: #3d6cf0; color: white; }
QPushButton#primary { background: #3d6cf0; border-color: #3d6cf0; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #4f7cff; }
QPushButton#primary:disabled { background: #2b3550; color: #8890a0; }
QPushButton#record { background: #c62828; border: none; color: white; font-size: 13pt; font-weight: 600; border-radius: 10px; }
QPushButton#record:hover { background: #e53935; }
QPushButton#record[active="true"] { background: #ffffff; color: #c62828; }
QPushButton#record:disabled { background: #4a2a2a; color: #999; }
QComboBox, QDoubleSpinBox { background: #23272e; border: 1px solid #353b46; border-radius: 6px; padding: 5px 8px; }
QListWidget { background: #15171b; border: 1px solid #2a2e36; border-radius: 8px; padding: 4px; }
QListWidget::item { padding: 7px; border-radius: 6px; }
QListWidget::item:selected { background: #2c3d66; }
QLabel#hint { color: #9aa4b2; font-size: 9pt; }
#statusbar { background: #111317; border-top: 1px solid #23262c; }
QLabel#pill { background: #23272e; border-radius: 9px; padding: 3px 10px; margin-left: 6px; }
QLabel#pill[ok="true"] { color: #58d68d; }
QProgressBar { background: #23272e; border: none; border-radius: 5px; height: 10px; text-align: center; }
QProgressBar::chunk { background: #3d6cf0; border-radius: 5px; }
QCheckBox { spacing: 8px; }
"""


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None or "pythonw" in sys.executable.lower():
        log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
    print(f"\n=== RealDenseFace Studio {datetime.now()} ===", flush=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#16181d"))
    palette.setColor(QPalette.Base, QColor("#23272e"))
    palette.setColor(QPalette.Text, QColor("#e6e6e6"))
    palette.setColor(QPalette.WindowText, QColor("#e6e6e6"))
    palette.setColor(QPalette.Button, QColor("#2a2f38"))
    palette.setColor(QPalette.ButtonText, QColor("#e6e6e6"))
    palette.setColor(QPalette.Highlight, QColor("#3d6cf0"))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)

    from setup_wizard import SETUP_STYLE, SetupDialog, setup_complete

    if not setup_complete():
        dialog = SetupDialog()
        dialog.setStyleSheet(STYLE + SETUP_STYLE)
        if dialog.exec() != SetupDialog.Accepted:
            sys.exit(0)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
