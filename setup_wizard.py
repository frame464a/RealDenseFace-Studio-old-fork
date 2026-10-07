"""First-run setup for RealDenseFace Studio: model weights + FLAME head model."""
from __future__ import annotations

import common.fix_chumpy  # noqa: F401  (needed to unpickle the FLAME model)

import pickle
import shutil
import threading
import time
import traceback
import webbrowser
import zipfile
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

ROOT = Path(__file__).resolve().parent
WEIGHTS_DRIVE_ID = "1eLlN_BCFMn0jSo0lNwBFqtI6igTh7Nc3"
WEIGHTS_ZIP_SIZE = 1_093_003_008
WEIGHT_FILES = [
    "weights/realdenseface/vits.pth",
    "weights/realdenseface/vitb.pth",
    "weights/facebox/face_box.pth",
    "weights/flame/flame_assets.npz",
]
FLAME_PATH = ROOT / "weights" / "flame" / "flame2023.pkl"
FLAME_URL = "https://flame.is.tue.mpg.de/download.php"


def weights_ok() -> bool:
    return all((ROOT / f).is_file() for f in WEIGHT_FILES)


def flame_ok() -> bool:
    return FLAME_PATH.is_file()


def setup_complete() -> bool:
    return weights_ok() and flame_ok()


def install_weights_zip(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.startswith("weights/") and not n.endswith("/")]
        missing = [f for f in WEIGHT_FILES if f not in names]
        if missing:
            raise ValueError(f"This zip does not look like the RealDenseFace weights (missing {', '.join(missing)}).")
        for name in names:
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 * 1024 * 1024)


def validate_flame(data: bytes) -> None:
    model = pickle.loads(data, encoding="latin1")
    v_template = getattr(model["v_template"], "shape", None)
    shapedirs = getattr(model["shapedirs"], "shape", None)
    if v_template != (5023, 3) or shapedirs is None or shapedirs[-1] < 400:
        raise ValueError("This file is not a FLAME 2023 model.")


def install_flame(path: Path) -> None:
    """Accept the FLAME 2023 zip or the flame2023.pkl inside it."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            matches = [n for n in zf.namelist() if Path(n).name == "flame2023.pkl"]
            if not matches:
                names = ", ".join(Path(n).name for n in zf.namelist() if n.endswith(".pkl")) or "none"
                raise ValueError(
                    "flame2023.pkl was not found in this zip (found: " + names + ").\n\n"
                    "Download 'FLAME 2023 (revised eye region, improved expressions, versions w/ and w/o jaw rotation)', "
                    "not 'FLAME 2023 Open'."
                )
            data = zf.read(matches[0])
    else:
        data = path.read_bytes()
    validate_flame(data)
    FLAME_PATH.parent.mkdir(parents=True, exist_ok=True)
    FLAME_PATH.write_bytes(data)


class DownloadWorker(QThread):
    progress = Signal(int, int)
    done = Signal()
    failed = Signal(str)

    def run(self) -> None:
        try:
            import gdown

            download_dir = ROOT / "tools" / "download"
            download_dir.mkdir(parents=True, exist_ok=True)
            zip_path = download_dir / "realdenseface_weights.zip"
            result: dict = {}

            def fetch() -> None:
                try:
                    result["path"] = gdown.download(id=WEIGHTS_DRIVE_ID, output=str(zip_path), quiet=True, resume=True)
                except Exception as exc:  # noqa: BLE001
                    result["error"] = exc

            thread = threading.Thread(target=fetch, daemon=True)
            thread.start()
            while thread.is_alive():
                size = sum(p.stat().st_size for p in download_dir.glob("realdenseface_weights.zip*") if p.is_file())
                self.progress.emit(min(size, WEIGHTS_ZIP_SIZE), WEIGHTS_ZIP_SIZE)
                time.sleep(0.5)
            if "error" in result or not result.get("path"):
                raise RuntimeError(
                    f"Download failed: {result.get('error', 'unknown error')}\n\n"
                    "Google Drive sometimes limits downloads of popular files. Download the zip in your browser "
                    "from the link in the README, then use 'I already have the zip…'."
                )
            self.progress.emit(-1, -1)  # extracting
            install_weights_zip(zip_path)
            zip_path.unlink(missing_ok=True)
            self.done.emit()
        except Exception:
            self.failed.emit(traceback.format_exc(limit=1))


class StepCard(QFrame):
    def __init__(self, number: str, title: str, text: str) -> None:
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        head = QHBoxLayout()
        self.badge = QLabel(number)
        self.badge.setObjectName("badge")
        self.badge.setFixedSize(28, 28)
        self.badge.setAlignment(Qt.AlignCenter)
        title_label = QLabel(title)
        title_label.setObjectName("cardtitle")
        head.addWidget(self.badge)
        head.addWidget(title_label, 1)
        layout.addLayout(head)
        body = QLabel(text)
        body.setWordWrap(True)
        body.setObjectName("hint")
        body.setOpenExternalLinks(True)
        layout.addWidget(body)
        self.buttons = QHBoxLayout()
        layout.addLayout(self.buttons)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def set_done(self, done: bool, text: str = "") -> None:
        self.badge.setText("✓" if done else self.badge.text())
        self.badge.setProperty("done", done)
        self.badge.style().polish(self.badge)
        self.status.setText(text)


class SetupDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RealDenseFace Studio: setup")
        self.setMinimumWidth(620)
        self.worker: DownloadWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        title = QLabel("Welcome to RealDenseFace Studio")
        title.setObjectName("title")
        layout.addWidget(title)
        intro = QLabel("Two things are needed before the first start. You only do this once.")
        intro.setObjectName("hint")
        layout.addWidget(intro)

        self.weights_card = StepCard(
            "1",
            "Download the tracking models (1.1 GB)",
            "The pretrained RealDenseFace networks by Li, Shao and Zhou (Zhejiang University).",
        )
        self.download_btn = QPushButton("⬇  Download")
        self.download_btn.setObjectName("primary")
        self.weights_zip_btn = QPushButton("I already have the zip…")
        self.weights_card.buttons.addWidget(self.download_btn)
        self.weights_card.buttons.addWidget(self.weights_zip_btn)
        layout.addWidget(self.weights_card)

        self.flame_card = StepCard(
            "2",
            "Get the FLAME 2023 head model (free)",
            "FLAME's license doesn't allow sharing it, so you download it yourself:<br>"
            "1. Register at the FLAME website and accept the license.<br>"
            "2. Under <b>Downloads</b>, get <b>FLAME 2023 (revised eye region, improved expressions, "
            "versions w/ and w/o jaw rotation)</b>. Not “FLAME 2023 Open”.<br>"
            "3. Select the downloaded zip here.",
        )
        self.flame_site_btn = QPushButton("🌐  Open FLAME website")
        self.flame_file_btn = QPushButton("Select downloaded file…")
        self.flame_file_btn.setObjectName("primary")
        self.flame_card.buttons.addWidget(self.flame_site_btn)
        self.flame_card.buttons.addWidget(self.flame_file_btn)
        layout.addWidget(self.flame_card)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.quit_btn = QPushButton("Quit")
        self.continue_btn = QPushButton("Start RealDenseFace Studio  →")
        self.continue_btn.setObjectName("primary")
        bottom.addWidget(self.quit_btn)
        bottom.addWidget(self.continue_btn)
        layout.addLayout(bottom)

        self.download_btn.clicked.connect(self.download_weights)
        self.weights_zip_btn.clicked.connect(self.pick_weights_zip)
        self.flame_site_btn.clicked.connect(lambda: webbrowser.open(FLAME_URL))
        self.flame_file_btn.clicked.connect(self.pick_flame)
        self.quit_btn.clicked.connect(self.reject)
        self.continue_btn.clicked.connect(self.accept)
        self.refresh()

    def refresh(self) -> None:
        w, f = weights_ok(), flame_ok()
        if w:
            self.weights_card.set_done(True, "Installed.")
        if f:
            self.flame_card.set_done(True, "Installed.")
        busy = self.worker is not None and self.worker.isRunning()
        self.download_btn.setEnabled(not w and not busy)
        self.weights_zip_btn.setEnabled(not w and not busy)
        self.flame_file_btn.setEnabled(not f)
        self.continue_btn.setEnabled(w and f and not busy)

    def download_weights(self) -> None:
        self.worker = DownloadWorker()
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_weights_done)
        self.worker.failed.connect(self.on_weights_failed)
        self.worker.finished.connect(self.refresh)
        self.weights_card.progress.setVisible(True)
        self.weights_card.progress.setRange(0, 0)
        self.weights_card.status.setText("Starting download…")
        self.worker.start()
        self.refresh()

    def on_progress(self, done: int, total: int) -> None:
        bar = self.weights_card.progress
        if done < 0:
            bar.setRange(0, 0)
            self.weights_card.status.setText("Unpacking…")
            return
        bar.setRange(0, 1000)
        bar.setValue(int(1000 * done / total))
        self.weights_card.status.setText(f"{done / 1e9:.2f} / {total / 1e9:.2f} GB")

    def on_weights_done(self) -> None:
        self.weights_card.progress.setVisible(False)
        self.refresh()

    def on_weights_failed(self, message: str) -> None:
        self.weights_card.progress.setVisible(False)
        self.weights_card.status.setText("Download failed.")
        QMessageBox.warning(self, "Download failed", message[-2000:])

    def pick_weights_zip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select the RealDenseFace weights zip", "", "Zip (*.zip);;All files (*)")
        if not path:
            return
        try:
            self.weights_card.status.setText("Unpacking…")
            self.repaint()
            install_weights_zip(Path(path))
        except Exception as exc:  # noqa: BLE001
            self.weights_card.status.setText("")
            QMessageBox.warning(self, "Not the right file", str(exc))
        self.refresh()

    def pick_flame(self) -> None:
        downloads = str(Path.home() / "Downloads")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FLAME2023.zip or flame2023.pkl", downloads, "FLAME (*.zip *.pkl);;All files (*)"
        )
        if not path:
            return
        try:
            self.flame_card.status.setText("Checking…")
            self.repaint()
            install_flame(Path(path))
        except Exception as exc:  # noqa: BLE001
            self.flame_card.status.setText("")
            QMessageBox.warning(self, "Not the right file", str(exc))
        self.refresh()

    def reject(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            if QMessageBox.question(self, "Quit", "A download is running. Quit anyway?") != QMessageBox.Yes:
                return
        super().reject()


SETUP_STYLE = """
QDialog { background: #16181d; }
QFrame#card { background: #1c1f25; border: 1px solid #2a2e36; border-radius: 10px; padding: 6px; }
QLabel#cardtitle { font-size: 11pt; font-weight: 600; }
QLabel#badge { background: #2a2f38; border-radius: 14px; font-weight: 700; }
QLabel#badge[done="true"] { background: #2e7d4f; color: white; }
QLabel a { color: #7aa2ff; }
"""
