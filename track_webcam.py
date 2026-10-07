"""Live webcam FLAME tracking with RealDenseFace.

Keys (in the preview window):
    R      start / stop recording (each take is saved to --output_dir as an npz)
    D      re-detect the face (use after the face left the frame)
    T      reset tracker (re-estimates identity from scratch)
    V      cycle view: overlay / shading / camera only
    Q/Esc  quit

Each saved take holds per-frame FLAME vertices plus the FLAME parameters and camera,
and can be turned into USD / OBJ / PC2 with export_animation.py.
"""
from __future__ import annotations

import common.fix_chumpy  # noqa: F401  (must import before FLAME model load)

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from camera import PerspectiveCamera
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
from track_video_online import build_camera


VIEW_MODES = ("overlay", "shading", "camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RealDenseFace online tracking on a live webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--mirror", action="store_true", help="Mirror the webcam image (selfie view).")
    parser.add_argument("--output_dir", type=Path, default=Path("output/webcam"))
    parser.add_argument("--save_video", action="store_true", help="Also save the raw webcam frames of each take as mp4.")
    parser.add_argument("--model_config", type=str, default="configs/model/vits.yaml")
    parser.add_argument("--model_weights", type=str, default="weights/realdenseface/vits.pth")
    parser.add_argument("--fitting_config", type=str, default="configs/fitting/video_online.yaml")
    parser.add_argument("--flame_model", type=str, default="weights/flame/flame2023.pkl")
    parser.add_argument("--flame_assets", type=str, default="weights/flame/flame_assets.npz")
    parser.add_argument("--facebox_weights", type=str, default="weights/facebox/face_box.pth")
    parser.add_argument("--num_expressions", type=int, choices=(50, 100), default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--fov_y_deg", type=float, default=40.0, help="Vertical FOV of your webcam.")
    parser.add_argument("--camera_distance", type=float, default=1.0)
    return parser.parse_args()


class Take:
    def __init__(self, output_dir: Path, frame_size: tuple[int, int], save_video: bool) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.npz_path = output_dir / f"take_{stamp}.npz"
        self.vertices: list[np.ndarray] = []
        self.xs: list[np.ndarray] = []
        self.times: list[float] = []
        self.writer = None
        if save_video:
            self.writer = cv2.VideoWriter(
                str(output_dir / f"take_{stamp}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, frame_size
            )

    def add(self, vertices: torch.Tensor, x: torch.Tensor, frame_rgb: np.ndarray, t: float) -> None:
        self.vertices.append(vertices.detach().cpu().numpy().astype(np.float32))
        self.xs.append(x.detach().reshape(-1).cpu().numpy().astype(np.float32))
        self.times.append(t)
        if self.writer is not None:
            self.writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def save(
        self,
        identity: torch.Tensor,
        camera: PerspectiveCamera,
        fov_y_deg: float,
        camera_distance: float,
        num_expressions: int,
    ) -> None:
        if self.writer is not None:
            self.writer.release()
        if len(self.vertices) < 2:
            print("Take too short, not saved.")
            return
        times = np.asarray(self.times, dtype=np.float64)
        fps = float((len(times) - 1) / max(times[-1] - times[0], 1e-6))
        np.savez(
            self.npz_path,
            vertices=np.stack(self.vertices),
            identity=identity.detach().cpu().numpy().astype(np.float32),
            x=np.stack(self.xs),
            timestamps=times - times[0],
            fps=np.float32(fps),
            camera_fov_y=np.float32(fov_y_deg),
            camera_rot=np.diag([1.0, -1.0, -1.0]).astype(np.float32),
            camera_pos=np.array([0.0, 0.0, camera_distance], dtype=np.float32),
            image_width=np.int32(camera.width),
            image_height=np.int32(camera.height),
            num_expressions=np.int32(num_expressions),
        )
        print(f"Saved take: {self.npz_path}  ({len(self.vertices)} frames, ~{fps:.1f} fps)")


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        raise OSError(f"Failed to open webcam {args.camera}")
    ok, frame = cap.read()
    if not ok:
        raise OSError("Webcam returned no frame.")
    img_h, img_w = frame.shape[:2]
    print(f"Webcam {args.camera}: {img_w}x{img_h}")

    camera = build_camera(img_w, img_h, args.fov_y_deg, args.camera_distance, 0.01, 100.0)
    flame_config = FLAMEConfig(
        flame_model_path=args.flame_model,
        flame_assets_path=args.flame_assets,
        num_expressions=int(args.num_expressions),
    )
    print("Loading model (first run with torch.compile can take a few minutes)...")
    inferencer = RealDenseFaceInferencer(
        RealDenseFaceInferenceConfig(
            model_config_path=args.model_config,
            model_weights_path=args.model_weights,
            flame=flame_config,
            device=args.device,
            compile_model=not args.no_compile,
            output_type="torch",
        ),
        FaceBoxConfig(model_weights_path=args.facebox_weights, device=args.device),
    )
    optimizer = OnlineGNOptimizer(
        load_fitting_config(args.fitting_config, OnlineGNOptimizerConfig, flame=flame_config, device=args.device)
    )
    optimizer.refresh_camera_matrices(camera)
    visualizer = Visualizer(flame_assets_path=args.flame_assets, device=args.device)

    window = "RealDenseFace webcam  [R]ec  [D]etect  [T]reset  [V]iew  [Q]uit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, img_w, img_h)

    face_bbox = None
    take: Take | None = None
    view_mode = 0
    fps_smooth = 0.0
    t_prev = time.perf_counter()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            t_now = time.perf_counter()

            inference = inferencer.infer_image(frame_rgb, face_bbox=face_bbox)
            x = optimizer.track(inference)
            face_bbox = inference.face_bbox
            bw, bh = face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1]
            if bw < 20 or bh < 20 or bw > img_w * 1.5:
                face_bbox = None  # lost track, re-detect next frame

            vertices, _ = optimizer.decode(x)
            if take is not None:
                take.add(vertices, x, frame_rgb, t_now)

            mode = VIEW_MODES[view_mode]
            if mode == "camera":
                display = frame_rgb
            else:
                overlay, shading = visualizer.vis_flame_shading_overlay(
                    camera, frame_rgb, vertices, alpha=0.5, return_shading=True
                )
                display = overlay if mode == "overlay" else shading
            display = cv2.cvtColor(np.ascontiguousarray(display), cv2.COLOR_RGB2BGR)

            dt = t_now - t_prev
            t_prev = t_now
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(dt, 1e-6))
            cv2.putText(display, f"{fps_smooth:5.1f} fps  view:{mode}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if take is not None:
                cv2.circle(display, (img_w - 30, 30), 12, (0, 0, 255), -1)
                cv2.putText(display, f"REC {len(take.vertices)}", (img_w - 170, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                if take is None:
                    take = Take(args.output_dir, (img_w, img_h), args.save_video)
                    print("Recording...")
                else:
                    take.save(optimizer.identity, camera, args.fov_y_deg, args.camera_distance, args.num_expressions)
                    take = None
            elif key == ord("d"):
                face_bbox = None
            elif key == ord("t"):
                optimizer.reset()
                face_bbox = None
            elif key == ord("v"):
                view_mode = (view_mode + 1) % len(VIEW_MODES)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if take is not None:
            take.save(optimizer.identity, camera, args.fov_y_deg, args.camera_distance, args.num_expressions)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
