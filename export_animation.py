"""Export RealDenseFace tracking results as animated geometry.

Input: an npz from track_video_offline.py (--output_npz) or track_webcam.py (a take).

Formats (pick with --format, default usd):
    usd      one .usdc/.usda file: FLAME mesh with time-sampled points + UVs, and the
             tracking camera. Imports into Blender, Houdini, Maya, Unreal, C4D.
    pc2      base mesh .obj (with UVs) + .pc2 point cache (Blender "Mesh Cache"
             modifier, 3ds Max "Point Cache", Houdini, C4D).
    objseq   one .obj per frame.
    mp4      rendered video: mesh over the camera footage, mesh only, or side by side.

Units are meters, Y-up, camera at +Z looking down -Z (matches the tracker's world space).
"""
from __future__ import annotations

import common.fix_chumpy  # noqa: F401  (must import before FLAME model load)

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from common.types import FittingOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RealDenseFace tracking npz to animated geometry.")
    parser.add_argument("--input", type=Path, required=True, help="Tracking npz.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path: .usd/.usdc/.usda for usd, .pc2 for pc2, a directory for objseq.")
    parser.add_argument("--format", choices=("usd", "pc2", "objseq", "mp4"), default="usd")
    parser.add_argument("--video_style", choices=VIDEO_STYLES, default="overlay", help="mp4 only.")
    parser.add_argument("--footage", type=str, default=None,
                        help="mp4 only: camera footage to draw on (default: the take's camera.mp4 / source video).")
    parser.add_argument("--fps", type=float, default=None, help="Override frame rate (default: from npz, else 30).")
    parser.add_argument("--head_space", action="store_true",
                        help="Remove global head rotation/translation (expression + jaw/neck/eyes only).")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale applied to geometry and camera (e.g. 100 for cm).")
    parser.add_argument("--no_camera", action="store_true", help="USD only: do not write the tracking camera.")
    parser.add_argument("--flame_model", type=str, default="weights/flame/flame2023.pkl")
    parser.add_argument("--flame_assets", type=str, default="weights/flame/flame_assets.npz")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def decode_vertices(
    data,
    head_space: bool = False,
    flame_model: str = "weights/flame/flame2023.pkl",
    flame_assets: str = "weights/flame/flame_assets.npz",
    device: str = "cuda",
) -> np.ndarray:
    """Return [T, 5023, 3] vertices, decoding FLAME parameters when needed."""
    if "vertices" in data and not head_space:
        return np.asarray(data["vertices"], dtype=np.float32)

    from tracker import FLAMEConfig, GNFlameOptimizer, GNFlameOptimizerConfig

    num_expressions = int(data["num_expressions"])
    flame_config = FLAMEConfig(
        flame_model_path=flame_model,
        flame_assets_path=flame_assets,
        num_expressions=num_expressions,
    )
    optimizer = GNFlameOptimizer(GNFlameOptimizerConfig(flame=flame_config, device=device))

    x = torch.from_numpy(np.asarray(data["x"], dtype=np.float32)).reshape(-1, num_expressions + 18)
    if head_space:
        pose_start = num_expressions
        x[:, pose_start:pose_start + 3] = 0.0  # global (root) rotation
        x[:, num_expressions + 15:] = 0.0  # global translation
    identity = torch.from_numpy(np.asarray(data["identity"], dtype=np.float32))

    chunks = []
    for start in range(0, x.shape[0], 512):
        vertices, _ = optimizer.decode(FittingOutput(identity=identity, x=x[start:start + 512]))
        chunks.append(vertices.reshape(-1, 5023, 3).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray | None, uv_faces: np.ndarray | None) -> None:
    lines = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in vertices]
    if uvs is not None:
        lines += [f"vt {t[0]:.6f} {t[1]:.6f}" for t in uvs]
        lines += [
            f"f {a + 1}/{ta + 1} {b + 1}/{tb + 1} {c + 1}/{tc + 1}"
            for (a, b, c), (ta, tb, tc) in zip(faces, uv_faces)
        ]
    else:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pc2(path: Path, frames: np.ndarray, fps: float) -> None:
    num_frames, num_points, _ = frames.shape
    with path.open("wb") as f:
        f.write(struct.pack("<12siiffi", b"POINTCACHE2\0", 1, num_points, 0.0, 1.0, num_frames))
        f.write(np.ascontiguousarray(frames, dtype="<f4").tobytes())


def write_usd(path: Path, frames: np.ndarray, faces: np.ndarray, uvs: np.ndarray, uv_faces: np.ndarray,
              fps: float, camera: dict | None) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0 / camera["scale"] if camera else 1.0)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(frames) - 1)

    root = UsdGeom.Xform.Define(stage, "/face")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/face/flame")
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).tolist()))
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray.FromNumpy(uvs.astype(np.float32)))
    st.SetIndices(Vt.IntArray(uv_faces.reshape(-1).tolist()))

    points = mesh.CreatePointsAttr()
    extent = mesh.CreateExtentAttr()
    for i, frame in enumerate(frames):
        points.Set(Vt.Vec3fArray.FromNumpy(frame), i)
        extent.Set(Vt.Vec3fArray([Gf.Vec3f(*frame.min(0).tolist()), Gf.Vec3f(*frame.max(0).tolist())]), i)

    if camera is not None:
        cam = UsdGeom.Camera.Define(stage, "/face/tracking_camera")
        vertical_aperture = 24.0  # mm
        aspect = camera["width"] / camera["height"]
        cam.CreateVerticalApertureAttr(vertical_aperture)
        cam.CreateHorizontalApertureAttr(vertical_aperture * aspect)
        cam.CreateFocalLengthAttr(0.5 * vertical_aperture / np.tan(np.radians(camera["fov_y_deg"]) * 0.5))
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01 * camera["scale"], 100.0 * camera["scale"]))
        # Tracker camera is OpenCV-style (x right, y down, z forward) = rot diag(1,-1,-1);
        # USD cameras look down -Z with +Y up, so flip y/z to get the USD camera frame.
        rot = camera["rot"] @ np.diag([1.0, -1.0, -1.0])
        xform = np.eye(4)
        xform[:3, :3] = rot.T  # USD uses row vectors
        xform[3, :3] = camera["pos"] * camera["scale"]
        cam.AddTransformOp().Set(Gf.Matrix4d(xform.tolist()))

    stage.GetRootLayer().Save()


VIDEO_STYLES = ("overlay", "mesh", "side_by_side")


def find_footage(input_npz: Path) -> Path | None:
    """Camera footage that belongs to a take: camera.mp4 next to it, or the source video of an offline take."""
    take_dir = Path(input_npz).parent
    if (take_dir / "camera.mp4").exists():
        return take_dir / "camera.mp4"
    info_path = take_dir / "info.json"
    if info_path.exists():
        import json

        source = json.loads(info_path.read_text(encoding="utf-8")).get("source_path")
        if source and Path(source).exists():
            return Path(source)
    return None


def export_video(
    input_npz: str | Path,
    output: str | Path,
    style: str = "overlay",
    footage: str | Path | None = None,
    fps: float | None = None,
    flame_model: str = "weights/flame/flame2023.pkl",
    flame_assets: str = "weights/flame/flame_assets.npz",
    device: str = "cuda",
    progress=None,
) -> Path:
    """Render a take to an H.264 mp4: mesh overlaid on the footage, mesh only, or footage | mesh side by side."""
    import cv2
    import imageio_ffmpeg

    from camera import PerspectiveCamera
    from visualization import Visualizer

    if style not in VIDEO_STYLES:
        raise ValueError(f"Unknown video style: {style}")
    output = Path(output).with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(input_npz)
    frames = decode_vertices(data, False, flame_model, flame_assets, device)
    fps = fps or (float(data["fps"]) if "fps" in data else 30.0)
    width, height = int(data["image_width"]), int(data["image_height"])
    fov = float(data["camera_fov_y"]) if "camera_fov_y" in data else float("nan")
    camera = PerspectiveCamera(
        fov_y=np.radians(30.0 if np.isnan(fov) else fov),
        rot=np.asarray(data["camera_rot"], dtype=np.float32).reshape(3, 3),
        pos=np.asarray(data["camera_pos"], dtype=np.float32).reshape(3),
        width=width,
        height=height,
    )
    visualizer = Visualizer(flame_assets_path=flame_assets, device=device)

    footage = Path(footage) if footage else find_footage(Path(input_npz))
    cap = cv2.VideoCapture(str(footage)) if footage else None
    if cap is None and style != "mesh":
        raise FileNotFoundError("This take has no camera footage; only the 'mesh' video style is available.")

    out_w = width * 2 if style == "side_by_side" else width
    out_h = height
    out_w, out_h = out_w + out_w % 2, out_h + out_h % 2  # H.264 needs even sizes
    writer = imageio_ffmpeg.write_frames(
        str(output), (out_w, out_h), fps=fps, codec="libx264", quality=8, macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p"],
    )
    writer.send(None)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    try:
        for i, vertices in enumerate(frames):
            image = blank
            if cap is not None:
                ok, bgr = cap.read()
                if ok:
                    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if image.shape[:2] != (height, width):
                        image = cv2.resize(image, (width, height))
            image = np.ascontiguousarray(image)
            overlay, mesh = visualizer.vis_flame_shading_overlay(camera, image, vertices, alpha=0.55, return_shading=True)
            if style == "overlay":
                out = overlay
            elif style == "mesh":
                out = mesh
            else:
                out = np.concatenate([image, mesh], axis=1)
            if out.shape[:2] != (out_h, out_w):
                out = cv2.copyMakeBorder(out, 0, out_h - out.shape[0], 0, out_w - out.shape[1], cv2.BORDER_CONSTANT)
            writer.send(np.ascontiguousarray(out))
            if progress is not None:
                progress(i + 1, len(frames))
    finally:
        writer.close()
        if cap is not None:
            cap.release()
    print(f"Saved video: {output}")
    return output


def export(
    input_npz: str | Path,
    output: str | Path,
    fmt: str = "usd",
    fps: float | None = None,
    head_space: bool = False,
    scale: float = 1.0,
    with_camera: bool = True,
    flame_model: str = "weights/flame/flame2023.pkl",
    flame_assets: str = "weights/flame/flame_assets.npz",
    device: str = "cuda",
) -> Path:
    """Export a tracking npz. Returns the main output path."""
    output = Path(output)
    data = np.load(input_npz)
    frames = decode_vertices(data, head_space, flame_model, flame_assets, device) * scale
    fps = fps or (float(data["fps"]) if "fps" in data else 30.0)

    assets = np.load(flame_assets)
    faces = np.asarray(assets["faces"], dtype=np.int64)
    uvs = np.asarray(assets["uvs"], dtype=np.float32)
    uv_faces = np.asarray(assets["uv_faces"], dtype=np.int64)
    print(f"{frames.shape[0]} frames @ {fps:.2f} fps, {frames.shape[1]} vertices")

    if fmt == "usd":
        if output.suffix.lower() not in (".usd", ".usdc", ".usda"):
            output = output.with_suffix(".usdc")
        output.parent.mkdir(parents=True, exist_ok=True)
        camera = None
        fov = float(data["camera_fov_y"]) if "camera_fov_y" in data else float("nan")
        if with_camera and not head_space:
            camera = dict(
                fov_y_deg=30.0 if np.isnan(fov) else fov,
                rot=np.asarray(data["camera_rot"], dtype=np.float64).reshape(3, 3),
                pos=np.asarray(data["camera_pos"], dtype=np.float64).reshape(3),
                width=int(data["image_width"]),
                height=int(data["image_height"]),
                scale=scale,
            )
        write_usd(output, frames, faces, uvs, uv_faces, fps, camera)
        print(f"Saved USD: {output}")
    elif fmt == "pc2":
        output = output.with_suffix(".pc2")
        output.parent.mkdir(parents=True, exist_ok=True)
        base = output.with_suffix(".obj")
        write_obj(base, frames[0], faces, uvs, uv_faces)
        write_pc2(output, frames, fps)
        print(f"Saved base mesh: {base}")
        print(f"Saved point cache: {output}  (frame rate {fps:.2f})")
    elif fmt == "objseq":
        output.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            write_obj(output / f"face_{i:05d}.obj", frame, faces, uvs, uv_faces)
        print(f"Saved {len(frames)} OBJs to {output}")
    else:
        raise ValueError(f"Unknown export format: {fmt}")
    return output


def main() -> None:
    args = parse_args()
    if args.format == "mp4":
        export_video(args.input, args.output, args.video_style, args.footage, args.fps,
                     args.flame_model, args.flame_assets, args.device)
        return
    export(
        args.input,
        args.output,
        fmt=args.format,
        fps=args.fps,
        head_space=args.head_space,
        scale=args.scale,
        with_camera=not args.no_camera,
        flame_model=args.flame_model,
        flame_assets=args.flame_assets,
        device=args.device,
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
