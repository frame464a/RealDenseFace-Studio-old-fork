<h1 align="center">RealDenseFace Studio</h1>

<p align="center">
  Real-time 3D face tracking for artists: track your face live from a webcam, or any face in a video,<br>
  and export the animated head mesh to Blender, Houdini, Maya, Unreal or Cinema 4D.
</p>

<p align="center">
  <img src="assets/teaser.jpg" width="100%" alt="RealDenseFace teaser">
</p>

> **Unofficial community project.** Not affiliated with or endorsed by the RealDenseFace authors.

RealDenseFace Studio is a free desktop app built on
[**RealDenseFace**](https://gapszju.github.io/RealDenseFace/) by Linzhou Li, Tianjia Shao and Kun Zhou
(Zhejiang University): a state-of-the-art method that fits the [FLAME](https://flame.is.tue.mpg.de/) head
model to video in real time. All the research credit goes to them. This project adds an app, an installer
and exporters around their code.

## Features

- **Live webcam tracking** with the 3D mesh drawn over your face (~30–60 fps, depending on your camera).
- **Record takes** with one key (Space) and keep the camera footage with them.
- **Process video files** in best quality (whole-clip solve with the larger model, runs on a second GPU if you have one).
- **Export** each take as:
  - **USD (.usdc)**: animated mesh with UVs plus the tracking camera; imports into Blender, Houdini, Maya, Unreal, C4D
  - **OBJ + PC2 point cache**: Blender *Mesh Cache* modifier, 3ds Max, C4D, Houdini
  - **OBJ sequence**
  - **Video (.mp4)**: mesh over the footage, mesh only, or side by side
- Options: units (m / cm / mm), **head-only** export (removes head rotation and position, keeps expressions).

## Requirements

- Windows 10 or 11, 64-bit
- An **NVIDIA RTX GPU** (20-series or newer) with driver **570 or newer**. AMD, Intel and Mac aren't supported (the solver is CUDA code).
- About 10 GB of free disk space
- A free account on the [FLAME website](https://flame.is.tue.mpg.de/) (the app walks you through it)

## Install

1. Download this repository (**Code → Download ZIP**) and unpack it to a short folder path without spaces, e.g. `C:\RealDenseFaceStudio`.
2. Double-click **`install.bat`**. It downloads a private copy of Python and PyTorch into that folder
   (about 4 GB, takes 5–15 minutes). Nothing is installed system-wide; delete the folder to uninstall.
3. Start **RealDenseFace Studio** from the desktop shortcut (or `RealDenseFace Studio.bat`).

### First start

A setup window asks for two things, once:

1. **Tracking models**: click **Download** (1.1 GB).
2. **FLAME 2023 head model**: its license doesn't allow redistribution, so you get it yourself:
   register at [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/), go to **Downloads**, download
   **FLAME 2023 (revised eye region, improved expressions, versions w/ and w/o jaw rotation)**
   (*not* "FLAME 2023 Open"), then select the zip in the setup window.

The very first launch afterwards takes a few minutes while the models are optimized for your GPU.

## Using it

| Step | What to do |
|---|---|
| **1 Source** | Pick your webcam and press **Start webcam**, or **Open video file…** to play a clip with live tracking. |
| **2 View** | Overlay / Mesh / Camera / Side by side (keys **1–4**). **Reset tracking** (key **R**) if the fit drifts. **Camera FOV**: most webcams are 35–50°. |
| **3 Record** | **Space** starts/stops a take. **Process a video file (best quality)…** tracks a whole clip offline. |
| **4 Takes & export** | Select a take, choose a format, **Export…**. |

### Importing into your 3D app

- **Blender**: *File → Import → Universal Scene Description* (the `.usdc`), or import the `.obj` and add a
  *Mesh Cache* modifier pointing at the `.pc2`.
- **Houdini**: *File → Import → USD* or a USD Import / LOP node; or File SOP with the OBJ + a point-cache workflow.
- **Maya**: enable the *mayaUsdPlugin* and import the `.usdc`.
- **Unreal**: import the `.usdc` via the USD Stage editor.

Scenes are Y-up, meters by default, with the camera at the position of the real camera, so the mesh
lines up with your footage through the exported camera.

## License and credits

- **App, installer and exporters**: MIT, see [LICENSE](LICENSE).
- **RealDenseFace code**: MIT, © 2026 Li Linzhou. Original README: [README_RealDenseFace.md](README_RealDenseFace.md).
  Paper: [arXiv:2608.09238](https://arxiv.org/abs/2608.09238).
- **Pretrained weights**: downloaded from the authors' release; they were trained on research datasets
  (NeRSemble, Ava-256, FaceScape) that are licensed for non-commercial use.
- **FLAME 2023**: © Max Planck Institute for Intelligent Systems, under the
  [FLAME license](https://flame.is.tue.mpg.de/modellicense.html) (non-commercial scientific research use).
- Third-party components: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Because of the FLAME license and the training data, use this tool and its output for non-commercial
purposes only.** For commercial use, check the licenses above and contact the rights holders.

If you use this in research, please cite the RealDenseFace paper:

```bibtex
@article{li2026realdenseface,
  title   = {RealDenseFace: Real-time Monocular 3D Face Reconstruction from Dense UV-space Priors},
  author  = {Li, Linzhou and Shao, Tianjia and Zhou, Kun},
  journal = {arXiv preprint arXiv:2608.09238},
  year    = {2026}
}
```

## Changes compared to the original repository

- New: desktop app (`app.py`), setup wizard, installer, exporter (`export_animation.py`), webcam CLI (`track_webcam.py`).
- Fix: the face detector now letterboxes frames instead of stretching them to a square, which made it
  miss faces in wide (16:9) video ([LinzhouLi/RealDenseFace#4](https://github.com/LinzhouLi/RealDenseFace/issues/4)).
- `track_video_offline.py` also stores the video frame rate in its `.npz` and falls back to uncompiled mode
  if `torch.compile` fails.
