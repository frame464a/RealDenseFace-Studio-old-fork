# Third-party notices

RealDenseFace Studio includes or downloads the following components. Each is used under its own license.

## Included in this repository

| Component | License | Notes |
|---|---|---|
| [RealDenseFace](https://github.com/LinzhouLi/RealDenseFace) | MIT | © 2026 Li Linzhou. The tracking code this app is built on. |
| [GLM](https://github.com/g-truc/glm) (`cuda_extensions/third_party/glm`) | MIT / Happy Bunny | See `cuda_extensions/third_party/glm/copying.txt`. |

## Downloaded by the installer or the app

| Component | License | Notes |
|---|---|---|
| RealDenseFace pretrained weights | No license stated by the authors | Downloaded from the authors' Google Drive link. Trained on NeRSemble, Ava-256 and FaceScape, which are licensed for non-commercial use. |
| [FLAME 2023](https://flame.is.tue.mpg.de/) | [FLAME license](https://flame.is.tue.mpg.de/modellicense.html) | Not redistributed; each user downloads it after registering. Non-commercial scientific research use. |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | NVIDIA Source Code License | Shipped as a prebuilt wheel built from the unmodified source. |
| [PyTorch](https://pytorch.org/), torchvision | BSD-3-Clause | |
| [PySide6 / Qt](https://www.qt.io/qt-for-python) | LGPL-3.0 | Used as an unmodified, dynamically linked library. |
| [OpenUSD (usd-core)](https://openusd.org/) | Tomorrow Open Source Technology License 1.0 | |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | BSD-2-Clause; bundled FFmpeg binary under GPL | Used as a separate executable to encode mp4 files. |
| [timm](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 | |
| [triton-windows](https://github.com/woct0rdho/triton-windows) | MIT | |
| [Alembic](https://github.com/alembic/alembic) | BSD-3-Clause | Statically linked into the prebuilt `abc_writer` wheel. © Lucasfilm Ltd. and Sony Pictures Imageworks. |
| [Imath](https://github.com/AcademySoftwareFoundation/Imath) | BSD-3-Clause | Statically linked into `abc_writer`. © Contributors to the OpenEXR Project. |
| [pybind11](https://github.com/pybind/pybind11) | BSD-3-Clause | Used to build `abc_writer`. |
| [chumpy](https://github.com/mattloper/chumpy) | MIT | Needed to read the FLAME model file. |
| NumPy, SciPy, OpenCV, tqdm, roma, PyYAML, trimesh, pygrabber, gdown | BSD / MIT / Apache-2.0 | |
| [uv](https://github.com/astral-sh/uv) | MIT or Apache-2.0 | Used by the installer to set up Python. |

Full license texts for the components linked into the prebuilt wheels are in [third_party_licenses/](third_party_licenses/).
