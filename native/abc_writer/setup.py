"""Build the Alembic writer extension.

Needs static Imath + Alembic libraries; point ALEMBIC_PREFIX at their install prefix:
    set ALEMBIC_PREFIX=C:/path/to/prefix
    pip wheel . --no-build-isolation
"""
import os

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

prefix = os.environ.get("ALEMBIC_PREFIX")
if not prefix:
    raise RuntimeError("Set ALEMBIC_PREFIX to the Alembic/Imath install prefix.")

ext = Pybind11Extension(
    "abc_writer",
    ["src/abc_writer.cpp"],
    include_dirs=[os.path.join(prefix, "include"), os.path.join(prefix, "include", "Imath")],
    library_dirs=[os.path.join(prefix, "lib")],
    libraries=["Alembic", "Imath-3_1"],
    cxx_std=17,
)

setup(name="abc_writer", version="0.1.0", ext_modules=[ext])
