// Minimal Alembic writer for RealDenseFace Studio: one animated triangle mesh (+ UVs) and an
// optional static camera, written with the official Alembic library (Ogawa backend).
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace Alembic::AbcGeom;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

void write_mesh_sequence_with_camera(
    const std::string &path,
    FloatArray frames, IntArray faces, FloatArray uvs, IntArray uv_faces, double fps,
    double fov_y_deg, int width, int height, py::sequence position, double scale)
{
    if (frames.ndim() != 3 || frames.shape(2) != 3) throw std::invalid_argument("frames must be [T, V, 3]");
    if (faces.ndim() != 2 || faces.shape(1) != 3) throw std::invalid_argument("faces must be [F, 3]");
    if (uv_faces.ndim() != 2 || uv_faces.shape(0) != faces.shape(0) || uv_faces.shape(1) != 3)
        throw std::invalid_argument("uv_faces must be [F, 3]");
    if (uvs.ndim() != 2 || uvs.shape(1) != 2) throw std::invalid_argument("uvs must be [U, 2]");
    const size_t num_frames = static_cast<size_t>(frames.shape(0));
    const size_t num_verts = static_cast<size_t>(frames.shape(1));
    const size_t num_faces = static_cast<size_t>(faces.shape(0));
    if (fps <= 0.0) throw std::invalid_argument("fps must be positive");

    std::vector<int32_t> indices(num_faces * 3);
    std::vector<uint32_t> uv_idx(num_faces * 3);
    const int32_t *f = faces.data();
    const int32_t *uf = uv_faces.data();
    for (size_t i = 0; i < num_faces; ++i) {
        indices[i * 3 + 0] = f[i * 3 + 0];
        indices[i * 3 + 1] = f[i * 3 + 2];
        indices[i * 3 + 2] = f[i * 3 + 1];
        uv_idx[i * 3 + 0] = static_cast<uint32_t>(uf[i * 3 + 0]);
        uv_idx[i * 3 + 1] = static_cast<uint32_t>(uf[i * 3 + 2]);
        uv_idx[i * 3 + 2] = static_cast<uint32_t>(uf[i * 3 + 1]);
    }
    std::vector<int32_t> counts(num_faces, 3);
    const double px = position[0].cast<double>(), py_ = position[1].cast<double>(), pz = position[2].cast<double>();

    py::gil_scoped_release release;
    OArchive archive = CreateArchiveWithInfo(Alembic::AbcCoreOgawa::WriteArchive(), path, "RealDenseFace Studio", "FLAME face tracking");
    TimeSamplingPtr ts(new TimeSampling(1.0 / fps, 1.0 / fps));  // first sample at frame 1
    uint32_t ts_index = archive.addTimeSampling(*ts);

    OXform face_xform(archive.getTop(), "face");
    face_xform.getSchema().set(XformSample());
    OPolyMesh mesh(face_xform, "flame", ts_index);
    OPolyMeshSchema &schema = mesh.getSchema();
    const V2f *uv_data = reinterpret_cast<const V2f *>(uvs.data());
    const V3f *pos = reinterpret_cast<const V3f *>(frames.data());
    for (size_t t = 0; t < num_frames; ++t) {
        P3fArraySample positions(pos + t * num_verts, num_verts);
        if (t == 0) {
            OV2fGeomParam::Sample uv_sample(
                V2fArraySample(uv_data, static_cast<size_t>(uvs.shape(0))),
                UInt32ArraySample(uv_idx.data(), uv_idx.size()), kFacevaryingScope);
            schema.set(OPolyMeshSchema::Sample(
                positions, Int32ArraySample(indices.data(), indices.size()),
                Int32ArraySample(counts.data(), counts.size()), uv_sample));
        } else {
            schema.set(OPolyMeshSchema::Sample(positions));
        }
    }

    if (width > 0 && height > 0) {
        OXform cam_xform(archive.getTop(), "tracking_camera");
        XformSample xs;
        xs.setTranslation(V3d(px * scale, py_ * scale, pz * scale));
        cam_xform.getSchema().set(xs);
        OCamera cam(cam_xform, "tracking_cameraShape");
        CameraSample cs;
        const double vertical_aperture_mm = 24.0;
        const double aspect = static_cast<double>(width) / static_cast<double>(height);
        cs.setFocalLength(0.5 * vertical_aperture_mm / std::tan(fov_y_deg * 3.14159265358979323846 / 360.0));
        cs.setVerticalAperture(vertical_aperture_mm / 10.0);  // Alembic apertures are in cm
        cs.setHorizontalAperture(vertical_aperture_mm * aspect / 10.0);
        cs.setNearClippingPlane(0.01 * scale);
        cs.setFarClippingPlane(100.0 * scale);
        cam.getSchema().set(cs);
    }
}

PYBIND11_MODULE(abc_writer, m) {
    m.doc() = "Minimal Alembic writer (animated mesh + camera) for RealDenseFace Studio";
    m.def("write", &write_mesh_sequence_with_camera,
          py::arg("path"), py::arg("frames"), py::arg("faces"), py::arg("uvs"), py::arg("uv_faces"),
          py::arg("fps"), py::arg("fov_y_deg") = 30.0, py::arg("width") = 0, py::arg("height") = 0,
          py::arg("position") = py::make_tuple(0.0, 0.0, 1.0), py::arg("scale") = 1.0,
          "Write an animated mesh (frames [T,V,3]) with UVs and an optional camera (width/height > 0).");
}
