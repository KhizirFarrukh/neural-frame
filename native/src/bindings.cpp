// Python bindings for neuralframe._native. All heavy work runs with the GIL released.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "neuralframe/color.hpp"
#include "neuralframe/resample.hpp"
#include "neuralframe/ringbuffer.hpp"
#include "neuralframe/threadpool.hpp"
#include "neuralframe/version.hpp"

namespace py = pybind11;

#ifndef NEURALFRAME_VERSION_INFO
#define NEURALFRAME_VERSION_INFO "dev"
#endif

namespace {

template <typename T>
using carray = py::array_t<T, py::array::c_style>;
using f32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

nf::ColorParams params(double kr, double kb, double yo, double ys, double co, double cs, int bit_depth, bool gbr) {
  nf::ColorParams p;
  p.kr_d = kr;
  p.kb_d = kb;
  p.y_offset = yo;
  p.y_scale = ys;
  p.c_offset = co;
  p.c_scale = cs;
  p.bit_depth = bit_depth;
  p.gbr = gbr;
  return p;
}

void require(bool cond, const char* what) {
  if (!cond) throw std::invalid_argument(what);
}

template <typename T>
f32 yuv_to_rgb(carray<T> y, carray<T> u, carray<T> v, double kr, double kb, double yo, double ys, double co,
               double cs, int bit_depth, bool gbr, bool h_cosited, bool v_cosited) {
  require(y.ndim() == 2 && u.ndim() == 2 && v.ndim() == 2, "planes must be 2-D");
  require(u.shape(0) == v.shape(0) && u.shape(1) == v.shape(1), "chroma planes differ in shape");
  const int h = static_cast<int>(y.shape(0)), w = static_cast<int>(y.shape(1));
  const int ch = static_cast<int>(u.shape(0)), cw = static_cast<int>(u.shape(1));
  const int log2_cw = cw < w ? 1 : 0, log2_ch = ch < h ? 1 : 0;
  require(!gbr || (cw == w && ch == h), "GBR planes must be full resolution");
  f32 out({3, h, w});
  const T *py_ = y.data(), *pu = u.data(), *pv = v.data();
  float* po = out.mutable_data();
  const auto p = params(kr, kb, yo, ys, co, cs, bit_depth, gbr);
  const nf::Siting s{h_cosited, v_cosited};
  {
    py::gil_scoped_release release;
    nf::yuv_to_rgb<T>(py_, pu, pv, w, h, cw, ch, log2_cw, log2_ch, p, s, po);
  }
  return out;
}

template <typename T>
std::tuple<carray<T>, carray<T>, carray<T>> rgb_to_yuv_t(const f32& rgb, double kr, double kb, double yo, double ys,
                                                         double co, double cs, int bit_depth, bool gbr, int log2_cw,
                                                         int log2_ch, bool h_cosited, bool v_cosited) {
  require(rgb.ndim() == 3 && rgb.shape(0) == 3, "rgb must be (3, H, W)");
  const int h = static_cast<int>(rgb.shape(1)), w = static_cast<int>(rgb.shape(2));
  const int cw = gbr ? w : (w + (1 << log2_cw) - 1) >> log2_cw;
  const int ch = gbr ? h : (h + (1 << log2_ch) - 1) >> log2_ch;
  carray<T> y({h, w}), u({ch, cw}), v({ch, cw});
  const float* src = rgb.data();
  T *py_ = y.mutable_data(), *pu = u.mutable_data(), *pv = v.mutable_data();
  const auto p = params(kr, kb, yo, ys, co, cs, bit_depth, gbr);
  const nf::Siting s{h_cosited, v_cosited};
  {
    py::gil_scoped_release release;
    nf::rgb_to_yuv<T>(src, w, h, gbr ? 0 : log2_cw, gbr ? 0 : log2_ch, p, s, py_, pu, pv);
  }
  return {y, u, v};
}

py::tuple rgb_to_yuv(const f32& rgb, double kr, double kb, double yo, double ys, double co, double cs, int bit_depth,
                     bool gbr, int log2_cw, int log2_ch, bool h_cosited, bool v_cosited) {
  if (bit_depth <= 8) {
    auto [y, u, v] = rgb_to_yuv_t<std::uint8_t>(rgb, kr, kb, yo, ys, co, cs, bit_depth, gbr, log2_cw, log2_ch,
                                               h_cosited, v_cosited);
    return py::make_tuple(y, u, v);
  }
  auto [y, u, v] = rgb_to_yuv_t<std::uint16_t>(rgb, kr, kb, yo, ys, co, cs, bit_depth, gbr, log2_cw, log2_ch,
                                              h_cosited, v_cosited);
  return py::make_tuple(y, u, v);
}

f32 resize(const f32& img, int out_h, int out_w, const carray<std::int32_t>& idx_x, const f32& w_x,
           const carray<std::int32_t>& idx_y, const f32& w_y) {
  require(img.ndim() == 3, "image must be (C, H, W)");
  const int c = static_cast<int>(img.shape(0)), h = static_cast<int>(img.shape(1)), w = static_cast<int>(img.shape(2));
  require(idx_x.ndim() == 2 && idx_x.shape(0) == out_w && w_x.shape(0) == out_w, "x table shape mismatch");
  require(idx_y.ndim() == 2 && idx_y.shape(0) == out_h && w_y.shape(0) == out_h, "y table shape mismatch");
  f32 out({c, out_h, out_w});
  const float* src = img.data();
  float* dst = out.mutable_data();
  const std::int32_t *ix = idx_x.data(), *iy = idx_y.data();
  const float *wx = w_x.data(), *wy = w_y.data();
  const int tx = static_cast<int>(idx_x.shape(1)), ty = static_cast<int>(idx_y.shape(1));
  {
    py::gil_scoped_release release;
    nf::resize(src, c, h, w, dst, out_h, out_w, ix, wx, tx, iy, wy, ty);
  }
  return out;
}

f32 apply_gamut(const f32& rgb, const f32& m, const f32& to_linear, const f32& to_code) {
  require(rgb.ndim() == 3 && rgb.shape(0) == 3, "rgb must be (3, H, W)");
  require(m.size() == 9, "matrix must be 3x3");
  require(to_linear.size() == to_code.size() && to_linear.size() >= 2, "bad transfer tables");
  f32 out({static_cast<py::ssize_t>(3), rgb.shape(1), rgb.shape(2)});
  const std::size_t pixels = static_cast<std::size_t>(rgb.shape(1) * rgb.shape(2));
  const float *src = rgb.data(), *pm = m.data(), *lin = to_linear.data(), *enc = to_code.data();
  float* dst = out.mutable_data();
  const std::size_t n = static_cast<std::size_t>(to_linear.size());
  {
    py::gil_scoped_release release;
    nf::apply_gamut(src, dst, pixels, pm, lin, enc, n);
  }
  return out;
}

void merge_tile(py::array_t<float, py::array::c_style> canvas, const f32& tile, int y0, int x0, const f32& wy,
                const f32& wx) {
  require(canvas.ndim() == 3 && tile.ndim() == 3 && canvas.shape(0) == tile.shape(0), "shape mismatch");
  const int c = static_cast<int>(canvas.shape(0)), h = static_cast<int>(canvas.shape(1)),
            w = static_cast<int>(canvas.shape(2));
  const int th = static_cast<int>(tile.shape(1)), tw = static_cast<int>(tile.shape(2));
  require(y0 >= 0 && x0 >= 0 && y0 + th <= h && x0 + tw <= w, "tile outside canvas");
  require(wy.size() == th && wx.size() == tw, "weight length mismatch");
  float* dst = canvas.mutable_data();
  const float *src = tile.data(), *pwy = wy.data(), *pwx = wx.data();
  py::gil_scoped_release release;
  nf::merge_tile(dst, c, h, w, src, th, tw, y0, x0, pwy, pwx);
}

void normalize(py::array_t<float, py::array::c_style> canvas, const f32& total) {
  require(canvas.ndim() == 3 && total.ndim() == 2 && total.shape(0) == canvas.shape(1) &&
              total.shape(1) == canvas.shape(2),
          "shape mismatch");
  float* dst = canvas.mutable_data();
  const float* t = total.data();
  const int c = static_cast<int>(canvas.shape(0)), h = static_cast<int>(canvas.shape(1)),
            w = static_cast<int>(canvas.shape(2));
  py::gil_scoped_release release;
  nf::normalize(dst, c, h, w, t);
}

f32 blend(const f32& a, const f32& b, double weight_a) {
  require(a.size() == b.size(), "blend operands differ in size");
  std::vector<py::ssize_t> shape(a.shape(), a.shape() + a.ndim());
  f32 out(shape);
  const float wa = static_cast<float>(weight_a);
  const float wb = static_cast<float>(1.0 - weight_a);
  const float *pa = a.data(), *pb = b.data();
  float* po = out.mutable_data();
  const std::size_t n = static_cast<std::size_t>(a.size());
  {
    py::gil_scoped_release release;
    nf::blend(pa, pb, po, n, wa, wb);
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "NeuralFrame native hot-path operations (C++17, multi-threaded, GIL released)";
  m.attr("__version__") = NEURALFRAME_VERSION_INFO;
  m.def("build_info", &nf::build_info, "Compiler and build flavour of the native module");
  m.def("set_threads", [](unsigned n) { nf::ThreadPool::instance().set_threads(n); }, py::arg("n"));
  m.def("get_threads", [] { return nf::ThreadPool::instance().threads(); });

  m.def("yuv_to_rgb", &yuv_to_rgb<std::uint8_t>, py::arg("y"), py::arg("u"), py::arg("v"), py::kw_only(),
        py::arg("kr"), py::arg("kb"), py::arg("y_offset"), py::arg("y_scale"), py::arg("c_offset"),
        py::arg("c_scale"), py::arg("bit_depth"), py::arg("gbr"), py::arg("h_cosited"), py::arg("v_cosited"));
  m.def("yuv_to_rgb", &yuv_to_rgb<std::uint16_t>, py::arg("y"), py::arg("u"), py::arg("v"), py::kw_only(),
        py::arg("kr"), py::arg("kb"), py::arg("y_offset"), py::arg("y_scale"), py::arg("c_offset"),
        py::arg("c_scale"), py::arg("bit_depth"), py::arg("gbr"), py::arg("h_cosited"), py::arg("v_cosited"));
  m.def("rgb_to_yuv", &rgb_to_yuv, py::arg("rgb"), py::kw_only(), py::arg("kr"), py::arg("kb"),
        py::arg("y_offset"), py::arg("y_scale"), py::arg("c_offset"), py::arg("c_scale"), py::arg("bit_depth"),
        py::arg("gbr"), py::arg("log2_cw"), py::arg("log2_ch"), py::arg("h_cosited"), py::arg("v_cosited"));
  m.def("resize", &resize, py::arg("img"), py::arg("out_h"), py::arg("out_w"), py::arg("idx_x"), py::arg("w_x"),
        py::arg("idx_y"), py::arg("w_y"));
  m.def("apply_gamut", &apply_gamut, py::arg("rgb"), py::arg("matrix"), py::arg("to_linear"), py::arg("to_code"));
  m.def("merge_tile", &merge_tile, py::arg("canvas"), py::arg("tile"), py::arg("y0"), py::arg("x0"),
        py::arg("wy"), py::arg("wx"));
  m.def("normalize", &normalize, py::arg("canvas"), py::arg("total"));
  m.def("blend", &blend, py::arg("a"), py::arg("b"), py::arg("weight_a"));

  py::class_<nf::RingBuffer>(m, "RingBuffer")
      .def(py::init<std::size_t, std::size_t>(), py::arg("slots"), py::arg("slot_bytes"))
      .def("acquire_write",
           [](nf::RingBuffer& rb, int timeout_ms) {
             py::gil_scoped_release release;
             return rb.acquire_write(timeout_ms);
           },
           py::arg("timeout_ms") = -1)
      .def("commit_write", &nf::RingBuffer::commit_write, py::arg("slot"), py::arg("tag") = 0)
      .def("acquire_read",
           [](nf::RingBuffer& rb, int timeout_ms) {
             std::int64_t tag = 0;
             int slot;
             {
               py::gil_scoped_release release;
               slot = rb.acquire_read(timeout_ms, &tag);
             }
             return py::make_tuple(slot, tag);
           },
           py::arg("timeout_ms") = -1)
      .def("release_read", &nf::RingBuffer::release_read, py::arg("slot"))
      .def("close", &nf::RingBuffer::close)
      .def("view",
           [](py::object self, int slot) {
             auto& rb = self.cast<nf::RingBuffer&>();
             std::uint8_t* ptr = rb.slot_ptr(slot);
             // Zero-copy view; ``self`` is the base object, so the buffer outlives every view.
             return py::array_t<std::uint8_t>({static_cast<py::ssize_t>(rb.slot_bytes())}, {1}, ptr, self);
           },
           py::arg("slot"))
      .def_property_readonly("slots", &nf::RingBuffer::slots)
      .def_property_readonly("slot_bytes", &nf::RingBuffer::slot_bytes)
      .def_property_readonly("nbytes", &nf::RingBuffer::bytes)
      .def_property_readonly("data_ptr", [](const nf::RingBuffer& rb) { return reinterpret_cast<std::uintptr_t>(rb.data()); })
      .def_property_readonly("ready", &nf::RingBuffer::ready)
      .def_property_readonly("closed", &nf::RingBuffer::closed);
}
