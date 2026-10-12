#pragma once

#include <cstddef>
#include <cstdint>

namespace nf {

// Matrix / range description in double precision, exactly as Python holds it (KR_KB and
// reference.range_params); kernels derive their float32 constants the same way NumPy does, so the
// native path matches the reference bit for bit.
struct ColorParams {
  double kr_d = 0.2126;
  double kb_d = 0.0722;
  double y_offset = 16.0;
  double y_scale = 219.0;
  double c_offset = 128.0;
  double c_scale = 224.0;
  int bit_depth = 8;
  bool gbr = false;  // planar GBR (no matrix, full range)
};

struct Siting {
  bool h_cosited = true;   // chroma sample at the left luma sample (MPEG-2/H.264 "left")
  bool v_cosited = false;  // chroma sample between two luma rows
};

// YCbCr planes (uint8 or uint16, LSB-aligned) -> planar RGB float32 (3, h, w) in [0, 1].
template <typename T>
void yuv_to_rgb(const T* y, const T* u, const T* v, int width, int height, int chroma_w, int chroma_h,
                int log2_cw, int log2_ch, const ColorParams& p, const Siting& s, float* rgb);

// Planar RGB float32 (3, h, w) -> quantised YCbCr planes (uint8 or uint16) with chroma downsampling.
template <typename T>
void rgb_to_yuv(const float* rgb, int width, int height, int log2_cw, int log2_ch, const ColorParams& p,
                const Siting& s, T* y, T* u, T* v);

// Linear-light gamut conversion through the same uniform transfer tables as the reference.
void apply_gamut(const float* rgb_in, float* rgb_out, std::size_t pixels, const float* m3x3,
                 const float* to_linear, const float* to_code, std::size_t lut_size);

}  // namespace nf
