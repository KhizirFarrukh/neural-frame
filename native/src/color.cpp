// YCbCr <-> RGB conversion with siting-aware chroma resampling.
//
// Every expression mirrors neuralframe/ops/reference.py operation by operation (same float32 constants,
// same evaluation order, horizontal chroma pass before vertical), and FP contraction is disabled so the
// compiler cannot fuse multiply-adds. That makes the native path bit-identical to the NumPy reference.

#include "neuralframe/color.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "neuralframe/threadpool.hpp"

#if defined(_MSC_VER)
#pragma fp_contract(off)
#endif

namespace nf {

namespace {

struct Coefficients {
  float y_off, y_scale, c_off, c_scale;
  float cr_to_r, cb_to_b, cr_to_g, cb_to_g;  // yuv -> rgb
  float kr, kg, kb, inv_b, inv_r;            // rgb -> yuv (inv_* are the divisors 2(1-kb), 2(1-kr))
  float peak;
};

Coefficients make_coefficients(const ColorParams& p) {
  const double kr = p.kr_d;
  const double kb = p.kb_d;
  const double kg = 1.0 - kr - kb;
  Coefficients c{};
  c.y_off = static_cast<float>(p.y_offset);
  c.y_scale = static_cast<float>(p.y_scale);
  c.c_off = static_cast<float>(p.c_offset);
  c.c_scale = static_cast<float>(p.c_scale);
  c.cr_to_r = static_cast<float>(2.0 * (1.0 - kr));
  c.cb_to_b = static_cast<float>(2.0 * (1.0 - kb));
  c.cr_to_g = static_cast<float>(2.0 * kr * (1.0 - kr) / kg);
  c.cb_to_g = static_cast<float>(2.0 * kb * (1.0 - kb) / kg);
  c.kr = static_cast<float>(kr);
  c.kg = static_cast<float>(kg);
  c.kb = static_cast<float>(kb);
  c.inv_b = static_cast<float>(2.0 * (1.0 - kb));
  c.inv_r = static_cast<float>(2.0 * (1.0 - kr));
  c.peak = static_cast<float>((1 << p.bit_depth) - 1);
  return c;
}

inline float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

// Horizontal 2x chroma upsampling of one row (normalised floats), like reference._up2(axis=1).
void up2_row(const float* c, int m, float* out, int n_out, bool cosited) {
  for (int i = 0; i < m; ++i) {
    const float cur = c[i];
    const float nxt = c[std::min(i + 1, m - 1)];
    float even, odd;
    if (cosited) {
      even = cur;
      odd = 0.5f * (cur + nxt);
    } else {
      const float prev = c[std::max(i - 1, 0)];
      even = 0.75f * cur + 0.25f * prev;
      odd = 0.75f * cur + 0.25f * nxt;
    }
    if (2 * i < n_out) out[2 * i] = even;
    if (2 * i + 1 < n_out) out[2 * i + 1] = odd;
  }
}

// Horizontal 2x downsampling of one row, like reference._down2(axis=1).
void down2_row(const float* x, int n, float* out, bool cosited) {
  const int m = (n + 1) / 2;
  for (int i = 0; i < m; ++i) {
    const float centre = x[std::min(2 * i, n - 1)];
    const float right = x[std::min(2 * i + 1, n - 1)];
    if (cosited) {
      const float left = x[std::clamp(2 * i - 1, 0, n - 1)];
      out[i] = 0.25f * left + 0.5f * centre + 0.25f * right;
    } else {
      out[i] = 0.5f * (centre + right);
    }
  }
}

template <typename T>
inline T quantise(float x, float offset, float scale, float peak) {
  float q = std::floor(x * scale + offset + 0.5f);
  q = q < 0.0f ? 0.0f : (q > peak ? peak : q);
  return static_cast<T>(q);
}

}  // namespace

template <typename T>
void yuv_to_rgb(const T* y, const T* u, const T* v, int width, int height, int chroma_w, int chroma_h,
                int log2_cw, int log2_ch, const ColorParams& p, const Siting& s, float* rgb) {
  const Coefficients k = make_coefficients(p);
  const std::size_t plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  float* r_out = rgb;
  float* g_out = rgb + plane;
  float* b_out = rgb + 2 * plane;

  if (p.gbr) {
    parallel_rows(static_cast<std::size_t>(height), [&](std::size_t r0, std::size_t r1) {
      for (std::size_t row = r0; row < r1; ++row) {
        for (int x = 0; x < width; ++x) {
          const std::size_t i = row * width + x;
          r_out[i] = clamp01(static_cast<float>(v[i]) / k.peak);
          g_out[i] = clamp01(static_cast<float>(y[i]) / k.peak);
          b_out[i] = clamp01(static_cast<float>(u[i]) / k.peak);
        }
      }
    });
    return;
  }

  parallel_rows(static_cast<std::size_t>(height), [&](std::size_t r0, std::size_t r1) {
    std::vector<float> norm(static_cast<std::size_t>(chroma_w));
    // horizontally upsampled chroma rows: [cb_a, cb_b, cr_a, cr_b] (a = main row, b = vertical neighbour)
    std::vector<float> rows(4 * static_cast<std::size_t>(width));
    float* cb_a = rows.data();
    float* cb_b = cb_a + width;
    float* cr_a = cb_b + width;
    float* cr_b = cr_a + width;

    auto load_row = [&](const T* plane_ptr, int j, float* dst) {
      const T* src = plane_ptr + static_cast<std::size_t>(j) * chroma_w;
      for (int i = 0; i < chroma_w; ++i) norm[i] = (static_cast<float>(src[i]) - k.c_off) / k.c_scale;
      if (log2_cw == 1) {
        up2_row(norm.data(), chroma_w, dst, width, s.h_cosited);
      } else {
        std::copy(norm.begin(), norm.begin() + width, dst);
      }
    };

    for (std::size_t row = r0; row < r1; ++row) {
      const int yrow = static_cast<int>(row);
      // Vertical position: which chroma rows contribute and with what weights (reference._up2(axis=0)).
      int ja = yrow, jb = yrow;
      float wa = 1.0f, wb = 0.0f;
      bool half = false;  // cosited odd rows: 0.5 * (a + b)
      if (log2_ch == 1) {
        const int j = yrow >> 1;
        const bool odd = (yrow & 1) != 0;
        ja = j;
        if (s.v_cosited) {
          if (odd) {
            jb = std::min(j + 1, chroma_h - 1);
            half = true;
          } else {
            jb = j;
          }
        } else {
          jb = odd ? std::min(j + 1, chroma_h - 1) : std::max(j - 1, 0);
          wa = 0.75f;
          wb = 0.25f;
        }
      }
      load_row(u, ja, cb_a);
      load_row(v, ja, cr_a);
      const bool need_b = log2_ch == 1 && (half || wb != 0.0f);
      if (need_b) {
        load_row(u, jb, cb_b);
        load_row(v, jb, cr_b);
      }
      const T* yrow_ptr = y + row * width;
      const std::size_t base = row * width;
      for (int x = 0; x < width; ++x) {
        float cb, cr;
        if (!need_b) {
          cb = cb_a[x];
          cr = cr_a[x];
        } else if (half) {
          cb = 0.5f * (cb_a[x] + cb_b[x]);
          cr = 0.5f * (cr_a[x] + cr_b[x]);
        } else {
          cb = wa * cb_a[x] + wb * cb_b[x];
          cr = wa * cr_a[x] + wb * cr_b[x];
        }
        const float yf = (static_cast<float>(yrow_ptr[x]) - k.y_off) / k.y_scale;
        const float r = yf + k.cr_to_r * cr;
        const float b = yf + k.cb_to_b * cb;
        const float g = yf - k.cr_to_g * cr - k.cb_to_g * cb;
        r_out[base + x] = clamp01(r);
        g_out[base + x] = clamp01(g);
        b_out[base + x] = clamp01(b);
      }
    }
  });
}

template <typename T>
void rgb_to_yuv(const float* rgb, int width, int height, int log2_cw, int log2_ch, const ColorParams& p,
                const Siting& s, T* y, T* u, T* v) {
  const Coefficients k = make_coefficients(p);
  const std::size_t plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const float* r_in = rgb;
  const float* g_in = rgb + plane;
  const float* b_in = rgb + 2 * plane;

  if (p.gbr) {
    parallel_rows(static_cast<std::size_t>(height), [&](std::size_t r0, std::size_t r1) {
      for (std::size_t i = r0 * width; i < r1 * width; ++i) {
        y[i] = quantise<T>(clamp01(g_in[i]), 0.0f, k.peak, k.peak);
        u[i] = quantise<T>(clamp01(b_in[i]), 0.0f, k.peak, k.peak);
        v[i] = quantise<T>(clamp01(r_in[i]), 0.0f, k.peak, k.peak);
      }
    });
    return;
  }

  // Luma.
  parallel_rows(static_cast<std::size_t>(height), [&](std::size_t r0, std::size_t r1) {
    for (std::size_t i = r0 * width; i < r1 * width; ++i) {
      const float r = clamp01(r_in[i]), g = clamp01(g_in[i]), b = clamp01(b_in[i]);
      const float yf = k.kr * r + k.kg * g + k.kb * b;
      y[i] = quantise<T>(yf, k.y_off, k.y_scale, k.peak);
    }
  });

  // Chroma: per output chroma row, horizontally downsample the contributing luma rows, then combine.
  const int cw = (width + (1 << log2_cw) - 1) >> log2_cw;
  const int ch = (height + (1 << log2_ch) - 1) >> log2_ch;
  parallel_rows(static_cast<std::size_t>(ch), [&](std::size_t c0, std::size_t c1) {
    std::vector<float> cb_full(static_cast<std::size_t>(width)), cr_full(static_cast<std::size_t>(width));
    std::vector<float> rows(6 * static_cast<std::size_t>(cw));  // cb/cr for up to three luma rows
    auto chroma_row = [&](int yy, float* cb_dst, float* cr_dst) {
      const std::size_t base = static_cast<std::size_t>(yy) * width;
      for (int x = 0; x < width; ++x) {
        const float r = clamp01(r_in[base + x]), g = clamp01(g_in[base + x]), b = clamp01(b_in[base + x]);
        const float yf = k.kr * r + k.kg * g + k.kb * b;
        cb_full[x] = (b - yf) / k.inv_b;
        cr_full[x] = (r - yf) / k.inv_r;
      }
      if (log2_cw == 1) {
        down2_row(cb_full.data(), width, cb_dst, s.h_cosited);
        down2_row(cr_full.data(), width, cr_dst, s.h_cosited);
      } else {
        std::copy(cb_full.begin(), cb_full.end(), cb_dst);
        std::copy(cr_full.begin(), cr_full.end(), cr_dst);
      }
    };
    float* cb0 = rows.data();
    float* cr0 = cb0 + cw;
    float* cb1 = cr0 + cw;
    float* cr1 = cb1 + cw;
    float* cb2 = cr1 + cw;
    float* cr2 = cb2 + cw;
    for (std::size_t j = c0; j < c1; ++j) {
      const int jj = static_cast<int>(j);
      T* u_row = u + j * cw;
      T* v_row = v + j * cw;
      if (log2_ch == 0) {
        chroma_row(jj, cb1, cr1);
        for (int i = 0; i < cw; ++i) {
          u_row[i] = quantise<T>(cb1[i], k.c_off, k.c_scale, k.peak);
          v_row[i] = quantise<T>(cr1[i], k.c_off, k.c_scale, k.peak);
        }
        continue;
      }
      const int centre = std::min(2 * jj, height - 1);
      const int right = std::min(2 * jj + 1, height - 1);
      chroma_row(centre, cb1, cr1);
      chroma_row(right, cb2, cr2);
      if (s.v_cosited) {
        chroma_row(std::clamp(2 * jj - 1, 0, height - 1), cb0, cr0);
        for (int i = 0; i < cw; ++i) {
          const float cbv = 0.25f * cb0[i] + 0.5f * cb1[i] + 0.25f * cb2[i];
          const float crv = 0.25f * cr0[i] + 0.5f * cr1[i] + 0.25f * cr2[i];
          u_row[i] = quantise<T>(cbv, k.c_off, k.c_scale, k.peak);
          v_row[i] = quantise<T>(crv, k.c_off, k.c_scale, k.peak);
        }
      } else {
        for (int i = 0; i < cw; ++i) {
          u_row[i] = quantise<T>(0.5f * (cb1[i] + cb2[i]), k.c_off, k.c_scale, k.peak);
          v_row[i] = quantise<T>(0.5f * (cr1[i] + cr2[i]), k.c_off, k.c_scale, k.peak);
        }
      }
    }
  });
}

void apply_gamut(const float* rgb_in, float* rgb_out, std::size_t pixels, const float* m,
                 const float* to_linear, const float* to_code, std::size_t lut_size) {
  const float scale = static_cast<float>(lut_size - 1);
  const int last = static_cast<int>(lut_size) - 2;
  auto lookup = [&](const float* lut, float x) {
    const float pos = (x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x)) * scale;
    const int i0 = std::min(static_cast<int>(pos), last);
    const float frac = pos - static_cast<float>(i0);
    return lut[i0] + (lut[i0 + 1] - lut[i0]) * frac;
  };
  const float* r_in = rgb_in;
  const float* g_in = rgb_in + pixels;
  const float* b_in = rgb_in + 2 * pixels;
  constexpr std::size_t kBlock = 1 << 15;
  const std::size_t blocks = (pixels + kBlock - 1) / kBlock;
  parallel_rows(blocks, [&](std::size_t b0, std::size_t b1) {
    for (std::size_t i = b0 * kBlock; i < std::min(b1 * kBlock, pixels); ++i) {
      const float lr = lookup(to_linear, r_in[i]);
      const float lg = lookup(to_linear, g_in[i]);
      const float lb = lookup(to_linear, b_in[i]);
      for (int c = 0; c < 3; ++c) {
        const float val = m[3 * c] * lr + m[3 * c + 1] * lg + m[3 * c + 2] * lb;
        rgb_out[c * pixels + i] = lookup(to_code, val);
      }
    }
  });
}

template void yuv_to_rgb<std::uint8_t>(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*, int, int,
                                       int, int, int, int, const ColorParams&, const Siting&, float*);
template void yuv_to_rgb<std::uint16_t>(const std::uint16_t*, const std::uint16_t*, const std::uint16_t*, int,
                                        int, int, int, int, int, const ColorParams&, const Siting&, float*);
template void rgb_to_yuv<std::uint8_t>(const float*, int, int, int, int, const ColorParams&, const Siting&,
                                       std::uint8_t*, std::uint8_t*, std::uint8_t*);
template void rgb_to_yuv<std::uint16_t>(const float*, int, int, int, int, const ColorParams&, const Siting&,
                                        std::uint16_t*, std::uint16_t*, std::uint16_t*);

}  // namespace nf
