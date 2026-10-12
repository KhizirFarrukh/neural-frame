// Separable resampling, feathered tile merging and temporal blending.
// Accumulation order matches the NumPy reference (tap 0 first, starting from 0.0f), FP contraction off.

#include "neuralframe/resample.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "neuralframe/threadpool.hpp"

#if defined(_MSC_VER)
#pragma fp_contract(off)
#endif

namespace nf {

void resize(const float* in, int channels, int height, int width, float* out, int out_h, int out_w,
            const std::int32_t* idx_x, const float* w_x, int taps_x, const std::int32_t* idx_y,
            const float* w_y, int taps_y) {
  const bool do_x = idx_x != nullptr && out_w != width;
  const bool do_y = idx_y != nullptr && out_h != height;
  const std::size_t in_plane = static_cast<std::size_t>(height) * width;

  // Horizontal pass into a temporary (C, H, OW) image (skipped when width is unchanged).
  std::vector<float> tmp;
  const float* stage = in;
  int stage_w = width;
  if (do_x) {
    tmp.assign(static_cast<std::size_t>(channels) * height * out_w, 0.0f);
    parallel_rows(static_cast<std::size_t>(channels) * height, [&](std::size_t r0, std::size_t r1) {
      for (std::size_t r = r0; r < r1; ++r) {
        const float* src = in + r * width;
        float* dst = tmp.data() + r * out_w;
        for (int ox = 0; ox < out_w; ++ox) {
          const std::int32_t* ix = idx_x + static_cast<std::size_t>(ox) * taps_x;
          const float* wx = w_x + static_cast<std::size_t>(ox) * taps_x;
          float acc = 0.0f;
          for (int k = 0; k < taps_x; ++k) acc += src[ix[k]] * wx[k];
          dst[ox] = acc;
        }
      }
    });
    stage = tmp.data();
    stage_w = out_w;
  }

  // Vertical pass.
  if (do_y) {
    const std::size_t stage_plane = static_cast<std::size_t>(height) * stage_w;
    parallel_rows(static_cast<std::size_t>(channels) * out_h, [&](std::size_t r0, std::size_t r1) {
      for (std::size_t r = r0; r < r1; ++r) {
        const std::size_t c = r / out_h;
        const int oy = static_cast<int>(r % out_h);
        const std::int32_t* iy = idx_y + static_cast<std::size_t>(oy) * taps_y;
        const float* wy = w_y + static_cast<std::size_t>(oy) * taps_y;
        const float* plane = stage + c * stage_plane;
        float* dst = out + (c * out_h + oy) * static_cast<std::size_t>(out_w);
        for (int ox = 0; ox < out_w; ++ox) dst[ox] = 0.0f;
        for (int k = 0; k < taps_y; ++k) {
          const float* src = plane + static_cast<std::size_t>(iy[k]) * stage_w;
          const float wk = wy[k];
          for (int ox = 0; ox < out_w; ++ox) dst[ox] += src[ox] * wk;
        }
      }
    });
  } else if (do_x) {
    std::memcpy(out, tmp.data(), tmp.size() * sizeof(float));
  } else {
    std::memcpy(out, in, static_cast<std::size_t>(channels) * in_plane * sizeof(float));
  }
}

void merge_tile(float* canvas, int channels, int height, int width, const float* tile, int tile_h, int tile_w,
                int y0, int x0, const float* wy, const float* wx) {
  (void)height;
  parallel_rows(static_cast<std::size_t>(channels) * tile_h, [&](std::size_t r0, std::size_t r1) {
    for (std::size_t r = r0; r < r1; ++r) {
      const std::size_t c = r / tile_h;
      const int i = static_cast<int>(r % tile_h);
      const float* src = tile + (c * tile_h + i) * static_cast<std::size_t>(tile_w);
      float* dst = canvas + (c * height + y0 + i) * static_cast<std::size_t>(width) + x0;
      for (int j = 0; j < tile_w; ++j) {
        const float weight = wy[i] * wx[j];
        dst[j] += src[j] * weight;
      }
    }
  });
}

void normalize(float* canvas, int channels, int height, int width, const float* total) {
  const std::size_t plane = static_cast<std::size_t>(height) * width;
  parallel_rows(static_cast<std::size_t>(channels) * height, [&](std::size_t r0, std::size_t r1) {
    for (std::size_t r = r0; r < r1; ++r) {
      float* row = canvas + r * width;
      const float* t = total + (r % height) * static_cast<std::size_t>(width);
      for (int x = 0; x < width; ++x) row[x] = row[x] / t[x];
    }
  });
  (void)plane;
}

void blend(const float* a, const float* b, float* out, std::size_t n, float wa, float wb) {
  constexpr std::size_t kBlock = 1 << 16;
  parallel_rows((n + kBlock - 1) / kBlock, [&](std::size_t b0, std::size_t b1) {
    const std::size_t end = std::min(b1 * kBlock, n);
    for (std::size_t i = b0 * kBlock; i < end; ++i) out[i] = a[i] * wa + b[i] * wb;
  });
}

}  // namespace nf
