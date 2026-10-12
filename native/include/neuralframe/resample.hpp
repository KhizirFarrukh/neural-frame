#pragma once

#include <cstddef>
#include <cstdint>

namespace nf {

// Separable resampling of planar float32 images (C, H, W) -> (C, OH, OW) with precomputed tables:
// for every output pixel ``taps`` source indices and weights (zero-padded rows). The tables come from
// neuralframe.ops.reference.resample_weights, so native and NumPy results are bit-identical.
// Either table may be null to skip that axis (sizes must then match).
void resize(const float* in, int channels, int height, int width, float* out, int out_h, int out_w,
            const std::int32_t* idx_x, const float* w_x, int taps_x, const std::int32_t* idx_y,
            const float* w_y, int taps_y);

// canvas[c, y0+i, x0+j] += tile[c, i, j] * (wy[i] * wx[j])   (feathered tile accumulation)
void merge_tile(float* canvas, int channels, int height, int width, const float* tile, int tile_h, int tile_w,
                int y0, int x0, const float* wy, const float* wx);

// canvas[c, y, x] /= total[y, x]
void normalize(float* canvas, int channels, int height, int width, const float* total);

// out = a * wa + b * wb  (temporal cross-fade of two frames; wa/wb computed by the caller)
void blend(const float* a, const float* b, float* out, std::size_t n, float wa, float wb);

}  // namespace nf
