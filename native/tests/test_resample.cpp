#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "neuralframe/resample.hpp"

TEST(Resample, ConstantImageStaysConstant) {
  // 2-tap linear tables for 4 -> 7 upsampling along x; identity along y.
  const int c = 2, h = 3, w = 4, ow = 7;
  std::vector<float> img(c * h * w, 0.25f), out(c * h * ow);
  std::vector<std::int32_t> ix(ow * 2);
  std::vector<float> wx(ow * 2);
  for (int o = 0; o < ow; ++o) {
    const float pos = (o + 0.5f) * w / ow - 0.5f;
    const int i0 = pos < 0 ? 0 : static_cast<int>(pos);
    const int i1 = i0 + 1 < w ? i0 + 1 : w - 1;
    const float f = pos < 0 ? 0.0f : pos - i0;
    ix[2 * o] = i0;
    ix[2 * o + 1] = i1;
    wx[2 * o] = 1.0f - f;
    wx[2 * o + 1] = f;
  }
  nf::resize(img.data(), c, h, w, out.data(), h, ow, ix.data(), wx.data(), 2, nullptr, nullptr, 0);
  for (float v : out) EXPECT_NEAR(v, 0.25f, 1e-7);
}

TEST(Resample, MergeTilesIsPartitionOfUnity) {
  const int c = 1, h = 4, w = 6;
  std::vector<float> canvas(c * h * w, 0.0f), total(h * w, 0.0f);
  std::vector<float> tile(c * 4 * 4, 2.0f), wy(4, 1.0f), wa = {1, 1, 0.75f, 0.25f}, wb = {0.25f, 0.75f, 1, 1};
  nf::merge_tile(canvas.data(), c, h, w, tile.data(), 4, 4, 0, 0, wy.data(), wa.data());
  nf::merge_tile(canvas.data(), c, h, w, tile.data(), 4, 4, 0, 2, wy.data(), wb.data());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float t = 0;
      if (x < 4) t += wa[x];
      if (x >= 2) t += wb[x - 2];
      total[y * w + x] = t;
    }
  nf::normalize(canvas.data(), c, h, w, total.data());
  for (float v : canvas) EXPECT_FLOAT_EQ(v, 2.0f);
}

TEST(Resample, BlendCrossFade) {
  std::vector<float> a(100000, 1.0f), b(100000, 3.0f), out(100000);
  nf::blend(a.data(), b.data(), out.data(), a.size(), 0.25f, 0.75f);
  for (float v : out) EXPECT_FLOAT_EQ(v, 2.5f);
}
