#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "neuralframe/color.hpp"

namespace {

nf::ColorParams bt709_limited(int bit_depth = 8) {
  nf::ColorParams p;
  const double s = static_cast<double>(1 << (bit_depth - 8));
  p.kr_d = 0.2126;
  p.kb_d = 0.0722;
  p.y_offset = 16.0 * s;
  p.y_scale = 219.0 * s;
  p.c_offset = 128.0 * s;
  p.c_scale = 224.0 * s;
  p.bit_depth = bit_depth;
  return p;
}

std::vector<float> gradient_rgb(int w, int h) {
  std::vector<float> rgb(3 * static_cast<std::size_t>(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * w + x;
      rgb[i] = 0.5f + 0.4f * std::sin(x / 7.0f);
      rgb[w * h + i] = 0.5f + 0.4f * std::cos(y / 5.0f);
      rgb[2 * w * h + i] = 0.5f + 0.3f * std::sin((x + y) / 9.0f);
    }
  return rgb;
}

}  // namespace

TEST(Color, ReferenceLevels) {
  const int w = 4, h = 2;
  std::vector<std::uint8_t> y(w * h, 235), u(2 * 1, 128), v(2 * 1, 128);
  std::vector<float> rgb(3 * w * h);
  nf::yuv_to_rgb<std::uint8_t>(y.data(), u.data(), v.data(), w, h, 2, 1, 1, 1, bt709_limited(), {}, rgb.data());
  for (float c : rgb) EXPECT_FLOAT_EQ(c, 1.0f);
  std::fill(y.begin(), y.end(), 16);
  nf::yuv_to_rgb<std::uint8_t>(y.data(), u.data(), v.data(), w, h, 2, 1, 1, 1, bt709_limited(), {}, rgb.data());
  for (float c : rgb) EXPECT_FLOAT_EQ(c, 0.0f);
}

TEST(Color, RoundTrip444IsWithinOneCode) {
  const int w = 37, h = 23;
  const auto rgb = gradient_rgb(w, h);
  std::vector<std::uint16_t> y(w * h), u(w * h), v(w * h), y2(w * h), u2(w * h), v2(w * h);
  const auto p = bt709_limited(10);
  nf::rgb_to_yuv<std::uint16_t>(rgb.data(), w, h, 0, 0, p, {}, y.data(), u.data(), v.data());
  std::vector<float> back(rgb.size());
  nf::yuv_to_rgb<std::uint16_t>(y.data(), u.data(), v.data(), w, h, w, h, 0, 0, p, {}, back.data());
  nf::rgb_to_yuv<std::uint16_t>(back.data(), w, h, 0, 0, p, {}, y2.data(), u2.data(), v2.data());
  for (int i = 0; i < w * h; ++i) {
    EXPECT_LE(std::abs(int(y[i]) - int(y2[i])), 1);
    EXPECT_LE(std::abs(int(u[i]) - int(u2[i])), 1);
    EXPECT_LE(std::abs(int(v[i]) - int(v2[i])), 1);
  }
}

TEST(Color, Subsampled420OddSizes) {
  const int w = 33, h = 19, cw = 17, ch = 10;
  const auto rgb = gradient_rgb(w, h);
  std::vector<std::uint8_t> y(w * h), u(cw * ch), v(cw * ch);
  const auto p = bt709_limited();
  for (bool hc : {true, false})
    for (bool vc : {true, false}) {
      const nf::Siting s{hc, vc};
      nf::rgb_to_yuv<std::uint8_t>(rgb.data(), w, h, 1, 1, p, s, y.data(), u.data(), v.data());
      std::vector<float> back(rgb.size());
      nf::yuv_to_rgb<std::uint8_t>(y.data(), u.data(), v.data(), w, h, cw, ch, 1, 1, p, s, back.data());
      double err = 0;
      for (std::size_t i = 0; i < rgb.size(); ++i) err += std::abs(back[i] - rgb[i]);
      EXPECT_LT(err / rgb.size(), 2.0 / 255.0);
    }
}

TEST(Color, GamutIdentityKeepsValues) {
  const std::size_t n = 1024;
  std::vector<float> lut(n);
  for (std::size_t i = 0; i < n; ++i) lut[i] = static_cast<float>(i) / (n - 1);
  const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<float> in(3 * 50), out(3 * 50);
  for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i % 50) / 49.0f;
  nf::apply_gamut(in.data(), out.data(), 50, identity, lut.data(), lut.data(), n);
  for (std::size_t i = 0; i < in.size(); ++i) EXPECT_NEAR(out[i], in[i], 1e-6);
}
