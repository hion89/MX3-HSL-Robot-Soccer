// range_noise.hpp — 깊이 카메라 검출의 거리별 오차 모델 (측정 공분산용).
// 매듭 (거리, σ) 사이는 선형, 마지막 매듭 뒤는 마지막 기울기로 외삽, scale 배.
// 기본값은 실기 D405+YOLO 경험치 (2026-10-31 사용자): 2.5 m 에서 10 cm 까지 선형, 4 m 에서 50 cm 로 급증.
#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace hsl
{

struct RangeNoise
{
  std::vector<double> d{0.0, 2.5, 4.0}, s{0.005, 0.10, 0.50};
  double scale = 1.3;     // 모델에 없는 오차(TF 시각, 목 흔들림)를 덮는 여유

  static RangeNoise fromKnots(const std::vector<double> & flat, double scale)
  {
    if(flat.size() < 4 || flat.size() % 2) throw std::runtime_error("range sigma 매듭은 [d0, s0, d1, s1, ...]");
    RangeNoise r;
    r.d.clear(); r.s.clear();
    for(size_t i = 0; i < flat.size(); i += 2) { r.d.push_back(flat[i]); r.s.push_back(flat[i + 1]); }
    r.scale = scale;
    return r;
  }

  double sigma(double dist) const
  {
    double v;
    if(dist <= d.front()) v = s.front();
    else {
      size_t i = 1;
      while(i < d.size() - 1 && dist > d[i]) ++i;
      v = s[i - 1] + (s[i] - s[i - 1]) * (dist - d[i - 1]) / (d[i] - d[i - 1]);   // 끝 구간은 외삽
    }
    return scale * std::max(v, 0.005);
  }
};

}  // namespace hsl
