// pose2d.hpp — B-Human Pose2f 대응 (double, m/rad).
#pragma once
#include <Eigen/Dense>
#include <cmath>

namespace hsl
{

inline double normalizeAngle(double a)
{
  a = std::fmod(a + M_PI, 2.0 * M_PI);
  if(a < 0) a += 2.0 * M_PI;
  return a - M_PI;
}

struct Pose2D
{
  double rotation = 0.0;
  Eigen::Vector2d translation = Eigen::Vector2d::Zero();

  Pose2D() = default;
  Pose2D(double rot, const Eigen::Vector2d & t) : rotation(rot), translation(t) {}
  Pose2D(double rot, double x, double y) : rotation(rot), translation(x, y) {}

  Eigen::Matrix2d R() const
  {
    const double c = std::cos(rotation), s = std::sin(rotation);
    return (Eigen::Matrix2d() << c, -s, s, c).finished();
  }
  // 이 pose 기준 점 → 바깥 좌표
  Eigen::Vector2d operator*(const Eigen::Vector2d & p) const { return R() * p + translation; }
  // 합성: this ∘ other
  Pose2D operator+(const Pose2D & o) const
  {
    return Pose2D(normalizeAngle(rotation + o.rotation), translation + R() * o.translation);
  }
  Pose2D inverse() const
  {
    const Eigen::Vector2d t = -(R().transpose() * translation);
    return Pose2D(-rotation, t);
  }
};

}  // namespace hsl
