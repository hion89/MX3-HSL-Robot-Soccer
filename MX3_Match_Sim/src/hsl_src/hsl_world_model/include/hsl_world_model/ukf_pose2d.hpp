// Copyright (c) 2026 B-Human. All rights reserved. — 이 파일은 B-Human code release 2026 의 일부를 이식한 것이다.
// "This product includes software developed by B-Human (http://www.b-human.de)." (B-Human License.md 조건 1·3)
// ukf_pose2d.hpp — B-Human Tools/Modeling/UKFPose2D 의 이식 (B-Human 2026 code release, BSD 계열 라이선스).
// 원본: Src/Tools/Modeling/UKFPose2D.cpp (Tim Laue, Colin Graf). 수식은 그대로, 단위만 mm → m, float → double.
//   상태 (x, y, θ), 시그마점 7 개 (평균 ± 숄레스키 열), 측정 갱신 3 종(랜드마크/선/절대 pose).
#pragma once
#include "hsl_world_model/pose2d.hpp"
#include <Eigen/Dense>

namespace hsl
{

inline void fixCovariance3(Eigen::Matrix3d & c)
{
  c(0, 1) = c(1, 0) = (c(0, 1) + c(1, 0)) * 0.5;
  c(0, 2) = c(2, 0) = (c(0, 2) + c(2, 0)) * 0.5;
  c(1, 2) = c(2, 1) = (c(1, 2) + c(2, 1)) * 0.5;
}

class UKFPose2D
{
public:
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();

  Pose2D getPose() const { return Pose2D(mean.z(), mean.head<2>()); }

  // B-Human UKFPose2D::motionUpdate 그대로
  void motionUpdate(const Pose2D & odo, const Pose2D & processDev, const Pose2D & odoDev,
                    const Eigen::Vector2d & odoRotDev)
  {
    generateSigmaPoints();
    for(int i = 0; i < 7; ++i) {
      const double c = std::cos(sp[i].z()), s = std::sin(sp[i].z());
      const Eigen::Vector2d o(c * odo.translation.x() - s * odo.translation.y(),
                              s * odo.translation.x() + c * odo.translation.y());
      sp[i] += Eigen::Vector3d(o.x(), o.y(), odo.rotation);
    }
    mean = sp[0];
    for(int i = 1; i < 7; ++i) mean += sp[i];
    mean /= 7.0;
    cov.setZero();
    for(int i = 0; i < 7; ++i) {
      const Eigen::Vector3d d = sp[i] - mean;
      cov += d * d.transpose();
    }
    cov *= 0.5;
    fixCovariance3(cov);
    cov(0, 0) += processDev.translation.x() * processDev.translation.x();
    cov(1, 1) += processDev.translation.y() * processDev.translation.y();
    cov(2, 2) += processDev.rotation * processDev.rotation;
    const double c = std::cos(mean.z()), s = std::sin(mean.z());
    const Eigen::Vector2d o(c * odo.translation.x() - s * odo.translation.y(),
                            s * odo.translation.x() + c * odo.translation.y());
    cov(0, 0) += std::pow(o.x() * odoDev.translation.x(), 2);
    cov(1, 1) += std::pow(o.y() * odoDev.translation.y(), 2);
    cov(2, 2) += std::pow(odo.rotation * odoDev.rotation, 2);
    cov(2, 2) += std::pow(o.x() * odoRotDev.x(), 2);
    cov(2, 2) += std::pow(o.y() * odoRotDev.y(), 2);
    mean.z() = normalizeAngle(mean.z());
  }

  // 알려진 랜드마크(필드 좌표) 를 로봇 기준 reading 으로 관측
  void landmarkSensorUpdate(const Eigen::Vector2d & landmark, const Eigen::Vector2d & reading,
                            const Eigen::Matrix2d & readingCov)
  {
    generateSigmaPoints();
    Eigen::Vector2d r[7];
    for(int i = 0; i < 7; ++i) {
      const Pose2D p(sp[i].z(), sp[i].head<2>());
      r[i] = p.inverse() * landmark;
    }
    Eigen::Vector2d rm = r[0];
    for(int i = 1; i < 7; ++i) rm += r[i];
    rm /= 7.0;
    Eigen::Matrix<double, 2, 3> crossCov = Eigen::Matrix<double, 2, 3>::Zero();
    for(int i = 0; i < 3; ++i) {
      const Eigen::Vector2d d1 = r[i * 2 + 1] - rm;
      crossCov += d1 * l.col(i).transpose();
      const Eigen::Vector2d d2 = r[i * 2 + 2] - rm;
      crossCov += d2 * (-l.col(i)).transpose();
    }
    crossCov *= 0.5;
    Eigen::Matrix2d readCov = Eigen::Matrix2d::Zero();
    for(int i = 0; i < 7; ++i) {
      const Eigen::Vector2d d = r[i] - rm;
      readCov += d * d.transpose();
    }
    readCov *= 0.5;
    const Eigen::Matrix<double, 3, 2> K = crossCov.transpose() * (readCov + readingCov).inverse();
    mean += K * (reading - rm);
    mean.z() = normalizeAngle(mean.z());
    cov -= K * crossCov;
    fixCovariance3(cov);
  }

  // 선 측정 (B-Human UKFPose2D::lineSensorUpdate 그대로).
  // reading = (선에 수직인 좌표, 방향): x 축에 평행한 선이면 (y, θ), y 축에 평행한 선이면 (x, θ).
  void lineSensorUpdate(bool parallelToX, const Eigen::Vector2d & reading, const Eigen::Matrix2d & readingCov)
  {
    generateSigmaPoints();
    Eigen::Vector2d r[7];
    for(int i = 0; i < 7; ++i)
      r[i] = parallelToX ? Eigen::Vector2d(sp[i].y(), sp[i].z()) : Eigen::Vector2d(sp[i].x(), sp[i].z());
    Eigen::Vector2d rm = r[0];
    for(int i = 1; i < 7; ++i) rm += r[i];
    rm /= 7.0;
    Eigen::Matrix<double, 2, 3> crossCov = Eigen::Matrix<double, 2, 3>::Zero();
    for(int i = 0; i < 3; ++i) {
      crossCov += (r[i * 2 + 1] - rm) * l.col(i).transpose();
      crossCov += (r[i * 2 + 2] - rm) * (-l.col(i)).transpose();
    }
    crossCov *= 0.5;
    Eigen::Matrix2d rc = Eigen::Matrix2d::Zero();
    for(int i = 0; i < 7; ++i) {
      const Eigen::Vector2d d = r[i] - rm;
      rc += d * d.transpose();
    }
    rc *= 0.5;
    rm.y() = normalizeAngle(rm.y());
    const Eigen::Matrix<double, 3, 2> K = crossCov.transpose() * (rc + readingCov).inverse();
    Eigen::Vector2d innov = reading - rm;
    innov.y() = normalizeAngle(innov.y());
    mean += K * innov;
    mean.z() = normalizeAngle(mean.z());
    cov -= K * crossCov;
    fixCovariance3(cov);
  }

  // 절대 pose 측정 (B-Human poseSensorUpdate)
  void poseSensorUpdate(const Eigen::Vector3d & reading, const Eigen::Matrix3d & readingCov)
  {
    generateSigmaPoints();
    Eigen::Vector3d pm = Eigen::Vector3d::Zero();
    for(int i = 0; i < 7; ++i) pm += sp[i];
    pm /= 7.0;
    Eigen::Matrix3d crossCov = Eigen::Matrix3d::Zero();
    for(int i = 0; i < 3; ++i) {
      crossCov += (sp[i * 2 + 1] - pm) * l.col(i).transpose();
      crossCov += (sp[i * 2 + 2] - pm) * (-l.col(i)).transpose();
    }
    crossCov *= 0.5;
    Eigen::Matrix3d pc = Eigen::Matrix3d::Zero();
    for(int i = 0; i < 7; ++i) {
      const Eigen::Vector3d d = sp[i] - pm;
      pc += d * d.transpose();
    }
    pc *= 0.5;
    pm.z() = normalizeAngle(pm.z());
    const Eigen::Matrix3d K = crossCov.transpose() * (pc + readingCov).inverse();
    Eigen::Vector3d innov = reading - pm;
    innov.z() = normalizeAngle(innov.z());
    mean += K * innov;
    mean.z() = normalizeAngle(mean.z());
    cov -= K * crossCov;
    fixCovariance3(cov);
  }

protected:
  Eigen::Vector3d sp[7];
  Eigen::Matrix3d l = Eigen::Matrix3d::Zero();

  void generateSigmaPoints()
  {
    const double a11 = cov(0, 0), a21 = (cov(1, 0) + cov(0, 1)) * 0.5, a31 = (cov(2, 0) + cov(0, 2)) * 0.5;
    const double a22 = cov(1, 1), a32 = (cov(2, 1) + cov(1, 2)) * 0.5, a33 = cov(2, 2);
    l.setZero();
    double & l11 = l(0, 0); double & l21 = l(1, 0); double & l31 = l(2, 0);
    double & l22 = l(1, 1); double & l32 = l(2, 1); double & l33 = l(2, 2);
    l11 = std::sqrt(std::max(a11, 0.0)); if(l11 == 0.0) l11 = 1e-10;
    l21 = a21 / l11; l31 = a31 / l11;
    l22 = std::sqrt(std::max(a22 - l21 * l21, 0.0)); if(l22 == 0.0) l22 = 1e-10;
    l32 = (a32 - l31 * l21) / l22;
    l33 = std::sqrt(std::max(a33 - l31 * l31 - l32 * l32, 0.0));
    sp[0] = mean;
    for(int i = 0; i < 3; ++i) {
      sp[1 + 2 * i] = mean + l.col(i);
      sp[2 + 2 * i] = mean - l.col(i);
    }
  }
};

}  // namespace hsl
