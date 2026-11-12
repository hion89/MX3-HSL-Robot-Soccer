// ball_tracker_node — 공 추적 + 팀 공 (B-Human BallLocator / TeamBallLocator 의 단순화 이식).
//
// 자기 공: 오도메트리 좌표(odom)에서 [x, y, vx, vy] 칼만 필터.
//   odom 은 다리 오도메트리라 짧은 시간엔 거의 고정이다 → 로봇이 돌거나 걸어도 공 추정이 흔들리지 않는다.
//   예측: 등감속 (구름 마찰, B-Human ballFriction 대응).  갱신: 거리 제곱에 비례하는 깊이 노이즈.
//   마할라노비스 게이트 밖 측정이 연속 N 번이면 재초기화 (B-Human 의 다중 가설 대신 단일 + 재시작).
//   '보여야 하는데 안 보임' 시간 = 카메라 시야·깊이 범위 안인데 검출이 없는 누적 시간 (BallLost 판단용).
// 팀 공: 자기 공 + /hsl/team_state 팀원 공을 현재 시각으로 예측해 0.6 m 군집, 가장 큰 군집 평균.
//   자기 공이 최근(own_ball_trust_time) 이면 자기 공을 우선 (B-Human: 본인이 본 공을 가장 신뢰).
// 출력: /hsl/ball (BallEstimate), /hsl/team_ball (TeamBall)
#include <Eigen/Dense>
#include <algorithm>
#include <deque>
#include <memory>

#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "hsl_interfaces/msg/ball_estimate.hpp"
#include "hsl_world_model/range_noise.hpp"
#include "hsl_interfaces/msg/team_ball.hpp"
#include "hsl_interfaces/msg/team_state.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "vision_msgs/msg/detection3_d_array.hpp"

using Vec4 = Eigen::Matrix<double, 4, 1>;
using Mat4 = Eigen::Matrix<double, 4, 4>;

class BallTracker : public rclcpp::Node
{
public:
  BallTracker() : Node("hsl_ball_tracker"), tf_buf_(get_clock()), tf_listener_(tf_buf_)
  {
    friction_ = declare_parameter<double>("ball_friction", 0.5);              // 등감속 [m/s²]
    process_pos_ = declare_parameter<double>("process_noise_pos", 0.02);       // [m/√s]
    process_vel_ = declare_parameter<double>("process_noise_vel", 0.6);        // [m/s/√s]
    range_noise_ = hsl::RangeNoise::fromKnots(
      declare_parameter<std::vector<double>>("range_sigma_knots", {0.0, 0.005, 2.5, 0.10, 4.0, 0.50}),
      declare_parameter<double>("range_sigma_scale", 1.3));
    sigma_bearing_ = declare_parameter<double>("sigma_bearing", 0.02);
    gate_ = declare_parameter<double>("mahalanobis_gate", 13.8);              // χ²(2) 99.9 %
    max_outliers_ = declare_parameter<int>("max_consecutive_outliers", 3);
    ball_timeout_ = declare_parameter<double>("ball_timeout", 6.0);
    max_speed_ = declare_parameter<double>("max_ball_speed", 6.0);
    fov_h_ = declare_parameter<double>("camera_fov_h_deg", 87.0) * M_PI / 360.0;   // 반각
    fov_v_ = declare_parameter<double>("camera_fov_v_deg", 58.0) * M_PI / 360.0;
    min_depth_ = declare_parameter<double>("camera_min_depth", 0.30);
    max_depth_ = declare_parameter<double>("camera_max_depth", 4.3);
    camera_frame_ = declare_parameter<std::string>("camera_frame", "camera_color_optical_frame");
    team_cluster_ = declare_parameter<double>("team_cluster_radius", 0.6);
    team_max_age_ = declare_parameter<double>("team_ball_max_age", 3.0);
    team_min_validity_ = declare_parameter<double>("team_min_pose_validity", 0.3);
    own_trust_ = declare_parameter<double>("own_ball_trust_time", 1.0);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    const double rate = declare_parameter<double>("update_rate", 50.0);

    obstacle_ttl_ = declare_parameter<double>("obstacle_ttl", 1.5);
    ball_pub_ = create_publisher<hsl_interfaces::msg::BallEstimate>("/hsl/ball", 10);
    obstacle_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("/hsl/obstacles", 10);
    team_ball_pub_ = create_publisher<hsl_interfaces::msg::TeamBall>("/hsl/team_ball", 10);
    det_sub_ = create_subscription<vision_msgs::msg::Detection3DArray>("/perception/detections_3d", 10,
      [this](vision_msgs::msg::Detection3DArray::SharedPtr m) { pending_.push_back(m); if(pending_.size() > 5) pending_.pop_front(); });
    team_sub_ = create_subscription<hsl_interfaces::msg::TeamState>("/hsl/team_state", 10,
      [this](hsl_interfaces::msg::TeamState::SharedPtr m) { team_ = m; });
    state_sub_ = create_subscription<std_msgs::msg::String>("/robot_state", 10,
      [this](std_msgs::msg::String::SharedPtr m) { robot_state_ = m->data; });
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate), std::bind(&BallTracker::cycle, this));
    last_cycle_ = now();
  }

private:
  // 등감속 예측 (속도 방향 반대로 friction, 0 을 지나치지 않음)
  static void predictState(Vec4 & x, double dt, double friction)
  {
    const Eigen::Vector2d v = x.tail<2>();
    const double s = v.norm();
    if(s < 1e-6) return;
    const double t_stop = s / friction, t = std::min(dt, t_stop);
    const Eigen::Vector2d dir = v / s;
    x.head<2>() += v * t - 0.5 * friction * dir * t * t;
    x.tail<2>() = dir * std::max(0.0, s - friction * t);
  }

  void predict(double dt)
  {
    if(!initialized_ || dt <= 0) return;
    predictState(x_, dt, friction_);
    Mat4 F = Mat4::Identity();
    F(0, 2) = F(1, 3) = dt;
    Mat4 Q = Mat4::Zero();
    Q(0, 0) = Q(1, 1) = process_pos_ * process_pos_ * dt;
    Q(2, 2) = Q(3, 3) = process_vel_ * process_vel_ * dt;
    P_ = F * P_ * F.transpose() + Q;
  }

  // 검출 시각의 TF 가 들어온 것만 처리 (자기위치 노드와 같은 이유). 아직이면 다음 주기에.
  void processPending()
  {
    while(!pending_.empty()) {
      const auto msg = pending_.front();
      const rclcpp::Time st(msg->header.stamp);
      if(!tf_buf_.canTransform(odom_frame_, msg->header.frame_id, st) ||
         !tf_buf_.canTransform(base_frame_, msg->header.frame_id, st)) {
        if((now() - st).seconds() > 0.25) { pending_.pop_front(); continue; }   // 포기
        return;
      }
      pending_.pop_front();
      onDetections(msg);
    }
  }

  void onDetections(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
  {
    if(!robot_state_.empty() && robot_state_ != "NORMAL") return;
    geometry_msgs::msg::TransformStamped T_ob, T_bc;
    try {
      T_ob = tf_buf_.lookupTransform(odom_frame_, msg->header.frame_id, msg->header.stamp, rclcpp::Duration::from_seconds(0.0));
      T_bc = tf_buf_.lookupTransform(base_frame_, msg->header.frame_id, msg->header.stamp, rclcpp::Duration::from_seconds(0.0));
    } catch(const std::exception &) { return; }
    // 로봇 검출 → 장애물 (odom 좌표, obstacle_ttl 동안 유지; B-Human ObstacleModel 단순화)
    const Eigen::Isometry3d iso_ob = tf2::transformToEigen(T_ob);
    const double now_s = now().seconds();
    for(const auto & d : msg->detections) {
      if(d.results.empty() || d.results[0].hypothesis.class_id != "robot") continue;
      const auto & q = d.bbox.center.position;
      const Eigen::Vector2d o = (iso_ob * Eigen::Vector3d(q.x, q.y, q.z)).head<2>();
      obstacles_.erase(std::remove_if(obstacles_.begin(), obstacles_.end(),
        [&](const Obstacle & e) { return (e.p - o).norm() < 0.4; }), obstacles_.end());
      obstacles_.push_back({o, now_s});
    }
    // 가장 확신 높은 공 하나 (B-Human 도 프레임당 BallPercept 1 개)
    const vision_msgs::msg::Detection3D * best = nullptr;
    double best_score = -1;
    for(const auto & d : msg->detections) {
      if(d.results.empty() || d.results[0].hypothesis.class_id != "ball") continue;
      if(d.results[0].hypothesis.score > best_score) { best_score = d.results[0].hypothesis.score; best = &d; }
    }
    if(!best) return;
    const auto & p = best->bbox.center.position;
    const Eigen::Vector3d pc(p.x, p.y, p.z);
    const Eigen::Vector3d po = tf2::transformToEigen(T_ob) * pc;
    const Eigen::Vector3d pb = tf2::transformToEigen(T_bc) * pc;
    const Eigen::Vector2d z = po.head<2>();

    // 측정 공분산: 몸통 기준 거리/방위 → odom 으로 회전
    const double dist = pb.head<2>().norm();
    const double sr = range_noise_.sigma(dist), st = std::max(0.01, dist * sigma_bearing_);
    const Eigen::Vector2d dir_o = (z - tf2::transformToEigen(T_ob).translation().head<2>()).normalized();
    Eigen::Matrix2d Rr;
    Rr << dir_o.x(), -dir_o.y(), dir_o.y(), dir_o.x();
    const Eigen::Matrix2d R = Rr * Eigen::Vector2d(sr * sr, st * st).asDiagonal() * Rr.transpose();

    const rclcpp::Time stamp = msg->header.stamp;
    if(!initialized_) { init(z, R, stamp); return; }
    Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
    H(0, 0) = H(1, 1) = 1.0;
    const Eigen::Vector2d innov = z - x_.head<2>();
    const Eigen::Matrix2d S = H * P_ * H.transpose() + R;
    const double m2 = innov.transpose() * S.inverse() * innov;
    if(m2 > gate_) {
      if(++outliers_ >= max_outliers_) {
        RCLCPP_INFO(get_logger(), "공 재초기화 (게이트 밖 %d 연속, 거리 %.2f m)", outliers_, innov.norm());
        init(z, R, stamp);
      }
      return;
    }
    outliers_ = 0;
    const Eigen::Matrix<double, 4, 2> K = P_ * H.transpose() * S.inverse();
    x_ += K * innov;
    P_ = (Mat4::Identity() - K * H) * P_;
    P_ = 0.5 * (P_ + P_.transpose());
    const double s = x_.tail<2>().norm();
    if(s > max_speed_) x_.tail<2>() *= max_speed_ / s;
    last_seen_ = stamp;
    disappeared_ = 0.0;
  }

  void init(const Eigen::Vector2d & z, const Eigen::Matrix2d & R, const rclcpp::Time & stamp)
  {
    x_ << z.x(), z.y(), 0.0, 0.0;
    P_.setZero();
    P_.topLeftCorner<2, 2>() = R;
    P_(2, 2) = P_(3, 3) = 1.0;
    initialized_ = true;
    outliers_ = 0;
    last_seen_ = stamp;
    disappeared_ = 0.0;
  }

  bool lookup(const std::string & target, const std::string & source, Eigen::Isometry3d & out)
  {
    try {
      out = tf2::transformToEigen(tf_buf_.lookupTransform(target, source, tf2::TimePointZero));
      return true;
    } catch(const std::exception &) { return false; }
  }

  void cycle()
  {
    const rclcpp::Time t = now();
    const double dt = (t - last_cycle_).seconds();
    last_cycle_ = t;
    predict(dt);
    processPending();

    hsl_interfaces::msg::BallEstimate b;
    b.header.stamp = t;
    b.header.frame_id = map_frame_;
    const double since_seen = initialized_ ? (t - last_seen_).seconds() : 1e3;
    b.valid = initialized_ && since_seen < ball_timeout_;
    b.time_since_seen = static_cast<float>(std::min(since_seen, 1e3));

    Eigen::Isometry3d T_bo, T_mo, T_co;
    const bool have_base = lookup(base_frame_, odom_frame_, T_bo);
    const bool have_map = lookup(map_frame_, odom_frame_, T_mo);
    if(initialized_) {
      const Eigen::Vector3d po(x_(0), x_(1), 0.0), vo(x_(2), x_(3), 0.0);
      if(have_base) {
        const Eigen::Vector3d pb = T_bo * po;
        b.rel_x = pb.x(); b.rel_y = pb.y();
      }
      if(lookup(camera_frame_, odom_frame_, T_co)) {
        const Eigen::Vector3d pc = T_co * Eigen::Vector3d(x_(0), x_(1), 0.065);
        b.cam_valid = true;
        b.cam_x = pc.x(); b.cam_y = pc.y(); b.cam_z = pc.z();
        // 보여야 하는데 안 보이는 시간: 예측 위치가 카메라 시야·깊이 범위 안인데 검출이 없다
        const bool visible = pc.z() > min_depth_ && pc.z() < max_depth_ &&
                             std::abs(std::atan2(pc.x(), pc.z())) < fov_h_ * 0.85 &&
                             std::abs(std::atan2(pc.y(), pc.z())) < fov_v_ * 0.85;
        if(since_seen > 0.15 && visible) disappeared_ += dt;
      }
      const Eigen::Isometry3d Tm = have_map ? T_mo : Eigen::Isometry3d::Identity();
      const Eigen::Vector3d pm = Tm * po, vm = Tm.linear() * vo;
      b.field_x = pm.x(); b.field_y = pm.y(); b.vel_x = vm.x(); b.vel_y = vm.y();
      const Eigen::Matrix2d Rm = Tm.linear().topLeftCorner<2, 2>();
      const Eigen::Matrix2d C = Rm * P_.topLeftCorner<2, 2>() * Rm.transpose();
      b.covariance = {static_cast<float>(C(0, 0)), static_cast<float>(C(0, 1)),
                      static_cast<float>(C(1, 0)), static_cast<float>(C(1, 1))};
    }
    b.time_since_disappeared = static_cast<float>(disappeared_);
    ball_pub_->publish(b);
    // 장애물 (몸통 좌표) — 행동 노드는 TF 를 안 듣는다 (Python TF 리스너가 로봇당 CPU 15~30 % 를 먹었다)
    const double now_s = t.seconds();
    obstacles_.erase(std::remove_if(obstacles_.begin(), obstacles_.end(),
      [&](const Obstacle & e) { return now_s - e.t > obstacle_ttl_; }), obstacles_.end());
    if(have_base && ++obstacle_div_ % 5 == 0) {
      geometry_msgs::msg::PoseArray pa;
      pa.header.stamp = t;
      pa.header.frame_id = base_frame_;
      for(const auto & e : obstacles_) {
        const Eigen::Vector3d pb = T_bo * Eigen::Vector3d(e.p.x(), e.p.y(), 0.0);
        geometry_msgs::msg::Pose ps;
        ps.position.x = pb.x(); ps.position.y = pb.y(); ps.orientation.w = 1.0;
        pa.poses.push_back(ps);
      }
      obstacle_pub_->publish(pa);
    }
    publishTeamBall(b, t);
  }

  struct Obs { Eigen::Vector2d p, v; double age; bool own; };

  void publishTeamBall(const hsl_interfaces::msg::BallEstimate & own, const rclcpp::Time & t)
  {
    std::vector<Obs> obs;
    if(own.valid) obs.push_back({{own.field_x, own.field_y}, {own.vel_x, own.vel_y}, own.time_since_seen, true});
    if(team_) {
      for(const auto & m : team_->teammates) {
        if(!m.ball_valid || m.penalized || m.fallen || m.pose_validity < team_min_validity_) continue;
        const double age = m.ball_age + std::max(0.0, (t - rclcpp::Time(m.received)).seconds());
        if(age > team_max_age_) continue;
        Vec4 s; s << m.ball_x, m.ball_y, m.ball_vel_x, m.ball_vel_y;
        predictState(s, age, friction_);
        obs.push_back({s.head<2>(), s.tail<2>(), age, false});
      }
    }
    hsl_interfaces::msg::TeamBall tb;
    tb.header.stamp = t;
    tb.header.frame_id = map_frame_;
    if(!obs.empty()) {
      // 자기 공을 최근에 봤으면 그대로 (팀 정보는 자기 시야 밖일 때 쓴다)
      int best = -1, best_n = 0;
      double best_age = 1e9;
      std::vector<int> best_members;
      for(size_t i = 0; i < obs.size(); ++i) {
        std::vector<int> mem;
        for(size_t j = 0; j < obs.size(); ++j)
          if((obs[j].p - obs[i].p).norm() < team_cluster_) mem.push_back(static_cast<int>(j));
        const int n = static_cast<int>(mem.size());
        const bool own_fresh = obs[i].own && obs[i].age < own_trust_;
        const bool better = own_fresh || (n > best_n || (n == best_n && obs[i].age < best_age));
        if(best >= 0 && obs[best].own && obs[best].age < own_trust_) continue;
        if(better) { best = static_cast<int>(i); best_n = n; best_age = obs[i].age; best_members = mem; }
      }
      Eigen::Vector2d p = Eigen::Vector2d::Zero(), v = Eigen::Vector2d::Zero();
      double wsum = 0.0;
      for(int j : best_members) {
        const double w = 1.0 / (0.2 + obs[j].age);
        p += w * obs[j].p; v += w * obs[j].v; wsum += w;
        if(obs[j].own) tb.own_ball_used = true;
      }
      p /= wsum; v /= wsum;
      tb.valid = true;
      tb.field_x = p.x(); tb.field_y = p.y(); tb.vel_x = v.x(); tb.vel_y = v.y();
      tb.num_contributors = static_cast<uint8_t>(best_members.size());
    }
    team_ball_pub_->publish(tb);
  }

  hsl::RangeNoise range_noise_;
  double friction_, process_pos_, process_vel_, sigma_bearing_, gate_, ball_timeout_, max_speed_;
  double fov_h_, fov_v_, min_depth_, max_depth_, team_cluster_, team_max_age_, team_min_validity_, own_trust_;
  int max_outliers_;
  std::string camera_frame_, map_frame_, odom_frame_, base_frame_, robot_state_;
  bool initialized_ = false;
  int outliers_ = 0;
  Vec4 x_ = Vec4::Zero();
  Mat4 P_ = Mat4::Identity();
  double disappeared_ = 0.0;
  rclcpp::Time last_seen_{0, 0, RCL_ROS_TIME}, last_cycle_{0, 0, RCL_ROS_TIME};
  hsl_interfaces::msg::TeamState::SharedPtr team_;
  std::deque<vision_msgs::msg::Detection3DArray::SharedPtr> pending_;
  struct Obstacle { Eigen::Vector2d p; double t; };
  std::vector<Obstacle> obstacles_;
  double obstacle_ttl_ = 1.5;
  int obstacle_div_ = 0;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr obstacle_pub_;
  tf2_ros::Buffer tf_buf_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<hsl_interfaces::msg::BallEstimate>::SharedPtr ball_pub_;
  rclcpp::Publisher<hsl_interfaces::msg::TeamBall>::SharedPtr team_ball_pub_;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr det_sub_;
  rclcpp::Subscription<hsl_interfaces::msg::TeamState>::SharedPtr team_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BallTracker>());
  rclcpp::shutdown();
  return 0;
}
