// Copyright (c) 2026 B-Human. All rights reserved. — 이 파일은 B-Human code release 2026 의 일부를 이식한 것이다.
// "This product includes software developed by B-Human (http://www.b-human.de)." (B-Human License.md 조건 1·3)
// self_localizer_node — B-Human SelfLocator 의 이식 (B-Human 2026 code release, BSD 계열 라이선스).
// 원본: Src/Modules/Modeling/SelfLocator/SelfLocator.cpp, UKFRobotPoseHypothesis.cpp,
//       Src/Modules/Modeling/PerceptRegistrationProvider (랜드마크 등록), AlternativeRobotPoseProvider (리셋용 대안 pose).
//
// 구조 (B-Human 과 같음):
//   샘플 N 개(기본 12), 각 샘플 = UKF (x, y, θ) + validity.
//   매 주기: 오도메트리로 motionUpdate (B-Human 노이즈 모델 그대로)
//            → 가설마다 '자기 pose 기준으로' 랜드마크를 최근접 등록 → UKF 갱신 + validity(설명한 비율) 이동평균
//            → validity 가중 low-variance 재표본화 → 최선 샘플이 RobotPose
//            → 대안 pose(여러 프레임 누적)가 현재 추정과 크게 다르면 최악 샘플을 교체 (sensor resetting)
// 우리 쪽 차이:
//   - 인식은 YOLO 의 L/T/X 교차점·골대뿐 (선 없음) → 선 갱신 없음, 랜드마크 + '프레임 절대 pose' 측정 사용.
//   - 프레임 절대 pose = 같은 프레임의 랜드마크 2 개 이상 기하 정합으로 구한 pose (B-Human 의 absolute pose
//     measurement 역할). 센터 X 교차점 0.75 m 주기 같은 단일 랜드마크 오매칭을 기하로 걸러낸다.
//   - 경기 상태: INITIAL·START 버튼 = 사이드라인 진입 pose, 페널티 복귀 = 양쪽 터치라인 pose, READY/SET = 우리 진영.
#include <Eigen/Dense>
#include <algorithm>
#include <deque>
#include <memory>
#include <optional>
#include <random>

#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "hsl_interfaces/msg/ball_estimate.hpp"
#include "hsl_interfaces/msg/line_segment_array.hpp"
#include "hsl_interfaces/msg/robot_pose_estimate.hpp"
#include "hsl_interfaces/msg/team_state.hpp"
#include "hsl_world_model/field_model.hpp"
#include "hsl_world_model/range_noise.hpp"
#include "hsl_world_model/ukf_pose2d.hpp"
#include "humanoid_behavior_interfaces/msg/game_state.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "vision_msgs/msg/detection3_d_array.hpp"

using namespace hsl;
using GameState = humanoid_behavior_interfaces::msg::GameState;

struct Percept
{
  LandmarkType type;
  Eigen::Vector2d pos;     // base_link 기준 [m]
  Eigen::Matrix2d cov;
};

// 선 인식 (base_link 기준 선분). B-Human RegisteredLine 의 '인식' 쪽 값들
struct LinePercept
{
  Eigen::Vector2d start, end, center, dir;
  Eigen::Matrix2d cov_center;   // 선분 중심점의 공분산
  double length;
};

struct Sample : public UKFPose2D
{
  int id = 0;
  double validity = 0.0, weighting = 0.0;
  void init(const Pose2D & p, const Pose2D & dev, int i, double v)
  {
    id = i; validity = v;
    mean << p.translation.x(), p.translation.y(), p.rotation;
    cov.setZero();
    cov(0, 0) = dev.translation.x() * dev.translation.x();
    cov(1, 1) = dev.translation.y() * dev.translation.y();
    cov(2, 2) = dev.rotation * dev.rotation;
  }
};

class SelfLocator : public rclcpp::Node
{
public:
  SelfLocator() : Node("hsl_self_localizer"), rng_(std::random_device{}()), tf_buf_(get_clock()), tf_listener_(tf_buf_)
  {
    // ── B-Human Config/Scenarios/Default/selfLocator.cfg (mm → m) ──
    n_samples_ = declare_parameter<int>("number_of_samples", 12);
    default_dev_ = poseParam("default_pose_deviation", {0.5, 0.5, 0.30});
    walk_in_dev_ = poseParam("walk_in_pose_deviation", {0.2, 0.1, 0.30});
    penalty_dev_ = poseParam("return_from_penalty_pose_deviation", {0.7, 0.2, 0.30});
    process_dev_ = poseParam("filter_process_deviation", {0.002, 0.002, 0.002});
    odo_dev_ = poseParam("odometry_deviation", {0.2, 0.2, 0.3});
    const auto ord = declare_parameter<std::vector<double>>("odometry_rotation_deviation", {1.57, 1.57});
    odo_rot_dev_ = Eigen::Vector2d(ord[0], ord[1]);   // rad/m (B-Human 0.00157 rad/mm)
    moved_dist_rot_noise_ = declare_parameter<double>("moved_dist_weight_rotation_noise", 0.3);
    moved_angle_rot_noise_ = declare_parameter<double>("moved_angle_weight_rotation_noise", 0.25);
    major_w_ = declare_parameter<double>("major_dir_trans_weight", 2.0);
    minor_w_ = declare_parameter<double>("minor_dir_trans_weight", 1.0);
    base_validity_w_ = declare_parameter<double>("base_validity_weighting", 0.1);
    validity_frames_ = declare_parameter<int>("number_of_considered_frames_for_validity", 60);
    vf_landmark_ = declare_parameter<double>("validity_factor_landmark_measurement", 3.0);
    vf_pose_ = declare_parameter<double>("validity_factor_pose_measurement", 5.0);
    // ── 선 (B-Human perceptRegistrationProvider.cfg / selfLocator.cfg, mm → m) ──
    use_lines_ = declare_parameter<bool>("use_lines", true);
    line_corridor_ = declare_parameter<double>("line_association_corridor", 0.4);
    line_max_ang_ = declare_parameter<double>("line_max_angular_deviation", 15.0 * M_PI / 180.0);
    min_line_length_ = declare_parameter<double>("minimum_line_length", 0.4);
    vf_line_ = declare_parameter<double>("validity_factor_line_measurement", 1.0);
    min_obs_reset_ = declare_parameter<int>("min_number_of_observations_for_resetting", 10);
    reset_min_span_ = declare_parameter<double>("min_time_span_for_resetting", 0.6);
    reset_min_support_ = declare_parameter<int>("min_landmarks_for_resetting", 3);
    reset_trans_ = declare_parameter<double>("translational_deviation_for_resetting", 0.5);
    reset_rot_ = declare_parameter<double>("rotational_deviation_for_resetting", 0.52);
    superb_validity_ = declare_parameter<double>("min_validity_for_superb", 0.8);
    superb_trans_ = declare_parameter<double>("max_translation_deviation_for_superb", 0.10);
    superb_rot_ = declare_parameter<double>("max_rotation_deviation_for_superb", 0.175);
    max_gyro_z_ = declare_parameter<double>("max_gyro_z_for_measurements", 0.87);   // 50°/s
    // ── PerceptRegistrationProvider ──
    max_intersection_dev_ = declare_parameter<double>("max_intersection_deviation", 0.5);
    max_goalpost_dev_ = declare_parameter<double>("max_goalpost_deviation", 0.6);
    pose_assoc_dist_ = declare_parameter<double>("global_pose_association_max_distance", 1.0);
    pose_assoc_ang_ = declare_parameter<double>("global_pose_association_max_angle", 0.35);
    // ── 인식 노이즈 (D405 깊이: 거리 제곱에 비례) ──
    range_noise_ = RangeNoise::fromKnots(
      declare_parameter<std::vector<double>>("percept_range_sigma_knots", {0.0, 0.005, 2.5, 0.10, 4.0, 0.50}),
      declare_parameter<double>("percept_range_sigma_scale", 1.3));
    sigma_bearing_ = declare_parameter<double>("percept_sigma_bearing", 0.02);   // [rad] 방위 σ (픽셀 0.5° + 여유)
    max_percept_dist_ = declare_parameter<double>("max_percept_distance", 4.5);
    // ── 대안 pose (프레임 절대 pose) ──
    pair_dist_tol_ = declare_parameter<double>("pair_distance_tolerance", 0.25);
    explain_tol_ = declare_parameter<double>("candidate_explain_tolerance", 0.35);
    alt_window_s_ = declare_parameter<double>("alternative_pose_window", 3.0);
    max_tf_wait_ = declare_parameter<double>("max_tf_wait", 0.25);
    mirror_frames_ = declare_parameter<int>("mirror_detection_frames", 20);
    cand_max_dist_ = declare_parameter<double>("candidate_max_percept_distance", 2.5);
    best_switch_margin_ = declare_parameter<double>("best_switch_validity_margin", 0.05);
    mirror_min_validity_ = declare_parameter<double>("mirror_teammate_min_validity", 0.5);
    // ── 진입 pose (사이드라인, relocalize_button_node 와 같은 규약) ──
    const auto field_profile = declare_parameter<std::string>("field_profile", "s_field");
    field_ = FieldModel::forProfile(field_profile);
    walk_in_x_inset_ = declare_parameter<double>("walk_in_x_inset", 0.0);
    walk_in_side_ = declare_parameter<int>("walk_in_side", 1);              // +1 = 왼쪽(+y) 사이드라인
    // Practice carpet extends only 0.20 m beyond the touchline. Keep reset seed inside it.
    const double default_walk_in_outside = std::min(0.4, field_.border_y * 0.5);
    walk_in_outside_ = declare_parameter<double>("walk_in_outside_margin", default_walk_in_outside);
    player_number_ = declare_parameter<int>("player_number", 1);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    const double rate = declare_parameter<double>("update_rate", 30.0);

    for(auto t : {LandmarkType::GOALPOST, LandmarkType::X, LandmarkType::T, LandmarkType::L})
      model_[static_cast<int>(t)] = field_.landmarks(t);
    model_lines_ = field_.lines();

    samples_.resize(n_samples_);
    resetAll(walkInPose(), walk_in_dev_, 0.5);

    pose_pub_ = create_publisher<hsl_interfaces::msg::RobotPoseEstimate>("/hsl/robot_pose", 10);
    tf_pub_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    line_sub_ = create_subscription<hsl_interfaces::msg::LineSegmentArray>("/perception/field_lines", 10,
      [this](hsl_interfaces::msg::LineSegmentArray::SharedPtr m) { last_lines_ = m; });
    det_sub_ = create_subscription<vision_msgs::msg::Detection3DArray>("/perception/detections_3d", 10,
      [this](vision_msgs::msg::Detection3DArray::SharedPtr m) { last_det_ = m; });
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>("/imu/data_50hz", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Imu::SharedPtr m) { gyro_z_ = m->angular_velocity.z; });
    gc_sub_ = create_subscription<GameState>("/game_state", 10, std::bind(&SelfLocator::onGameState, this, std::placeholders::_1));
    start_sub_ = create_subscription<std_msgs::msg::Empty>("/buttons/start_event", 10,
      [this](std_msgs::msg::Empty::SharedPtr) {
        resetAll(walkInPose(), walk_in_dev_, 0.5);
        own_half_until_ = now() + rclcpp::Duration::from_seconds(5.0);
        RCLCPP_INFO(get_logger(), "START → 진입 pose 로 재시작");
      });
    state_sub_ = create_subscription<std_msgs::msg::String>("/robot_state", 10,
      [this](std_msgs::msg::String::SharedPtr m) { robot_state_ = m->data; });
    init_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", 10,
      [this](geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr m) {
        const auto & q = m->pose.pose.orientation;
        const double yaw = std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
        resetAll(Pose2D(yaw, m->pose.pose.position.x, m->pose.pose.position.y), Pose2D(0.1, 0.1, 0.1), 0.8);
        RCLCPP_INFO(get_logger(), "/initialpose (%.2f, %.2f, %.0f°)", m->pose.pose.position.x,
                    m->pose.pose.position.y, yaw * 57.3);
      });
    ball_sub_ = create_subscription<hsl_interfaces::msg::BallEstimate>("/hsl/ball", 10,
      [this](hsl_interfaces::msg::BallEstimate::SharedPtr m) { own_ball_ = m; });
    team_sub_ = create_subscription<hsl_interfaces::msg::TeamState>("/hsl/team_state", 10,
      [this](hsl_interfaces::msg::TeamState::SharedPtr m) { team_ = m; });
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate), std::bind(&SelfLocator::cycle, this));
    RCLCPP_INFO(get_logger(), "self localizer: %d samples, field %.1fx%.1f, player %d, walk-in inset %.2f",
                n_samples_, field_.length, field_.width, player_number_, walk_in_x_inset_);
    RCLCPP_INFO(get_logger(),
      "FIELD_GEOMETRY profile=%s circle_r=%.2f penalty=%.2fx%.2f goal_area=%.2fx%.2f "
      "goal_width=%.2f penalty_mark=%.2f carpet=%.2fx%.2f walk_in_outside=%.2f",
      field_profile.c_str(), field_.center_circle_radius, field_.penalty_area_depth, field_.penalty_area_width,
      field_.goal_area_depth, field_.goal_area_width, field_.goal_width, field_.penalty_mark_dist,
      field_.length + 2 * field_.border_x, field_.width + 2 * field_.border_y, walk_in_outside_);
  }

private:
  Pose2D poseParam(const std::string & name, std::vector<double> def)
  {
    const auto v = declare_parameter<std::vector<double>>(name, def);
    return Pose2D(v[2], v[0], v[1]);
  }

  Pose2D walkInPose() const
  {
    const double side = walk_in_side_ >= 0 ? 1.0 : -1.0;
    return Pose2D(-side * M_PI / 2, -(field_.hl() - walk_in_x_inset_), side * (field_.hw() + walk_in_outside_));
  }

  void resetAll(const Pose2D & p, const Pose2D & dev, double validity)
  {
    for(auto & s : samples_) s.init(p, dev, next_id_++, validity);
  }

  // ── 경기 상태 처리 (B-Human handleGameStateChanges 단순화) ──
  void onGameState(const GameState::SharedPtr m)
  {
    const bool penalized = m->team_found && player_number_ >= 1 &&
                           static_cast<size_t>(player_number_ - 1) < m->penalties.size() &&
                           m->penalties[player_number_ - 1] != 0;
    if(game_state_ != GameState::STATE_INITIAL && m->state == GameState::STATE_INITIAL)
      resetAll(walkInPose(), walk_in_dev_, 0.5);
    if(was_penalized_ && !penalized) {   // 페널티 복귀: 우리 진영 페널티마크 높이의 양쪽 터치라인
      const double x = -(field_.hl() - field_.penalty_mark_dist);
      for(int i = 0; i < n_samples_; ++i) {
        const double side = (i % 2 == 0) ? 1.0 : -1.0;
        samples_[i].init(Pose2D(-side * M_PI / 2, x, side * field_.hw()), penalty_dev_, next_id_++, 0.5);
      }
      own_half_until_ = now() + rclcpp::Duration::from_seconds(5.0);
      RCLCPP_INFO(get_logger(), "페널티 복귀 → 양쪽 터치라인 가설");
    }
    was_penalized_ = penalized;
    if(m->state == GameState::STATE_READY && game_state_ != GameState::STATE_READY)
      ready_from_initial_ = (game_state_ == GameState::STATE_INITIAL || game_state_ == 255);
    game_state_ = m->state;
  }

  // B-Human SideInformation: '우리 진영 확정' 은 사이드라인 진입(INITIAL → 첫 READY), SET, 페널티 복귀 직후뿐.
  // 골 뒤 READY 는 아니다 — 상대 진영에서 걸어 돌아오는 중이라 이 규칙을 걸면 맞는 가설을 지우고
  // 대칭(거울) 가설을 고른다 (4v4 시험에서 t36p4 가 이렇게 뒤집혀 끝까지 못 돌아옴).
  bool mustBeInOwnHalf()
  {
    return game_state_ == GameState::STATE_INITIAL || game_state_ == GameState::STATE_SET ||
           (game_state_ == GameState::STATE_READY && ready_from_initial_) || now() < own_half_until_;
  }

  // 좌우 반전 감지 (B-Human 도 팀 공으로 대칭 가설을 판정한다): 내가 본 공을 내 pose 로 필드에 옮긴 값이
  // 팀원들이 본 공과 멀고, 그 대칭점은 가깝다는 판정이 연속 mirror_frames 번이면 모든 가설을 뒤집는다.
  void checkMirror()
  {
    if(!own_ball_ || !team_ || !own_ball_->valid || own_ball_->time_since_seen > 0.5) return;
    Eigen::Vector2d tb = Eigen::Vector2d::Zero();
    int n = 0;
    const double now_s = now().seconds();
    for(const auto & t : team_->teammates) {
      if(!t.ball_valid || t.fallen || t.penalized || t.pose_validity < mirror_min_validity_) continue;
      const double age = t.ball_age + (now_s - rclcpp::Time(t.received).seconds());
      if(age > 1.0) continue;
      tb += Eigen::Vector2d(t.ball_x, t.ball_y); ++n;
    }
    if(n == 0) return;
    tb /= n;
    if(tb.norm() < 1.0) { mirror_count_ = 0; return; }       // 센터 근처 공은 대칭점과 구분이 안 된다
    const Pose2D p = best().getPose();
    const Eigen::Vector2d mine = p * Eigen::Vector2d(own_ball_->rel_x, own_ball_->rel_y);
    const double d = (mine - tb).norm(), dm = (-mine - tb).norm();
    if(d > 1.5 && dm < 0.7) ++mirror_count_;
    else if(d < 0.7) mirror_count_ = 0;
    if(mirror_count_ >= mirror_frames_) {
      for(auto & s : samples_) {
        s.mean.x() = -s.mean.x(); s.mean.y() = -s.mean.y(); s.mean.z() = normalizeAngle(s.mean.z() + M_PI);
      }
      mirror_count_ = 0;
      last_jump_ = now();
      RCLCPP_WARN(get_logger(), "좌우 반전 감지 (내 공 (%.1f,%.1f) vs 팀 공 (%.1f,%.1f), 팀원 %d) → 모든 가설 뒤집음",
                  mine.x(), mine.y(), tb.x(), tb.y(), n);
    }
  }

  // ── 인식: detections_3d → base_link 2D 랜드마크 ──
  // ── 선 인식: /perception/field_lines → base_link 선분 (검출 시각 TF, 랜드마크와 같은 규칙) ──
  std::vector<LinePercept> readLines()
  {
    std::vector<LinePercept> out;
    if(!use_lines_ || !last_lines_ || last_lines_.get() == used_lines_) return out;
    const auto & msg = *last_lines_;
    geometry_msgs::msg::TransformStamped T;
    try {
      T = tf_buf_.lookupTransform(base_frame_, msg.header.frame_id, msg.header.stamp, rclcpp::Duration::from_seconds(0.0));
    } catch(const std::exception &) {
      if((now() - rclcpp::Time(msg.header.stamp)).seconds() > max_tf_wait_) used_lines_ = last_lines_.get();
      return out;
    }
    used_lines_ = last_lines_.get();
    const Eigen::Isometry3d iso = tf2::transformToEigen(T);
    for(const auto & sg : msg.segments) {
      LinePercept lp;
      lp.start = (iso * Eigen::Vector3d(sg.start.x, sg.start.y, sg.start.z)).head<2>();
      lp.end = (iso * Eigen::Vector3d(sg.end.x, sg.end.y, sg.end.z)).head<2>();
      lp.length = (lp.end - lp.start).norm();
      if(lp.length < min_line_length_) continue;
      lp.center = 0.5 * (lp.start + lp.end);
      lp.dir = (lp.end - lp.start) / lp.length;
      const double dist = lp.center.norm();
      if(dist > max_percept_dist_) continue;
      const double sr = range_noise_.sigma(dist), st = std::max(0.02, dist * sigma_bearing_);
      const double a = std::atan2(lp.center.y(), lp.center.x()), c = std::cos(a), sn = std::sin(a);
      Eigen::Matrix2d R; R << c, -sn, sn, c;
      lp.cov_center = R * Eigen::Vector2d(sr * sr, st * st).asDiagonal() * R.transpose();
      out.push_back(lp);
    }
    return out;
  }

  // ── 선 등록: B-Human PerceptRegistrationProvider::getPointerToCorrespondingLineInWorldModel (호/원 처리 제외) ──
  const FieldModel::Line * registerLine(const Pose2D & pose, const LinePercept & lp) const
  {
    const Eigen::Vector2d a = pose * lp.start, b = pose * lp.end;
    const Eigen::Vector2d dir = (b - a).normalized();
    const bool parallel_x = std::abs(dir.x()) > std::abs(dir.y());
    const double corr2 = line_corridor_ * line_corridor_;
    for(const auto & m : model_lines_) {
      const bool m_parallel_x = std::abs(m.start.y() - m.end.y()) < 1e-6;
      if(m_parallel_x != parallel_x) continue;
      const Eigen::Vector2d md = m.end - m.start;
      const double mlen = md.norm();
      if(lp.length > 1.25 * mlen) continue;                    // 원래 선보다 길 수 없다
      auto seg_d2 = [&](const Eigen::Vector2d & p) {
        const double t = std::clamp((p - m.start).dot(md) / (mlen * mlen), 0.0, 1.0);
        return (p - (m.start + t * md)).squaredNorm();
      };
      const double da = seg_d2(a), db = seg_d2(b);
      const double ang = std::acos(std::clamp(std::abs(dir.dot(md / mlen)), 0.0, 1.0));
      const bool both = da < corr2 && db < corr2;
      const bool one = da < corr2 || db < corr2;
      if(!both || !(one && ang < line_max_ang_)) continue;
      // 센터서클 조각이 하프라인으로 잘못 붙는 일이 잦다 → 멀리서 본 짧은 하프라인은 받지 않음 (B-Human 과 같음)
      const double diam = 2.0 * field_.center_circle_radius;
      if(m.halfway && lp.length < diam && lp.start.norm() > diam && lp.end.norm() > diam) return nullptr;
      return &m;
    }
    return nullptr;
  }

  // ── 선 갱신: B-Human UKFRobotPoseHypothesis::updateByLine + RegisteredLine 생성자 ──
  static void updateByLine(Sample & s, const LinePercept & lp, const FieldModel::Line & m)
  {
    const bool parallel_x = std::abs(m.start.y() - m.end.y()) < 1e-6;
    // 로봇 원점의 선 위 수선의 발 (로봇 기준)
    const Eigen::Vector2d op = lp.start + lp.dir * (-lp.start).dot(lp.dir);
    double ang = normalizeAngle(-std::atan2(op.y(), op.x()) + (parallel_x ? M_PI / 2 : 0.0));
    const double alt = normalizeAngle(ang - M_PI);
    if(std::abs(normalizeAngle(alt - s.mean.z())) < std::abs(normalizeAngle(ang - s.mean.z()))) ang = alt;
    const double c = std::cos(ang), sn = std::sin(ang);
    Eigen::Matrix2d R; R << c, -sn, sn, c;
    const Eigen::Vector2d opw = R * op;
    const Eigen::Matrix2d cov = R * lp.cov_center * R.transpose();
    const double len2 = lp.length * lp.length;
    if(parallel_x) {
      const double var = cov(1, 1);
      const double avar = std::pow(std::atan(std::sqrt(4.0 * var / len2)), 2);
      s.lineSensorUpdate(true, Eigen::Vector2d(m.start.y() - opw.y(), ang), Eigen::Vector2d(var, avar).asDiagonal());
    } else {
      const double var = cov(0, 0);
      const double avar = std::pow(std::atan(std::sqrt(4.0 * var / len2)), 2);
      s.lineSensorUpdate(false, Eigen::Vector2d(m.start.x() - opw.x(), ang), Eigen::Vector2d(var, avar).asDiagonal());
    }
  }

  std::vector<Percept> readPercepts()
  {
    std::vector<Percept> out;
    if(!last_det_ || last_det_.get() == used_det_) return out;
    const auto & msg = *last_det_;
    geometry_msgs::msg::TransformStamped T;
    // 검출 시각의 몸통→카메라 변환 (목 각도 포함) 이 꼭 필요하다. '최신 TF' 로 대신하면 목이 빠르게 돌 때
    // (훑기 ~120°/s) 수십 ms 차이로 방향이 수 ° 씩 틀어진다 (4v4 시험에서 방향 오차 p90 12°).
    // 아직 그 시각 TF 가 안 왔으면 이번 주기는 건너뛰고 다음 주기에 다시 본다.
    try {
      T = tf_buf_.lookupTransform(base_frame_, msg.header.frame_id, msg.header.stamp, rclcpp::Duration::from_seconds(0.0));
    } catch(const std::exception &) {
      if((now() - rclcpp::Time(msg.header.stamp)).seconds() > max_tf_wait_) used_det_ = last_det_.get();   // 포기
      return out;
    }
    used_det_ = last_det_.get();
    const Eigen::Isometry3d iso = tf2::transformToEigen(T);
    for(const auto & d : msg.detections) {
      if(d.results.empty()) continue;
      const auto & cls = d.results[0].hypothesis.class_id;
      LandmarkType t;
      if(cls == "goalpost") t = LandmarkType::GOALPOST;
      else if(cls == "X-Intersection") t = LandmarkType::X;
      else if(cls == "T-Intersection") t = LandmarkType::T;
      else if(cls == "L-Intersection") t = LandmarkType::L;
      else continue;
      const auto & p = d.bbox.center.position;
      const Eigen::Vector3d pb = iso * Eigen::Vector3d(p.x, p.y, p.z);
      const Eigen::Vector2d q = pb.head<2>();
      const double dist = q.norm();
      if(dist < 0.05 || dist > max_percept_dist_) continue;
      const double sr = range_noise_.sigma(dist), st = std::max(0.02, dist * sigma_bearing_);
      const double a = std::atan2(q.y(), q.x()), c = std::cos(a), s = std::sin(a);
      Eigen::Matrix2d R; R << c, -s, s, c;
      out.push_back({t, q, R * Eigen::Vector2d(sr * sr, st * st).asDiagonal() * R.transpose()});
    }
    return out;
  }

  // ── 등록: 가설 pose 기준 최근접 같은 타입 랜드마크 (PerceptRegistrationProvider) ──
  bool registerPercept(const Pose2D & pose, const Percept & p, Eigen::Vector2d & model) const
  {
    const Eigen::Vector2d w = pose * p.pos;
    const double gate = p.type == LandmarkType::GOALPOST ? max_goalpost_dev_ : max_intersection_dev_;
    double best = gate * gate;
    bool found = false;
    for(const auto & m : model_.at(static_cast<int>(p.type))) {
      const double d2 = (w - m).squaredNorm();
      if(d2 < best) { best = d2; model = m; found = true; }
    }
    return found;
  }

  // ── 프레임 절대 pose: 랜드마크 쌍 기하 정합 (AlternativeRobotPoseProvider 역할) ──
  struct Candidate { Pose2D pose; int support; };
  std::vector<Candidate> frameCandidates(const std::vector<Percept> & ps) const
  {
    std::vector<Candidate> cands;
    int best_support = 2;
    for(size_t i = 0; i < ps.size(); ++i) {
      for(size_t j = i + 1; j < ps.size(); ++j) {
        const Eigen::Vector2d dp = ps[j].pos - ps[i].pos;
        const double lp = dp.norm();
        if(lp < 0.4) continue;
        for(const auto & m1 : model_.at(static_cast<int>(ps[i].type))) {
          for(const auto & m2 : model_.at(static_cast<int>(ps[j].type))) {
            const Eigen::Vector2d dm = m2 - m1;
            if(std::abs(dm.norm() - lp) > pair_dist_tol_) continue;
            const double th = normalizeAngle(std::atan2(dm.y(), dm.x()) - std::atan2(dp.y(), dp.x()));
            const Pose2D cand(th, m1 - Pose2D(th, 0, 0).R() * ps[i].pos);
            int support = 0;
            for(const auto & p : ps) {
              const Eigen::Vector2d w = cand * p.pos;
              for(const auto & m : model_.at(static_cast<int>(p.type)))
                if((w - m).norm() < explain_tol_) { ++support; break; }
            }
            if(!field_.insideCarpet(cand.translation) || support < best_support) continue;
            if(support > best_support) { cands.clear(); best_support = support; }
            bool dup = false;
            for(const auto & c : cands)
              if((c.pose.translation - cand.translation).norm() < 0.2 &&
                 std::abs(normalizeAngle(c.pose.rotation - cand.rotation)) < 0.2) { dup = true; break; }
            if(!dup) cands.push_back({cand, support});
          }
        }
      }
    }
    return cands;
  }

  void cycle()
  {
    // 1) 오도메트리 증분 (TF odom → base_link)
    Pose2D odo_now;
    bool have_odo = false;
    try {
      const auto T = tf_buf_.lookupTransform(odom_frame_, base_frame_, tf2::TimePointZero);
      const auto & q = T.transform.rotation;
      odo_now = Pose2D(std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
                       T.transform.translation.x, T.transform.translation.y);
      have_odo = true;
    } catch(const std::exception &) {}
    if(!have_odo) return;
    Pose2D delta;
    if(have_last_odo_) delta = last_odo_.inverse() + odo_now;
    last_odo_ = odo_now; have_last_odo_ = true;

    if(game_state_ == GameState::STATE_INITIAL) {   // B-Human: INITIAL 에선 진입 pose 고정
      resetAll(walkInPose(), walk_in_dev_, 0.5);
      publish(odo_now, 0);
      return;
    }

    motionUpdate(delta);
    const auto percepts = readPercepts();
    const auto lines = readLines();
    int used = 0;
    if((!percepts.empty() || !lines.empty()) && std::abs(gyro_z_) < max_gyro_z_) used = sensorUpdate(percepts, lines);

    // 진영 제약·카펫 밖 무효화 (B-Human sensorUpdate 끝부분)
    const bool own_half = mustBeInOwnHalf();
    for(auto & s : samples_) {
      if(own_half && s.mean.x() > 0.3) s.validity = 0.0;
      if(!field_.insideCarpet(s.mean.head<2>())) s.validity = 0.0;
    }
    double wsum = 0.0;
    for(auto & s : samples_) { s.weighting = base_validity_w_ + (1.0 - base_validity_w_) * s.validity; wsum += s.weighting; }
    avg_weighting_ = wsum / n_samples_;
    if(validities_updated_) resampling();
    validities_updated_ = false;
    if(!own_half) checkMirror();
    sensorResetting(own_half);
    publish(odo_now, used);
  }

  void motionUpdate(const Pose2D & odo)
  {
    const double tx = odo.translation.x(), ty = odo.translation.y(), dist = odo.translation.norm();
    const double rotErr = std::max(dist * moved_dist_rot_noise_, std::abs(odo.rotation) * moved_angle_rot_noise_);
    const double exErr = std::max(std::abs(tx * major_w_), std::abs(ty * minor_w_));
    const double eyErr = std::max(std::abs(ty * major_w_), std::abs(tx * minor_w_));
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for(auto & s : samples_) {
      const Eigen::Vector2d to((tx - exErr) + 2 * exErr * u(rng_), (ty - eyErr) + 2 * eyErr * u(rng_));
      const double ro = odo.rotation + (u(rng_) * 2 - 1) * rotErr;
      s.motionUpdate(Pose2D(ro, to), process_dev_, odo_dev_, odo_rot_dev_);
    }
  }

  int sensorUpdate(const std::vector<Percept> & percepts, const std::vector<LinePercept> & lines)
  {
    // 절대 pose 후보는 가까운(오차 작은) 랜드마크로만 만든다. 실기 깊이 오차(3 m 이상 σ 20~50 cm)를 넣은
    // 시험에서 먼 랜드마크로 만든 후보가 180° 돌아간 위치로 리셋을 일으켰다.
    std::vector<Percept> near_percepts;
    for(const auto & p : percepts) if(p.pos.norm() <= cand_max_dist_) near_percepts.push_back(p);
    const auto cands = frameCandidates(near_percepts);
    // 대안 pose 버퍼에는 '모호하지 않은' 프레임만 넣는다: 랜드마크 reset_min_support 개 이상이 설명되고,
    // 최고 지지 후보들이 (필드 점대칭 짝을 빼면) 한 곳뿐일 때. 골대 근처에서 랜드마크 2 개만 보고 만든 후보는
    // 좌우가 바뀐 짝과 구분이 안 되는데, 그걸로 리셋해 로봇이 장외로 걸어 나간 일이 있었다 (4v4 화면 경기).
    bool frame_unique = !cands.empty() && cands[0].support >= reset_min_support_;
    for(size_t i = 1; frame_unique && i < cands.size(); ++i) {
      const Pose2D & a = cands[0].pose;
      const Pose2D & b = cands[i].pose;
      const Pose2D tw(normalizeAngle(b.rotation + M_PI), -b.translation);
      const bool same = (a.translation - b.translation).norm() < 0.3 && std::abs(normalizeAngle(a.rotation - b.rotation)) < 0.3;
      const bool twin = (a.translation - tw.translation).norm() < 0.3 && std::abs(normalizeAngle(a.rotation - tw.rotation)) < 0.3;
      if(!same && !twin) frame_unique = false;
    }
    if(frame_unique) {   // 대안 pose 버퍼 (현재 추정에 가장 가까운 후보 = 짝 중 가까운 쪽)
      const Pose2D cur = best().getPose();
      auto pd = [&](const Pose2D & a) {   // 위치 + 방향 (1 rad ≈ 1 m)
        return (a.translation - cur.translation).norm() + std::abs(normalizeAngle(a.rotation - cur.rotation));
      };
      const Candidate * pick = &cands[0];
      for(const auto & c : cands)
        if(pd(c.pose) < pd(pick->pose)) pick = &c;
      alt_buf_.push_back({now(), pick->pose, last_odo_});
    }
    while(!alt_buf_.empty() && (now() - alt_buf_.front().stamp).seconds() > alt_window_s_) alt_buf_.pop_front();

    int used_best = 0;
    const int best_id = best().id;
    for(auto & s : samples_) {
      double num = 0.0, den = 0.0;
      const Pose2D sp = s.getPose();
      // 프레임 절대 pose (B-Human absolute pose measurement 대응).
      // 가설 근처(위치 pose_assoc_dist, 방향 pose_assoc_ang 안)에 후보가 '하나뿐' 일 때만 쓴다.
      // 둘 이상이면 모호 → 건너뜀. 센터서클 X 3 개는 (−1.3, 0) 에서 보면 30° 간격이라 위치만 보고 고르면
      // 30° 돌아간 별칭 후보를 잡아 방향을 끌고 간다 (1v0 시험에서 ±30° 튐 반복 → 이 규칙으로 수정).
      if(!cands.empty()) {
        const Candidate * near = nullptr;
        int n_near = 0;
        for(const auto & c : cands) {
          if((c.pose.translation - sp.translation).norm() < pose_assoc_dist_ &&
             std::abs(normalizeAngle(c.pose.rotation - sp.rotation)) < pose_assoc_ang_) { near = &c; ++n_near; }
        }
        if(n_near == 1) {
          Eigen::Matrix3d C = Eigen::Vector3d(0.15 * 0.15, 0.15 * 0.15, 0.12 * 0.12).asDiagonal();
          s.poseSensorUpdate(Eigen::Vector3d(near->pose.translation.x(), near->pose.translation.y(), near->pose.rotation), C);
        }
        if(n_near <= 1) {
          num += vf_pose_ * (n_near == 1 ? 1.0 : 0.0);
          den += vf_pose_;
        }
      }
      int reg = 0;
      for(const auto & p : percepts) {
        Eigen::Vector2d m;
        if(registerPercept(s.getPose(), p, m)) { s.landmarkSensorUpdate(m, p.pos, p.cov); ++reg; }
      }
      if(!percepts.empty()) {
        num += vf_landmark_ * static_cast<double>(reg) / percepts.size();
        den += vf_landmark_;
      }
      int reg_lines = 0;
      for(const auto & lp : lines) {
        if(const FieldModel::Line * m = registerLine(s.getPose(), lp)) { updateByLine(s, lp, *m); ++reg_lines; }
      }
      if(!lines.empty()) {
        num += vf_line_ * static_cast<double>(reg_lines) / lines.size();
        den += vf_line_;
      }
      if(s.id == best_id) used_best = reg + reg_lines;
      if(den <= 0.0) continue;
      const double cur = num / den;
      s.validity = (s.validity * (validity_frames_ - 1) + cur) / validity_frames_;
      validities_updated_ = true;
    }
    return used_best;
  }

  void resampling()
  {
    if(avg_weighting_ <= 0.0) return;
    std::vector<Sample> old = samples_;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double next = u(rng_) * avg_weighting_, sum = 0.0;
    int j = 0;
    for(int i = 0; i < n_samples_; ++i) {
      sum += old[i].weighting;
      while(sum > next && j < n_samples_) {
        samples_[j] = old[i];
        if(j > 0 && samples_[j].id == samples_[j - 1].id) samples_[j].id = next_id_++;   // 복제는 새 id
        ++j;
        next += avg_weighting_;
      }
    }
    for(; j < n_samples_; ++j) samples_[j] = old[n_samples_ - 1];
  }

  bool sensorResetting(bool own_half)
  {
    if(alt_buf_.empty() || robot_state_ != "NORMAL") return false;
    if(alt_buf_.back().stamp.nanoseconds() <= last_alt_used_.nanoseconds()) return false;
    // 버퍼의 각 관측을 '지금' 으로 옮긴다: 관측 pose ∘ (그때 오도메트리⁻¹ ∘ 지금 오도메트리).
    // (B-Human AlternativeRobotPoseProvider 도 오도메트리로 옮겨 누적한다. 안 옮기면 회전 중에
    //  창(3 s) 평균이 1~2 s 늦은 방향이 되어 30~50° 틀린 가설을 주입했다 — 4v0 시험에서 확인)
    std::vector<Pose2D> moved;
    moved.reserve(alt_buf_.size());
    for(const auto & a : alt_buf_) moved.push_back(a.pose + (a.odo.inverse() + last_odo_));
    const Pose2D & latest = moved.back();
    int count = 0;
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    double sc = 0.0, ss = 0.0, t_first = 1e18, t_last = -1e18;
    for(size_t i = 0; i < moved.size(); ++i) {
      const auto & a = moved[i];
      if((a.translation - latest.translation).norm() < 0.5 &&
         std::abs(normalizeAngle(a.rotation - latest.rotation)) < 0.45) {
        ++count; acc.head<2>() += a.translation; sc += std::cos(a.rotation); ss += std::sin(a.rotation);
        const double ts = alt_buf_[i].stamp.seconds();
        t_first = std::min(t_first, ts); t_last = std::max(t_last, ts);
      }
    }
    // 관측 수와 함께 '시간 폭' 도 요구 — 한 순간(0.2 s) 의 같은 오인식이 여러 프레임 반복된 것을 거른다
    if(count < min_obs_reset_ || t_last - t_first < reset_min_span_) return false;
    Pose2D alt(std::atan2(ss, sc), acc.head<2>() / count);
    const Pose2D mirror(normalizeAngle(alt.rotation + M_PI), -alt.translation);   // 필드 점대칭 짝
    const Pose2D cur = best().getPose();
    auto compatible = [&](const Pose2D & p, const Pose2D & a) {
      return std::abs(p.translation.x() - a.translation.x()) < reset_trans_ &&
             std::abs(p.translation.y() - a.translation.y()) < reset_trans_ &&
             std::abs(normalizeAngle(p.rotation - a.rotation)) < reset_rot_;
    };
    // B-Human: 우리 진영 강제가 아니면 대칭 짝도 '맞음' 으로 본다 (구분 불가 → 리셋 안 함)
    if(compatible(cur, alt) || (!own_half && compatible(cur, mirror))) return false;
    if(own_half) {
      if(alt.translation.x() > 0.3) alt = mirror;
      if(alt.translation.x() > 0.3) return false;
    } else if((mirror.translation - cur.translation).norm() < (alt.translation - cur.translation).norm()) {
      alt = mirror;
    }
    int worst = 0;
    for(int i = 1; i < n_samples_; ++i) if(samples_[i].validity < samples_[worst].validity) worst = i;
    samples_[worst].init(alt, default_dev_, next_id_++, std::max(0.5, avg_weighting_));
    last_alt_used_ = alt_buf_.back().stamp;
    last_jump_ = now();
    RCLCPP_INFO(get_logger(), "sensor resetting: (%.2f,%.2f,%.0f°) → 대안 (%.2f,%.2f,%.0f°) [%d 관측]",
                cur.translation.x(), cur.translation.y(), cur.rotation * 57.3,
                alt.translation.x(), alt.translation.y(), alt.rotation * 57.3, count);
    return true;
  }

  // 출력용 최선 가설 (히스테리시스): 지금 내는 가설이 살아 있고 점수가 최고와 best_switch_margin 안이면 유지.
  // 맞는 가설과 대칭(180°) 가설의 점수가 비슷할 때 출력이 0.1 s 마다 뒤집히는 것을 막는다.
  const Sample & best()
  {
    const Sample & top = bestRaw();
    for(const auto & s : samples_)
      if(s.id == out_id_ && s.validity >= top.validity - best_switch_margin_) return s;
    out_id_ = top.id;
    return top;
  }

  const Sample & bestRaw() const
  {
    const Sample * b = &samples_[0];
    for(const auto & s : samples_) if(s.validity > b->validity) b = &s;
    return *b;
  }

  void publish(const Pose2D & odo_now, int used)
  {
    const Sample & b = best();
    const Pose2D p = b.getPose();
    hsl_interfaces::msg::RobotPoseEstimate m;
    m.header.stamp = now();
    m.header.frame_id = map_frame_;
    m.x = p.translation.x(); m.y = p.translation.y(); m.theta = p.rotation;
    for(int r = 0; r < 3; ++r) for(int c = 0; c < 3; ++c) m.covariance[r * 3 + c] = b.cov(r, c);
    m.validity = b.validity;
    bool unimodal = true;
    for(const auto & s : samples_)
      if((s.getPose().translation - p.translation).norm() > 2.0 ||
         std::abs(normalizeAngle(s.getPose().rotation - p.rotation)) > 1.47) { unimodal = false; break; }
    const double tdev = std::sqrt(std::max(b.cov(0, 0), b.cov(1, 1))), rdev = std::sqrt(b.cov(2, 2));
    m.quality = !unimodal ? m.QUALITY_POOR
              : (b.validity >= superb_validity_ && tdev < superb_trans_ && rdev < superb_rot_) ? m.QUALITY_SUPERB
              : m.QUALITY_OKAY;
    m.num_hypotheses = n_samples_;
    m.num_landmarks_used = used;
    m.last_jump = last_jump_;
    pose_pub_->publish(m);
    // map → odom = robot_pose ∘ odom_pose⁻¹
    const Pose2D mo = p + odo_now.inverse();
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = m.header.stamp;
    t.header.frame_id = map_frame_;
    t.child_frame_id = odom_frame_;
    t.transform.translation.x = mo.translation.x();
    t.transform.translation.y = mo.translation.y();
    tf2::Quaternion q; q.setRPY(0, 0, mo.rotation);
    t.transform.rotation.x = q.x(); t.transform.rotation.y = q.y(); t.transform.rotation.z = q.z(); t.transform.rotation.w = q.w();
    tf_pub_->sendTransform(t);
  }

  // params
  int n_samples_, validity_frames_, min_obs_reset_, walk_in_side_, player_number_;
  Pose2D default_dev_, walk_in_dev_, penalty_dev_, process_dev_, odo_dev_;
  Eigen::Vector2d odo_rot_dev_;
  double moved_dist_rot_noise_, moved_angle_rot_noise_, major_w_, minor_w_, base_validity_w_, vf_landmark_, vf_pose_;
  double reset_trans_, reset_rot_, superb_validity_, superb_trans_, superb_rot_, max_gyro_z_;
  double max_intersection_dev_, max_goalpost_dev_, pose_assoc_dist_, pose_assoc_ang_;
  RangeNoise range_noise_;
  double sigma_bearing_, max_percept_dist_, pair_dist_tol_, explain_tol_, alt_window_s_, max_tf_wait_;
  double walk_in_x_inset_, walk_in_outside_;
  FieldModel field_;
  std::map<int, std::vector<Eigen::Vector2d>> model_;
  std::string map_frame_, odom_frame_, base_frame_, robot_state_ = "NORMAL";
  // state
  std::vector<Sample> samples_;
  int next_id_ = 0;
  double avg_weighting_ = 0.0, gyro_z_ = 0.0;
  bool validities_updated_ = false, have_last_odo_ = false, was_penalized_ = false, ready_from_initial_ = true;
  int mirror_count_ = 0, mirror_frames_ = 20, reset_min_support_ = 3, out_id_ = -1;
  double cand_max_dist_ = 2.5, best_switch_margin_ = 0.05;
  double reset_min_span_ = 0.6;
  double mirror_min_validity_ = 0.5;
  hsl_interfaces::msg::BallEstimate::SharedPtr own_ball_;
  hsl_interfaces::msg::TeamState::SharedPtr team_;
  rclcpp::Subscription<hsl_interfaces::msg::BallEstimate>::SharedPtr ball_sub_;
  rclcpp::Subscription<hsl_interfaces::msg::TeamState>::SharedPtr team_sub_;
  Pose2D last_odo_;
  uint8_t game_state_ = 255;
  rclcpp::Time own_half_until_{0, 0, RCL_ROS_TIME}, last_alt_used_{0, 0, RCL_ROS_TIME}, last_jump_{0, 0, RCL_ROS_TIME};
  struct AltObs { rclcpp::Time stamp; Pose2D pose; Pose2D odo; };   // 관측 시점의 오도메트리 같이 저장
  std::deque<AltObs> alt_buf_;
  std::mt19937 rng_;
  bool use_lines_ = true;
  double line_corridor_ = 0.4, line_max_ang_ = 0.26, min_line_length_ = 0.4, vf_line_ = 1.0;
  std::vector<FieldModel::Line> model_lines_;
  hsl_interfaces::msg::LineSegmentArray::SharedPtr last_lines_;
  const hsl_interfaces::msg::LineSegmentArray * used_lines_ = nullptr;
  rclcpp::Subscription<hsl_interfaces::msg::LineSegmentArray>::SharedPtr line_sub_;
  vision_msgs::msg::Detection3DArray::SharedPtr last_det_;
  const vision_msgs::msg::Detection3DArray * used_det_ = nullptr;
  tf2_ros::Buffer tf_buf_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Publisher<hsl_interfaces::msg::RobotPoseEstimate>::SharedPtr pose_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_pub_;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr det_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<GameState>::SharedPtr gc_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr start_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr init_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SelfLocator>());
  rclcpp::shutdown();
  return 0;
}
