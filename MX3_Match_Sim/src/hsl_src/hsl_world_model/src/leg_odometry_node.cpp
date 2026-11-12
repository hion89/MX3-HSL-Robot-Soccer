// leg_odometry_node — 다리 기구학 오도메트리 (B-Human Odometer/WalkingEngine odometryOffset 대응).
//
// 기존 walk_odometry 는 /cmd_vel '명령' 을 적분해서, 로봇이 명령대로 안 움직이면(정렬 중 진동, 밀림,
// 킥 러너가 다리를 잡은 동안) 측위가 흘렀다 (시뮬에서 5 분에 11 m). 여기서는 실제 관절각으로
// 디딤발 위치를 계산해, 디딤발이 바닥에 고정돼 있다고 보고 몸통 이동량을 구한다.
//
//   1) /joint_states → KDL FK → 양 발바닥 중심점 (base_link 기준)
//   2) IMU 자세에서 yaw 를 뺀 기울기로 회전 → '수평 정렬된 몸통 좌표' 의 발 위치
//   3) 더 낮은 발 = 디딤발 (히스테리시스). 몸통 이동 = −(직전 프레임 디딤발의 위치 변화).
//   4) heading = IMU yaw (시작 시 0). 위치는 heading 으로 회전해 누적.
// 출력: /hsl/odom (nav_msgs/Odometry), TF odom → base_link
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>

#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/transform_broadcaster.h"

using std::placeholders::_1;

class LegOdometry : public rclcpp::Node
{
public:
  LegOdometry() : Node("hsl_leg_odometry")
  {
    urdf_path_ = declare_parameter<std::string>("urdf_path", "");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    left_foot_ = declare_parameter<std::string>("left_foot_link", "l_ankle_roll_link");
    right_foot_ = declare_parameter<std::string>("right_foot_link", "r_ankle_roll_link");
    const auto sole = declare_parameter<std::vector<double>>("sole_offset", {0.057, 0.0, -0.04});
    sole_ = KDL::Vector(sole[0], sole[1], sole[2]);
    hysteresis_ = declare_parameter<double>("stance_hysteresis", 0.004);
    max_step_ = declare_parameter<double>("max_step_per_update", 0.05);   // 이상치 방어 [m]
    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    const auto imu_topic = declare_parameter<std::string>("imu_topic", "/imu/data");

    std::ifstream f(urdf_path_);
    if(!f) throw std::runtime_error("URDF 를 열 수 없음: " + urdf_path_);
    std::stringstream ss; ss << f.rdbuf();
    KDL::Tree tree;
    if(!kdl_parser::treeFromString(ss.str(), tree)) throw std::runtime_error("URDF 파싱 실패");
    if(!tree.getChain(base_frame_, left_foot_, chain_l_) || !tree.getChain(base_frame_, right_foot_, chain_r_))
      throw std::runtime_error("다리 체인을 못 찾음");
    fk_l_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_l_);
    fk_r_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_r_);

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/hsl/odom", 10);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(imu_topic, rclcpp::SensorDataQoS(),
      std::bind(&LegOdometry::onImu, this, _1));
    js_sub_ = create_subscription<sensor_msgs::msg::JointState>("/joint_states", 10,
      std::bind(&LegOdometry::onJoints, this, _1));
    state_sub_ = create_subscription<std_msgs::msg::String>("/robot_state", 10,
      [this](std_msgs::msg::String::SharedPtr m) { robot_state_ = m->data; });
    RCLCPP_INFO(get_logger(), "leg odometry: %s (%u/%u joints), sole=(%.3f,%.3f,%.3f)", urdf_path_.c_str(),
                chain_l_.getNrOfJoints(), chain_r_.getNrOfJoints(), sole_.x(), sole_.y(), sole_.z());
  }

private:
  void onImu(const sensor_msgs::msg::Imu::SharedPtr m)
  {
    q_ = Eigen::Quaterniond(m->orientation.w, m->orientation.x, m->orientation.y, m->orientation.z);
    have_imu_ = true;
  }

  bool footPos(const KDL::Chain & chain, KDL::ChainFkSolverPos_recursive & fk, Eigen::Vector3d & out)
  {
    KDL::JntArray q(chain.getNrOfJoints());
    unsigned j = 0;
    for(const auto & seg : chain.segments) {
      if(seg.getJoint().getType() == KDL::Joint::None) continue;
      auto it = joints_.find(seg.getJoint().getName());
      if(it == joints_.end()) return false;
      q(j++) = it->second;
    }
    KDL::Frame fr;
    if(fk.JntToCart(q, fr) < 0) return false;
    const KDL::Vector p = fr * sole_;
    out = Eigen::Vector3d(p.x(), p.y(), p.z());
    return true;
  }

  void onJoints(const sensor_msgs::msg::JointState::SharedPtr m)
  {
    for(size_t i = 0; i < m->name.size() && i < m->position.size(); ++i) joints_[m->name[i]] = m->position[i];
    if(!have_imu_) return;
    Eigen::Vector3d pl, pr;
    if(!footPos(chain_l_, *fk_l_, pl) || !footPos(chain_r_, *fk_r_, pr)) return;

    // yaw 를 뺀 기울기만으로 수평 정렬 (heading 좌표)
    const Eigen::Matrix3d Rwb = q_.toRotationMatrix();
    const double yaw = std::atan2(Rwb(1, 0), Rwb(0, 0));
    const Eigen::Matrix3d Rtilt = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * Rwb;
    const Eigen::Vector3d hl = Rtilt * pl, hr = Rtilt * pr;

    if(!yaw0_set_) { yaw0_ = yaw; yaw0_set_ = true; }
    const double heading = std::atan2(std::sin(yaw - yaw0_), std::cos(yaw - yaw0_));

    // 이동량은 '직전 프레임의 디딤발' 로 계산한다 (그 발은 이번 프레임까지 바닥에 있었다).
    // 처음엔 디딤발이 바뀌는 프레임을 통째로 건너뛰었는데, 걸음마다 한 프레임씩 잃어 보행 이동량을 8 %,
    // 킥 중엔 15 % 적게 잡았다 (시뮬 정답 비교). B-Human 도 직전 지지발 기준으로 오도메트리를 낸다.
    const bool moving_ok = robot_state_.empty() || robot_state_ == "NORMAL";
    if(stance_ >= 0 && moving_ok) {
      const Eigen::Vector3d & cur = stance_ == 0 ? hl : hr;
      const Eigen::Vector3d & prev = stance_ == 0 ? prev_l_ : prev_r_;
      Eigen::Vector2d d = -(cur - prev).head<2>();    // 몸통 이동 (heading 좌표)
      if(d.norm() < max_step_) {
        const double c = std::cos(heading), s = std::sin(heading);
        x_ += c * d.x() - s * d.y();
        y_ += s * d.x() + c * d.y();
      }
    }
    if(stance_ < 0) stance_ = hl.z() < hr.z() ? 0 : 1;
    else if(stance_ == 0 && hr.z() < hl.z() - hysteresis_) stance_ = 1;
    else if(stance_ == 1 && hl.z() < hr.z() - hysteresis_) stance_ = 0;
    prev_l_ = hl;
    prev_r_ = hr;

    rclcpp::Time stamp = m->header.stamp;
    if(stamp.nanoseconds() == 0) stamp = now();
    const Eigen::Quaterniond qh(Eigen::AngleAxisd(heading, Eigen::Vector3d::UnitZ()));
    nav_msgs::msg::Odometry od;
    od.header.stamp = stamp;
    od.header.frame_id = odom_frame_;
    od.child_frame_id = base_frame_;
    od.pose.pose.position.x = x_;
    od.pose.pose.position.y = y_;
    od.pose.pose.position.z = -std::min(hl.z(), hr.z());
    od.pose.pose.orientation.w = qh.w(); od.pose.pose.orientation.z = qh.z();
    odom_pub_->publish(od);
    if(publish_tf_) {
      geometry_msgs::msg::TransformStamped t;
      t.header = od.header;
      t.child_frame_id = base_frame_;
      t.transform.translation.x = x_;
      t.transform.translation.y = y_;
      t.transform.translation.z = od.pose.pose.position.z;
      t.transform.rotation = od.pose.pose.orientation;
      tf_->sendTransform(t);
    }
  }

  std::string urdf_path_, base_frame_, odom_frame_, left_foot_, right_foot_, robot_state_;
  KDL::Chain chain_l_, chain_r_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_l_, fk_r_;
  KDL::Vector sole_;
  double hysteresis_, max_step_;
  bool publish_tf_;
  std::map<std::string, double> joints_;
  Eigen::Quaterniond q_ = Eigen::Quaterniond::Identity();
  bool have_imu_ = false, yaw0_set_ = false;
  double yaw0_ = 0.0, x_ = 0.0, y_ = 0.0;
  int stance_ = -1;
  Eigen::Vector3d prev_l_ = Eigen::Vector3d::Zero(), prev_r_ = Eigen::Vector3d::Zero();
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LegOdometry>());
  rclcpp::shutdown();
  return 0;
}
