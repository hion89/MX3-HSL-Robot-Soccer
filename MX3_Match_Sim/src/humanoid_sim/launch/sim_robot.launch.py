#!/usr/bin/env python3
"""시뮬 경기용 로봇 1 대 스택 — match.launch.py 와 같은 구성에서 하드웨어만 뺀다.

  ROS_DOMAIN_ID=2 ros2 launch humanoid_sim sim_robot.launch.py robot:=robo02

match.launch.py 와의 차이 (나머지 노드·인자는 동일):
  - bringup.launch.py 대신 control.launch.py 를 직접 include → robot 인자를 넘길 수 있다
    (bringup 은 robot 을 안 넘겨서 hostname 으로 robot_params 섹션을 고른다)
  - dynamixel_system.launch.py 제외 → 하드웨어는 MuJoCo 브리지(sim_bridge)가 대신한다.
    단 그 안의 소프트웨어 어댑터 neck_control / neck_scan 은 여기서 직접 띄운다.
  - perception: 카메라·YOLO OFF (브리지가 /perception/detections_3d 를 정답 위치로 낸다)
  - depth_obstacle 제외 (깊이 영상이 없다)
  - game_controller / strategy / team_comm 에 robot 인자 전달 (hostname 대신)
  - 재진입 x_inset 을 reentry_positions.yaml 에서 robot 이름으로 찾아 넘긴다 (vision_localization 은 hostname 만 본다)
  - ekf_localization.launch.py 대신 그 안의 3 개를 직접 띄운다 — walk_odometry 에 시뮬 보정(odom_calib_sim.yaml)을 얹기 위해
  - twist_mux 는 control.launch 의 것을 끄고(use_twist_mux:=false) 같은 yaml·리매핑으로 여기서 띄운다.
    차이는 use_stamped=false 하나 — Jazzy twist_mux 는 기본 true(TwistStamped)라 스택의 Twist 와 끊긴다
    (Humble 실기에는 이 파라미터가 없다).
    ★ 전역 SetParameter 로 주면 안 된다: '/**' 키가 먼저 생겨 yaml 의 노드별 값이 런치 dict 를 이긴다
      (field_landmark_localizer 가 s_field 대신 yaml 의 practice 로 뜨는 것을 확인함).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
import yaml
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParametersFromFile


def _inc(pkg, f, args=None):
    path = os.path.join(get_package_share_directory(pkg), "launch", f)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                    launch_arguments=(args or {}).items())


def _reentry_inset(robot):
    cfg = os.path.join(get_package_share_directory("humanoid_localization"), "config", "reentry_positions.yaml")
    with open(cfg, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return float((data.get(robot) or {}).get("x_inset_m", 0.0))


def _neck_tracker(context):
    """neck_ball_tracker.launch.py 와 같은 노드. neck_yaw_lock_deg 를 주면 yaml 위에 그 값만 덮어쓴다
    (트래커는 실행 중 파라미터 변경을 받지 않아서 기동 시에 준다). −71 이면 yaw 고정 끔."""
    lock = LaunchConfiguration("neck_yaw_lock_deg").perform(context).strip()
    if not lock:
        return _inc("humanoid_serial_bridge", "neck_ball_tracker.launch.py")
    yml = os.path.join(get_package_share_directory("humanoid_serial_bridge"), "config", "neck_ball_tracker.yaml")
    return Node(package="humanoid_serial_bridge", executable="neck_ball_tracker_node",
                name="neck_ball_tracker_node", output="screen",
                parameters=[yml, {"yaw_lock_pitch_deg": float(lock)}])


def _stack(context):
    robot = LaunchConfiguration("robot")
    field = LaunchConfiguration("field_profile")
    rviz = LaunchConfiguration("rviz")
    inset = str(_reentry_inset(robot.perform(context)))
    loc_share = get_package_share_directory("humanoid_localization")
    sim_share = get_package_share_directory("humanoid_sim")
    return [
        Node(package="twist_mux", executable="twist_mux", name="twist_mux", output="screen",
             parameters=[os.path.join(get_package_share_directory("humanoid_bringup"), "config",
                                      "twist_mux_topics.yaml"), {"use_stamped": False}],
             remappings=[("cmd_vel_out", "/cmd_vel_raw")]),

        # 1) 제어 — bringup.launch.py 기본값 그대로 (use_kick=true 는 match 와 같음)
        #    보행 러너 CSV 로그만 끈다 (config/runner_sim.yaml, 노드 이름 키라 이 범위의 다른 노드는 영향 없음)
        GroupAction(scoped=True, actions=[
            SetParametersFromFile(os.path.join(sim_share, "config", "runner_sim.yaml")),
            _inc("humanoid_bringup", "control.launch.py", {
                "robot": robot, "use_teleop": "false", "use_kick": "true", "use_twist_mux": "false",
                "kick_resume_hold": "0.0", "kick_ball_x": "0.30", "kick_cmd_vx": "0.30",
                "kick_duration": "2.0", "kick_start_hold": "0.3", "kick_strict_gate": "true",
                "gravity_source": "derive", "fall_manager_imu_topic": "/imu/data_50hz"})]),

        # 1-b) dynamixel_system.launch.py 의 소프트웨어 어댑터 (기본값 그대로)
        Node(package="humanoid_serial_bridge", executable="neck_control_node", name="neck_control",
             output="screen", parameters=[{
                 "yaw_deg_min": -90.0, "yaw_deg_max": 90.0, "pitch_deg_min": -60.0, "pitch_deg_max": -25.0,
                 "yaw_joint_name": "neck_yaw_joint", "pitch_joint_name": "neck_pitch_joint",
                 "yaw_sign": 1.0, "pitch_sign": -1.0, "default_speed_rpm": 10.0, "speed_rpm_max": 30.0}]),
        Node(package="humanoid_serial_bridge", executable="neck_scan_node", name="neck_scan",
             output="screen", parameters=[{
                 "yaw_min_deg": -60.0, "yaw_max_deg": 60.0, "sweep_period_sec": 3.0, "moving_speed_raw": 90}]),

        # 2) 측위 (match 와 같은 field_profile)
        # ekf_localization.launch.py 와 같은 3 개 (비전 측위 / walk_odometry / ekf) — walk_odometry 에만
        # 시뮬 보정(config/odom_calib_sim.yaml)을 원본 위에 얹는다 (같은 노드 키라 뒤 파일이 이긴다).
        _inc("humanoid_localization", "vision_localization.launch.py",
             {"publish_map_odom_identity": "false", "start_field_lines": "true", "rviz": rviz,
              "field_profile": field, "reentry_x_inset_m": inset}),
        Node(package="humanoid_localization", executable="walk_odometry_node", name="walk_odometry_node",
             output="screen",
             parameters=[os.path.join(loc_share, "config", "odom_calib.yaml"),
                         os.path.join(sim_share, "config", "odom_calib_sim.yaml")],
             remappings=[("/initialpose", "/initialpose_walk_odom_disabled")]),
        Node(package="robot_localization", executable="ekf_node", name="ekf_filter_node", output="screen",
             parameters=[os.path.join(loc_share, "config", "ekf.yaml")],
             remappings=[("set_pose", "/initialpose")]),
        # 3) 인지 — RSP + 공 map 투영만 (카메라·YOLO 는 브리지가 대신)
        _inc("humanoid_perception", "perception_bringup.launch.py",
             {"enable_camera": "false", "enable_yolo": "false"}),
        # 4)~8) 행동
        _inc("humanoid_localization", "pose_goal.launch.py", {"avoid_enable": "true"}),
        _neck_tracker(context),
        _inc("humanoid_localization", "ball_follow.launch.py"),
        _inc("humanoid_localization", "vision_servo.launch.py"),
        # 9)~11) 경기
        _inc("humanoid_game_controller", "game_controller.launch.py", {"robot": robot}),
        _inc("humanoid_strategy", "strategy.launch.py", {"robot": robot, "field_profile": field}),
        _inc("humanoid_strategy", "team_comm.launch.py", {"robot": robot}),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="robo02"),
        DeclareLaunchArgument("field_profile", default_value="s_field"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("neck_yaw_lock_deg", default_value="",
                              description="비우면 yaml 값(-40). -71 = 숙여도 yaw 고정 안 함"),
        OpaqueFunction(function=_stack),
    ])
