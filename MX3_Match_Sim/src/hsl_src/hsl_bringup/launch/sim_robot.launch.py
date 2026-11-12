#!/usr/bin/env python3
"""시뮬 로봇 1 대 = mx3_src 하위 계층 (하드웨어는 MuJoCo 브리지) + hsl 상위 계층.

  ROS_DOMAIN_ID=11 ros2 launch hsl_bringup sim_robot.launch.py team_number:=35 player_number:=1

하위 계층 (mx3_src 그대로, humanoid_sim/sim_robot.launch.py 와 같은 방식):
  control.launch.py (보행·킥 러너, 횡보정, fall_manager) + twist_mux(use_stamped=false, Jazzy 용),
  neck_control, perception_bringup (RSP 만 — 카메라·YOLO 는 브리지가 정답+노이즈로 대신), game_controller
빠진 것 (hsl 이 대체): vision/EKF 측위, walk_odometry, pose_goal, neck_ball_tracker, ball_follow,
  visual_servo, strategy, team_comm, depth_obstacle
robot 인자 = robot_params.yaml 섹션. 기본 'sim' = common 만 (실기 기체별 vy_trim 등은 시뮬에 안 맞는다).
보행 정책 = 최종 v0_6 model_13500 (hsl_bringup/models/walk_v0_6_13500.onnx, ~/Documents/XM_Walking_Policy 와 같은 가중치).
  mx3_src 러너 사본 models/walk_policy.onnx 는 v0_8 18400 이라 쓰지 않는다 (2026-11-01). policy_variant 는 기본 v0_2 그대로
  (관측 240, 러너 코드상 v1s 와 같은 경로, 걸음 2.0 Hz — v0_6 학습 범위 1.7~2.4 안).
킥 정책 = v1_5b model_22300 (hsl_bringup/models/walkkick_v1_5b_22300.onnx, 2026-11-02 부터). 경기 시뮬 5 판씩 비교에서
  mx3_src 러너 사본 models/walkkick_v1_policy.onnx (W90B70 18200) 보다 킥 중 넘어짐 8.5 → 5.3 %, 공 이동 1.85 → 2.20 m.
  실기 러너 사본은 아직 18200 — kick_actor_path:=runner 이면 그걸 쓴다 (Jazzy ros2 launch 는 빈 값 ':=' 을 거부). 다른 모델은 관측 265 같은 walk_kick v1 계열만.
  control.launch.py 에 경로 인자가 없어 킥 노드 이름으로 매긴 params 파일을 이 include 범위에만 건다 — 다른 노드는
  키가 안 맞아 영향 없음.
보행 러너 CSV 로그 (~/policy_logs) 는 시뮬에서 끈다 (2026-11-02, 판마다 8 대가 쓰여 이틀에 7.6 GB). 같은 params 파일로 덮고,
  켜려면 walk_csv_log:=true. 실기 control.launch.py 는 그대로 켜져 있다.
"""
import hashlib
import os
import tempfile

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParametersFromFile


def _inc(pkg, f, args=None):
    path = os.path.join(get_package_share_directory(pkg), "launch", f)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=(args or {}).items())


def _runner_overrides(context):
    params = {}
    if LaunchConfiguration("walk_csv_log").perform(context).lower() != "true":
        # enable_csv_log 은 control.launch.py 가 노드 dict 로 주는 값이다. launch 는 이 dict 를 '/**' 키 파일로 이 파일 뒤에 붙이고,
        # rcl 은 노드 이름 항목을 처음 나온 순서대로 적용한다 (뒤가 이김, Jazzy 실측). 그래서 빈 '/**' 를 먼저 세워 자리를
        # 앞으로 당기고, 그 뒤 '/**/actor_onnx_walking_node' 로 덮는다. control.launch.py 노드들은 dict 만 받아서
        # '/**' 가 앞에 와도 다른 값은 안 바뀐다 (4 노드 파라미터 덤프 비교로 확인 — 노드별 yaml 이 생기면 다시 볼 것).
        params["/**"] = {"ros__parameters": {}}
        params["/**/actor_onnx_walking_node"] = {"ros__parameters": {"enable_csv_log": False}}
    path = LaunchConfiguration("kick_actor_path").perform(context)
    if path not in ("", "runner"):
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise RuntimeError(f"kick_actor_path 없음: {path}")
        params["/actor_onnx_walkkick_v1_node"] = {"ros__parameters": {"actor_path": path}}
    if not params:
        return []
    # 내용이 같으면 같은 파일 (로봇 8 대 × 판마다 /tmp 에 쌓이지 않게). 동시에 쓰니 rename 으로 바꿔 넣는다.
    text = yaml.safe_dump(params, sort_keys=False)
    name = os.path.join(tempfile.gettempdir(), f"hsl_runner_params_{hashlib.md5(text.encode()).hexdigest()[:12]}.yaml")
    with tempfile.NamedTemporaryFile("w", dir=tempfile.gettempdir(), prefix=".hsl_runner_params_", delete=False) as f:
        f.write(text)
    os.replace(f.name, name)
    return [SetParametersFromFile(name)]


def generate_launch_description():
    robot = LaunchConfiguration("robot")
    team = LaunchConfiguration("team_number")
    player = LaunchConfiguration("player_number")
    out = LaunchConfiguration("node_output")
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="sim"),
        DeclareLaunchArgument("team_number", default_value="35"),
        DeclareLaunchArgument("player_number", default_value="1"),
        DeclareLaunchArgument("field_profile", default_value="s_field"),
        DeclareLaunchArgument("node_output", default_value="screen"),
        DeclareLaunchArgument("team_send_rate_hz", default_value="", description="비우면 hsl.yaml"),
        DeclareLaunchArgument("team_event_send", default_value="", description="비우면 hsl.yaml"),
        DeclareLaunchArgument("walk_actor_path", description="보행 정책 ONNX (기본 v0_6 13500)",
                              default_value=os.path.join(get_package_share_directory("hsl_bringup"), "models",
                                                         "walk_v0_6_13500.onnx")),
        DeclareLaunchArgument("kick_actor_path", description="킥 정책 ONNX (기본 v1_5b 22300, runner = 러너 사본 18200)",
                              default_value=os.path.join(get_package_share_directory("hsl_bringup"), "models",
                                                         "walkkick_v1_5b_22300.onnx")),
        DeclareLaunchArgument("kick_duration", default_value="2.0",
                              description="킥 러너 구동 시간 [s] (실기 기본 2.0). 보행 대기 상한 = 이 값 + 1.0, hsl kick_timeout 6.0 보다 작게"),
        DeclareLaunchArgument("walk_csv_log", default_value="false",
                              description="보행 러너 CSV 로그 (~/policy_logs). 시뮬 기본 끔"),
        Node(package="twist_mux", executable="twist_mux", name="twist_mux", output=out,
             parameters=[os.path.join(get_package_share_directory("humanoid_bringup"), "config",
                                      "twist_mux_topics.yaml"), {"use_stamped": False}],
             remappings=[("cmd_vel_out", "/cmd_vel_raw")]),
        GroupAction(scoped=True, actions=[
            OpaqueFunction(function=_runner_overrides),
            _inc("humanoid_bringup", "control.launch.py", {
                "robot": robot, "use_teleop": "false", "use_kick": "true", "use_twist_mux": "false",
                "kick_resume_hold": "0.0", "kick_ball_x": "0.30", "kick_cmd_vx": "0.30",
                "kick_duration": LaunchConfiguration("kick_duration"), "kick_start_hold": "0.3",
                "kick_strict_gate": "true",
                "gravity_source": "derive", "fall_manager_imu_topic": "/imu/data_50hz", "node_output": out,
                "walk_actor_path": LaunchConfiguration("walk_actor_path")})]),
        Node(package="humanoid_serial_bridge", executable="neck_control_node", name="neck_control",
             output=out, parameters=[{
                 "yaw_deg_min": -90.0, "yaw_deg_max": 90.0, "pitch_deg_min": -60.0, "pitch_deg_max": -25.0,
                 "yaw_joint_name": "neck_yaw_joint", "pitch_joint_name": "neck_pitch_joint",
                 "yaw_sign": 1.0, "pitch_sign": -1.0,
                 # 실기 dynamixel_system 기본값 10 rpm 그대로 — hsl_behavior 가 neck_speed_rpm(15) 으로 덮어쓰는 경로까지 시험
                 "default_speed_rpm": 10.0, "speed_rpm_max": 30.0}]),
        _inc("humanoid_perception", "perception_bringup.launch.py",
             {"enable_camera": "false", "enable_yolo": "false"}),
        _inc("humanoid_game_controller", "game_controller.launch.py",
             {"robot": robot, "team_number": team, "player_number": player}),
        _inc("hsl_bringup", "hsl_stack.launch.py", {
            "team_number": team, "player_number": player,
            "field_profile": LaunchConfiguration("field_profile"), "node_output": out,
            "team_send_rate_hz": LaunchConfiguration("team_send_rate_hz"),
            "team_event_send": LaunchConfiguration("team_event_send")}),
    ])
