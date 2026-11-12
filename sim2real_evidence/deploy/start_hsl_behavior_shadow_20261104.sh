#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/robo04/sim2real_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1

if pgrep -f '^/usr/bin/python3 .*/hsl_behavior/lib/hsl_behavior/behavior( |$)' >/dev/null; then
    printf 'An hsl_behavior process already exists; inspect it before starting another.\n' >&2
    exit 1
fi

mkdir -p /home/robo04/sim2real_ws/runtime_logs
TASK_LOG="/home/robo04/sim2real_ws/runtime_logs/hsl_behavior_shadow_$(date +%Y%m%d_%H%M%S).log"
printf 'LOG=%s\n' "$TASK_LOG"

ros2 run hsl_behavior behavior --ros-args \
  --params-file /home/robo04/sim2real_ws/install/hsl_bringup/share/hsl_bringup/config/hsl.yaml \
  -p field_profile:=practice \
  -p player_number:=4 \
  -p default_game_state:=playing \
  -p enable_kick:=false \
  -p neck_speed_rpm:=0.0 \
  -p stop_via_teleop_topic:=false \
  -p kick_node:=hsl_shadow_kick_target \
  -r /game_state:=/hsl/shadow/game_state \
  -r cmd_vel_behavior:=/hsl/shadow/cmd_vel_behavior \
  -r cmd_vel_teleop:=/hsl/shadow/cmd_vel_teleop \
  -r /neck/yaw_deg:=/hsl/shadow/neck/yaw_deg \
  -r /neck/pitch_deg:=/hsl/shadow/neck/pitch_deg \
  -r /neck/speed_rpm:=/hsl/shadow/neck/speed_rpm \
  -r /kick_trigger:=/hsl/shadow/kick_trigger \
  -r /hsl/strategy_state:=/hsl/shadow/strategy_state \
  -r /hsl/behavior_debug:=/hsl/shadow/behavior_debug \
  2>&1 | tee "$TASK_LOG"
