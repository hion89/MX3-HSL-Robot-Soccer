#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/robo04/sim2real_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
cd /home/robo04/sim2real_ws
mkdir -p runtime_logs
# Keep the lock for the lifetime of the foreground launch, including the log pipe.
exec 9>/tmp/robo04_real_base.lock
flock -n 9 || { printf 'Base launch is already running.\n' >&2; exit 1; }
TASK_LOG="runtime_logs/base_$(date +%Y%m%d_%H%M%S).log"
printf 'BASE_LOG=%s/%s\n' "$PWD" "$TASK_LOG"
ros2 launch hsl_bringup real_base.launch.py 2>&1 | tee "$TASK_LOG"
