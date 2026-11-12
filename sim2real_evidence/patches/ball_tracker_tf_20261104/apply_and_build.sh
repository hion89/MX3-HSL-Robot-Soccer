#!/usr/bin/env bash
set -eo pipefail

TASK_BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
TASK_WS=/home/robo04/sim2real_ws

source /opt/ros/humble/setup.bash
source "$TASK_WS/install/setup.bash"
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1

/usr/bin/python3 "$TASK_BUNDLE/apply.py" --workspace "$TASK_WS" --apply

cd "$TASK_WS"
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAKEFLAGS=-j1
colcon build --base-paths src --packages-select hsl_world_model \
    --symlink-install --executor sequential --event-handlers console_direct+
source "$TASK_WS/install/setup.bash"

# The test process and its C++ child use domain 77. The real robot stays in 42.
/usr/bin/python3 "$TASK_BUNDLE/test_tf_transport.py"

printf '\nStarting the reviewed ball tracker in robot domain 42.\n'
exec ros2 run hsl_world_model ball_tracker --ros-args \
    --params-file "$TASK_WS/install/hsl_bringup/share/hsl_bringup/config/hsl.yaml"
