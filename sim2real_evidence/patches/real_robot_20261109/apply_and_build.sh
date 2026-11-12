#!/usr/bin/env bash
set -eo pipefail
TASK_BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source /opt/ros/humble/setup.bash
source /home/robo04/sim2real_ws/install/setup.bash
cd /home/robo04/sim2real_ws
/usr/bin/python3 "$TASK_BUNDLE/apply.py" --apply
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAKEFLAGS=-j1
colcon build --base-paths src \
  --packages-select humanoid_policy_runner humanoid_bringup hsl_behavior hsl_bringup \
  --symlink-install --executor sequential --event-handlers console_direct+
printf '\nREAL_ROLE_INTEGRATION_INSTALLED\n'
