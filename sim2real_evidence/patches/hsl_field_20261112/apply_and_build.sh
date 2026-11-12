#!/usr/bin/env bash
set -eo pipefail
TASK_BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python3 "$TASK_BUNDLE/apply_field.py"
source /opt/ros/humble/setup.bash
source /home/robo04/sim2real_ws/install/setup.bash
cd /home/robo04/sim2real_ws
python3 "$TASK_BUNDLE/verify_field.py" --source
c++ -std=c++17 -I /usr/include/eigen3 -I src/hsl_src/hsl_world_model/include \
    "$TASK_BUNDLE/check_field.cpp" -o "$TASK_BUNDLE/check_field"
"$TASK_BUNDLE/check_field"
colcon build --base-paths src --symlink-install --packages-select hsl_world_model hsl_behavior
source install/setup.bash
python3 "$TASK_BUNDLE/verify_field.py"
printf '\nFIELD_UPDATE_BUILT_AND_VERIFIED\n'
