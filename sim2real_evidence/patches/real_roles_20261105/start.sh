#!/usr/bin/env bash
set -eo pipefail
TASK_BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source /opt/ros/humble/setup.bash
source /home/robo04/sim2real_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
export PYTHONPATH="/home/robo04/sim2real_ws/.policy-python/numpy1:${PYTHONPATH:-}"
/usr/bin/python3 - "$TASK_BUNDLE" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import apply
apply.ensure_stopped()
PY
cd /home/robo04/sim2real_ws
mkdir -p runtime_logs
ros2 launch hsl_bringup real_role_trial.launch.py \
  2>&1 | tee "runtime_logs/real_roles_$(date +%Y%m%d_%H%M%S).log"
