#!/usr/bin/env bash
set -eo pipefail
TASK_BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source /opt/ros/humble/setup.bash
source /home/robo04/sim2real_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
if [ -z "${DISPLAY:-}" ]; then
    printf 'DISPLAY is empty. Open a new MobaXterm SSH session with X11-forwarding enabled.\n' >&2
    exit 1
fi
if ! ros2 pkg prefix rviz2 >/dev/null 2>&1; then
    printf 'RViz missing. Install once: sudo apt install ros-humble-rviz2\n' >&2
    exit 1
fi
TASK_PROFILE=$(timeout 12s ros2 param get /hsl_self_localizer field_profile)
printf 'LOCALIZER: %s\nDISPLAY=%s\n' "$TASK_PROFILE" "$DISPLAY"
case "$TASK_PROFILE" in
    'String value is: practice') ;;
    *) printf 'Expected localizer field_profile=practice; observer not started.\n' >&2; exit 1 ;;
esac
# Do not copy the NumPy override from the policy shell into this native GUI.
export QT_X11_NO_MITSHM=1
exec ros2 launch "$TASK_BUNDLE/view.launch.py"
