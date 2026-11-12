#!/usr/bin/env bash
set -eo pipefail
TASK_BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source /opt/ros/humble/setup.bash
source "${SIM2REAL_WS:-$HOME/sim2real_ws}/install/setup.bash"
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
if [ -z "${DISPLAY:-}" ]; then
    printf 'DISPLAY is empty. Enable MobaXterm X11 forwarding and X server.\n' >&2
    exit 1
fi
ros2 pkg prefix rviz2 >/dev/null
TASK_PROFILE=$(timeout 12s ros2 param get /hsl_self_localizer field_profile)
case "$TASK_PROFILE" in
    'String value is: practice') export HSL_VIEW_FIELD_PROFILE=practice ;;
    'String value is: s_field') export HSL_VIEW_FIELD_PROFILE=s_field ;;
    *) printf 'Cannot read supported field profile: %s\n' "$TASK_PROFILE" >&2; exit 1 ;;
esac
export QT_X11_NO_MITSHM=1
printf 'FIELD=%s ROBOT_NAME=%s DISPLAY=%s\n' "$HSL_VIEW_FIELD_PROFILE" "${ROBOT_NAME:-auto-hostname}" "$DISPLAY"
exec ros2 launch "$TASK_BUNDLE/view.launch.py"
