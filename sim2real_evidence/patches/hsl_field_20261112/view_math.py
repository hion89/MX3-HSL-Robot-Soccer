"""Pure display helpers. No ROS imports or control side effects."""
import json
import math
import re


def finite(values):
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError, OverflowError):
        return False


def robot_name(value):
    return re.sub(r'[^A-Za-z0-9_-]', '_', str(value).strip())[:32] or 'unknown_robot'


def behavior_data(text):
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('Behavior debug must be an object')
    target = data.get('target')
    if target is not None:
        if not isinstance(target, list) or len(target) != 3 or not finite(target):
            raise ValueError('Invalid map target')
        data['target'] = [float(v) for v in target]
    return data


def transform_xyz(xyz, translation, quaternion):
    """Transform a point using quaternion xyzw, independent of ROS helpers."""
    if not finite((*xyz, *translation, *quaternion)):
        raise ValueError('Non-finite transform')
    x, y, z = map(float, xyz)
    qx, qy, qz, qw = map(float, quaternion)
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm < 1e-9:
        raise ValueError('Zero transform quaternion')
    qx, qy, qz, qw = (v/norm for v in (qx, qy, qz, qw))
    # v' = v + 2*w*(q cross v) + 2*q cross (q cross v).
    ux, uy, uz = qy*z-qz*y, qz*x-qx*z, qx*y-qy*x
    vx, vy, vz = qy*uz-qz*uy, qz*ux-qx*uz, qx*uy-qy*ux
    return (x+2*qw*ux+2*vx+float(translation[0]),
            y+2*qw*uy+2*vy+float(translation[1]),
            z+2*qw*uz+2*vz+float(translation[2]))
