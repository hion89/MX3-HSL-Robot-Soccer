# 로봇에서 실행: 킥 시험 bag 의 모터 포화·추종 오차 (읽기 전용). python3 kick_bag_motors.py <bag_dir>
import sys, json, math, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
ty = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
d = {k: [] for k in ty}
while r.has_next():
    n, raw, t = r.read_next(); d[n].append((t, deserialize_message(raw, ty[n])))
t0 = d["/kick_trigger"][0][0]
for t, m in d["/motor_diagnostics"]:
    tt = (t - t0) / 1e9
    if 0.3 <= tt <= 2.4:
        j = json.loads(m.data)
        big = ["%s pwm%+d %+.1fA" % (x["name"].replace("_joint", ""), x["pwm"], x["current_ma"] / 1000)
               for x in j["motors"] if abs(x["pwm"]) > 500 or abs(x["current_ma"]) > 2500]
        v = min(x["voltage_v"] for x in j["motors"])
        print("%.2fs Vmin %.1f: %s" % (tt, v, ", ".join(big)))
def ser(top):
    return [((t - t0) / 1e9, dict(zip(m.name, m.position))) for t, m in d[top] if m.name]
g, s = ser("/joint_goals"), ser("/joint_states")
for j in ["l_pelvis_pitch_joint", "l_knee_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint", "l_pelvis_roll_joint",
          "r_pelvis_roll_joint", "r_pelvis_pitch_joint"]:
    row = []
    for tq in (0.75, 1.0, 1.25, 1.5, 1.75, 1.9):
        gv = min(g, key=lambda x: abs(x[0] - tq))[1].get(j); sv = min(s, key=lambda x: abs(x[0] - tq))[1].get(j)
        if gv is not None and sv is not None:
            row.append("%.2f g%+4.0f a%+4.0f" % (tq, math.degrees(gv), math.degrees(sv)))
    print("%-21s %s" % (j.replace("_joint", ""), " | ".join(row)))
