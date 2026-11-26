# 로봇에서 실행 (읽기 전용): 킥 bag 의 /motor_fast 시계열 (트리거 기준). python3 kick_fast_series.py <bag>
import sys, math
import numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
ty = {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}
d = {k: [] for k in ty}
while r.has_next():
    n, raw, t = r.read_next(); d[n].append((t, deserialize_message(raw, ty[n])))
t0 = d["/kick_trigger"][0][0]
s = lambda t: (t - t0) / 1e9
mf = [(s(t), dict(zip(m.name, zip(m.position, m.effort)))) for t, m in d["/motor_fast"]]
js = [(s(t), dict(zip(m.name, zip(m.position, m.velocity)))) for t, m in d["/joint_states"] if m.name]
jg = [(s(t), dict(zip(m.name, m.position))) for t, m in d["/joint_goals"] if m.name]
near = lambda ser, tq: min(ser, key=lambda x: abs(x[0] - tq))[1]
J = ["l_pelvis_roll_joint", "l_pelvis_pitch_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint", "l_knee_pitch_joint",
     "r_pelvis_roll_joint", "r_pelvis_pitch_joint"]
print("t[s]  " + "  ".join("%-17s" % j.replace("_joint", "").replace("pelvis", "hip") for j in J) + "   (PWM% / A / goal-act deg / vel deg/s)")
for tq in np.arange(0.6, 2.61, 0.1):
    a, b, g = near(mf, tq), near(js, tq), near(jg, tq)
    row = []
    for j in J:
        pw, cu = a.get(j, (0, 0)); pos, vel = b.get(j, (0, 0)); gg = g.get(j, pos)
        row.append("%+4.0f %+4.1f %+4.0f %+4.0f" % (100 * pw, cu, math.degrees(gg - pos), math.degrees(vel)))
    print("%.1f   " % tq + "  ".join("%-17s" % x for x in row))
sat = {j: [x[0] for x in mf if 0 <= x[0] <= 3 and abs(x[1].get(j, (0, 0))[0]) >= 0.95] for j in J}
for j in J:
    if sat[j]: print("PWM>=95%%: %-20s %d samples, %.2f-%.2f s" % (j, len(sat[j]), min(sat[j]), max(sat[j])))
st = []
for t, m in d["/robot_state"]:
    if not st or st[-1][1] != m.data: st.append((round(s(t), 2), m.data))
print("robot_state:", st, "| kick_done:", [round(s(t), 2) for t, _ in d["/kick_done"]])
def rp(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    return math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))), math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x)))))
im = [(s(t), *rp(m.orientation)) for t, m in d["/imu/data"]]
b0 = [x for x in im if -1 < x[0] < 0]; r0 = np.mean([x[1] for x in b0]); p0 = np.mean([x[2] for x in b0])
for a_, b_ in ((0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 5)):
    w = [x for x in im if a_ <= x[0] < b_]
    if w: print("  %.1f-%.1fs roll %+6.1f/%+6.1f pitch %+6.1f/%+6.1f" % (a_, b_, max(x[1]-r0 for x in w), min(x[1]-r0 for x in w), max(x[2]-p0 for x in w), min(x[2]-p0 for x in w)))
